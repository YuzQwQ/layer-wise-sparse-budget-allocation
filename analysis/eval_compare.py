"""
Cross-experiment result comparison.

Reads training loss curves from each experiment's metrics log (wandb CSV or
torchtitan JSON logs), computes perplexity on a held-out set, and produces a
single comparison table + plot.

Usage:
    # After all experiments finish:
    python analysis/eval_compare.py \
        --exp_dirs \
            exp/baseline_uniform_k16 \
            exp/phase1_monotone_inc \
            exp/phase1_monotone_dec \
            exp/phase1_u_shape \
            exp/phase2a_tool_w0 \
            exp/phase2b_gate_sched \
        --tokenizer fla-hub/transformer-1.3B-100B \
        --eval_dataset HuggingFaceFW/fineweb-edu \
        --eval_dataset_name sample-100BT \
        --n_eval_samples 512 \
        --seq_len 8192 \
        --output_dir analysis/comparison
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import native_sparse_attention  # noqa


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp_dirs",          nargs="+", required=True)
    p.add_argument("--tokenizer",         required=True)
    p.add_argument("--eval_dataset",      default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--eval_dataset_name", default="sample-100BT")
    p.add_argument("--n_eval_samples",    type=int, default=512)
    p.add_argument("--seq_len",           type=int, default=8192)
    p.add_argument("--batch_size",        type=int, default=1)
    p.add_argument("--output_dir",        default="analysis/comparison")
    p.add_argument("--device",            default="cuda")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Loss-curve helpers                                                           #
# --------------------------------------------------------------------------- #

def load_loss_curve(exp_dir: str):
    """Try to load step→loss from torchtitan metrics JSON files."""
    exp_path = Path(exp_dir)
    steps, losses = [], []

    # torchtitan writes one JSON per log step inside the dump folder
    json_files = sorted(exp_path.glob("metrics_step_*.json"))
    if json_files:
        for jf in json_files:
            try:
                d = json.loads(jf.read_text())
                steps.append(d.get("step", 0))
                losses.append(d.get("optim/global_avg_loss", float("nan")))
            except Exception:
                continue
        return steps, losses

    # Fallback: look for a single metrics.jsonl
    jsonl = exp_path / "metrics.jsonl"
    if jsonl.exists():
        for line in jsonl.read_text().splitlines():
            try:
                d = json.loads(line)
                steps.append(d.get("step", len(steps)))
                losses.append(d.get("loss", float("nan")))
            except Exception:
                continue
        return steps, losses

    return [], []


# --------------------------------------------------------------------------- #
# Perplexity evaluation                                                        #
# --------------------------------------------------------------------------- #

def collect_eval_samples(args, tok):
    ds = load_dataset(
        args.eval_dataset, name=args.eval_dataset_name,
        split="train", streaming=True, trust_remote_code=True,
    )
    samples, buf = [], []
    # Use a fixed offset to avoid overlapping with training data seen early
    skip = 50_000
    skipped = 0
    for ex in ds:
        ids = tok.encode(ex.get("text") or ex.get("content", ""),
                         add_special_tokens=False)
        if skipped < skip:
            skipped += len(ids)
            continue
        buf.extend(ids)
        while len(buf) >= args.seq_len:
            samples.append(torch.tensor(buf[:args.seq_len], dtype=torch.long))
            buf = buf[args.seq_len:]
        if len(samples) >= args.n_eval_samples:
            break
    return samples[:args.n_eval_samples]


@torch.no_grad()
def evaluate_ppl(model, samples, args, device):
    model.eval()
    total_nll, total_tokens = 0.0, 0
    for i in range(0, len(samples), args.batch_size):
        batch = torch.stack(samples[i:i + args.batch_size]).to(device)
        labels = batch.clone()
        out = model(input_ids=batch, labels=labels)
        # loss is mean NLL over non-padding tokens
        total_nll += out.loss.item() * batch.numel()
        total_tokens += batch.numel()
    ppl = float(np.exp(total_nll / total_tokens))
    return ppl


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    print("Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    print(f"Collecting {args.n_eval_samples} eval samples...")
    eval_samples = collect_eval_samples(args, tok)

    results = {}   # exp_name -> {ppl, budget_mode, mean_k, steps, losses}

    for exp_dir in args.exp_dirs:
        exp_name = Path(exp_dir).name
        print(f"\n=== {exp_name} ===")

        # Find latest checkpoint
        ckpt_dirs = sorted(Path(exp_dir).glob("step_*"))
        if not ckpt_dirs:
            print(f"  No checkpoint found in {exp_dir}, skipping.")
            continue
        ckpt = str(ckpt_dirs[-1])
        print(f"  Checkpoint: {ckpt}")

        model = AutoModelForCausalLM.from_pretrained(
            ckpt, torch_dtype=torch.bfloat16
        ).to(device).eval()

        cfg = model.config
        budget_mode = getattr(cfg, "budget_mode", "uniform")
        bc = getattr(cfg, "block_counts", 16)
        mean_k = float(np.mean(bc)) if isinstance(bc, list) else float(bc)

        ppl = evaluate_ppl(model, eval_samples, args, device)
        steps, losses = load_loss_curve(exp_dir)

        results[exp_name] = {
            "ppl": ppl,
            "budget_mode": budget_mode,
            "mean_k": mean_k,
            "steps": steps,
            "losses": losses,
        }
        print(f"  PPL={ppl:.3f}  budget_mode={budget_mode}  mean_k={mean_k:.1f}")

        del model
        torch.cuda.empty_cache()

    if not results:
        print("No results collected.")
        return

    # ----------------------------------------------------------------------- #
    # Plot 1: training loss curves                                             #
    # ----------------------------------------------------------------------- #
    fig, ax = plt.subplots(figsize=(12, 5))
    for name, r in results.items():
        if r["steps"]:
            ax.plot(r["steps"], r["losses"], label=name, alpha=0.85)
    ax.set_xlabel("Step"); ax.set_ylabel("Train loss")
    ax.set_title("Training loss curves")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "loss_curves.png"), dpi=150)
    plt.close()

    # ----------------------------------------------------------------------- #
    # Plot 2: PPL bar chart                                                    #
    # ----------------------------------------------------------------------- #
    names = list(results.keys())
    ppls  = [results[n]["ppl"] for n in names]
    colors = ["#4C72B0" if "baseline" in n
              else "#DD8452" if "phase1" in n
              else "#55A868" if "phase2a" in n
              else "#C44E52"
              for n in names]

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.5), 5))
    bars = ax.bar(names, ppls, color=colors, edgecolor="white")
    for bar, ppl in zip(bars, ppls):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{ppl:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Perplexity (lower is better)")
    ax.set_title("Held-out perplexity across experiments")
    plt.xticks(rotation=20, ha="right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "ppl_comparison.png"), dpi=150)
    plt.close()

    # ----------------------------------------------------------------------- #
    # Print summary table                                                      #
    # ----------------------------------------------------------------------- #
    baseline_ppl = results.get(
        next((n for n in names if "baseline" in n), names[0])
    )["ppl"]

    print("\n" + "=" * 70)
    print(f"{'Experiment':<35} {'Mode':<20} {'mean_k':>6}  {'PPL':>7}  {'ΔPPL':>7}")
    print("-" * 70)
    for name in names:
        r = results[name]
        delta = r["ppl"] - baseline_ppl
        sign = "+" if delta >= 0 else ""
        print(f"{name:<35} {r['budget_mode']:<20} {r['mean_k']:>6.1f}  "
              f"{r['ppl']:>7.3f}  {sign}{delta:>6.3f}")
    print("=" * 70)

    # Save table as JSON
    table_path = os.path.join(args.output_dir, "summary.json")
    with open(table_path, "w") as f:
        json.dump(
            {n: {k: v for k, v in r.items() if k not in ("steps", "losses")}
             for n, r in results.items()},
            f, indent=2
        )
    print(f"\nSummary JSON saved to {table_path}")
    print(f"Plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
