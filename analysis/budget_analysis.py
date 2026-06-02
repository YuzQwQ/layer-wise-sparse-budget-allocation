"""
Per-token dynamic budget distribution analysis for Phase 2 models.

Verifies that query_aware_tool / query_aware_gate actually produces
varying block counts across tokens and layers (not collapsed to a constant).

Usage:
    python analysis/budget_analysis.py \
        --checkpoint exp/phase2b_gate_sched \
        --tokenizer  fla-hub/transformer-1.3B-100B \
        --n_samples  128 \
        --seq_len    8192 \
        --output_dir analysis/budget_plots
"""

import argparse
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import native_sparse_attention  # noqa
from native_sparse_attention.modeling_nsa import NativeSparseAttention


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",   required=True)
    p.add_argument("--tokenizer",    required=True)
    p.add_argument("--dataset",      default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--dataset_name", default="sample-100BT")
    p.add_argument("--n_samples",    type=int, default=128)
    p.add_argument("--seq_len",      type=int, default=8192)
    p.add_argument("--batch_size",   type=int, default=1)
    p.add_argument("--output_dir",   default="analysis/budget_plots")
    p.add_argument("--device",       default="cuda")
    return p.parse_args()


def collect_samples(args, tok):
    ds = load_dataset(
        args.dataset, name=args.dataset_name,
        split="train", streaming=True, trust_remote_code=True,
    )
    samples, buf = [], []
    for ex in ds:
        buf.extend(tok.encode(ex.get("text") or ex.get("content", ""),
                              add_special_tokens=False))
        while len(buf) >= args.seq_len:
            samples.append(torch.tensor(buf[:args.seq_len], dtype=torch.long))
            buf = buf[args.seq_len:]
        if len(samples) >= args.n_samples:
            break
    return samples[:args.n_samples]


class BudgetRecorder:
    """Records the effective block_counts tensor produced each forward pass."""

    def __init__(self, model, budget_mode: str, budget_min_k: int, budget_max_k: int):
        self.budget_mode = budget_mode
        self.min_k = budget_min_k
        self.max_k = budget_max_k
        # layer_idx -> list of (B, T, H_kv) int tensors
        self.records: dict = defaultdict(list)
        self._hooks = []

        for li, layer in enumerate(model.layers):
            attn: NativeSparseAttention = layer.attn
            hook = attn.g_proj.register_forward_hook(
                self._make_hook(li, attn)
            )
            self._hooks.append(hook)

    def _make_hook(self, layer_idx: int, attn: NativeSparseAttention):
        def hook(mod, inp, out):
            if self.budget_mode not in ("query_aware_tool", "query_aware_gate"):
                return
            g = out.detach().float().sigmoid()
            B, T, _ = g.shape
            g = g.reshape(B, T, attn.num_heads, 3)
            g_slc = g[..., 1]  # (B, T, HQ)

            if self.budget_mode == "query_aware_gate":
                # reduce HQ → H_kv
                g_kv = g_slc.reshape(
                    B, T, attn.num_kv_heads, attn.num_kv_groups
                ).mean(-1)
                budget_cont = g_kv * (self.max_k - self.min_k) + self.min_k
            else:
                # query_aware_tool: use budget_head if available
                if hasattr(attn, "budget_head"):
                    # recompute from the stored input; inp[0] is hidden_states
                    hs = inp[0].detach().float()
                    sig = torch.sigmoid(attn.budget_head(hs))
                    budget_cont = sig * (self.max_k - self.min_k) + self.min_k
                else:
                    return

            dynamic_k = budget_cont.round().long().clamp(self.min_k, self.max_k)
            self.records[layer_idx].append(dynamic_k.cpu().numpy())

        return hook

    def remove(self):
        for h in self._hooks:
            h.remove()

    def summary(self):
        """Returns per-layer mean and std of dynamic_k across all samples."""
        result = {}
        for li, arrs in self.records.items():
            all_k = np.concatenate([a.reshape(-1) for a in arrs])
            result[li] = {"mean": float(all_k.mean()), "std": float(all_k.std()),
                          "min": int(all_k.min()), "max": int(all_k.max()),
                          "all": all_k}
        return result


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    print("Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    print(f"Loading model from {args.checkpoint}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, torch_dtype=torch.bfloat16
    ).to(device).eval()

    cfg = model.config
    budget_mode = getattr(cfg, "budget_mode", "uniform")
    budget_min_k = getattr(cfg, "budget_min_k", 4)
    budget_max_k = getattr(cfg, "budget_max_k", 32)

    print(f"Budget mode: {budget_mode}  [{budget_min_k}, {budget_max_k}]")

    if budget_mode not in ("query_aware_tool", "query_aware_gate"):
        print("Model is not in a query_aware mode — nothing to plot for dynamic budget.")
        return

    recorder = BudgetRecorder(model.model, budget_mode, budget_min_k, budget_max_k)

    print(f"Processing {args.n_samples} samples...")
    samples = collect_samples(args, tok)
    with torch.no_grad():
        for i in range(0, len(samples), args.batch_size):
            batch = torch.stack(samples[i:i + args.batch_size]).to(device)
            model(input_ids=batch)

    recorder.remove()
    summary = recorder.summary()
    layers = sorted(summary.keys())

    # ------------------------------------------------------------------- #
    # Plot 1: mean ± std per layer                                         #
    # ------------------------------------------------------------------- #
    means = [summary[l]["mean"] for l in layers]
    stds  = [summary[l]["std"]  for l in layers]

    fig, ax = plt.subplots(figsize=(14, 4))
    x = np.arange(len(layers))
    ax.bar(x, means, yerr=stds, capsize=3, color="#DD8452", alpha=0.8)
    ax.axhline(np.mean(means), color="red", ls="--", label=f"global mean={np.mean(means):.1f}")
    ax.set_xticks(x); ax.set_xticklabels(layers, fontsize=8)
    ax.set_xlabel("Layer"); ax.set_ylabel("dynamic_k (blocks selected)")
    ax.set_title(f"Per-layer mean ± std of dynamic block counts  [{budget_mode}]")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    out1 = os.path.join(args.output_dir, "mean_std_per_layer.png")
    plt.tight_layout(); plt.savefig(out1, dpi=150); plt.close()
    print(f"Saved {out1}")

    # ------------------------------------------------------------------- #
    # Plot 2: histogram of dynamic_k across all layers                     #
    # ------------------------------------------------------------------- #
    all_k = np.concatenate([summary[l]["all"] for l in layers])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(all_k, bins=range(budget_min_k, budget_max_k + 2),
            color="#4C72B0", edgecolor="white", rwidth=0.85)
    ax.set_xlabel("dynamic_k"); ax.set_ylabel("Count")
    ax.set_title("Histogram of selected block counts across all layers & tokens")
    out2 = os.path.join(args.output_dir, "dynamic_k_histogram.png")
    plt.tight_layout(); plt.savefig(out2, dpi=150); plt.close()
    print(f"Saved {out2}")

    # ------------------------------------------------------------------- #
    # Print verdict                                                         #
    # ------------------------------------------------------------------- #
    global_std = float(all_k.std())
    print(f"\nGlobal dynamic_k: mean={all_k.mean():.2f}, std={global_std:.2f}, "
          f"min={all_k.min()}, max={all_k.max()}")
    if global_std < 0.5:
        print("WARNING: budget variance is very low — model may have collapsed to "
              "a constant k. Consider increasing budget_aux_loss_weight.")
    else:
        print("Budget shows meaningful variation across tokens/layers. ✓")


if __name__ == "__main__":
    main()
