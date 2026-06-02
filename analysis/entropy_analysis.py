"""
Per-layer gate analysis for NSA baseline.

Measures mean g_slc (selection gate) per layer as a proxy for how much
each layer relies on precise block selection. High g_slc → layer needs
more blocks. Output drives the Phase 1 layer-aware budget profiles.

Usage (on server after training baseline):
    python analysis/entropy_analysis.py \
        --checkpoint exp/baseline_uniform_k16 \
        --tokenizer  fla-hub/transformer-1.3B-100B \
        --n_samples  256 \
        --seq_len    8192 \
        --output     analysis/entropy_per_layer.png
"""

import argparse
from collections import defaultdict

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
    p.add_argument("--checkpoint",   required=True)
    p.add_argument("--tokenizer",    required=True)
    p.add_argument("--dataset",      default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--dataset_name", default="sample-100BT")
    p.add_argument("--n_samples",    type=int, default=256)
    p.add_argument("--seq_len",      type=int, default=8192)
    p.add_argument("--batch_size",   type=int, default=1)
    p.add_argument("--output",       default="analysis/entropy_per_layer.png")
    p.add_argument("--device",       default="cuda")
    return p.parse_args()


def collect_samples(args, tokenizer):
    ds = load_dataset(
        args.dataset, name=args.dataset_name,
        split="train", streaming=True, trust_remote_code=True,
    )
    samples, buf = [], []
    for ex in ds:
        buf.extend(tokenizer.encode(ex.get("text") or ex.get("content", ""),
                                    add_special_tokens=False))
        while len(buf) >= args.seq_len:
            samples.append(torch.tensor(buf[:args.seq_len], dtype=torch.long))
            buf = buf[args.seq_len:]
        if len(samples) >= args.n_samples:
            break
    return samples[:args.n_samples]


def main():
    args = parse_args()
    device = torch.device(args.device)

    print("Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    print(f"Loading model from {args.checkpoint}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, torch_dtype=torch.bfloat16
    ).to(device).eval()

    num_layers = model.config.num_hidden_layers
    records = defaultdict(lambda: {"g_cmp": [], "g_slc": [], "g_swa": []})

    # Register hooks on g_proj of each layer
    hooks = []
    for li, layer in enumerate(model.model.layers):
        num_heads = layer.attn.num_heads

        def make_hook(idx, nh):
            def hook(mod, inp, out):
                g = out.detach().float().sigmoid()
                g = g.reshape(g.shape[0], g.shape[1], nh, 3)
                records[idx]["g_cmp"].append(g[..., 0].mean().item())
                records[idx]["g_slc"].append(g[..., 1].mean().item())
                records[idx]["g_swa"].append(g[..., 2].mean().item())
            return hook

        hooks.append(layer.attn.g_proj.register_forward_hook(make_hook(li, num_heads)))

    print(f"Processing {args.n_samples} samples...")
    samples = collect_samples(args, tok)
    with torch.no_grad():
        for i in range(0, len(samples), args.batch_size):
            batch = torch.stack(samples[i:i + args.batch_size]).to(device)
            model(input_ids=batch)
            if i % (args.batch_size * 32) == 0:
                print(f"  {i + len(batch)}/{len(samples)}")

    for h in hooks:
        h.remove()

    # Summarise
    layers = sorted(records.keys())
    g_slc = np.array([np.mean(records[l]["g_slc"]) for l in layers])
    g_cmp = np.array([np.mean(records[l]["g_cmp"]) for l in layers])
    g_swa = np.array([np.mean(records[l]["g_swa"]) for l in layers])

    # Plot
    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(layers))
    ax.bar(x - 0.25, g_cmp, 0.25, label="g_cmp (compression)", color="#4C72B0")
    ax.bar(x,         g_slc, 0.25, label="g_slc (selection)",   color="#DD8452")
    ax.bar(x + 0.25, g_swa, 0.25, label="g_swa (sliding)",     color="#55A868")
    ax.set_xticks(x); ax.set_xticklabels(layers, fontsize=8)
    ax.set_xlabel("Layer"); ax.set_ylabel("Mean gate (sigmoid)")
    ax.set_title("Per-layer gate magnitudes — high g_slc → more budget needed")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Plot saved to {args.output}")

    # Derive a budget-neutral block_counts list from g_slc ranks
    base_k = model.config.block_counts if isinstance(model.config.block_counts, int) else 16
    slc_n = (g_slc - g_slc.min()) / (g_slc.max() - g_slc.min() + 1e-8)
    k_raw = base_k * 0.5 + slc_n * base_k               # [0.5*base .. 1.5*base]
    k_r4  = (np.round(k_raw / 4) * 4).astype(int)       # align to multiple of 4
    # Shift so mean equals base_k
    diff  = base_k - int(round(k_r4.mean()))
    k_adj = np.clip(k_r4 + diff, 4, None).tolist()

    print("\n--- Data-driven block_counts suggestion ---")
    print(f"  block_counts = {k_adj}")
    print(f"  mean = {np.mean(k_adj):.2f}  (target = {base_k})")
    print("  → Save this as a new config with budget_mode='layer_aware'")


if __name__ == "__main__":
    main()
