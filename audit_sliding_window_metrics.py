"""Audit sliding-window diagnostic metric consistency.

This script recomputes all-token loss and fixed-cutoff after-active losses
from the same logits/labels/forward path.  It is an audit utility, not a new
benchmark or optimized sparse runtime.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch

from evaluate_h2o_kv_retention import load_model_and_tokenizer, load_split_tokens, set_seed, write_json
from evaluate_sliding_window_attention import (
    METHODS,
    ce_sum_from_logits,
    check_window_match,
    custom_forward_logits,
    hf_dense_loss,
    layer_groups,
    layer_windows,
    make_sliding_window_mask,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["evaluate", "sanity", "mask_sanity"], default="evaluate")
    parser.add_argument("--model_name", default="facebook/opt-350m")
    parser.add_argument("--dataset", default="json")
    parser.add_argument("--dataset_name", default="")
    parser.add_argument("--data_files", default=None)
    parser.add_argument("--val_split", default="validation")
    parser.add_argument("--test_split", default="test")
    parser.add_argument("--val_skip_tokens", type=int, default=0)
    parser.add_argument("--test_skip_tokens", type=int, default=8_000_000)
    parser.add_argument("--text_column", default="text")
    parser.add_argument("--method", choices=METHODS, default="uniform")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--avg_window_size", type=int, default=512)
    parser.add_argument("--val_tokens", type=int, default=393_216)
    parser.add_argument("--test_tokens", type=int, default=393_216)
    parser.add_argument("--sanity_tokens", type=int, default=2048)
    parser.add_argument("--sanity_tolerance", type=float, default=0.001)
    parser.add_argument("--require_exact_tokens", action="store_true")
    parser.add_argument("--cutoffs", default="512,768")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float16", "float32", "bfloat16"], default="float16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache_dir", default=None)
    return parser.parse_args()


def parse_cutoffs(value: str) -> list[int]:
    cutoffs = []
    for item in value.split(","):
        item = item.strip()
        if item:
            cutoffs.append(int(item))
    return sorted(set(cutoffs))


@torch.no_grad()
def evaluate_split_audit(
    model,
    tokens: torch.Tensor,
    *,
    windows: list[int],
    device: torch.device,
    seq_len: int,
    cutoffs: list[int],
) -> dict[str, Any]:
    start_time = time.time()
    sums = {"all": 0.0}
    counts = {"all": 0}
    for cutoff in cutoffs:
        sums[f"after_{cutoff}"] = 0.0
        counts[f"after_{cutoff}"] = 0
    sequences = 0
    for offset in range(0, tokens.numel(), seq_len):
        ids = tokens[offset : offset + seq_len]
        if ids.numel() != seq_len:
            continue
        input_ids = ids.unsqueeze(0).to(device)
        logits = custom_forward_logits(model, input_ids, windows=windows, full_window=False)
        loss, ntok = ce_sum_from_logits(logits, input_ids)
        sums["all"] += loss
        counts["all"] += ntok
        for cutoff in cutoffs:
            loss_c, ntok_c = ce_sum_from_logits(logits, input_ids, start_target_index=cutoff)
            key = f"after_{cutoff}"
            sums[key] += loss_c
            counts[key] += ntok_c
        sequences += 1
    seconds = time.time() - start_time
    metrics: dict[str, Any] = {
        "sequences": sequences,
        "seconds": seconds,
        "tok_per_s": counts["all"] / seconds if seconds > 0 else 0.0,
        "loss": sums["all"] / max(counts["all"], 1),
        "tokens": counts["all"],
    }
    metrics["ppl"] = math.exp(metrics["loss"]) if metrics["loss"] < 50 else float("inf")
    for cutoff in cutoffs:
        key = f"after_{cutoff}"
        loss_value = sums[key] / max(counts[key], 1)
        metrics[f"loss_{key}"] = loss_value
        metrics[f"ppl_{key}"] = math.exp(loss_value) if loss_value < 50 else float("inf")
        metrics[f"tokens_{key}"] = counts[key]
    return metrics


def run_sanity(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer, device = load_model_and_tokenizer(args)
    tokens, meta = load_split_tokens(args, tokenizer, args.val_split, max(args.sanity_tokens, args.seq_len), window_name="sanity", skip_tokens=args.val_skip_tokens)
    ids = tokens[: args.seq_len].unsqueeze(0).to(device)
    dense = hf_dense_loss(model, ids)
    num_layers = len(model.model.decoder.layers)
    logits = custom_forward_logits(model, ids, windows=[args.seq_len] * num_layers, full_window=True)
    loss_sum, ntok = ce_sum_from_logits(logits, ids)
    custom = loss_sum / ntok
    result = {
        "mode": "sanity",
        "split_meta": meta,
        "hf_dense_loss": dense,
        "custom_full_window_loss": custom,
        "abs_diff": abs(custom - dense),
        "relative_diff": abs(custom - dense) / max(abs(dense), 1e-12),
        "tolerance": args.sanity_tolerance,
        "passed": abs(custom - dense) <= args.sanity_tolerance,
        "evaluation_context_length": args.seq_len,
    }
    write_json(out_dir / "audit_sanity_summary.json", result)
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(2)


def run_mask_sanity(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    query_positions = [0, 1, 127, 511, 512, 767, 768, 1024, 2047]
    num_layers = 24
    rows = []
    for method in METHODS:
        windows = layer_windows(num_layers, method, args.avg_window_size)
        groups = layer_groups(num_layers)
        for layer_idx in [0, 5, 6, 11, 12, 17, 18, 23]:
            window = windows[layer_idx]
            mask = make_sliding_window_mask(args.seq_len, window, torch.device("cpu"), torch.float32)[0, 0]
            for q in query_positions:
                accessible = int((mask[q] == 0).sum().item())
                expected = min(q + 1, window)
                first_key = max(0, q - window + 1)
                rows.append(
                    {
                        "method": method,
                        "layer_id": layer_idx,
                        "group_id": groups[layer_idx],
                        "window_size": window,
                        "query_position": q,
                        "expected_first_key": first_key,
                        "expected_accessible_keys": expected,
                        "actual_accessible_keys": accessible,
                        "passed": accessible == expected,
                    }
                )
    result = {"mode": "mask_sanity", "rows": rows, "passed": all(row["passed"] for row in rows)}
    write_json(out_dir / "audit_mask_sanity_summary.json", result)
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(2)


def run_evaluate(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cutoffs = parse_cutoffs(args.cutoffs)
    model, tokenizer, device = load_model_and_tokenizer(args)
    num_layers = len(model.model.decoder.layers)
    windows = layer_windows(num_layers, args.method, args.avg_window_size)
    check_window_match(windows, args.avg_window_size)
    val_tokens, val_meta = load_split_tokens(args, tokenizer, args.val_split, args.val_tokens, window_name="validation", skip_tokens=args.val_skip_tokens)
    test_tokens, test_meta = load_split_tokens(args, tokenizer, args.test_split, args.test_tokens, window_name="test", skip_tokens=args.test_skip_tokens)
    val = evaluate_split_audit(model, val_tokens, windows=windows, device=device, seq_len=args.seq_len, cutoffs=cutoffs)
    test = evaluate_split_audit(model, test_tokens, windows=windows, device=device, seq_len=args.seq_len, cutoffs=cutoffs)
    profile = {
        "method": args.method,
        "num_layers": num_layers,
        "evaluation_context_length": args.seq_len,
        "avg_window_size": sum(windows) / len(windows),
        "window_sizes": windows,
        "group_assignments": layer_groups(num_layers),
        "cutoffs": cutoffs,
    }
    summary = {
        "method": args.method,
        "val_loss": val["loss"],
        "val_ppl": val["ppl"],
        "test_loss": test["loss"],
        "test_ppl": test["ppl"],
        "seq_len": args.seq_len,
        "evaluation_context_length": args.seq_len,
        "avg_window_size": profile["avg_window_size"],
        "cutoffs": cutoffs,
        "tok_per_s": (val["tokens"] + test["tokens"]) / max(val["seconds"] + test["seconds"], 1e-12),
    }
    for cutoff in cutoffs:
        summary[f"val_loss_after_{cutoff}"] = val[f"loss_after_{cutoff}"]
        summary[f"test_loss_after_{cutoff}"] = test[f"loss_after_{cutoff}"]
        summary[f"val_tokens_after_{cutoff}"] = val[f"tokens_after_{cutoff}"]
        summary[f"test_tokens_after_{cutoff}"] = test[f"tokens_after_{cutoff}"]
    write_json(out_dir / "audit_metrics.json", {"validation": val, "test": test, "val_cache": val_meta, "test_cache": test_meta})
    write_json(out_dir / "audit_profile.json", profile)
    write_json(out_dir / "audit_summary.json", summary)
    print(json.dumps(summary, indent=2))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if args.mode == "sanity":
        run_sanity(args)
    elif args.mode == "mask_sanity":
        run_mask_sanity(args)
    else:
        run_evaluate(args)


if __name__ == "__main__":
    main()
    import os

    os._exit(0)
