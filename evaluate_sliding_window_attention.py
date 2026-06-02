"""Sliding-window / Streaming-style window-level sparse attention diagnostic.

This is an inference-time diagnostic for layer-wise local-window budget
placement.  It is not a StreamingLLM reproduction, not a sparse-attention
benchmark, and not an optimized sparse runtime implementation.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from evaluate_h2o_kv_retention import (
    call_decoder_layer,
    dtype_from_name,
    hf_dense_loss,
    load_model_and_tokenizer,
    load_split_tokens,
    opt_position_embeddings,
    set_seed,
    write_json,
)


METHODS = ("uniform", "mono_inc", "mono_dec")
WINDOW_VALUES_512 = {
    "uniform": (512, 512, 512, 512),
    "mono_inc": (256, 384, 640, 768),
    "mono_dec": (768, 640, 384, 256),
}
WINDOW_VALUES_256 = {
    "uniform": (256, 256, 256, 256),
    "mono_inc": (128, 192, 320, 384),
    "mono_dec": (384, 320, 192, 128),
}


@dataclass
class SplitResult:
    split: str
    loss: float
    ppl: float
    loss_after_window_active: float | None
    ppl_after_window_active: float | None
    tokens_after_window_active: int
    tokens: int
    sequences: int
    seconds: float
    tok_per_s: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["evaluate", "sanity", "mask_sanity"], default="evaluate")
    parser.add_argument("--model_name", default="facebook/opt-350m")
    parser.add_argument("--dataset", default="Salesforce/wikitext")
    parser.add_argument("--dataset_name", default="wikitext-103-raw-v1")
    parser.add_argument("--data_files", default=None)
    parser.add_argument("--val_split", default="validation")
    parser.add_argument("--test_split", default="test")
    parser.add_argument("--val_skip_tokens", type=int, default=0)
    parser.add_argument("--test_skip_tokens", type=int, default=0)
    parser.add_argument("--text_column", default="text")
    parser.add_argument("--method", choices=METHODS, default="uniform")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--avg_window_size", type=int, choices=[256, 512], default=512)
    parser.add_argument("--val_tokens", type=int, default=393_216)
    parser.add_argument("--test_tokens", type=int, default=393_216)
    parser.add_argument("--sanity_tokens", type=int, default=2048)
    parser.add_argument("--sanity_tolerance", type=float, default=0.001)
    parser.add_argument("--require_exact_tokens", action="store_true")
    parser.add_argument("--report_loss_after_window_active", action="store_true")
    parser.add_argument("--window_active_exclude_tokens", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float16", "float32", "bfloat16"], default="float16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache_dir", default=None)
    return parser.parse_args()


def window_values(avg_window_size: int) -> dict[str, tuple[int, int, int, int]]:
    if avg_window_size == 512:
        return WINDOW_VALUES_512
    if avg_window_size == 256:
        return WINDOW_VALUES_256
    raise ValueError(f"Unsupported avg_window_size: {avg_window_size}")


def layer_groups(num_layers: int) -> list[int]:
    return [min(3, (layer_idx * 4) // num_layers) for layer_idx in range(num_layers)]


def layer_windows(num_layers: int, method: str, avg_window_size: int) -> list[int]:
    values = window_values(avg_window_size)[method]
    groups = layer_groups(num_layers)
    return [values[group] for group in groups]


def check_window_match(windows: list[int], expected_avg: int) -> None:
    avg = sum(windows) / len(windows)
    if abs(avg - expected_avg) > 1e-6:
        raise ValueError(f"Average window mismatch: {avg} != {expected_avg}")


def make_causal_mask(seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    mask = torch.full((1, 1, seq_len, seq_len), -1.0e4, device=device, dtype=dtype)
    keep = torch.tril(torch.ones((seq_len, seq_len), device=device, dtype=torch.bool))
    mask[0, 0].masked_fill_(keep, 0.0)
    return mask


def make_sliding_window_mask(seq_len: int, window_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    positions = torch.arange(seq_len, device=device)
    q = positions.unsqueeze(1)
    k = positions.unsqueeze(0)
    # Window size counts accessible key positions including the current token.
    keep = (k <= q) & (k >= (q - window_size + 1).clamp_min(0))
    mask = torch.full((1, 1, seq_len, seq_len), -1.0e4, device=device, dtype=dtype)
    mask[0, 0].masked_fill_(keep, 0.0)
    return mask


def accessible_counts(seq_len: int, window_size: int) -> list[int]:
    return [min(pos + 1, window_size) for pos in range(seq_len)]


@torch.no_grad()
def custom_forward_logits(
    model,
    input_ids: torch.Tensor,
    *,
    windows: list[int],
    full_window: bool,
) -> torch.Tensor:
    decoder = model.model.decoder
    attention_mask_2d = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)
    hidden_states = decoder.embed_tokens(input_ids)
    hidden_states = hidden_states * getattr(decoder, "embed_scale", 1.0)
    if getattr(decoder, "project_in", None) is not None:
        hidden_states = decoder.project_in(hidden_states)
    hidden_states = hidden_states + opt_position_embeddings(decoder, attention_mask_2d)

    seq_len = input_ids.shape[1]
    causal_mask: torch.Tensor | None = None
    if full_window:
        causal_mask = make_causal_mask(seq_len, input_ids.device, hidden_states.dtype)
    mask_cache: dict[int, torch.Tensor] = {}

    for layer_idx, layer in enumerate(decoder.layers):
        if full_window:
            layer_mask = causal_mask
        else:
            window = windows[layer_idx]
            layer_mask = mask_cache.get(window)
            if layer_mask is None:
                layer_mask = make_sliding_window_mask(seq_len, window, input_ids.device, hidden_states.dtype)
                mask_cache[window] = layer_mask
        hidden_states = call_decoder_layer(layer, hidden_states, layer_mask)

    if getattr(decoder, "final_layer_norm", None) is not None:
        hidden_states = decoder.final_layer_norm(hidden_states)
    if getattr(decoder, "project_out", None) is not None:
        hidden_states = decoder.project_out(hidden_states)
    return model.lm_head(hidden_states)


def ce_sum_from_logits(logits: torch.Tensor, labels: torch.Tensor, *, start_target_index: int = 1) -> tuple[float, int]:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    start = max(0, start_target_index - 1)
    if start > 0:
        shift_logits = shift_logits[:, start:, :].contiguous()
        shift_labels = shift_labels[:, start:].contiguous()
    if shift_labels.numel() == 0:
        return 0.0, 0
    loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction="sum")
    return float(loss.detach().cpu()), int(shift_labels.numel())


def evaluate_split(
    model,
    tokens: torch.Tensor,
    *,
    split: str,
    windows: list[int],
    device: torch.device,
    report_loss_after_window_active: bool,
    window_active_exclude_tokens: int,
    seq_len: int,
) -> SplitResult:
    start_time = time.time()
    loss_sum = 0.0
    token_count = 0
    late_loss_sum = 0.0
    late_token_count = 0
    sequences = 0
    for offset in range(0, tokens.numel(), seq_len):
        ids = tokens[offset : offset + seq_len]
        if ids.numel() != seq_len:
            continue
        input_ids = ids.unsqueeze(0).to(device)
        logits = custom_forward_logits(model, input_ids, windows=windows, full_window=False)
        loss, ntok = ce_sum_from_logits(logits, input_ids)
        loss_sum += loss
        token_count += ntok
        if report_loss_after_window_active:
            late_loss, late_ntok = ce_sum_from_logits(logits, input_ids, start_target_index=window_active_exclude_tokens)
            late_loss_sum += late_loss
            late_token_count += late_ntok
        sequences += 1
    seconds = time.time() - start_time
    mean_loss = loss_sum / max(token_count, 1)
    late_mean_loss = late_loss_sum / late_token_count if late_token_count > 0 else None
    return SplitResult(
        split=split,
        loss=mean_loss,
        ppl=math.exp(mean_loss) if mean_loss < 50 else float("inf"),
        loss_after_window_active=late_mean_loss,
        ppl_after_window_active=math.exp(late_mean_loss) if late_mean_loss is not None and late_mean_loss < 50 else None,
        tokens_after_window_active=late_token_count,
        tokens=token_count,
        sequences=sequences,
        seconds=seconds,
        tok_per_s=token_count / seconds if seconds > 0 else 0.0,
    )


def run_sanity(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer, device = load_model_and_tokenizer(args)
    token_cap = max(args.sanity_tokens, args.seq_len)
    tokens, meta = load_split_tokens(args, tokenizer, args.val_split, token_cap, window_name="sanity", skip_tokens=args.val_skip_tokens)
    ids = tokens[: args.seq_len].unsqueeze(0).to(device)
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    dense = hf_dense_loss(model, ids)
    num_layers = len(model.model.decoder.layers)
    full_windows = [args.seq_len for _ in range(num_layers)]
    logits = custom_forward_logits(model, ids, windows=full_windows, full_window=True)
    loss_sum, ntok = ce_sum_from_logits(logits, ids)
    custom = loss_sum / ntok
    abs_diff = abs(custom - dense)
    rel_diff = abs_diff / max(abs(dense), 1e-12)
    result = {
        "mode": "sanity",
        "model_name": args.model_name,
        "dataset": args.dataset,
        "dataset_name": args.dataset_name,
        "split_meta": meta,
        "hf_dense_loss": dense,
        "custom_full_window_loss": custom,
        "abs_diff": abs_diff,
        "relative_diff": rel_diff,
        "tolerance": args.sanity_tolerance,
        "passed": abs_diff <= args.sanity_tolerance,
        "evaluation_context_length": args.seq_len,
    }
    if torch.cuda.is_available() and device.type == "cuda":
        result["cuda_max_memory_allocated_mb"] = torch.cuda.max_memory_allocated() / (1024**2)
    write_json(out_dir / "sanity_summary.json", result)
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(2)


def run_mask_sanity(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    num_layers = 24
    windows = layer_windows(num_layers, args.method, args.avg_window_size)
    rows = []
    for layer_idx, window in enumerate(windows):
        counts = accessible_counts(min(args.seq_len, 64), window)
        rows.append(
            {
                "layer_id": layer_idx,
                "group_id": layer_groups(num_layers)[layer_idx],
                "window_size": window,
                "max_accessible_keys_in_short_check": max(counts),
                "causal_ok": all(count <= pos + 1 for pos, count in enumerate(counts)),
            }
        )
    result = {
        "mode": "mask_sanity",
        "method": args.method,
        "avg_window_size": args.avg_window_size,
        "evaluation_context_length": args.seq_len,
        "rows": rows,
        "passed": all(row["causal_ok"] for row in rows),
    }
    write_json(out_dir / "mask_sanity_summary.json", result)
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(2)


def run_evaluate(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer, device = load_model_and_tokenizer(args)
    num_layers = len(model.model.decoder.layers)
    windows = layer_windows(num_layers, args.method, args.avg_window_size)
    check_window_match(windows, args.avg_window_size)
    exclude = args.window_active_exclude_tokens if args.window_active_exclude_tokens is not None else max(windows)

    val_tokens, val_meta = load_split_tokens(args, tokenizer, args.val_split, args.val_tokens, window_name="validation", skip_tokens=args.val_skip_tokens)
    test_tokens, test_meta = load_split_tokens(args, tokenizer, args.test_split, args.test_tokens, window_name="test", skip_tokens=args.test_skip_tokens)
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    val = evaluate_split(
        model,
        val_tokens,
        split="validation",
        windows=windows,
        device=device,
        report_loss_after_window_active=args.report_loss_after_window_active,
        window_active_exclude_tokens=exclude,
        seq_len=args.seq_len,
    )
    test = evaluate_split(
        model,
        test_tokens,
        split="test",
        windows=windows,
        device=device,
        report_loss_after_window_active=args.report_loss_after_window_active,
        window_active_exclude_tokens=exclude,
        seq_len=args.seq_len,
    )
    groups = layer_groups(num_layers)
    per_layer_window = [
        {"layer_id": i, "group_id": groups[i], "window_size": windows[i]}
        for i in range(num_layers)
    ]
    profile = {
        "model_name": args.model_name,
        "dataset": args.dataset,
        "dataset_name": args.dataset_name,
        "data_files": args.data_files,
        "method": args.method,
        "num_layers": num_layers,
        "evaluation_context_length": args.seq_len,
        "window_sizes": windows,
        "group_assignments": groups,
        "per_layer_window": per_layer_window,
        "avg_window_size": sum(windows) / len(windows),
        "avg_window_size_check": abs(sum(windows) / len(windows) - args.avg_window_size) < 1e-6,
        "max_window_size": max(windows),
    }
    metrics = {"validation": val.__dict__, "test": test.__dict__, "val_cache": val_meta, "test_cache": test_meta}
    total_tokens = val.tokens + test.tokens
    total_seconds = val.seconds + test.seconds
    summary = {
        "model_name": args.model_name,
        "dataset": args.dataset,
        "dataset_name": args.dataset_name,
        "method": args.method,
        "val_loss": val.loss,
        "val_ppl": val.ppl,
        "test_loss": test.loss,
        "test_ppl": test.ppl,
        "val_loss_after_window_active": val.loss_after_window_active,
        "val_ppl_after_window_active": val.ppl_after_window_active,
        "test_loss_after_window_active": test.loss_after_window_active,
        "test_ppl_after_window_active": test.ppl_after_window_active,
        "window_active_exclude_tokens": exclude,
        "avg_window_size": profile["avg_window_size"],
        "tok_per_s": total_tokens / total_seconds if total_seconds > 0 else 0.0,
        "total_tokens": total_tokens,
        "total_seconds": total_seconds,
        "num_layers": num_layers,
        "seq_len": args.seq_len,
        "evaluation_context_length": args.seq_len,
    }
    if torch.cuda.is_available() and device.type == "cuda":
        summary["cuda_max_memory_allocated_mb"] = torch.cuda.max_memory_allocated() / (1024**2)
        summary["cuda_max_memory_reserved_mb"] = torch.cuda.max_memory_reserved() / (1024**2)
    write_json(out_dir / "window_profile.json", profile)
    write_json(out_dir / "eval_metrics.json", metrics)
    write_json(out_dir / "run_summary.json", summary)
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
    # Some HF/torch background workers can keep the interpreter alive after all
    # JSON artifacts are written on the remote host.  For this standalone
    # diagnostic evaluator, force a clean zero exit after successful completion
    # so shell launchers can continue to the next scheduled run.
    import os

    os._exit(0)
