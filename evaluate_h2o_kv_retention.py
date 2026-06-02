"""H2O-style token-level KV retention diagnostic for OPT models.

This script is intentionally an evaluation-time diagnostic, not an H2O
reproduction or an optimized sparse-attention implementation.  It evaluates
whether layer-wise placement of a fixed average heavy-hitter KV retention
budget changes validation loss / perplexity under matched average retained
budget.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


METHODS = ("uniform", "mono_inc", "mono_dec")
HH_VALUES = {
    "uniform": (128, 128, 128, 128),
    "mono_inc": (64, 96, 160, 192),
    "mono_dec": (192, 160, 96, 64),
}


@dataclass
class SplitResult:
    split: str
    loss: float
    ppl: float
    loss_after_cache_full: float | None
    ppl_after_cache_full: float | None
    tokens_after_cache_full: int
    tokens: int
    sequences: int
    seconds: float
    tok_per_s: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["evaluate", "sanity"], default="evaluate")
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
    parser.add_argument("--chunk_size", type=int, default=64)
    parser.add_argument("--recent_size", type=int, default=128)
    parser.add_argument("--avg_heavy_hitter_size", type=int, default=128)
    parser.add_argument("--val_tokens", type=int, default=393_216)
    parser.add_argument("--test_tokens", type=int, default=393_216)
    parser.add_argument("--smoke_tokens", type=int, default=8192)
    parser.add_argument("--sanity_tokens", type=int, default=2048)
    parser.add_argument("--sanity_tolerance", type=float, default=0.02)
    parser.add_argument("--require_exact_tokens", action="store_true")
    parser.add_argument("--report_loss_after_cache_full", action="store_true")
    parser.add_argument("--cache_full_exclude_tokens", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float16", "float32", "bfloat16"], default="float16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--hf_token", default=None)
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def layer_budgets(num_layers: int, method: str) -> list[int]:
    values = HH_VALUES[method]
    budgets: list[int] = []
    for layer_idx in range(num_layers):
        group = min(3, (layer_idx * 4) // num_layers)
        budgets.append(values[group])
    return budgets


def layer_groups(num_layers: int) -> list[int]:
    return [min(3, (layer_idx * 4) // num_layers) for layer_idx in range(num_layers)]


def check_budget_match(budgets: list[int], expected_avg: int) -> None:
    avg = sum(budgets) / len(budgets)
    if abs(avg - expected_avg) > 1e-6:
        raise ValueError(f"Average HH budget mismatch: {avg} != {expected_avg}")


def load_model_and_tokenizer(args: argparse.Namespace):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = dtype_from_name(args.dtype)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model_kwargs: dict[str, Any] = {
        "cache_dir": args.cache_dir,
        "trust_remote_code": False,
    }
    if device.type == "cuda":
        model_kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir, use_fast=True)
    model.eval().to(device)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, device


def split_skip_tokens(args: argparse.Namespace, split: str) -> int:
    if split == args.val_split:
        return int(args.val_skip_tokens)
    if split == args.test_split:
        return int(args.test_skip_tokens)
    return 0


def iter_json_gz_rows(paths: list[Path]):
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)


def load_split_tokens(
    args: argparse.Namespace,
    tokenizer,
    split: str,
    token_cap: int,
    *,
    window_name: str | None = None,
    skip_tokens: int | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    from datasets import load_dataset

    tokens: list[int] = []
    eos = tokenizer.eos_token_id
    window_name = window_name or split
    skip = int(split_skip_tokens(args, split) if skip_tokens is None else skip_tokens)
    skipped = 0
    data_source = "hf_dataset"
    if args.data_files:
        data_source = "json_gz_files"
        paths = [Path(item) for item in args.data_files.split(",") if item.strip()]
        row_iter = iter_json_gz_rows(paths)
    else:
        dataset = load_dataset(args.dataset, args.dataset_name, split=split, cache_dir=args.cache_dir)
        row_iter = iter(dataset)

    for row in row_iter:
        text = row.get(args.text_column, "")
        if not text:
            continue
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if encoded:
            if eos is not None:
                encoded = encoded + [eos]
            if skipped < skip:
                need = skip - skipped
                if len(encoded) <= need:
                    skipped += len(encoded)
                    continue
                encoded = encoded[need:]
                skipped += need
            tokens.extend(encoded)
        if len(tokens) >= token_cap:
            break
    if args.require_exact_tokens and len(tokens) < token_cap:
        raise RuntimeError(
            f"Window {window_name} split {split} has {len(tokens)} tokens after skip={skip}; "
            f"required exactly {token_cap}"
        )
    usable = token_cap if args.require_exact_tokens else (min(len(tokens), token_cap) // args.seq_len) * args.seq_len
    if usable % args.seq_len != 0:
        raise RuntimeError(f"Token window {window_name} has {usable} tokens, not divisible by seq_len={args.seq_len}")
    tokens = tokens[:usable]
    if args.require_exact_tokens and len(tokens) != token_cap:
        raise RuntimeError(f"Window {window_name} actual tokens {len(tokens)} != requested exact tokens {token_cap}")
    if usable < args.seq_len:
        raise RuntimeError(f"Window {window_name} split {split} produced too few tokens: {usable}")
    meta = {
        "dataset": args.dataset,
        "dataset_name": args.dataset_name,
        "data_files": args.data_files,
        "data_source": data_source,
        "window_name": window_name,
        "split": split,
        "skip_tokens": skip,
        "requested_tokens": token_cap,
        "actual_tokens": usable,
        "seq_len": args.seq_len,
        "evaluation_context_length": args.seq_len,
        "num_sequences": usable // args.seq_len,
        "eos_between_documents": eos is not None,
        "require_exact_tokens": bool(args.require_exact_tokens),
    }
    return torch.tensor(tokens, dtype=torch.long), meta


def opt_position_embeddings(decoder, attention_mask: torch.Tensor) -> torch.Tensor:
    try:
        return decoder.embed_positions(attention_mask, 0)
    except TypeError:
        return decoder.embed_positions(attention_mask)


def make_causal_mask(seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    mask = torch.full((1, 1, seq_len, seq_len), -1.0e4, device=device, dtype=dtype)
    keep = torch.tril(torch.ones((seq_len, seq_len), device=device, dtype=torch.bool))
    mask[0, 0].masked_fill_(keep, 0.0)
    return mask


def attention_qk(self_attn, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    bsz, seq_len, _ = hidden_states.shape
    num_heads = self_attn.num_heads
    head_dim = self_attn.head_dim
    scaling = getattr(self_attn, "scaling", head_dim**-0.5)
    q = self_attn.q_proj(hidden_states) * scaling
    k = self_attn.k_proj(hidden_states)
    q = q.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2).contiguous()
    k = k.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2).contiguous()
    return q, k


def update_layer_stats(stats: dict[str, float], retained_count: int, recent_count: int, hh_count: int) -> None:
    stats["chunks"] += 1.0
    stats["retained_sum"] += float(retained_count)
    stats["recent_sum"] += float(recent_count)
    stats["heavy_hitter_sum"] += float(hh_count)
    stats["retained_max"] = max(stats["retained_max"], float(retained_count))


def make_h2o_mask(
    self_attn,
    hidden_states: torch.Tensor,
    *,
    heavy_hitter_size: int,
    recent_size: int,
    chunk_size: int,
    layer_stats: dict[str, float],
) -> torch.Tensor:
    device = hidden_states.device
    dtype = hidden_states.dtype
    _, seq_len, _ = hidden_states.shape
    q, k = attention_qk(self_attn, hidden_states)
    q = q[0]
    k = k[0]
    mask = torch.full((1, 1, seq_len, seq_len), -1.0e4, device=device, dtype=dtype)
    retained: set[int] = set()
    scores: dict[int, float] = {}

    for start in range(0, seq_len, chunk_size):
        end = min(seq_len, start + chunk_size)
        allowed = sorted(retained.union(range(start, end)))
        allowed_t = torch.tensor(allowed, device=device, dtype=torch.long)
        q_chunk = q[:, start:end, :]
        k_allowed = k[:, allowed_t, :]
        logits = torch.einsum("hqd,hkd->hqk", q_chunk.float(), k_allowed.float())
        query_positions = torch.arange(start, end, device=device).unsqueeze(1)
        allowed_positions = allowed_t.unsqueeze(0)
        row_allowed = allowed_positions <= query_positions
        logits = logits.masked_fill(~row_allowed.unsqueeze(0), -1.0e9)
        probs = torch.softmax(logits, dim=-1)
        probs = probs.masked_fill(~row_allowed.unsqueeze(0), 0.0)
        mass = probs.sum(dim=(0, 1)).detach().cpu().tolist()

        for local_row in range(end - start):
            row_indices = allowed_t[row_allowed[local_row]]
            mask[0, 0, start + local_row, row_indices] = 0.0

        for token_idx, value in zip(allowed, mass):
            scores[token_idx] = scores.get(token_idx, 0.0) + float(value)

        recent_start = max(0, end - recent_size)
        recent = set(range(recent_start, end))
        candidates = [idx for idx in allowed if idx < recent_start]
        top_hh = sorted(candidates, key=lambda idx: scores.get(idx, 0.0), reverse=True)[:heavy_hitter_size]
        retained = recent.union(top_hh)

        # Tokens not retained are deliberately removed from the score table so
        # evicted keys cannot re-enter the cache later.
        scores = {idx: scores.get(idx, 0.0) for idx in retained}
        update_layer_stats(layer_stats, len(retained), len(recent), len(top_hh))

    return mask


def call_decoder_layer(layer, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    try:
        outputs = layer(
            hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=None,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
        )
    except TypeError:
        outputs = layer(
            hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=None,
            output_attentions=False,
            use_cache=False,
        )
    return outputs[0] if isinstance(outputs, (tuple, list)) else outputs


@torch.no_grad()
def custom_forward_logits(
    model,
    input_ids: torch.Tensor,
    *,
    method: str,
    recent_size: int,
    heavy_hitter_budgets: list[int],
    chunk_size: int,
    full_retention: bool,
    retention_stats: list[dict[str, float]] | None = None,
) -> torch.Tensor:
    decoder = model.model.decoder
    attention_mask_2d = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)
    hidden_states = decoder.embed_tokens(input_ids)
    embed_scale = getattr(decoder, "embed_scale", 1.0)
    hidden_states = hidden_states * embed_scale
    if getattr(decoder, "project_in", None) is not None:
        hidden_states = decoder.project_in(hidden_states)
    hidden_states = hidden_states + opt_position_embeddings(decoder, attention_mask_2d)

    causal_mask: torch.Tensor | None = None
    if full_retention:
        causal_mask = make_causal_mask(input_ids.shape[1], input_ids.device, hidden_states.dtype)

    for layer_idx, layer in enumerate(decoder.layers):
        if full_retention:
            layer_mask = causal_mask
        else:
            layer_stats = retention_stats[layer_idx] if retention_stats is not None else {
                "chunks": 0.0,
                "retained_sum": 0.0,
                "recent_sum": 0.0,
                "heavy_hitter_sum": 0.0,
                "retained_max": 0.0,
            }
            layer_mask = make_h2o_mask(
                layer.self_attn,
                hidden_states,
                heavy_hitter_size=heavy_hitter_budgets[layer_idx],
                recent_size=recent_size,
                chunk_size=chunk_size,
                layer_stats=layer_stats,
            )
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
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="sum",
    )
    return float(loss.detach().cpu()), int(shift_labels.numel())


@torch.no_grad()
def hf_dense_loss(model, input_ids: torch.Tensor) -> float:
    outputs = model(input_ids=input_ids, labels=input_ids)
    return float(outputs.loss.detach().cpu())


def init_retention_stats(num_layers: int) -> list[dict[str, float]]:
    return [
        {"chunks": 0.0, "retained_sum": 0.0, "recent_sum": 0.0, "heavy_hitter_sum": 0.0, "retained_max": 0.0}
        for _ in range(num_layers)
    ]


def summarize_retention_stats(stats: list[dict[str, float]], budgets: list[int], recent_size: int) -> list[dict[str, Any]]:
    summary = []
    groups = layer_groups(len(stats))
    for layer_idx, row in enumerate(stats):
        chunks = max(row["chunks"], 1.0)
        summary.append(
            {
                "layer": layer_idx,
                "group_id": groups[layer_idx],
                "heavy_hitter_size": budgets[layer_idx],
                "recent_size": recent_size,
                "target_total_budget": recent_size + budgets[layer_idx],
                "actual_retained_mean": row["retained_sum"] / chunks,
                "actual_recent_mean": row["recent_sum"] / chunks,
                "actual_heavy_hitter_mean": row["heavy_hitter_sum"] / chunks,
                "actual_retained_max": row["retained_max"],
                "chunks": int(row["chunks"]),
            }
        )
    return summary


def evaluate_split(
    model,
    tokens: torch.Tensor,
    *,
    split: str,
    method: str,
    recent_size: int,
    budgets: list[int],
    chunk_size: int,
    device: torch.device,
    retention_stats: list[dict[str, float]],
    report_loss_after_cache_full: bool,
    cache_full_exclude_tokens: int,
) -> SplitResult:
    start_time = time.time()
    loss_sum = 0.0
    token_count = 0
    late_loss_sum = 0.0
    late_token_count = 0

    sequences = 0
    for offset in range(0, tokens.numel(), evaluate_split.seq_len):
        ids = tokens[offset : offset + evaluate_split.seq_len]
        if ids.numel() != evaluate_split.seq_len:
            continue
        input_ids = ids.unsqueeze(0).to(device)
        logits = custom_forward_logits(
            model,
            input_ids,
            method=method,
            recent_size=recent_size,
            heavy_hitter_budgets=budgets,
            chunk_size=chunk_size,
            full_retention=False,
            retention_stats=retention_stats,
        )
        loss, ntok = ce_sum_from_logits(logits, input_ids)
        loss_sum += loss
        token_count += ntok
        if report_loss_after_cache_full:
            late_loss, late_ntok = ce_sum_from_logits(
                logits,
                input_ids,
                start_target_index=cache_full_exclude_tokens,
            )
            late_loss_sum += late_loss
            late_token_count += late_ntok
        sequences += 1
    seconds = time.time() - start_time
    mean_loss = loss_sum / max(token_count, 1)
    late_mean_loss = late_loss_sum / late_token_count if late_token_count > 0 else None
    ppl = math.exp(mean_loss) if mean_loss < 50 else float("inf")
    late_ppl = math.exp(late_mean_loss) if late_mean_loss is not None and late_mean_loss < 50 else None
    return SplitResult(
        split=split,
        loss=mean_loss,
        ppl=ppl,
        loss_after_cache_full=late_mean_loss,
        ppl_after_cache_full=late_ppl,
        tokens_after_cache_full=late_token_count,
        tokens=token_count,
        sequences=sequences,
        seconds=seconds,
        tok_per_s=token_count / seconds if seconds > 0 else 0.0,
    )


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def run_sanity(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer, device = load_model_and_tokenizer(args)
    token_cap = max(args.sanity_tokens, args.seq_len)
    tokens, meta = load_split_tokens(
        args,
        tokenizer,
        args.val_split,
        token_cap,
        window_name="sanity",
        skip_tokens=args.val_skip_tokens,
    )
    ids = tokens[: args.seq_len].unsqueeze(0).to(device)
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    dense = hf_dense_loss(model, ids)
    logits = custom_forward_logits(
        model,
        ids,
        method="uniform",
        recent_size=args.recent_size,
        heavy_hitter_budgets=layer_budgets(len(model.model.decoder.layers), "uniform"),
        chunk_size=args.chunk_size,
        full_retention=True,
        retention_stats=None,
    )
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
        "custom_full_retention_loss": custom,
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


def run_evaluate(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer, device = load_model_and_tokenizer(args)
    num_layers = len(model.model.decoder.layers)
    budgets = layer_budgets(num_layers, args.method)
    check_budget_match(budgets, args.avg_heavy_hitter_size)
    evaluate_split.seq_len = args.seq_len
    cache_full_exclude = args.cache_full_exclude_tokens
    if cache_full_exclude is None:
        cache_full_exclude = args.recent_size + max(budgets)

    val_tokens, val_meta = load_split_tokens(
        args,
        tokenizer,
        args.val_split,
        args.val_tokens,
        window_name="validation",
        skip_tokens=args.val_skip_tokens,
    )
    test_tokens, test_meta = load_split_tokens(
        args,
        tokenizer,
        args.test_split,
        args.test_tokens,
        window_name="test",
        skip_tokens=args.test_skip_tokens,
    )
    retention_stats = init_retention_stats(num_layers)

    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    val = evaluate_split(
        model,
        val_tokens,
        split="validation",
        method=args.method,
        recent_size=args.recent_size,
        budgets=budgets,
        chunk_size=args.chunk_size,
        device=device,
        retention_stats=retention_stats,
        report_loss_after_cache_full=args.report_loss_after_cache_full,
        cache_full_exclude_tokens=cache_full_exclude,
    )
    test = evaluate_split(
        model,
        test_tokens,
        split="test",
        method=args.method,
        recent_size=args.recent_size,
        budgets=budgets,
        chunk_size=args.chunk_size,
        device=device,
        retention_stats=retention_stats,
        report_loss_after_cache_full=args.report_loss_after_cache_full,
        cache_full_exclude_tokens=cache_full_exclude,
    )
    groups = layer_groups(num_layers)
    per_layer_budget = [
        {
            "layer_id": layer_idx,
            "group_id": groups[layer_idx],
            "heavy_hitter_budget": budgets[layer_idx],
            "recent_size": args.recent_size,
            "total_retained_budget": args.recent_size + budgets[layer_idx],
        }
        for layer_idx in range(num_layers)
    ]
    profile = {
        "model_name": args.model_name,
        "dataset": args.dataset,
        "dataset_name": args.dataset_name,
        "data_files": args.data_files,
        "method": args.method,
        "num_layers": num_layers,
        "evaluation_context_length": args.seq_len,
        "recent_size": args.recent_size,
        "heavy_hitter_budgets": budgets,
        "group_assignments": groups,
        "per_layer_budget": per_layer_budget,
        "avg_heavy_hitter_budget": sum(budgets) / len(budgets),
        "avg_total_retained_budget": args.recent_size + sum(budgets) / len(budgets),
        "avg_heavy_hitter_budget_check": abs(sum(budgets) / len(budgets) - args.avg_heavy_hitter_size) < 1e-6,
        "avg_total_retained_budget_check": abs(
            args.recent_size + sum(budgets) / len(budgets) - (args.recent_size + args.avg_heavy_hitter_size)
        )
        < 1e-6,
        "layer_profiles": summarize_retention_stats(retention_stats, budgets, args.recent_size),
    }
    metrics = {
        "validation": val.__dict__,
        "test": test.__dict__,
        "val_cache": val_meta,
        "test_cache": test_meta,
    }
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
        "val_loss_after_cache_full": val.loss_after_cache_full,
        "val_ppl_after_cache_full": val.ppl_after_cache_full,
        "test_loss_after_cache_full": test.loss_after_cache_full,
        "test_ppl_after_cache_full": test.ppl_after_cache_full,
        "cache_full_exclude_tokens": cache_full_exclude,
        "recent_size": args.recent_size,
        "avg_heavy_hitter_budget": profile["avg_heavy_hitter_budget"],
        "avg_total_retained_budget": profile["avg_total_retained_budget"],
        "tok_per_s": total_tokens / total_seconds if total_seconds > 0 else 0.0,
        "total_tokens": total_tokens,
        "total_seconds": total_seconds,
        "num_layers": num_layers,
        "seq_len": args.seq_len,
        "evaluation_context_length": args.seq_len,
        "chunk_size": args.chunk_size,
    }
    if torch.cuda.is_available() and device.type == "cuda":
        summary["cuda_max_memory_allocated_mb"] = torch.cuda.max_memory_allocated() / (1024**2)
        summary["cuda_max_memory_reserved_mb"] = torch.cuda.max_memory_reserved() / (1024**2)

    write_json(out_dir / "retention_profile.json", profile)
    write_json(out_dir / "eval_metrics.json", metrics)
    write_json(out_dir / "run_summary.json", summary)
    print(json.dumps(summary, indent=2))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if args.mode == "sanity":
        run_sanity(args)
    else:
        run_evaluate(args)


if __name__ == "__main__":
    main()
