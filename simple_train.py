"""
Self-contained single-GPU training script for NSA budget-allocation experiments.
Supports streaming large-data training plus cached validation/test subsets.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable

# Use HuggingFace mirror for Chinese servers
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent))

DEFAULTS = dict(
    seq_len=2048,
    batch_size=2,
    grad_accum=4,
    steps=2048,
    warmup_steps=128,
    lr=3e-4,
    lr_min_ratio=0.1,
    max_norm=1.0,
    seed=42,
    log_every=20,
    save_every=512,
    dataset="wikitext",
    dataset_name="wikitext-103-v1",
    tokenizer_path="gpt2",
    dump_dir="exp",
    budget_aux_loss_weight=0.0,
    budget_aux_mode="variance",
    budget_aux_target_min=40.0,
    budget_aux_target_max=60.0,
    budget_aux_target_weight=0.0,
    budget_shuffle_mode="none",
    budget_center_mode="none",
    budget_random_mode="none",
    budget_clamp_mode="none",
    budget_profile_path="",
    budget_profile_out="",
    save_budget_profile=False,
    save_final_ckpt_for_attribution=False,
    train_split="train",
    train_data_files="",
    train_tokens=0,
    train_skip_tokens=0,
    cache_train_subset=True,
    val_split="",
    val_data_files="",
    test_split="",
    test_data_files="",
    eval_every=0,
    val_tokens=0,
    val_skip_tokens=0,
    test_tokens=0,
    test_skip_tokens=0,
    stream_shuffle_buffer=10000,
    save_final_ckpt=True,
    text_column="text",
    eval_cache_dir="",
    eval_cache_seed=0,
    budget_diagnostics=False,
)

TEXT_FALLBACK_COLUMNS = ("text", "content")


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got: {value}")


def optional_str(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized or normalized.lower() in {"none", "null"}:
        return None
    return normalized


def parse_data_files(value: str | None):
    normalized = optional_str(value)
    if normalized is None:
        return None
    parts = [part.strip() for part in normalized.split(",") if part.strip()]
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else parts


def get_lr(step: int, warmup: int, total: int, lr: float, lr_min: float) -> float:
    if step < warmup:
        return lr * step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_min + (lr - lr_min) * cosine


def round_float(value: float, ndigits: int = 6) -> float:
    return round(float(value), ndigits)


def tensor_quantiles(flat: torch.Tensor) -> dict[str, float]:
    if flat.numel() == 0:
        return {}
    qs = torch.tensor([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99], device=flat.device)
    values = torch.quantile(flat.float(), qs)
    names = ["p10", "p25", "p50", "p75", "p90", "p95", "p99"]
    return {name: round_float(value.item()) for name, value in zip(names, values)}


def add_tensor_stats(
    row: dict[str, Any],
    prefix: str,
    tensors: list[torch.Tensor],
    include_layer: bool,
    include_quantiles: bool,
) -> torch.Tensor | None:
    if not tensors:
        return None
    stack = torch.stack([tensor.detach().float() for tensor in tensors], dim=0)
    flat = stack.reshape(-1)
    row[f"{prefix}_mean"] = round_float(flat.mean().item())
    row[f"{prefix}_var"] = round_float(flat.var(unbiased=False).item())
    row[f"{prefix}_min"] = round_float(flat.min().item())
    row[f"{prefix}_max"] = round_float(flat.max().item())
    if include_layer:
        layer_dims = tuple(range(1, stack.ndim))
        row[f"layer_{prefix}_mean"] = [round_float(value) for value in stack.mean(dim=layer_dims).tolist()]
        row[f"layer_{prefix}_var"] = [
            round_float(value) for value in stack.var(dim=layer_dims, unbiased=False).tolist()
        ]
    if include_quantiles:
        quantiles = tensor_quantiles(flat)
        row[f"{prefix}_quantiles"] = quantiles
        for name, value in quantiles.items():
            row[f"{prefix}_{name}"] = value
    return stack


def layer_base_budget_value(attn) -> float | None:
    value = getattr(attn, "block_counts", None)
    if value is None:
        return None
    if torch.is_tensor(value):
        return float(value.detach().float().mean().item())
    if isinstance(value, (list, tuple)):
        return float(sum(value) / max(len(value), 1))
    return float(value)


def collect_budget_diagnostics(model, include_layer: bool = False, include_quantiles: bool = False) -> dict[str, Any]:
    row: dict[str, Any] = {"has_budget": False}
    layers = getattr(getattr(model, "model", None), "layers", [])
    budget_tensors: list[torch.Tensor] = []
    actual_k_tensors: list[torch.Tensor] = []
    base_values: list[float] = []
    dynamic_base_values: list[float] = []
    block_sizes: list[float] = []
    window_sizes: list[float] = []
    layer_indices: list[int] = []

    with torch.no_grad():
        for layer_idx, layer in enumerate(layers):
            attn = getattr(layer, "attn", None)
            if attn is None:
                continue
            base_value = layer_base_budget_value(attn)
            if base_value is not None:
                base_values.append(base_value)
            block_sizes.append(float(getattr(attn, "block_size", 1) or 1))
            window_sizes.append(float(getattr(attn, "window_size", 0) or 0))
            budget_tensor = getattr(attn, "_budget_tensor", None)
            actual_k_tensor = getattr(attn, "_dynamic_k_tensor", None)
            if budget_tensor is None:
                continue
            budget_tensors.append(budget_tensor)
            layer_indices.append(layer_idx)
            if actual_k_tensor is not None:
                actual_k_tensors.append(actual_k_tensor)
            if base_value is not None:
                dynamic_base_values.append(base_value)

        if not budget_tensors:
            if base_values:
                static = torch.tensor(base_values, dtype=torch.float32)
                row["dynamic_layers"] = 0
                row["actual_k_mean"] = round_float(static.mean().item())
                row["actual_k_var"] = round_float(static.var(unbiased=False).item())
                row["actual_k_min"] = round_float(static.min().item())
                row["actual_k_max"] = round_float(static.max().item())
                if include_layer:
                    row["layer_actual_k_mean"] = [round_float(value) for value in static.tolist()]
                    row["layer_actual_k_var"] = [0.0 for _ in base_values]
                if include_quantiles:
                    quantiles = tensor_quantiles(static)
                    row["actual_k_quantiles"] = quantiles
                    for name, value in quantiles.items():
                        row[f"actual_k_{name}"] = value
                block_size = sum(block_sizes) / max(len(block_sizes), 1)
                seq_len = 2048.0
                kv_access_mean = static.mean() * block_size
                row["estimated_selected_blocks_mean"] = round_float(static.mean().item())
                row["estimated_kv_access_mean"] = round_float(kv_access_mean.item())
                row["estimated_sliding_window_tokens_mean"] = round_float(sum(window_sizes) / max(len(window_sizes), 1))
                row["estimated_sparse_attention_budget_ratio"] = round_float(kv_access_mean.item() / seq_len)
            return row

        row["has_budget"] = True
        row["dynamic_layers"] = len(budget_tensors)
        row["layer_indices"] = layer_indices
        budget_stack = add_tensor_stats(row, "budget", budget_tensors, include_layer, include_quantiles)
        actual_k_stack = add_tensor_stats(row, "actual_k", actual_k_tensors, include_layer, include_quantiles)

        if actual_k_stack is not None:
            block_size = sum(block_sizes) / max(len(block_sizes), 1)
            seq_len = float(actual_k_stack.shape[2]) if actual_k_stack.ndim >= 3 else 1.0
            selected_blocks_mean = actual_k_stack.float().mean()
            kv_access_mean = selected_blocks_mean * block_size
            window_tokens_mean = sum(window_sizes) / max(len(window_sizes), 1)
            row["estimated_selected_blocks_mean"] = round_float(selected_blocks_mean.item())
            row["estimated_kv_access_mean"] = round_float(kv_access_mean.item())
            row["estimated_sliding_window_tokens_mean"] = round_float(window_tokens_mean)
            row["estimated_sparse_attention_budget_ratio"] = round_float(kv_access_mean.item() / max(seq_len, 1.0))

        if actual_k_stack is not None and len(dynamic_base_values) == actual_k_stack.shape[0]:
            base = torch.tensor(dynamic_base_values, device=actual_k_stack.device, dtype=actual_k_stack.dtype).view(
                -1, *([1] * (actual_k_stack.ndim - 1))
            )
            delta = actual_k_stack - base
            row["delta_vs_mono_mean"] = round_float(delta.mean().item())
            row["delta_vs_mono_abs_mean"] = round_float(delta.abs().mean().item())
            if include_layer:
                layer_dims = tuple(range(1, delta.ndim))
                row["layer_delta_vs_mono_mean"] = [round_float(value) for value in delta.mean(dim=layer_dims).tolist()]
                row["layer_delta_vs_mono_abs_mean"] = [
                    round_float(value) for value in delta.abs().mean(dim=layer_dims).tolist()
                ]

        return row


def update_budget_profile_accumulator(accumulator: dict[int, dict[str, Any]], model) -> None:
    layers = getattr(getattr(model, "model", None), "layers", [])
    with torch.no_grad():
        for layer_idx, layer in enumerate(layers):
            attn = getattr(layer, "attn", None)
            if attn is None:
                continue
            actual_k = getattr(attn, "_dynamic_k_tensor", None)
            if actual_k is None:
                continue
            flat = actual_k.detach().long().reshape(-1).cpu()
            if flat.numel() == 0:
                continue
            entry = accumulator.setdefault(
                layer_idx,
                {
                    "counts": {},
                    "samples": 0,
                    "block_size": int(getattr(attn, "block_size", 1) or 1),
                    "block_counts": layer_base_budget_value(attn),
                    "budget_min_k": int(getattr(attn, "budget_min_k", 0)),
                    "budget_max_k": int(getattr(attn, "budget_max_k", 0)),
                    "num_kv_heads": int(getattr(attn, "num_kv_heads", 0)),
                },
            )
            values, counts = torch.unique(flat, return_counts=True)
            for value, count in zip(values.tolist(), counts.tolist()):
                key = str(int(value))
                entry["counts"][key] = int(entry["counts"].get(key, 0)) + int(count)
            entry["samples"] = int(entry.get("samples", 0)) + int(flat.numel())


def write_budget_profile(
    path: Path,
    accumulator: dict[int, dict[str, Any]],
    *,
    args,
    cfg_dict: dict[str, Any],
    step: int,
) -> dict[str, Any]:
    layers = []
    for layer_idx in sorted(accumulator):
        entry = accumulator[layer_idx]
        counts = {str(key): int(value) for key, value in sorted(entry["counts"].items(), key=lambda item: int(item[0]))}
        total = max(sum(counts.values()), 1)
        values = torch.tensor([int(key) for key in counts], dtype=torch.float32)
        weights = torch.tensor([counts[str(int(value.item()))] for value in values], dtype=torch.float32)
        mean = float((values * weights).sum().item() / total)
        var = float((((values - mean) ** 2) * weights).sum().item() / total)
        expanded = torch.repeat_interleave(values, weights.long()) if total <= 2_000_000 else values
        quantiles = tensor_quantiles(expanded) if expanded.numel() else {}
        layers.append(
            {
                "layer_id": layer_idx,
                "counts": counts,
                "probabilities": {key: count / total for key, count in counts.items()},
                "samples": int(total),
                "mean": round_float(mean),
                "var": round_float(var),
                "p10": quantiles.get("p10"),
                "p50": quantiles.get("p50"),
                "p90": quantiles.get("p90"),
                "block_size": entry.get("block_size"),
                "block_counts": entry.get("block_counts"),
                "budget_min_k": entry.get("budget_min_k"),
                "budget_max_k": entry.get("budget_max_k"),
                "num_kv_heads": entry.get("num_kv_heads"),
            }
        )
    profile = {
        "format": "nsa_budget_profile_v1",
        "dataset": args.dataset,
        "dataset_name": args.dataset_name,
        "config": Path(args.config).name,
        "seed": args.seed,
        "step": step,
        "seq_len": args.seq_len,
        "num_layers": len(layers),
        "vocab_size": cfg_dict.get("vocab_size"),
        "layers": layers,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    return profile


def load_budget_replay_profile(path: Path) -> dict[str, Any]:
    profile = json.loads(path.read_text(encoding="utf-8"))
    if profile.get("format") != "nsa_budget_profile_v1":
        raise ValueError(f"Unsupported budget profile format in {path}")
    if not profile.get("layers"):
        raise ValueError(f"Budget profile has no layers: {path}")
    return profile


def configure_budget_controls(model, args, profile: dict[str, Any] | None = None) -> None:
    profile_layers = {}
    if profile is not None:
        profile_layers = {int(layer["layer_id"]): layer for layer in profile.get("layers", [])}
    for layer_idx, block in enumerate(model.model.layers):
        attn = getattr(block, "attn", None)
        if attn is None:
            continue
        attn.budget_shuffle_mode = args.budget_shuffle_mode
        attn.budget_center_mode = args.budget_center_mode
        attn.budget_random_mode = args.budget_random_mode
        attn.budget_clamp_mode = args.budget_clamp_mode
        if args.budget_random_mode == "layer_histogram_replay":
            layer_profile = profile_layers.get(layer_idx)
            if layer_profile is None:
                raise ValueError(f"Budget replay profile missing layer {layer_idx}")
            counts = layer_profile.get("counts", {})
            values = [int(key) for key in sorted(counts, key=lambda item: int(item))]
            probs = [float(counts[str(value)]) for value in values]
            if not values or sum(probs) <= 0:
                raise ValueError(f"Invalid replay distribution for layer {layer_idx}")
            attn.budget_replay_values = torch.tensor(values, dtype=torch.long)
            attn.budget_replay_probs = torch.tensor(probs, dtype=torch.float32)


def get_loss_breakdown(model, fallback_loss: torch.Tensor) -> dict[str, float]:
    breakdown = getattr(model, "_last_loss_breakdown", None) or {}
    total_loss = float(breakdown.get("total_loss", fallback_loss.detach().float().item()))
    lm_loss = float(breakdown.get("lm_loss", total_loss))
    aux_loss = float(breakdown.get("aux_loss", total_loss - lm_loss))
    return {
        "total_loss": total_loss,
        "lm_loss": lm_loss,
        "aux_loss": aux_loss,
        "target_band_loss": float(breakdown.get("target_band_loss", 0.0)),
        "budget_var_mean": float(breakdown.get("budget_var_mean", 0.0)),
    }


def build_tokenizer(path: str):
    from transformers import AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained(path)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        return tok
    except Exception as exc:
        print(f"[warn] tokenizer '{path}' unavailable ({exc}), trying tiktoken fallback")

    try:
        import tiktoken

        class TikTokenWrapper:
            def __init__(self):
                self._enc = tiktoken.get_encoding("cl100k_base")
                self.eos_token_id = self._enc.eot_token
                self.pad_token = "<|endoftext|>"

            def __call__(self, texts, truncation=False, return_attention_mask=False, **kwargs):
                if isinstance(texts, str):
                    texts = [texts]
                return {"input_ids": [self._enc.encode(t) for t in texts]}

        print("[info] using tiktoken cl100k_base (vocab=100277)")
        return TikTokenWrapper()
    except ImportError as exc:
        raise RuntimeError("No tokenizer available. Install tiktoken or ensure network access.") from exc


def infer_text_column(dataset_obj, requested: str | None) -> str:
    column_names = getattr(dataset_obj, "column_names", None)
    if column_names is None and getattr(dataset_obj, "features", None):
        column_names = list(dataset_obj.features.keys())
    if requested and column_names and requested in column_names:
        return requested
    if requested and not column_names:
        return requested
    if column_names:
        for name in TEXT_FALLBACK_COLUMNS:
            if name in column_names:
                return name
        for name in column_names:
            if name not in {"meta", "id"}:
                return name
    return requested or "text"


def dataset_kwargs(dataset: str, dataset_name: str | None, split: str, streaming: bool, data_files: str | None = None):
    kwargs: dict[str, Any] = {
        "path": dataset,
        "split": split,
        "streaming": streaming,
    }
    parsed_data_files = parse_data_files(data_files)
    if parsed_data_files is not None:
        kwargs["data_files"] = {split: parsed_data_files}
    if dataset_name is not None:
        kwargs["name"] = dataset_name
    if not streaming:
        kwargs["num_proc"] = 2
    return kwargs


def tokenize_batch(batch, tokenizer, text_column: str):
    texts = [t for t in batch[text_column] if isinstance(t, str) and t.strip()]
    if not texts:
        return {"input_ids": []}
    return tokenizer(texts, truncation=False, return_attention_mask=False)


def pack_batch(batch, seq_len: int, eos_id: int):
    ids: list[int] = []
    for row in batch["input_ids"]:
        if row:
            ids.extend(row)
            ids.append(eos_id)
    out = []
    for i in range(0, len(ids) - seq_len + 1, seq_len):
        out.append(ids[i : i + seq_len])
    return {"input_ids": out}


def build_training_dataset(args, tokenizer):
    from datasets import load_dataset

    streaming = args.dataset != "wikitext"
    ds = load_dataset(**dataset_kwargs(args.dataset, args.dataset_name, args.train_split, streaming, args.train_data_files))
    text_column = infer_text_column(ds, args.text_column)
    print(
        f"Loading training dataset {args.dataset} / {args.dataset_name or '-'} "
        f"split={args.train_split} streaming={streaming} text_column={text_column} "
        f"data_files={args.train_data_files or '-'}"
    )

    if streaming and args.stream_shuffle_buffer > 0:
        ds = ds.shuffle(seed=args.seed, buffer_size=args.stream_shuffle_buffer)

    eos_id = getattr(tokenizer, "eos_token_id", getattr(getattr(tokenizer, "_enc", None), "eot_token", 0))
    remove_columns = getattr(ds, "column_names", None) or [text_column]

    if streaming:
        ds = ds.map(lambda batch: tokenize_batch(batch, tokenizer, text_column), batched=True, remove_columns=remove_columns)
        ds = ds.map(lambda batch: pack_batch(batch, args.seq_len, eos_id), batched=True, batch_size=256)
    else:
        ds = ds.map(
            lambda batch: tokenize_batch(batch, tokenizer, text_column),
            batched=True,
            remove_columns=remove_columns,
            num_proc=2,
            desc="tokenize",
        )
        ds = ds.map(
            lambda batch: pack_batch(batch, args.seq_len, eos_id),
            batched=True,
            batch_size=4096,
            remove_columns=["input_ids"],
            num_proc=2,
            desc="pack",
        )
    return ds, text_column


class PackedTensorDataset(Dataset):
    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor

    def __len__(self) -> int:
        return int(self.tensor.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"input_ids": self.tensor[index]}


def collate(batch):
    rows = []
    for item in batch:
        ids = item["input_ids"]
        if not torch.is_tensor(ids):
            ids = torch.tensor(ids, dtype=torch.long)
        rows.append(ids.to(torch.long))
    return torch.stack(rows)


def autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def make_eval_cache_key(
    dataset: str,
    dataset_name: str | None,
    split: str,
    data_files: str | None,
    text_column: str,
    tokenizer_path: str,
    seq_len: int,
    target_tokens: int,
    skip_tokens: int,
    cache_seed: int,
) -> str:
    payload = {
        "dataset": dataset,
        "dataset_name": dataset_name,
        "split": split,
        "data_files": data_files,
        "text_column": text_column,
        "tokenizer_path": tokenizer_path,
        "seq_len": seq_len,
        "target_tokens": target_tokens,
        "skip_tokens": skip_tokens,
        "cache_seed": cache_seed,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return digest[:16]


def iter_text_batches(dataset_obj, text_column: str, batch_size: int = 64) -> Iterable[list[str]]:
    bucket: list[str] = []
    for example in dataset_obj:
        text = example.get(text_column)
        if isinstance(text, str) and text.strip():
            bucket.append(text)
        if len(bucket) >= batch_size:
            yield bucket
            bucket = []
    if bucket:
        yield bucket


def try_slice_eval_cache_from_existing(
    *,
    dataset: str,
    dataset_name: str | None,
    split: str,
    data_files: str | None,
    tokenizer_path: str,
    seq_len: int,
    target_tokens: int,
    skip_tokens: int,
    text_column: str,
    cache_dir: Path,
    cache_seed: int,
    cache_key: str,
    tensor_path: Path,
    meta_path: Path,
) -> dict[str, Any] | None:
    """Create a smaller cache by slicing an existing same-offset superset cache."""

    if not cache_dir.exists():
        return None
    target_sequences = target_tokens // seq_len
    if target_sequences <= 0:
        return None

    candidates: list[tuple[int, Path, Path, dict[str, Any]]] = []
    pattern = f"{split}_skip{skip_tokens}_tok*_seq{seq_len}_*.json"
    for candidate_meta_path in cache_dir.glob(pattern):
        if candidate_meta_path == meta_path:
            continue
        try:
            metadata = json.loads(candidate_meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        if metadata.get("dataset") != dataset:
            continue
        if metadata.get("dataset_name") != dataset_name:
            continue
        if metadata.get("split") != split:
            continue
        if metadata.get("data_files") != data_files:
            continue
        if metadata.get("tokenizer_path") != tokenizer_path:
            continue
        if int(metadata.get("seq_len", -1)) != seq_len:
            continue
        if int(metadata.get("skip_tokens", -1)) != skip_tokens:
            continue
        if int(metadata.get("cache_seed", -1)) != cache_seed:
            continue
        if metadata.get("text_column", text_column) != text_column:
            continue

        actual_tokens = int(metadata.get("actual_tokens", 0))
        if actual_tokens < target_sequences * seq_len:
            continue
        candidate_tensor_path = candidate_meta_path.with_suffix(".pt")
        if not candidate_tensor_path.is_file():
            continue
        candidates.append((actual_tokens, candidate_meta_path, candidate_tensor_path, metadata))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0])
    source_tokens, source_meta_path, source_tensor_path, source_meta = candidates[0]
    print(
        f"[cache] slicing {split} subset from existing cache {source_tensor_path} "
        f"(target_tokens={target_tokens}, skip_tokens={skip_tokens})"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    source_tensor = torch.load(source_tensor_path, map_location="cpu")
    if not torch.is_tensor(source_tensor) or source_tensor.ndim != 2 or int(source_tensor.shape[1]) != seq_len:
        raise RuntimeError(f"Unexpected source cache tensor shape in {source_tensor_path}")

    tensor = source_tensor[:target_sequences].clone()
    if int(tensor.shape[0]) != target_sequences:
        raise RuntimeError(
            f"Source cache is too small after load: wanted {target_sequences} rows, got {int(tensor.shape[0])}"
        )

    metadata = {
        "dataset": dataset,
        "dataset_name": dataset_name,
        "split": split,
        "text_column": text_column,
        "tokenizer_path": tokenizer_path,
        "seq_len": seq_len,
        "target_tokens": target_tokens,
        "target_packed_tokens": target_sequences * seq_len,
        "skip_tokens": skip_tokens,
        "actual_tokens": int(tensor.numel()),
        "num_sequences": int(tensor.shape[0]),
        "cache_seed": cache_seed,
        "cache_key": cache_key,
        "examples_consumed": source_meta.get("examples_consumed"),
        "source_cache_meta": str(source_meta_path),
        "source_cache_tokens": source_tokens,
        "source_cache_key": source_meta.get("cache_key"),
        "tensor_path": str(tensor_path),
        "meta_path": str(meta_path),
    }
    torch.save(tensor, tensor_path)
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def build_eval_cache(
    *,
    dataset: str,
    dataset_name: str | None,
    split: str,
    data_files: str | None,
    tokenizer,
    tokenizer_path: str,
    seq_len: int,
    target_tokens: int,
    skip_tokens: int = 0,
    text_column: str,
    cache_dir: Path,
    cache_seed: int,
):
    resolved_text_column = text_column or "text"
    eos_id = getattr(tokenizer, "eos_token_id", getattr(getattr(tokenizer, "_enc", None), "eot_token", 0))
    cache_key = make_eval_cache_key(
        dataset=dataset,
        dataset_name=dataset_name,
        split=split,
        data_files=data_files,
        text_column=resolved_text_column,
        tokenizer_path=tokenizer_path,
        seq_len=seq_len,
        target_tokens=target_tokens,
        skip_tokens=skip_tokens,
        cache_seed=cache_seed,
    )
    stem = f"{split}_skip{skip_tokens}_tok{target_tokens}_seq{seq_len}_{cache_key}"
    tensor_path = cache_dir / f"{stem}.pt"
    meta_path = cache_dir / f"{stem}.json"

    if tensor_path.is_file() and meta_path.is_file():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        metadata["tensor_path"] = str(tensor_path)
        metadata["meta_path"] = str(meta_path)
        metadata["text_column"] = resolved_text_column
        return metadata

    sliced_metadata = try_slice_eval_cache_from_existing(
        dataset=dataset,
        dataset_name=dataset_name,
        split=split,
        data_files=data_files,
        tokenizer_path=tokenizer_path,
        seq_len=seq_len,
        target_tokens=target_tokens,
        skip_tokens=skip_tokens,
        text_column=resolved_text_column,
        cache_dir=cache_dir,
        cache_seed=cache_seed,
        cache_key=cache_key,
        tensor_path=tensor_path,
        meta_path=meta_path,
    )
    if sliced_metadata is not None:
        return sliced_metadata

    from datasets import load_dataset

    ds = load_dataset(**dataset_kwargs(dataset, dataset_name, split, streaming=True, data_files=data_files))
    resolved_text_column = infer_text_column(ds, text_column)
    print(
        f"[cache] building {split} subset at {tensor_path} "
        f"(target_tokens={target_tokens}, skip_tokens={skip_tokens}, seq_len={seq_len}, "
        f"text_column={resolved_text_column}, data_files={data_files or '-'})"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    packed_rows: list[list[int]] = []
    token_buffer: list[int] = []
    target_sequences = max(1, target_tokens // seq_len)
    target_packed_tokens = target_sequences * seq_len
    consumed = 0
    skipped = 0
    examples = 0

    for texts in iter_text_batches(ds, resolved_text_column):
        encoded = tokenizer(texts, truncation=False, return_attention_mask=False)["input_ids"]
        for row in encoded:
            if not row:
                continue
            token_buffer.extend(row)
            token_buffer.append(eos_id)
            examples += 1
            while len(token_buffer) >= seq_len:
                if skipped + seq_len <= skip_tokens:
                    del token_buffer[:seq_len]
                    skipped += seq_len
                    continue
                if consumed + seq_len > target_packed_tokens:
                    break
                packed_rows.append(token_buffer[:seq_len])
                del token_buffer[:seq_len]
                consumed += seq_len
            if consumed >= target_packed_tokens:
                break
        if consumed >= target_packed_tokens:
            break

    if not packed_rows:
        raise RuntimeError(f"Failed to build any packed sequences for {dataset}:{split}")

    tensor = torch.tensor(packed_rows, dtype=torch.int32)
    metadata = {
        "dataset": dataset,
        "dataset_name": dataset_name,
        "split": split,
        "data_files": data_files,
        "text_column": resolved_text_column,
        "tokenizer_path": tokenizer_path,
        "seq_len": seq_len,
        "target_tokens": target_tokens,
        "skip_tokens": skip_tokens,
        "actual_tokens": int(tensor.numel()),
        "num_sequences": int(tensor.shape[0]),
        "cache_seed": cache_seed,
        "cache_key": cache_key,
        "examples_consumed": examples,
        "tensor_path": str(tensor_path),
        "meta_path": str(meta_path),
    }
    torch.save(tensor, tensor_path)
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def load_eval_dataset(cache_meta: dict[str, Any]) -> PackedTensorDataset:
    tensor = torch.load(cache_meta["tensor_path"], map_location="cpu")
    if not torch.is_tensor(tensor):
        raise RuntimeError(f"Unexpected cached tensor payload in {cache_meta['tensor_path']}")
    return PackedTensorDataset(tensor)


def evaluate_model(model, loader, device, split_name: str) -> dict[str, Any]:
    original_aux = getattr(model, "budget_aux_loss_weight", 0.0)
    original_target_weight = getattr(model, "budget_aux_target_weight", 0.0)
    model.budget_aux_loss_weight = 0.0
    model.budget_aux_target_weight = 0.0
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    total_sequences = 0
    start = time.time()

    with torch.no_grad():
        for ids in loader:
            ids = ids.to(device)
            with autocast_context(device):
                out = model(input_ids=ids, labels=ids)
            token_count = int(ids.numel())
            total_loss += float(out.loss.item()) * token_count
            total_tokens += token_count
            total_sequences += int(ids.shape[0])

    model.train()
    model.budget_aux_loss_weight = original_aux
    model.budget_aux_target_weight = original_target_weight

    mean_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(mean_loss, 20.0))
    return {
        "split": split_name,
        "loss": round(mean_loss, 6),
        "lm_loss": round(mean_loss, 6),
        "total_loss": round(mean_loss, 6),
        "aux_loss": 0.0,
        "ppl": round(ppl, 6),
        "tokens": total_tokens,
        "sequences": total_sequences,
        "elapsed": round(time.time() - start, 2),
    }


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to NSAConfig JSON")
    parser.add_argument("--name", required=True, help="experiment name")
    parser.add_argument("--seq_len", type=int, default=DEFAULTS["seq_len"])
    parser.add_argument("--batch_size", type=int, default=DEFAULTS["batch_size"])
    parser.add_argument("--grad_accum", type=int, default=DEFAULTS["grad_accum"])
    parser.add_argument("--steps", type=int, default=DEFAULTS["steps"])
    parser.add_argument("--warmup_steps", type=int, default=DEFAULTS["warmup_steps"])
    parser.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    parser.add_argument("--lr_min_ratio", type=float, default=DEFAULTS["lr_min_ratio"])
    parser.add_argument("--max_norm", type=float, default=DEFAULTS["max_norm"])
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    parser.add_argument("--log_every", type=int, default=DEFAULTS["log_every"])
    parser.add_argument("--save_every", type=int, default=DEFAULTS["save_every"])
    parser.add_argument("--dataset", default=DEFAULTS["dataset"])
    parser.add_argument("--dataset_name", default=DEFAULTS["dataset_name"])
    parser.add_argument("--tokenizer_path", default=DEFAULTS["tokenizer_path"])
    parser.add_argument("--dump_dir", default=DEFAULTS["dump_dir"])
    parser.add_argument("--budget_aux_loss_weight", type=float, default=DEFAULTS["budget_aux_loss_weight"])
    parser.add_argument("--budget_aux_mode", choices=["variance", "target_band"], default=DEFAULTS["budget_aux_mode"])
    parser.add_argument("--budget_aux_target_min", type=float, default=DEFAULTS["budget_aux_target_min"])
    parser.add_argument("--budget_aux_target_max", type=float, default=DEFAULTS["budget_aux_target_max"])
    parser.add_argument("--budget_aux_target_weight", type=float, default=DEFAULTS["budget_aux_target_weight"])
    parser.add_argument("--budget_shuffle_mode", choices=["none", "token"], default=DEFAULTS["budget_shuffle_mode"])
    parser.add_argument("--budget_center_mode", choices=["none", "mono_layer_mean"], default=DEFAULTS["budget_center_mode"])
    parser.add_argument(
        "--budget_random_mode",
        choices=["none", "layer_histogram_replay"],
        default=DEFAULTS["budget_random_mode"],
    )
    parser.add_argument("--budget_clamp_mode", choices=["none", "mono_layer_sum"], default=DEFAULTS["budget_clamp_mode"])
    parser.add_argument("--budget_profile_path", default=DEFAULTS["budget_profile_path"])
    parser.add_argument("--budget_profile_out", default=DEFAULTS["budget_profile_out"])
    parser.add_argument("--save_budget_profile", type=str2bool, default=DEFAULTS["save_budget_profile"])
    parser.add_argument(
        "--save_final_ckpt_for_attribution",
        type=str2bool,
        default=DEFAULTS["save_final_ckpt_for_attribution"],
    )
    parser.add_argument("--train_split", default=DEFAULTS["train_split"])
    parser.add_argument("--train_data_files", default=DEFAULTS["train_data_files"])
    parser.add_argument("--train_tokens", type=int, default=DEFAULTS["train_tokens"])
    parser.add_argument("--train_skip_tokens", type=int, default=DEFAULTS["train_skip_tokens"])
    parser.add_argument("--cache_train_subset", type=str2bool, default=DEFAULTS["cache_train_subset"])
    parser.add_argument("--val_split", default=DEFAULTS["val_split"])
    parser.add_argument("--val_data_files", default=DEFAULTS["val_data_files"])
    parser.add_argument("--test_split", default=DEFAULTS["test_split"])
    parser.add_argument("--test_data_files", default=DEFAULTS["test_data_files"])
    parser.add_argument("--eval_every", type=int, default=DEFAULTS["eval_every"])
    parser.add_argument("--val_tokens", type=int, default=DEFAULTS["val_tokens"])
    parser.add_argument("--val_skip_tokens", type=int, default=DEFAULTS["val_skip_tokens"])
    parser.add_argument("--test_tokens", type=int, default=DEFAULTS["test_tokens"])
    parser.add_argument("--test_skip_tokens", type=int, default=DEFAULTS["test_skip_tokens"])
    parser.add_argument("--stream_shuffle_buffer", type=int, default=DEFAULTS["stream_shuffle_buffer"])
    parser.add_argument("--save_final_ckpt", type=str2bool, default=DEFAULTS["save_final_ckpt"])
    parser.add_argument("--text_column", default=DEFAULTS["text_column"])
    parser.add_argument("--eval_cache_dir", default=DEFAULTS["eval_cache_dir"])
    parser.add_argument("--eval_cache_seed", type=int, default=DEFAULTS["eval_cache_seed"])
    parser.add_argument("--budget_diagnostics", type=str2bool, default=DEFAULTS["budget_diagnostics"])
    args = parser.parse_args()
    args.dataset_name = optional_str(args.dataset_name)
    args.val_split = optional_str(args.val_split)
    args.test_split = optional_str(args.test_split)
    args.eval_cache_dir = optional_str(args.eval_cache_dir)
    args.budget_profile_path = optional_str(args.budget_profile_path) or ""
    args.budget_profile_out = optional_str(args.budget_profile_out) or ""
    args.train_data_files = optional_str(args.train_data_files) or ""
    args.val_data_files = optional_str(args.val_data_files) or ""
    args.test_data_files = optional_str(args.test_data_files) or ""
    args.text_column = optional_str(args.text_column) or DEFAULTS["text_column"]
    return args


def main():
    args = parse_args()
    from native_sparse_attention import NSAConfig, NSAForCausalLM

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dump = Path(args.dump_dir) / args.name
    dump.mkdir(parents=True, exist_ok=True)
    log_path = dump / "train_log.jsonl"
    eval_log_path = dump / "eval_log.jsonl"
    test_metrics_path = dump / "test_metrics.json"
    summary_path = dump / "run_summary.json"
    budget_diag_path = dump / "budget_diagnostics.jsonl"
    if args.budget_diagnostics:
        budget_diag_path.write_text("", encoding="utf-8")
    print(f"Experiment: {args.name}  dump: {dump}")

    tokenizer = build_tokenizer(args.tokenizer_path)

    with open(args.config, encoding="utf-8") as handle:
        cfg_dict = json.load(handle)

    tok_vocab = getattr(tokenizer, "vocab_size", getattr(getattr(tokenizer, "_enc", None), "n_vocab", None))
    if tok_vocab and tok_vocab != cfg_dict.get("vocab_size"):
        aligned = ((tok_vocab + 63) // 64) * 64
        print(f"[info] vocab_size: config={cfg_dict['vocab_size']} tokenizer={tok_vocab} -> using {aligned}")
        cfg_dict["vocab_size"] = aligned

    try:
        from flash_attn import flash_attn_func  # noqa: F401
    except ImportError:
        if cfg_dict.get("window_size", 0) > 0:
            print(f"[warn] flash_attn not found -> overriding window_size {cfg_dict['window_size']} -> 0")
            cfg_dict["window_size"] = 0

    cfg_clean = {key: value for key, value in cfg_dict.items() if not key.startswith("_")}
    cfg = NSAConfig(**cfg_clean)
    model = NSAForCausalLM(cfg).to(device).to(torch.bfloat16)
    model.budget_aux_loss_weight = args.budget_aux_loss_weight
    model.budget_aux_mode = args.budget_aux_mode
    model.budget_aux_target_min = args.budget_aux_target_min
    model.budget_aux_target_max = args.budget_aux_target_max
    model.budget_aux_target_weight = args.budget_aux_target_weight
    replay_profile = None
    if args.budget_random_mode == "layer_histogram_replay":
        if not args.budget_profile_path:
            raise ValueError("--budget_profile_path is required for layer_histogram_replay")
        replay_profile = load_budget_replay_profile(Path(args.budget_profile_path))
        if int(replay_profile.get("seq_len", args.seq_len)) != int(args.seq_len):
            raise ValueError(f"Replay profile seq_len mismatch: {replay_profile.get('seq_len')} vs {args.seq_len}")
    configure_budget_controls(model, args, replay_profile)

    total_params = sum(param.numel() for param in model.parameters())
    dyn_layers = sum(1 for block in model.model.layers if hasattr(block.attn, "budget_head"))
    print(
        f"Model params: {total_params/1e6:.1f}M  "
        f"window_size={cfg.window_size}  "
        f"block_counts={cfg.block_counts}  "
        f"budget_mode={cfg.budget_mode}  "
        f"dynamic_layers={dyn_layers}  "
        f"aux_mode={args.budget_aux_mode}  "
        f"aux_loss_w={args.budget_aux_loss_weight}  "
        f"target_band=[{args.budget_aux_target_min},{args.budget_aux_target_max}]  "
        f"target_w={args.budget_aux_target_weight}  "
        f"shuffle={args.budget_shuffle_mode}  "
        f"center={args.budget_center_mode}  "
        f"random={args.budget_random_mode}  "
        f"clamp={args.budget_clamp_mode}"
    )

    with open(dump / "config.json", "w", encoding="utf-8") as handle:
        json.dump(cfg_dict, handle, indent=2)
    with open(dump / "train_args.json", "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)

    eval_cache_dir = Path(args.eval_cache_dir) if args.eval_cache_dir else dump / "eval_cache"
    train_cache_meta = None
    if args.cache_train_subset and args.train_tokens > 0:
        train_cache_meta = build_eval_cache(
            dataset=args.dataset,
            dataset_name=args.dataset_name,
            split=args.train_split,
            data_files=args.train_data_files,
            tokenizer=tokenizer,
            tokenizer_path=args.tokenizer_path,
            seq_len=args.seq_len,
            target_tokens=args.train_tokens,
            skip_tokens=args.train_skip_tokens,
            text_column=args.text_column,
            cache_dir=eval_cache_dir,
            cache_seed=args.eval_cache_seed,
        )
        train_ds = load_eval_dataset(train_cache_meta)
        resolved_text_column = train_cache_meta["text_column"]
        print(
            f"Using cached training subset: {train_cache_meta['tensor_path']} "
            f"({train_cache_meta['actual_tokens']} tokens)"
        )
    else:
        if args.train_tokens > 0:
            print(
                "Using streaming training dataset without train subset cache "
                f"(train_tokens={args.train_tokens}, train_skip_tokens={args.train_skip_tokens})."
            )
        train_ds, resolved_text_column = build_training_dataset(args, tokenizer)
    train_num_workers = 0 if args.dataset != "wikitext" else 2
    train_loader_kwargs = {
        "batch_size": args.batch_size,
        "collate_fn": collate,
        "num_workers": train_num_workers,
    }
    if train_num_workers > 0:
        train_loader_kwargs["prefetch_factor"] = 2
    train_loader = DataLoader(train_ds, **train_loader_kwargs)

    val_loader = None
    val_cache_meta = None
    if args.val_split and args.val_tokens > 0:
        val_cache_meta = build_eval_cache(
            dataset=args.dataset,
            dataset_name=args.dataset_name,
            split=args.val_split,
            data_files=args.val_data_files,
            tokenizer=tokenizer,
            tokenizer_path=args.tokenizer_path,
            seq_len=args.seq_len,
            target_tokens=args.val_tokens,
            skip_tokens=args.val_skip_tokens,
            text_column=resolved_text_column,
            cache_dir=eval_cache_dir,
            cache_seed=args.eval_cache_seed,
        )
        val_loader = DataLoader(load_eval_dataset(val_cache_meta), batch_size=args.batch_size, collate_fn=collate)

    test_loader = None
    test_cache_meta = None
    if args.test_split and args.test_tokens > 0:
        test_cache_meta = build_eval_cache(
            dataset=args.dataset,
            dataset_name=args.dataset_name,
            split=args.test_split,
            data_files=args.test_data_files,
            tokenizer=tokenizer,
            tokenizer_path=args.tokenizer_path,
            seq_len=args.seq_len,
            target_tokens=args.test_tokens,
            skip_tokens=args.test_skip_tokens,
            text_column=resolved_text_column,
            cache_dir=eval_cache_dir,
            cache_seed=args.eval_cache_seed,
        )
        test_loader = DataLoader(load_eval_dataset(test_cache_meta), batch_size=args.batch_size, collate_fn=collate)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        eps=1e-8,
    )

    model.train()
    step = 0
    micro_step = 0
    accum_total_loss = 0.0
    accum_lm_loss = 0.0
    accum_aux_loss = 0.0
    accum_target_band_loss = 0.0
    accum_budget_var_mean = 0.0
    last_logged_train = None
    last_eval = None
    last_budget_diag = None
    budget_profile_accumulator: dict[int, dict[str, Any]] = {}
    t0 = time.time()
    data_iter = iter(train_loader)

    optimizer.zero_grad()

    while step < args.steps:
        try:
            ids = next(data_iter).to(device)
        except StopIteration:
            data_iter = iter(train_loader)
            ids = next(data_iter).to(device)

        with autocast_context(device):
            out = model(input_ids=ids, labels=ids)
            loss = out.loss / args.grad_accum

        loss.backward()
        loss_breakdown = get_loss_breakdown(model, out.loss)
        accum_total_loss += loss_breakdown["total_loss"] / args.grad_accum
        accum_lm_loss += loss_breakdown["lm_loss"] / args.grad_accum
        accum_aux_loss += loss_breakdown["aux_loss"] / args.grad_accum
        accum_target_band_loss += loss_breakdown["target_band_loss"] / args.grad_accum
        accum_budget_var_mean += loss_breakdown["budget_var_mean"] / args.grad_accum
        micro_step += 1

        if micro_step % args.grad_accum == 0:
            step += 1
            cur_lr = get_lr(step, args.warmup_steps, args.steps, args.lr, args.lr * args.lr_min_ratio)
            for param_group in optimizer.param_groups:
                param_group["lr"] = cur_lr

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm).item()
            optimizer.step()
            optimizer.zero_grad()

            if step % args.log_every == 0 or step == 1:
                elapsed = time.time() - t0
                tokens = step * args.grad_accum * args.batch_size * args.seq_len
                tok_per_s = tokens / max(elapsed, 1e-6)
                row = {
                    "step": step,
                    "loss": round(accum_total_loss, 5),
                    "train_total_loss": round(accum_total_loss, 5),
                    "train_lm_loss": round(accum_lm_loss, 5),
                    "train_aux_loss": round(accum_aux_loss, 5),
                    "budget_aux_loss_weight": args.budget_aux_loss_weight,
                    "budget_aux_mode": args.budget_aux_mode,
                    "budget_aux_target_min": args.budget_aux_target_min,
                    "budget_aux_target_max": args.budget_aux_target_max,
                    "budget_aux_target_weight": args.budget_aux_target_weight,
                    "target_band_loss": round(accum_target_band_loss, 6),
                    "budget_var_mean_from_loss": round(accum_budget_var_mean, 6),
                    "lr": round(cur_lr, 8),
                    "grad_norm": round(grad_norm, 4),
                    "tokens": tokens,
                    "elapsed": round(elapsed, 1),
                    "tok_per_s": round(tok_per_s, 2),
                }
                if args.budget_diagnostics:
                    budget_diag = collect_budget_diagnostics(model, include_layer=False, include_quantiles=True)
                    if args.save_budget_profile:
                        update_budget_profile_accumulator(budget_profile_accumulator, model)
                    row["budget_has_dynamic"] = bool(budget_diag.get("has_budget", False))
                    for key in (
                        "budget_mean",
                        "budget_var",
                        "budget_min",
                        "budget_max",
                        "actual_k_mean",
                        "actual_k_var",
                        "actual_k_min",
                        "actual_k_max",
                        "actual_k_p10",
                        "actual_k_p25",
                        "actual_k_p50",
                        "actual_k_p75",
                        "actual_k_p90",
                        "estimated_selected_blocks_mean",
                        "estimated_kv_access_mean",
                        "estimated_sliding_window_tokens_mean",
                        "estimated_sparse_attention_budget_ratio",
                    ):
                        if key in budget_diag:
                            row[key] = budget_diag[key]
                    if torch.cuda.is_available():
                        row["cuda_max_memory_allocated_mb"] = round_float(torch.cuda.max_memory_allocated(device) / (1024**2))
                        row["cuda_max_memory_reserved_mb"] = round_float(torch.cuda.max_memory_reserved(device) / (1024**2))
                last_logged_train = row
                print(
                    f"step {step:5d}/{args.steps}  "
                    f"loss={accum_total_loss:.4f}  "
                    f"lm={accum_lm_loss:.4f}  "
                    f"aux={accum_aux_loss:.4f}  "
                    f"lr={cur_lr:.2e}  "
                    f"gnorm={grad_norm:.3f}  "
                    f"{tok_per_s/1e3:.1f}k tok/s"
                )
                append_jsonl(log_path, row)
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats(device)

                accum_total_loss = 0.0
                accum_lm_loss = 0.0
                accum_aux_loss = 0.0
                accum_target_band_loss = 0.0
                accum_budget_var_mean = 0.0

            if args.save_every > 0 and step % args.save_every == 0:
                ckpt = dump / f"ckpt_{step:06d}.pt"
                torch.save({"step": step, "model": model.state_dict(), "opt": optimizer.state_dict()}, ckpt)
                print(f"  [ckpt saved -> {ckpt}]")

            if val_loader is not None and args.eval_every > 0 and step % args.eval_every == 0:
                if args.budget_diagnostics:
                    last_budget_diag = collect_budget_diagnostics(model, include_layer=True, include_quantiles=True)
                    if args.save_budget_profile:
                        update_budget_profile_accumulator(budget_profile_accumulator, model)
                    last_budget_diag.update({"step": step, "event": "train_eval_boundary"})
                    append_jsonl(budget_diag_path, last_budget_diag)
                metrics = evaluate_model(model, val_loader, device, "validation")
                metrics["step"] = step
                if last_logged_train and "train_lm_loss" in last_logged_train:
                    metrics["recent_train_lm_loss"] = last_logged_train["train_lm_loss"]
                    metrics["generalization_gap"] = round(metrics["lm_loss"] - last_logged_train["train_lm_loss"], 6)
                last_eval = metrics
                print(
                    f"[eval] step={step}  val_loss={metrics['loss']:.4f}  "
                    f"val_ppl={metrics['ppl']:.3f}  {metrics['elapsed']:.1f}s"
                )
                append_jsonl(eval_log_path, metrics)

    if val_loader is not None and (last_eval is None or last_eval.get("step") != step):
        if args.budget_diagnostics:
            last_budget_diag = collect_budget_diagnostics(model, include_layer=True, include_quantiles=True)
            if args.save_budget_profile:
                update_budget_profile_accumulator(budget_profile_accumulator, model)
            last_budget_diag.update({"step": step, "event": "train_final_boundary"})
            append_jsonl(budget_diag_path, last_budget_diag)
        metrics = evaluate_model(model, val_loader, device, "validation")
        metrics["step"] = step
        if last_logged_train and "train_lm_loss" in last_logged_train:
            metrics["recent_train_lm_loss"] = last_logged_train["train_lm_loss"]
            metrics["generalization_gap"] = round(metrics["lm_loss"] - last_logged_train["train_lm_loss"], 6)
        last_eval = metrics
        print(
            f"[eval] step={step}  val_loss={metrics['loss']:.4f}  "
            f"val_ppl={metrics['ppl']:.3f}  {metrics['elapsed']:.1f}s"
        )
        append_jsonl(eval_log_path, metrics)

    final_test = None
    if test_loader is not None:
        final_test = evaluate_model(model, test_loader, device, "test")
        if args.save_budget_profile:
            update_budget_profile_accumulator(budget_profile_accumulator, model)
        final_test["step"] = step
        print(
            f"[test] step={step}  test_loss={final_test['loss']:.4f}  "
            f"test_ppl={final_test['ppl']:.3f}  {final_test['elapsed']:.1f}s"
        )
        with open(test_metrics_path, "w", encoding="utf-8") as handle:
            json.dump(final_test, handle, indent=2)

    saved_budget_profile = None
    if args.save_budget_profile:
        profile_out = Path(args.budget_profile_out) if args.budget_profile_out else dump / "budget_profile.json"
        saved_budget_profile = write_budget_profile(
            profile_out,
            budget_profile_accumulator,
            args=args,
            cfg_dict=cfg_dict,
            step=step,
        )
        print(f"  [budget profile saved -> {profile_out}]")

    if args.save_final_ckpt or args.save_final_ckpt_for_attribution:
        torch.save({"step": step, "model": model.state_dict()}, dump / "ckpt_final.pt")

    summary = {
        "run_name": args.name,
        "steps": step,
        "dataset": args.dataset,
        "dataset_name": args.dataset_name,
        "train_split": args.train_split,
        "train_data_files": args.train_data_files,
        "train_tokens": args.train_tokens,
        "train_skip_tokens": args.train_skip_tokens,
        "val_split": args.val_split,
        "val_data_files": args.val_data_files,
        "test_split": args.test_split,
        "test_data_files": args.test_data_files,
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "budget_aux_loss_weight": args.budget_aux_loss_weight,
        "budget_aux_mode": args.budget_aux_mode,
        "budget_aux_target_min": args.budget_aux_target_min,
        "budget_aux_target_max": args.budget_aux_target_max,
        "budget_aux_target_weight": args.budget_aux_target_weight,
        "budget_shuffle_mode": args.budget_shuffle_mode,
        "budget_center_mode": args.budget_center_mode,
        "budget_random_mode": args.budget_random_mode,
        "budget_clamp_mode": args.budget_clamp_mode,
        "budget_profile_path": args.budget_profile_path,
        "budget_profile_out": args.budget_profile_out,
        "save_budget_profile": args.save_budget_profile,
        "budget_diagnostics": args.budget_diagnostics,
        "save_final_ckpt": args.save_final_ckpt,
        "save_final_ckpt_for_attribution": args.save_final_ckpt_for_attribution,
        "saved_budget_profile": saved_budget_profile,
        "final_train": last_logged_train,
        "final_validation": last_eval,
        "final_test": final_test,
        "final_budget_diagnostics": last_budget_diag,
        "train_cache": train_cache_meta,
        "val_cache": val_cache_meta,
        "test_cache": test_cache_meta,
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"\nTraining done. Logs: {log_path}")


if __name__ == "__main__":
    main()
