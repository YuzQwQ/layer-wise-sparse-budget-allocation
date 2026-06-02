"""Token-level budget attribution for R9 mechanism diagnostics."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent))

from native_sparse_attention import NSAConfig, NSAForCausalLM
from simple_train import (
    autocast_context,
    build_eval_cache,
    build_tokenizer,
    collate,
    configure_budget_controls,
    load_budget_replay_profile,
    load_eval_dataset,
)


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.lower() in {"1", "true", "yes", "y"}


def round_float(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    if not math.isfinite(float(value)):
        return None
    return round(float(value), digits)


def pearson(x: torch.Tensor, y: torch.Tensor) -> float | None:
    if x.numel() < 2:
        return None
    x = x.float()
    y = y.float()
    x = x - x.mean()
    y = y - y.mean()
    denom = x.norm() * y.norm()
    if float(denom) == 0.0:
        return None
    return round_float(float((x * y).sum() / denom))


def ranks(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values.float(), stable=True)
    out = torch.empty_like(order, dtype=torch.float32)
    out[order] = torch.arange(values.numel(), dtype=torch.float32, device=values.device)
    return out


def spearman(x: torch.Tensor, y: torch.Tensor) -> float | None:
    if x.numel() < 2:
        return None
    return pearson(ranks(x), ranks(y))


class OnlineStats:
    def __init__(self) -> None:
        self.n = 0
        self.sum = 0.0
        self.sum_sq = 0.0

    def update(self, values: torch.Tensor) -> None:
        values = values.detach().float().cpu()
        if values.numel() == 0:
            return
        self.n += int(values.numel())
        self.sum += float(values.sum())
        self.sum_sq += float((values * values).sum())

    def mean(self) -> float | None:
        return None if self.n == 0 else self.sum / self.n

    def var(self) -> float | None:
        if self.n < 2:
            return None
        mean = self.sum / self.n
        return max(0.0, self.sum_sq / self.n - mean * mean)

    def as_dict(self) -> dict[str, Any]:
        return {"count": self.n, "mean": round_float(self.mean()), "var": round_float(self.var())}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dataset", default="allenai/c4")
    parser.add_argument("--dataset_name", default="en")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--data_files", default="")
    parser.add_argument("--eval_cache_dir", required=True)
    parser.add_argument("--tokens", type=int, default=262_144)
    parser.add_argument("--skip_tokens", type=int, default=0)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--tokenizer_path", default="gpt2")
    parser.add_argument("--text_column", default="text")
    parser.add_argument("--budget_shuffle_mode", choices=["none", "token"], default="none")
    parser.add_argument("--budget_center_mode", choices=["none", "mono_layer_mean"], default="none")
    parser.add_argument("--budget_random_mode", choices=["none", "layer_histogram_replay"], default="none")
    parser.add_argument("--budget_clamp_mode", choices=["none", "mono_layer_sum"], default="none")
    parser.add_argument("--budget_profile_path", default="")
    parser.add_argument("--max_sample_rows", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_model(args: argparse.Namespace, device: torch.device) -> NSAForCausalLM:
    cfg_dict = json.loads(Path(args.config).read_text(encoding="utf-8"))
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        if cfg_dict.get("window_size", 0) > 0:
            print(f"[warn] flash_attn not found -> overriding window_size {cfg_dict['window_size']} -> 0")
            cfg_dict["window_size"] = 0
    cfg_clean = {key: value for key, value in cfg_dict.items() if not key.startswith("_")}
    model = NSAForCausalLM(NSAConfig(**cfg_clean)).to(device).to(torch.bfloat16)
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state, strict=True)
    model.eval()

    replay_profile = None
    if args.budget_random_mode == "layer_histogram_replay":
        if not args.budget_profile_path:
            raise ValueError("--budget_profile_path is required for layer_histogram_replay")
        replay_profile = load_budget_replay_profile(Path(args.budget_profile_path))
    control_args = SimpleNamespace(
        budget_shuffle_mode=args.budget_shuffle_mode,
        budget_center_mode=args.budget_center_mode,
        budget_random_mode=args.budget_random_mode,
        budget_clamp_mode=args.budget_clamp_mode,
    )
    configure_budget_controls(model, control_args, replay_profile)
    return model


def build_loader(args: argparse.Namespace):
    tokenizer = build_tokenizer(args.tokenizer_path)
    cache_meta = build_eval_cache(
        dataset=args.dataset,
        dataset_name=args.dataset_name or None,
        split=args.split,
        data_files=args.data_files or None,
        tokenizer=tokenizer,
        tokenizer_path=args.tokenizer_path,
        seq_len=args.seq_len,
        target_tokens=args.tokens,
        skip_tokens=args.skip_tokens,
        text_column=args.text_column,
        cache_dir=Path(args.eval_cache_dir),
        cache_seed=0,
    )
    dataset = load_eval_dataset(cache_meta)
    sample_count = min(len(dataset), max(1, args.tokens // args.seq_len))
    dataset = Subset(dataset, range(sample_count))
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate), cache_meta


def collect_layer_budgets(model: NSAForCausalLM, seq_len_minus_one: int) -> list[torch.Tensor]:
    out = []
    for layer in model.model.layers:
        tensor = getattr(layer.attn, "_dynamic_k_tensor", None)
        if tensor is None:
            continue
        # Average across KV heads so each token-position has one comparable budget.
        out.append(tensor[:, :seq_len_minus_one].float().mean(dim=-1).detach().cpu())
    return out


def bucket_id(values: torch.Tensor, quantiles: list[float]) -> torch.Tensor:
    if values.numel() == 0:
        return torch.empty_like(values, dtype=torch.long)
    cuts = torch.quantile(values.float(), torch.tensor(quantiles, device=values.device))
    return torch.bucketize(values.float(), cuts)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(args, device)
    loader, cache_meta = build_loader(args)

    sample_path = output_dir / f"token_attribution_samples_{args.method}_s42.jsonl"
    summary_path = output_dir / f"token_attribution_summary_{args.method}_s42.json"

    rows_written = 0
    total_tokens = 0
    token_counter: Counter[int] = Counter()
    per_layer: dict[int, dict[str, list[torch.Tensor]]] = defaultdict(lambda: defaultdict(list))
    difficulty_stats: dict[str, OnlineStats] = defaultdict(OnlineStats)
    freq_stats: dict[str, OnlineStats] = defaultdict(OnlineStats)
    position_stats: dict[str, OnlineStats] = defaultdict(OnlineStats)

    sample_handle = sample_path.open("w", encoding="utf-8")
    try:
        with torch.no_grad():
            for batch in loader:
                ids = (batch["input_ids"] if isinstance(batch, dict) else batch).to(device)
                with autocast_context(device):
                    out = model(input_ids=ids, labels=None, logits_to_keep=None)
                logits = out.logits[:, :-1].float()
                targets = ids[:, 1:]
                log_probs = F.log_softmax(logits, dim=-1)
                target_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
                nll = -target_log_probs.detach().cpu()
                probs = log_probs.exp()
                entropy = -(probs * log_probs).sum(dim=-1).detach().cpu()
                top1 = probs.max(dim=-1).values.detach().cpu()
                target_cpu = targets.detach().cpu()
                budgets = collect_layer_budgets(model, targets.shape[1])
                if not budgets:
                    raise RuntimeError("No dynamic budget tensors were captured; attribution requires a dynamic NSA config.")

                token_counter.update(int(tok) for tok in target_cpu.reshape(-1).tolist())
                nll_flat = nll.reshape(-1)
                entropy_flat = entropy.reshape(-1)
                top1_flat = top1.reshape(-1)
                target_flat = target_cpu.reshape(-1)
                pos = torch.arange(targets.shape[1]).repeat(targets.shape[0])
                difficulty_bucket = bucket_id(nll_flat, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
                position_bucket = bucket_id(pos.float(), [0.25, 0.5, 0.75])

                for layer_id, budget in enumerate(budgets):
                    budget_flat = budget.reshape(-1)
                    per_layer[layer_id]["budget"].append(budget_flat)
                    per_layer[layer_id]["nll"].append(nll_flat)
                    per_layer[layer_id]["entropy"].append(entropy_flat)
                    per_layer[layer_id]["top1"].append(top1_flat)
                    per_layer[layer_id]["target"].append(target_flat)

                    for bucket in range(10):
                        mask = difficulty_bucket == bucket
                        if mask.any():
                            difficulty_stats[f"decile_{bucket}"].update(budget_flat[mask])
                    for bucket in range(4):
                        mask = position_bucket == bucket
                        if mask.any():
                            position_stats[f"quartile_{bucket}"].update(budget_flat[mask])

                    if rows_written < args.max_sample_rows:
                        write_n = min(int(budget_flat.numel()), args.max_sample_rows - rows_written)
                        for idx in range(write_n):
                            token_id = int(target_flat[idx])
                            freq = token_counter.get(token_id, 0)
                            row = {
                                "layer_id": layer_id,
                                "position": int(pos[idx]),
                                "target_token_id": token_id,
                                "token_nll": round_float(float(nll_flat[idx])),
                                "prediction_entropy": round_float(float(entropy_flat[idx])),
                                "top1_probability": round_float(float(top1_flat[idx])),
                                "dynamic_k": round_float(float(budget_flat[idx])),
                                "budget_cont": round_float(float(budget_flat[idx])),
                                "token_frequency_bucket": "online_seen_1" if freq <= 1 else ("online_seen_2_4" if freq <= 4 else "online_seen_5p"),
                                "position_bucket": int(position_bucket[idx]),
                            }
                            sample_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                        rows_written += write_n
                total_tokens += int(target_cpu.numel())
    finally:
        sample_handle.close()

    # Frequency buckets are computed after the pass; this is a subset-local proxy.
    all_freqs = torch.tensor([token_counter[int(tok)] for tok in token_counter], dtype=torch.float32)
    freq_cuts = torch.quantile(all_freqs, torch.tensor([0.25, 0.5, 0.75])) if all_freqs.numel() else torch.tensor([])

    layer_summaries = []
    all_spearman_nll = []
    all_spearman_entropy = []
    all_spearman_top1 = []
    for layer_id in sorted(per_layer):
        budget = torch.cat(per_layer[layer_id]["budget"])
        nll = torch.cat(per_layer[layer_id]["nll"])
        entropy = torch.cat(per_layer[layer_id]["entropy"])
        top1 = torch.cat(per_layer[layer_id]["top1"])
        targets = torch.cat(per_layer[layer_id]["target"])
        if freq_cuts.numel():
            freq_values = torch.tensor([token_counter[int(tok)] for tok in targets.tolist()], dtype=torch.float32)
            freq_bucket = torch.bucketize(freq_values, freq_cuts)
            for bucket in range(4):
                mask = freq_bucket == bucket
                if mask.any():
                    freq_stats[f"quartile_{bucket}"].update(budget[mask])
        nll_s = spearman(budget, nll)
        entropy_s = spearman(budget, entropy)
        top1_s = spearman(budget, top1)
        if nll_s is not None:
            all_spearman_nll.append(nll_s)
        if entropy_s is not None:
            all_spearman_entropy.append(entropy_s)
        if top1_s is not None:
            all_spearman_top1.append(top1_s)
        layer_summaries.append(
            {
                "layer_id": layer_id,
                "samples": int(budget.numel()),
                "budget_mean": round_float(float(budget.mean())),
                "budget_var": round_float(float(budget.var(unbiased=False))),
                "pearson_dynamic_k_nll": pearson(budget, nll),
                "spearman_dynamic_k_nll": nll_s,
                "pearson_dynamic_k_entropy": pearson(budget, entropy),
                "spearman_dynamic_k_entropy": entropy_s,
                "pearson_dynamic_k_top1_probability": pearson(budget, top1),
                "spearman_dynamic_k_top1_probability": top1_s,
            }
        )

    summary = {
        "method": args.method,
        "checkpoint": args.checkpoint,
        "cache_meta": cache_meta,
        "tokens_analyzed": total_tokens,
        "rows_written": rows_written,
        "frequency_source": "attribution_subset_online_counts",
        "frequency_quantile_cuts": [round_float(float(v)) for v in freq_cuts.tolist()] if freq_cuts.numel() else [],
        "mean_spearman_dynamic_k_nll": round_float(sum(all_spearman_nll) / len(all_spearman_nll)) if all_spearman_nll else None,
        "mean_spearman_dynamic_k_entropy": round_float(sum(all_spearman_entropy) / len(all_spearman_entropy)) if all_spearman_entropy else None,
        "mean_spearman_dynamic_k_top1_probability": round_float(sum(all_spearman_top1) / len(all_spearman_top1)) if all_spearman_top1 else None,
        "difficulty_decile_budget": {key: value.as_dict() for key, value in sorted(difficulty_stats.items())},
        "position_bucket_budget": {key: value.as_dict() for key, value in sorted(position_stats.items())},
        "frequency_bucket_budget": {key: value.as_dict() for key, value in sorted(freq_stats.items())},
        "layers": layer_summaries,
        "budget_shuffle_mode": args.budget_shuffle_mode,
        "budget_random_mode": args.budget_random_mode,
        "budget_clamp_mode": args.budget_clamp_mode,
        "budget_profile_path": args.budget_profile_path,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    combined_path = output_dir / "round9_attribution_summary.json"
    existing = {}
    if combined_path.exists():
        existing = json.loads(combined_path.read_text(encoding="utf-8"))
    existing[args.method] = summary
    combined_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[attribution] wrote {summary_path}")


if __name__ == "__main__":
    main()
