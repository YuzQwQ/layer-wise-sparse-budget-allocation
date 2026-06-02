"""Selected-attention probability mass behavior diagnostics.

This script is forward-only. It reuses existing NSA checkpoints, captures the
selected block indices, and offline recomputes the selected-branch QK softmax
for sampled query positions. The goal is behavioral evidence only: whether
selected attention probability mass follows the same depth-aligned pattern as
the selected block indices.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent))


GROUPS = ["layers_1_6", "layers_7_12", "layers_13_18", "layers_19_24"]
DISTANCE_BUCKETS = [
    ("0", 0, 0),
    ("1", 1, 1),
    ("2_3", 2, 3),
    ("4_7", 4, 7),
    ("8_15", 8, 15),
    ("16_plus", 16, 10**9),
]


def round_float(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    if not math.isfinite(float(value)):
        return None
    return round(float(value), digits)


def parse_run(value: str) -> tuple[str, int, Path, Path]:
    parts = value.split("|")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("--run must be formatted as method|seed|config_path|checkpoint_path")
    method, seed, config, checkpoint = parts
    return method, int(seed), Path(config), Path(checkpoint)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Selected attention probability mass behavior analysis.")
    parser.add_argument("--run", action="append", type=parse_run, required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tokenizer_path", default="gpt2")
    parser.add_argument("--dataset", default="allenai/c4")
    parser.add_argument("--dataset_name", default="en")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--data_files", default="")
    parser.add_argument("--text_column", default="text")
    parser.add_argument("--eval_cache_dir", required=True)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--tokens", type=int, default=262_144)
    parser.add_argument("--skip_tokens", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--far_distance_blocks", type=int, default=8)
    parser.add_argument("--max_query_positions_per_sequence", type=int, default=128)
    parser.add_argument("--max_sample_rows", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def layer_group(layer_id: int) -> str:
    if layer_id < 6:
        return "layers_1_6"
    if layer_id < 12:
        return "layers_7_12"
    if layer_id < 18:
        return "layers_13_18"
    return "layers_19_24"


class MassAccumulator:
    def __init__(self) -> None:
        self.query_heads = 0
        self.far_mass_sum = 0.0
        self.entropy_sum = 0.0
        self.entropy_norm_sum = 0.0
        self.top_block_mass_sum = 0.0
        self.top1_mass_sum = 0.0
        self.selected_token_count_sum = 0.0
        self.bucket_mass = {name: 0.0 for name, _, _ in DISTANCE_BUCKETS}

    def update(
        self,
        probs: torch.Tensor,
        distance_blocks: torch.Tensor,
        far_distance_blocks: int,
    ) -> None:
        if probs.numel() == 0:
            return
        probs = probs.float().cpu()
        distance_blocks = distance_blocks.long().cpu()
        n = int(probs.numel())
        entropy = float(-(probs * probs.clamp_min(1e-30).log()).sum().item())
        entropy_norm = entropy / math.log(max(2, n))
        block_masses = []
        for distance in distance_blocks.unique(sorted=True):
            block_masses.append(float(probs[distance_blocks == distance].sum().item()))
        self.query_heads += 1
        self.far_mass_sum += float(probs[distance_blocks >= far_distance_blocks].sum().item())
        self.entropy_sum += entropy
        self.entropy_norm_sum += entropy_norm
        self.top_block_mass_sum += max(block_masses) if block_masses else 0.0
        self.top1_mass_sum += float(probs.max().item())
        self.selected_token_count_sum += float(n)
        for name, low, high in DISTANCE_BUCKETS:
            mask = (distance_blocks >= low) & (distance_blocks <= high)
            self.bucket_mass[name] += float(probs[mask].sum().item()) if bool(mask.any()) else 0.0

    def merge(self, other: "MassAccumulator") -> None:
        self.query_heads += other.query_heads
        self.far_mass_sum += other.far_mass_sum
        self.entropy_sum += other.entropy_sum
        self.entropy_norm_sum += other.entropy_norm_sum
        self.top_block_mass_sum += other.top_block_mass_sum
        self.top1_mass_sum += other.top1_mass_sum
        self.selected_token_count_sum += other.selected_token_count_sum
        for key in self.bucket_mass:
            self.bucket_mass[key] += other.bucket_mass[key]

    def summary(self) -> dict[str, Any]:
        if self.query_heads == 0:
            return {
                "query_heads": 0,
                "far_attention_mass_ratio": None,
                "selected_attention_entropy": None,
                "selected_attention_entropy_norm": None,
                "top_selected_block_mass": None,
                "top_selected_token_mass": None,
                "selected_token_count_mean": None,
                "distance_bucket_mass": {name: None for name, _, _ in DISTANCE_BUCKETS},
            }
        return {
            "query_heads": self.query_heads,
            "far_attention_mass_ratio": round_float(self.far_mass_sum / self.query_heads),
            "selected_attention_entropy": round_float(self.entropy_sum / self.query_heads),
            "selected_attention_entropy_norm": round_float(self.entropy_norm_sum / self.query_heads),
            "top_selected_block_mass": round_float(self.top_block_mass_sum / self.query_heads),
            "top_selected_token_mass": round_float(self.top1_mass_sum / self.query_heads),
            "selected_token_count_mean": round_float(self.selected_token_count_sum / self.query_heads),
            "distance_bucket_mass": {
                name: round_float(value / self.query_heads)
                for name, value in self.bucket_mass.items()
            },
        }


def load_model(config_path: Path, checkpoint_path: Path, device: torch.device):
    from native_sparse_attention import NSAConfig, NSAForCausalLM

    cfg_dict = json.loads(config_path.read_text(encoding="utf-8"))
    state_payload = torch.load(checkpoint_path, map_location="cpu")
    state = state_payload["model"] if isinstance(state_payload, dict) and "model" in state_payload else state_payload
    embed_weight = state.get("model.embeddings.weight") if isinstance(state, dict) else None
    if torch.is_tensor(embed_weight) and int(embed_weight.shape[0]) != int(cfg_dict.get("vocab_size", 0)):
        cfg_dict["vocab_size"] = int(embed_weight.shape[0])
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        if cfg_dict.get("window_size", 0) > 0:
            print(f"[warn] flash_attn not found -> overriding window_size {cfg_dict['window_size']} -> 0")
            cfg_dict["window_size"] = 0
    cfg_clean = {key: value for key, value in cfg_dict.items() if not key.startswith("_")}
    model = NSAForCausalLM(NSAConfig(**cfg_clean)).to(device).to(torch.bfloat16)
    model.load_state_dict(state, strict=True)
    model.eval()
    for layer in model.model.layers:
        layer.attn._debug_record_selection = True
        layer.attn._debug_record_selection_tensors = True
    return model


def build_loader(args: argparse.Namespace):
    from simple_train import build_eval_cache, build_tokenizer, collate, load_eval_dataset

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


def sampled_positions(seq_len: int, max_positions: int) -> torch.Tensor:
    if max_positions <= 0 or max_positions >= seq_len:
        return torch.arange(seq_len, dtype=torch.long)
    # Avoid overrepresenting the first few causal positions while preserving
    # deterministic coverage across the whole context.
    return torch.linspace(0, seq_len - 1, steps=max_positions).round().long().unique()


def count_tensor_for_state(state: dict[str, Any]) -> torch.Tensor:
    block_indices = state["block_indices"].long()
    block_counts = state["block_counts"]
    bsz, seq_len, n_heads, _ = block_indices.shape
    if isinstance(block_counts, int):
        return torch.full((bsz, seq_len, n_heads), int(block_counts), dtype=torch.long)
    return block_counts.long()


def analyze_layer_state(
    state: dict[str, Any],
    positions: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[MassAccumulator, list[dict[str, Any]], int]:
    block_indices = state["block_indices"].long()
    block_counts = count_tensor_for_state(state)
    block_size = int(state["block_size"])
    seq_len = int(state["seq_len"])
    q = state["q"].float()
    k = state["k"].float()
    scale = float(state.get("scale", q.shape[-1] ** -0.5))
    bsz, _, num_kv_heads, max_selected = block_indices.shape
    num_query_heads = q.shape[2]
    kv_group = num_query_heads // num_kv_heads

    acc = MassAccumulator()
    rows: list[dict[str, Any]] = []
    future_violations = 0
    for b in range(bsz):
        for t in positions.tolist():
            query_block = t // block_size
            for kv_h in range(num_kv_heads):
                count = int(block_counts[b, t, kv_h].item())
                blocks = block_indices[b, t, kv_h, : min(count, max_selected)].long()
                blocks = blocks[(blocks >= 0) & ((blocks * block_size) <= t)]
                if blocks.numel() == 0:
                    continue
                token_chunks = []
                distance_chunks = []
                for block in blocks.tolist():
                    start = int(block) * block_size
                    end = min(start + block_size, seq_len, t + 1)
                    if end <= start:
                        continue
                    token_idx = torch.arange(start, end, dtype=torch.long)
                    token_chunks.append(token_idx)
                    distance_chunks.append(torch.full((end - start,), query_block - int(block), dtype=torch.long))
                if not token_chunks:
                    continue
                token_indices = torch.cat(token_chunks)
                distance_blocks = torch.cat(distance_chunks)
                future_violations += int((token_indices > t).sum().item())
                k_tokens = k[b, token_indices, kv_h, :]
                for group_offset in range(kv_group):
                    q_h = kv_h * kv_group + group_offset
                    logits = torch.matmul(k_tokens, q[b, t, q_h, :]) * scale
                    probs = torch.softmax(logits, dim=0)
                    acc.update(probs, distance_blocks, args.far_distance_blocks)
                    if len(rows) < args.max_sample_rows:
                        rows.append({
                            "batch_query_position": int(t),
                            "kv_head": int(kv_h),
                            "query_head": int(q_h),
                            "selected_tokens": int(probs.numel()),
                            "far_attention_mass_ratio": round_float(float(probs[distance_blocks >= args.far_distance_blocks].sum().item())),
                            "top_selected_block_mass": round_float(max(
                                float(probs[distance_blocks == d].sum().item())
                                for d in distance_blocks.unique(sorted=True)
                            )),
                            "top_selected_token_mass": round_float(float(probs.max().item())),
                        })
    return acc, rows, future_violations


def summarize_run(
    method: str,
    seed: int,
    config_path: Path,
    checkpoint_path: Path,
    loader: DataLoader,
    args: argparse.Namespace,
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    from simple_train import autocast_context

    run_key = f"{method}_s{seed}"
    model = load_model(config_path, checkpoint_path, device)
    by_layer = {i: MassAccumulator() for i in range(model.config.num_hidden_layers)}
    by_group = {group: MassAccumulator() for group in GROUPS}
    sample_path = output_dir / f"selected_attention_mass_samples_{run_key}.jsonl"
    rows_written = 0
    total_future_violations = 0
    positions = sampled_positions(args.seq_len, args.max_query_positions_per_sequence)

    with sample_path.open("w", encoding="utf-8") as sample_handle:
        with torch.no_grad():
            for batch_id, batch in enumerate(loader):
                ids = (batch["input_ids"] if isinstance(batch, dict) else batch).to(device)
                with autocast_context(device):
                    _ = model(input_ids=ids, labels=None, logits_to_keep=1)
                for layer_id, layer in enumerate(model.model.layers):
                    state = getattr(layer.attn, "_debug_selection_state", None)
                    if state is None or "q" not in state or "k" not in state:
                        raise RuntimeError(f"No tensor selection debug state captured for layer {layer_id}")
                    acc, rows, future_violations = analyze_layer_state(state, positions, args)
                    total_future_violations += future_violations
                    by_layer[layer_id].merge(acc)
                    by_group[layer_group(layer_id)].merge(acc)
                    for row in rows:
                        if rows_written >= args.max_sample_rows:
                            break
                        row.update({
                            "method": method,
                            "seed": seed,
                            "batch_id": batch_id,
                            "layer_id": layer_id,
                            "layer_group": layer_group(layer_id),
                        })
                        sample_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                        rows_written += 1

    return {
        "method": method,
        "seed": seed,
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "sampled_query_positions_per_sequence": int(positions.numel()),
        "future_violations": total_future_violations,
        "layer_summary": [
            {"layer_id": i, "layer_group": layer_group(i), **by_layer[i].summary()}
            for i in sorted(by_layer)
        ],
        "group_summary": {
            group: by_group[group].summary()
            for group in GROUPS
        },
    }


def mean_std(values: list[float | None]) -> tuple[float | None, float | None]:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not clean:
        return None, None
    tensor = torch.tensor(clean, dtype=torch.float32)
    std = float(tensor.std(unbiased=True).item()) if len(clean) > 1 else 0.0
    return float(tensor.mean().item()), std


def aggregate(summary: dict[str, Any]) -> dict[str, Any]:
    methods = sorted({run["method"] for run in summary["runs"].values()})
    aggregate_payload: dict[str, Any] = {"methods": {}}
    for method in methods:
        method_runs = [run for run in summary["runs"].values() if run["method"] == method]
        aggregate_payload["methods"][method] = {"groups": {}, "layers": []}
        for group in GROUPS:
            group_payload: dict[str, Any] = {}
            keys = [
                "far_attention_mass_ratio",
                "selected_attention_entropy",
                "selected_attention_entropy_norm",
                "top_selected_block_mass",
                "top_selected_token_mass",
                "selected_token_count_mean",
            ]
            for key in keys:
                mean, std = mean_std([run["group_summary"][group].get(key) for run in method_runs])
                group_payload[f"{key}_mean"] = round_float(mean)
                group_payload[f"{key}_std"] = round_float(std)
            buckets = {}
            for bucket, _, _ in DISTANCE_BUCKETS:
                mean, std = mean_std([
                    run["group_summary"][group].get("distance_bucket_mass", {}).get(bucket)
                    for run in method_runs
                ])
                buckets[bucket] = {"mean": round_float(mean), "std": round_float(std)}
            group_payload["distance_bucket_mass"] = buckets
            aggregate_payload["methods"][method]["groups"][group] = group_payload
        layer_count = len(method_runs[0]["layer_summary"]) if method_runs else 0
        for layer_id in range(layer_count):
            layer_payload = {"layer_id": layer_id, "layer_group": layer_group(layer_id)}
            for key in ["far_attention_mass_ratio", "selected_attention_entropy_norm", "top_selected_block_mass", "top_selected_token_mass"]:
                mean, std = mean_std([run["layer_summary"][layer_id].get(key) for run in method_runs])
                layer_payload[f"{key}_mean"] = round_float(mean)
                layer_payload[f"{key}_std"] = round_float(std)
            aggregate_payload["methods"][method]["layers"].append(layer_payload)
    return aggregate_payload


def plot_group_metric(aggregate_payload: dict[str, Any], metric: str, title: str, ylabel: str, output: Path) -> None:
    methods = list(aggregate_payload["methods"].keys())
    width = 0.8 / max(1, len(methods))
    x = list(range(len(GROUPS)))
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for m_idx, method in enumerate(methods):
        vals = [
            aggregate_payload["methods"][method]["groups"][group].get(f"{metric}_mean")
            for group in GROUPS
        ]
        errs = [
            aggregate_payload["methods"][method]["groups"][group].get(f"{metric}_std")
            for group in GROUPS
        ]
        vals = [float(v) if v is not None else 0.0 for v in vals]
        errs = [float(e) if e is not None else 0.0 for e in errs]
        offsets = [pos - 0.4 + width / 2 + m_idx * width for pos in x]
        ax.bar(offsets, vals, yerr=errs, width=width, capsize=3, label=method)
    ax.set_xticks(x)
    ax.set_xticklabels(["1-6", "7-12", "13-18", "19-24"])
    ax.set_title(title)
    ax.set_xlabel("Layer group")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_bucket_mass(aggregate_payload: dict[str, Any], output: Path) -> None:
    methods = list(aggregate_payload["methods"].keys())
    bucket_names = [name for name, _, _ in DISTANCE_BUCKETS]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True)
    axes = axes.flatten()
    for ax, group in zip(axes, GROUPS):
        for method in methods:
            vals = [
                aggregate_payload["methods"][method]["groups"][group]["distance_bucket_mass"][bucket]["mean"] or 0.0
                for bucket in bucket_names
            ]
            ax.plot(bucket_names, vals, marker="o", linewidth=1.4, label=method)
        ax.set_title(group.replace("layers_", "layers "))
        ax.set_xlabel("Distance bucket (blocks)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Attention mass")
    axes[2].set_ylabel("Attention mass")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Selected attention mass by distance bucket")
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def write_markdown(summary: dict[str, Any], output: Path) -> None:
    agg = summary["aggregate"]
    lines = [
        "# Selected Attention Probability Mass Behavior Analysis",
        "",
        "This is a probability-level behavioral diagnostic, not a causal mechanism proof.",
        "",
        "| Method | Group | Far attention mass | Norm. entropy | Top block mass | Top token mass | Selected tokens |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method, payload in agg["methods"].items():
        for group, stats in payload["groups"].items():
            lines.append(
                f"| {method} | {group} | "
                f"{stats.get('far_attention_mass_ratio_mean')} +/- {stats.get('far_attention_mass_ratio_std')} | "
                f"{stats.get('selected_attention_entropy_norm_mean')} +/- {stats.get('selected_attention_entropy_norm_std')} | "
                f"{stats.get('top_selected_block_mass_mean')} +/- {stats.get('top_selected_block_mass_std')} | "
                f"{stats.get('top_selected_token_mass_mean')} +/- {stats.get('top_selected_token_mass_std')} | "
                f"{stats.get('selected_token_count_mean_mean')} +/- {stats.get('selected_token_count_mean_std')} |"
            )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    missing = []
    for method, seed, config_path, checkpoint_path in args.run:
        if not config_path.exists():
            missing.append({"method": method, "seed": seed, "kind": "config", "path": str(config_path)})
        if not checkpoint_path.exists():
            missing.append({"method": method, "seed": seed, "kind": "checkpoint", "path": str(checkpoint_path)})
    if missing:
        report = {
            "status": "missing_required_files",
            "missing": missing,
            "note": "No training was started. Provide existing checkpoints or run checkpoint-producing experiments first.",
        }
        (output_dir / "selected_attention_mass_missing_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        raise SystemExit(2)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for NSA Triton kernels in selected-attention mass analysis.")

    device = torch.device("cuda")
    loader, cache_meta = build_loader(args)
    summary: dict[str, Any] = {
        "analysis": "selected_attention_probability_mass_behavior",
        "cache_meta": cache_meta,
        "seq_len": args.seq_len,
        "tokens": args.tokens,
        "sampled_query_positions_per_sequence": args.max_query_positions_per_sequence,
        "far_distance_blocks": args.far_distance_blocks,
        "far_distance_tokens": args.far_distance_blocks * 64,
        "runs": {},
    }
    for method, seed, config_path, checkpoint_path in args.run:
        run_key = f"{method}_s{seed}"
        print(f"[analyze] {run_key}")
        summary["runs"][run_key] = summarize_run(
            method=method,
            seed=seed,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            loader=loader,
            args=args,
            output_dir=output_dir,
            device=device,
        )
        torch.cuda.empty_cache()

    summary["aggregate"] = aggregate(summary)
    summary_path = output_dir / "selected_attention_mass_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(summary, output_dir / "selected_attention_mass_summary.md")
    plot_group_metric(
        summary["aggregate"],
        "far_attention_mass_ratio",
        "Far selected-attention mass ratio by layer group",
        f"Mass ratio (distance >= {args.far_distance_blocks} blocks)",
        output_dir / "far_attention_mass_by_layer_group.png",
    )
    plot_group_metric(
        summary["aggregate"],
        "selected_attention_entropy_norm",
        "Selected attention entropy by layer group",
        "Normalized entropy",
        output_dir / "selected_attention_entropy_by_layer_group.png",
    )
    plot_bucket_mass(summary["aggregate"], output_dir / "attention_mass_distance_bucket.png")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
