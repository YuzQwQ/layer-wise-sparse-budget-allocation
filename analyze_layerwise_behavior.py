"""Layer-wise sparse context access behavior diagnostics.

This script is intentionally forward-only. It does not train or update model
weights. It reads existing NSA checkpoints, enables opt-in selection debugging,
and summarizes selected-block distance patterns by layer and layer group.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent))


def round_float(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    if not math.isfinite(float(value)):
        return None
    return round(float(value), digits)


def parse_run(value: str) -> tuple[str, Path, Path]:
    parts = value.split("|")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--run must be formatted as method|config_path|checkpoint_path")
    method, config, checkpoint = parts
    return method, Path(config), Path(checkpoint)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Layer-wise sparse context access behavior analysis.")
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
    parser.add_argument("--max_sample_rows", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


class DistanceAccumulator:
    def __init__(self, max_distance_blocks: int):
        self.max_distance_blocks = max_distance_blocks
        self.total = 0
        self.sum_distance_blocks = 0.0
        self.sum_normalized_distance = 0.0
        self.far_count = 0
        self.hist = torch.zeros(max_distance_blocks + 1, dtype=torch.long)
        self.selected_count_sum = 0.0
        self.selected_count_den = 0
        self.eligible_selected_count_sum = 0.0
        self.eligible_selected_count_den = 0

    def update(
        self,
        distances: torch.Tensor,
        normalized: torch.Tensor,
        selected_count_mean: float,
        eligible_selected_count_mean: float | None,
        far_distance_blocks: int,
    ) -> None:
        if distances.numel() == 0:
            return
        distances = distances.long().cpu()
        normalized = normalized.float().cpu()
        self.total += int(distances.numel())
        self.sum_distance_blocks += float(distances.float().sum().item())
        self.sum_normalized_distance += float(normalized.sum().item())
        self.far_count += int((distances >= far_distance_blocks).sum().item())
        clipped = distances.clamp(0, self.max_distance_blocks)
        self.hist += torch.bincount(clipped, minlength=self.max_distance_blocks + 1)
        self.selected_count_sum += float(selected_count_mean)
        self.selected_count_den += 1
        if eligible_selected_count_mean is not None:
            self.eligible_selected_count_sum += float(eligible_selected_count_mean)
            self.eligible_selected_count_den += 1

    def summary(self, block_size: int) -> dict[str, Any]:
        if self.total == 0:
            return {
                "selected_items": 0,
                "mean_selected_distance_blocks": None,
                "mean_selected_distance_tokens": None,
                "mean_normalized_distance": None,
                "far_context_ratio": None,
                "selection_distance_entropy": None,
                "selection_distance_entropy_norm": None,
                "selected_count_mean": None,
                "eligible_selected_count_mean": None,
            }
        probs = self.hist.float() / self.hist.sum().clamp_min(1)
        nz = probs > 0
        entropy = float(-(probs[nz] * probs[nz].log()).sum().item())
        entropy_norm = entropy / math.log(max(2, int(nz.sum().item())))
        mean_blocks = self.sum_distance_blocks / self.total
        return {
            "selected_items": self.total,
            "mean_selected_distance_blocks": round_float(mean_blocks),
            "mean_selected_distance_tokens": round_float(mean_blocks * block_size),
            "mean_normalized_distance": round_float(self.sum_normalized_distance / self.total),
            "far_context_ratio": round_float(self.far_count / self.total),
            "selection_distance_entropy": round_float(entropy),
            "selection_distance_entropy_norm": round_float(entropy_norm),
            "selected_count_mean": round_float(self.selected_count_sum / max(1, self.selected_count_den)),
            "eligible_selected_count_mean": round_float(self.eligible_selected_count_sum / self.eligible_selected_count_den)
            if self.eligible_selected_count_den
            else None,
        }


def load_model(config_path: Path, checkpoint_path: Path, device: torch.device) -> NSAForCausalLM:
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


def layer_group(layer_id: int) -> str:
    if layer_id < 6:
        return "layers_1_6"
    if layer_id < 12:
        return "layers_7_12"
    if layer_id < 18:
        return "layers_13_18"
    return "layers_19_24"


def extract_valid_distances(
    state: dict[str, Any],
    far_distance_blocks: int,
) -> tuple[torch.Tensor, torch.Tensor, float, float | None, int]:
    block_indices = state["block_indices"].long()
    block_counts = state["block_counts"]
    block_size = int(state["block_size"])
    bsz, seq_len, n_heads, max_selected = block_indices.shape

    positions = torch.arange(seq_len, dtype=torch.long).view(1, seq_len, 1, 1)
    query_block = positions // block_size
    ranks = torch.arange(max_selected, dtype=torch.long).view(1, 1, 1, max_selected)
    if isinstance(block_counts, int):
        count_tensor = torch.full((bsz, seq_len, n_heads), int(block_counts), dtype=torch.long)
    else:
        count_tensor = block_counts.long()
    valid_rank = ranks < count_tensor.unsqueeze(-1)
    distances = query_block - block_indices
    valid = valid_rank & (block_indices >= 0) & (distances >= 0) & ((block_indices * block_size) <= positions)
    selected_count_mean = float(valid.sum().item()) / float(max(1, bsz * seq_len * n_heads))
    query_block_bth = (torch.arange(seq_len, dtype=torch.long).view(1, seq_len, 1) // block_size).expand(bsz, seq_len, n_heads)
    eligible = query_block_bth >= count_tensor
    valid_per_query = valid.sum(dim=-1).float()
    eligible_selected_count_mean = (
        float(valid_per_query[eligible].mean().item()) if bool(eligible.any().item()) else None
    )
    distance_flat = distances[valid]
    norm = distance_flat.float() / query_block.expand_as(block_indices)[valid].clamp_min(1).float()
    future_violations = int((valid_rank & ((block_indices * block_size) > positions)).sum().item())
    return distance_flat, norm, selected_count_mean, eligible_selected_count_mean, future_violations


def summarize_method(
    method: str,
    config_path: Path,
    checkpoint_path: Path,
    loader: DataLoader,
    args: argparse.Namespace,
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    from simple_train import autocast_context

    model = load_model(config_path, checkpoint_path, device)
    max_distance_blocks = math.ceil(args.seq_len / max(1, int(model.config.block_size)))
    by_layer = {i: DistanceAccumulator(max_distance_blocks) for i in range(model.config.num_hidden_layers)}
    by_group = {name: DistanceAccumulator(max_distance_blocks) for name in ["layers_1_6", "layers_7_12", "layers_13_18", "layers_19_24"]}
    sample_path = output_dir / f"layerwise_behavior_samples_{method}_s42.jsonl"

    rows_written = 0
    total_future_violations = 0
    block_size = int(model.config.block_size)
    with sample_path.open("w", encoding="utf-8") as sample_handle:
        with torch.no_grad():
            for batch_id, batch in enumerate(loader):
                ids = (batch["input_ids"] if isinstance(batch, dict) else batch).to(device)
                with autocast_context(device):
                    _ = model(input_ids=ids, labels=None, logits_to_keep=1)
                for layer_id, layer in enumerate(model.model.layers):
                    state = getattr(layer.attn, "_debug_selection_state", None)
                    if state is None:
                        raise RuntimeError(f"No selection debug state captured for layer {layer_id}")
                    distances, normalized, selected_count_mean, eligible_selected_count_mean, future_violations = extract_valid_distances(
                        state, args.far_distance_blocks
                    )
                    total_future_violations += future_violations
                    by_layer[layer_id].update(
                        distances,
                        normalized,
                        selected_count_mean,
                        eligible_selected_count_mean,
                        args.far_distance_blocks,
                    )
                    by_group[layer_group(layer_id)].update(
                        distances,
                        normalized,
                        selected_count_mean,
                        eligible_selected_count_mean,
                        args.far_distance_blocks,
                    )
                    if rows_written < args.max_sample_rows:
                        row = {
                            "method": method,
                            "batch_id": batch_id,
                            "layer_id": layer_id,
                            "layer_group": layer_group(layer_id),
                            "valid_selected_items": int(distances.numel()),
                            "selected_count_mean": round_float(selected_count_mean),
                            "eligible_selected_count_mean": round_float(eligible_selected_count_mean),
                            "mean_selected_distance_blocks": round_float(float(distances.float().mean().item())) if distances.numel() else None,
                            "mean_selected_distance_tokens": round_float(float(distances.float().mean().item()) * block_size) if distances.numel() else None,
                            "mean_normalized_distance": round_float(float(normalized.float().mean().item())) if normalized.numel() else None,
                            "far_context_ratio": round_float(float((distances >= args.far_distance_blocks).float().mean().item())) if distances.numel() else None,
                            "future_violations": future_violations,
                        }
                        sample_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                        rows_written += 1

    return {
        "method": method,
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "future_violations": total_future_violations,
        "layer_summary": [
            {"layer_id": i, "layer_group": layer_group(i), **by_layer[i].summary(block_size)}
            for i in sorted(by_layer)
        ],
        "group_summary": {
            group: acc.summary(block_size)
            for group, acc in by_group.items()
        },
    }


def plot_group_metric(summary: dict[str, Any], metric: str, title: str, ylabel: str, output: Path) -> None:
    groups = ["layers_1_6", "layers_7_12", "layers_13_18", "layers_19_24"]
    methods = list(summary["methods"].keys())
    width = 0.8 / max(1, len(methods))
    x = list(range(len(groups)))
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for m_idx, method in enumerate(methods):
        vals = [summary["methods"][method]["group_summary"][g].get(metric) for g in groups]
        vals = [float(v) if v is not None else 0.0 for v in vals]
        offsets = [pos - 0.4 + width / 2 + m_idx * width for pos in x]
        ax.bar(offsets, vals, width=width, label=method)
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


def plot_entropy_by_layer(summary: dict[str, Any], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for method, payload in summary["methods"].items():
        xs = [row["layer_id"] + 1 for row in payload["layer_summary"]]
        ys = [row["selection_distance_entropy_norm"] for row in payload["layer_summary"]]
        ax.plot(xs, ys, marker="o", linewidth=1.4, label=method)
    ax.set_title("Selection distance entropy by layer")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Normalized entropy")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_distance_heatmap(summary: dict[str, Any], output: Path) -> None:
    methods = list(summary["methods"].keys())
    matrix = []
    for method in methods:
        matrix.append([
            row["mean_selected_distance_blocks"] or 0.0
            for row in summary["methods"][method]["layer_summary"]
        ])
    fig, ax = plt.subplots(figsize=(9, 2.8 + len(methods) * 0.3))
    im = ax.imshow(matrix, aspect="auto", cmap="YlGnBu")
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods)
    ax.set_xticks([0, 5, 11, 17, 23])
    ax.set_xticklabels([1, 6, 12, 18, 24])
    ax.set_xlabel("Layer")
    ax.set_title("Mean selected distance by method and layer")
    fig.colorbar(im, ax=ax, label="Distance (blocks)")
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def write_markdown(summary: dict[str, Any], output: Path) -> None:
    lines = [
        "# Layer-wise Sparse Context Access Behavior Analysis",
        "",
        "This is a behavioral diagnostic, not a causal mechanism proof.",
        "",
        "| Method | Group | Mean distance blocks | Far ratio | Norm. entropy | Selected count mean | Eligible selected count mean |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method, payload in summary["methods"].items():
        for group, stats in payload["group_summary"].items():
            lines.append(
                f"| {method} | {group} | {stats.get('mean_selected_distance_blocks')} | "
                f"{stats.get('far_context_ratio')} | {stats.get('selection_distance_entropy_norm')} | "
                f"{stats.get('selected_count_mean')} | {stats.get('eligible_selected_count_mean')} |"
            )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    missing = []
    for method, config_path, checkpoint_path in args.run:
        if not config_path.exists():
            missing.append({"method": method, "kind": "config", "path": str(config_path)})
        if not checkpoint_path.exists():
            missing.append({"method": method, "kind": "checkpoint", "path": str(checkpoint_path)})
    if missing:
        report = {
            "status": "missing_required_files",
            "missing": missing,
            "note": "No training was started. Provide existing checkpoints or run a checkpoint-producing experiment first.",
        }
        (output_dir / "layerwise_behavior_missing_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        raise SystemExit(2)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for NSA Triton kernels in layer-wise behavior analysis.")

    device = torch.device("cuda")
    loader, cache_meta = build_loader(args)
    summary: dict[str, Any] = {
        "analysis": "layerwise_sparse_context_access_behavior",
        "cache_meta": cache_meta,
        "seq_len": args.seq_len,
        "tokens": args.tokens,
        "far_distance_blocks": args.far_distance_blocks,
        "far_distance_tokens": args.far_distance_blocks * 64,
        "methods": {},
    }
    for method, config_path, checkpoint_path in args.run:
        summary["methods"][method] = summarize_method(
            method=method,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            loader=loader,
            args=args,
            output_dir=output_dir,
            device=device,
        )
        torch.cuda.empty_cache()

    summary_path = output_dir / "layerwise_behavior_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(summary, output_dir / "layerwise_behavior_summary.md")
    plot_group_metric(
        summary,
        "mean_selected_distance_blocks",
        "Mean selected block distance by layer group",
        "Distance (blocks)",
        output_dir / "mean_selected_distance_by_layer_group.png",
    )
    plot_group_metric(
        summary,
        "far_context_ratio",
        "Far-context selection ratio by layer group",
        f"Ratio (distance >= {args.far_distance_blocks} blocks)",
        output_dir / "far_context_ratio_by_layer_group.png",
    )
    plot_entropy_by_layer(summary, output_dir / "selection_entropy_by_layer.png")
    plot_distance_heatmap(summary, output_dir / "layerwise_distance_heatmap.png")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
