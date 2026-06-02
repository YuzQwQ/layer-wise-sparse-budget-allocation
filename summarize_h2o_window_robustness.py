"""Summarize R26 H2O-style multi-window KV retention robustness diagnostics."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LOCAL_RESULTS_DIR = Path("results") / "round26-phase26-h2o-window-robustness-diagnostic"
METHOD_ORDER = ["uniform", "mono_inc", "mono_dec"]
DATASETS = ["c4", "owt"]
WINDOWS = ["A", "B", "C"]
COLORS = {"uniform": "#6F7A83", "mono_inc": "#2FA17D", "mono_dec": "#D97A2B"}


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def safe_float(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def fmt(value: float | None, digits: int = 4) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def fmt_signed(value: float | None, digits: int = 4) -> str:
    return "-" if value is None else f"{value:+.{digits}f}"


def collect(input_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for window in WINDOWS:
            window_rows = []
            for method in METHOD_ORDER:
                run_dir = input_dir / dataset / "formal" / f"window_{window}" / method
                summary = read_json(run_dir / "run_summary.json")
                profile = read_json(run_dir / "retention_profile.json")
                metrics = read_json(run_dir / "eval_metrics.json")
                if not summary:
                    continue
                val_cache = (metrics or {}).get("val_cache") or {}
                test_cache = (metrics or {}).get("test_cache") or {}
                row = {
                    "dataset_key": dataset,
                    "dataset": "C4" if dataset == "c4" else "OpenWebText",
                    "window_id": window,
                    "sparse_platform": "H2O-style KV retention diagnostic",
                    "sparse_unit": "KV tokens",
                    "model": summary.get("model_name", "facebook/opt-350m"),
                    "method": method,
                    "context_length": safe_float(summary.get("evaluation_context_length")),
                    "val_loss": safe_float(summary.get("val_loss")),
                    "test_ppl": safe_float(summary.get("test_ppl")),
                    "val_loss_after_cache_full": safe_float(summary.get("val_loss_after_cache_full")),
                    "test_ppl_after_cache_full": safe_float(summary.get("test_ppl_after_cache_full")),
                    "recent_size": safe_float(summary.get("recent_size")),
                    "avg_heavy_hitter_budget": safe_float(summary.get("avg_heavy_hitter_budget")),
                    "avg_total_retained_budget": safe_float(summary.get("avg_total_retained_budget")),
                    "tok_per_s": safe_float(summary.get("tok_per_s")),
                    "cuda_max_memory_allocated_mb": safe_float(summary.get("cuda_max_memory_allocated_mb")),
                    "layer_budgets": (profile or {}).get("heavy_hitter_budgets", []),
                    "val_skip_tokens": val_cache.get("skip_tokens"),
                    "test_skip_tokens": test_cache.get("skip_tokens"),
                    "val_actual_tokens": val_cache.get("actual_tokens"),
                    "test_actual_tokens": test_cache.get("actual_tokens"),
                }
                window_rows.append(row)
            baseline = next((row for row in window_rows if row["method"] == "uniform"), None)
            if baseline:
                for row in window_rows:
                    row["delta_val_vs_uniform"] = (
                        baseline["val_loss"] - row["val_loss"]
                        if baseline["val_loss"] is not None and row["val_loss"] is not None
                        else None
                    )
                    row["delta_ppl_vs_uniform"] = (
                        baseline["test_ppl"] - row["test_ppl"]
                        if baseline["test_ppl"] is not None and row["test_ppl"] is not None
                        else None
                    )
            rows.extend(window_rows)
    return rows


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset_key"], row["method"])].append(row)
    aggregates = []
    for (dataset, method), items in sorted(grouped.items()):
        def vals(key: str) -> list[float]:
            return [float(item[key]) for item in items if item.get(key) is not None]

        dval = vals("delta_val_vs_uniform")
        dppl = vals("delta_ppl_vs_uniform")
        aggregates.append(
            {
                "dataset_key": dataset,
                "dataset": "C4" if dataset == "c4" else "OpenWebText",
                "method": method,
                "num_windows": len(items),
                "val_loss_mean": mean(vals("val_loss")) if vals("val_loss") else None,
                "val_loss_std": stdev(vals("val_loss")) if len(vals("val_loss")) > 1 else 0.0,
                "test_ppl_mean": mean(vals("test_ppl")) if vals("test_ppl") else None,
                "test_ppl_std": stdev(vals("test_ppl")) if len(vals("test_ppl")) > 1 else 0.0,
                "delta_val_mean": mean(dval) if dval else None,
                "delta_val_std": stdev(dval) if len(dval) > 1 else 0.0,
                "delta_ppl_mean": mean(dppl) if dppl else None,
                "delta_ppl_std": stdev(dppl) if len(dppl) > 1 else 0.0,
                "wins_val_over_uniform": sum(1 for value in dval if value > 0),
                "wins_ppl_over_uniform": sum(1 for value in dppl if value > 0),
                "tok_per_s_mean": mean(vals("tok_per_s")) if vals("tok_per_s") else None,
                "avg_heavy_hitter_budget": mean(vals("avg_heavy_hitter_budget")) if vals("avg_heavy_hitter_budget") else None,
                "avg_total_retained_budget": mean(vals("avg_total_retained_budget")) if vals("avg_total_retained_budget") else None,
            }
        )
    return aggregates


def decisions(aggregates: list[dict[str, Any]]) -> dict[str, str]:
    result = {}
    for dataset in DATASETS:
        subset = [row for row in aggregates if row["dataset_key"] == dataset]
        if len(subset) < 3:
            result[dataset] = "incomplete"
            continue
        best = min(subset, key=lambda row: row["val_loss_mean"] if row["val_loss_mean"] is not None else float("inf"))
        if best["method"] == "mono_inc":
            result[dataset] = "limited_cross_operator_support_with_window_robustness"
        elif best["method"] == "uniform":
            result[dataset] = "uniform_strong_default_with_window_robustness"
        else:
            result[dataset] = "operator_and_window_sensitive_evidence"
    return result


def plot_delta(aggregates: list[dict[str, Any]], out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 6.4), dpi=180, sharex=True)
    datasets = ["C4", "OpenWebText"]
    x = range(len(datasets))
    width = 0.24
    offsets = {"uniform": -width, "mono_inc": 0.0, "mono_dec": width}
    for ax, key, err_key, title, ylabel, digits in [
        (axes[0], "delta_val_mean", "delta_val_std", "Validation loss improvement", "Delta Val = uniform - method", 4),
        (axes[1], "delta_ppl_mean", "delta_ppl_std", "Test PPL improvement", "Delta PPL = uniform - method", 3),
    ]:
        for method in METHOD_ORDER:
            values = []
            errors = []
            for dataset in DATASETS:
                row = next((item for item in aggregates if item["dataset_key"] == dataset and item["method"] == method), None)
                values.append(0.0 if row is None or row[key] is None else row[key])
                errors.append(0.0 if row is None or row[err_key] is None else row[err_key])
            xpos = [i + offsets[method] for i in x]
            ax.bar(xpos, values, width=width, yerr=errors, color=COLORS[method], alpha=0.86, label=method, capsize=3)
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.22)
    axes[1].set_xticks(list(x), datasets)
    axes[0].legend(frameon=False, ncols=3, loc="best")
    fig.suptitle("H2O-style multi-window robustness: improvement over uniform")
    fig.tight_layout()
    fig.savefig(out_dir / "h2o_window_robustness_delta_mean_std.png")
    plt.close(fig)


def write_markdown(rows: list[dict[str, Any]], aggregates: list[dict[str, Any]], decision_info: dict[str, str], out_path: Path) -> None:
    lines = [
        "# R26 H2O-style Evaluation-window Robustness Diagnostic",
        "",
        "This is inference-time cross-operator diagnostic evidence, not an H2O benchmark or reproduction.",
        "",
        "## Decisions",
        "",
    ]
    for dataset, status in decision_info.items():
        label = "C4" if dataset == "c4" else "OpenWebText"
        lines.append(f"- **{label}**: `{status}`")
    lines.extend(
        [
            "",
            "## Mean across deterministic windows",
            "",
            "| Dataset | Method | Windows | Val Loss mean+/-std (lower) | Test PPL mean+/-std (lower) | Delta Val mean+/-std (higher) | Delta PPL mean+/-std (higher) | Val wins | PPL wins | avg HH | avg retained | tok/s |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in aggregates:
        lines.append(
            "| {dataset} | {method} | {n} | {val}+/-{val_std} | {ppl}+/-{ppl_std} | {dval}+/-{dval_std} | {dppl}+/-{dppl_std} | {vw}/{n} | {pw}/{n} | {hh} | {total} | {tok} |".format(
                dataset=row["dataset"],
                method=row["method"],
                n=row["num_windows"],
                val=fmt(row["val_loss_mean"], 4),
                val_std=fmt(row["val_loss_std"], 4),
                ppl=fmt(row["test_ppl_mean"], 3),
                ppl_std=fmt(row["test_ppl_std"], 3),
                dval=fmt_signed(row["delta_val_mean"], 4),
                dval_std=fmt(row["delta_val_std"], 4),
                dppl=fmt_signed(row["delta_ppl_mean"], 3),
                dppl_std=fmt(row["delta_ppl_std"], 3),
                vw=row["wins_val_over_uniform"],
                pw=row["wins_ppl_over_uniform"],
                hh=fmt(row["avg_heavy_hitter_budget"], 0),
                total=fmt(row["avg_total_retained_budget"], 0),
                tok=fmt(row["tok_per_s_mean"], 1),
            )
        )
    lines.extend(
        [
            "",
            "## Per-window results",
            "",
            "| Dataset | Window | Val skip | Test skip | Method | Val Loss (lower) | Test PPL (lower) | Delta Val (higher) | Delta PPL (higher) | tokens |",
            "|---|---|---:|---:|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        lines.append(
            "| {dataset} | {window} | {vskip} | {tskip} | {method} | {val} | {ppl} | {dval} | {dppl} | {vtok}/{ttok} |".format(
                dataset=row["dataset"],
                window=row["window_id"],
                vskip=row["val_skip_tokens"],
                tskip=row["test_skip_tokens"],
                method=row["method"],
                val=fmt(row["val_loss"], 4),
                ppl=fmt(row["test_ppl"], 3),
                dval=fmt_signed(row.get("delta_val_vs_uniform"), 4),
                dppl=fmt_signed(row.get("delta_ppl_vs_uniform"), 3),
                vtok=row["val_actual_tokens"],
                ttok=row["test_actual_tokens"],
            )
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- Delta is defined as uniform - method; positive values indicate improvement over uniform.",
            "- H2O-style and sliding-window diagnostics do not involve training seeds; this table reports deterministic evaluation-window robustness.",
            "- Results should be interpreted as operator/window sensitivity evidence, not universal optimality.",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=LOCAL_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=LOCAL_RESULTS_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = args.output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    rows = collect(args.input_dir)
    aggregates = aggregate(rows)
    decision_info = decisions(aggregates)
    payload = {"decisions": decision_info, "rows": rows, "aggregates": aggregates}
    (args.output_dir / "round26_h2o_window_robustness_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_markdown(rows, aggregates, decision_info, args.output_dir / "round26_h2o_window_robustness_summary.md")
    if aggregates:
        plot_delta(aggregates, fig_dir)
    print(f"Wrote R26 H2O window robustness summary to {args.output_dir}")


if __name__ == "__main__":
    main()
