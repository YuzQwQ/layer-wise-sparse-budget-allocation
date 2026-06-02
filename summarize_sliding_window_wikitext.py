"""Summarize R25 sliding-window / Streaming-style diagnostic on WikiText-103."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LOCAL_RESULTS_DIR = Path("results") / "round25-phase25-sliding-window-wikitext-diagnostic"
SUMMARY_PREFIX = "round25_sliding_window_wikitext"
SUMMARY_TITLE = "R25 Sliding-window / Streaming-style Diagnostic on WikiText-103"
FORMAL_RUN_PREFIX = "r25_sliding_wikitext"
DATASET_SHORT_NAME = "WikiText-103"
METHOD_ORDER = ["uniform", "mono_inc", "mono_dec"]
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
    rows = []
    for method in METHOD_ORDER:
        run_dir = input_dir / "formal" / f"{FORMAL_RUN_PREFIX}_{method}"
        summary = read_json(run_dir / "run_summary.json")
        profile = read_json(run_dir / "window_profile.json")
        metrics = read_json(run_dir / "eval_metrics.json")
        if not summary:
            continue
        val_cache = (metrics or {}).get("val_cache") or {}
        test_cache = (metrics or {}).get("test_cache") or {}
        rows.append(
            {
                "sparse_platform": "Sliding-window-style diagnostic",
                "sparse_unit": "local attention window",
                "model": summary.get("model_name", "facebook/opt-350m"),
                "dataset": f"{summary.get('dataset', 'Salesforce/wikitext')}/{summary.get('dataset_name', 'wikitext-103-raw-v1')}",
                "context_length": safe_float(summary.get("evaluation_context_length")),
                "method": method,
                "val_loss": safe_float(summary.get("val_loss")),
                "test_ppl": safe_float(summary.get("test_ppl")),
                "val_loss_after_window_active": safe_float(summary.get("val_loss_after_window_active")),
                "test_ppl_after_window_active": safe_float(summary.get("test_ppl_after_window_active")),
                "avg_window_size": safe_float(summary.get("avg_window_size")),
                "tok_per_s": safe_float(summary.get("tok_per_s")),
                "cuda_max_memory_allocated_mb": safe_float(summary.get("cuda_max_memory_allocated_mb")),
                "window_sizes": (profile or {}).get("window_sizes", []),
                "val_actual_tokens": val_cache.get("actual_tokens"),
                "test_actual_tokens": test_cache.get("actual_tokens"),
            }
        )
    baseline = next((row for row in rows if row["method"] == "uniform"), None)
    if baseline:
        for row in rows:
            row["delta_val_vs_uniform"] = (
                baseline["val_loss"] - row["val_loss"] if baseline["val_loss"] is not None and row["val_loss"] is not None else None
            )
            row["delta_ppl_vs_uniform"] = (
                baseline["test_ppl"] - row["test_ppl"] if baseline["test_ppl"] is not None and row["test_ppl"] is not None else None
            )
    return rows


def decision(rows: list[dict[str, Any]]) -> dict[str, str]:
    if len(rows) < 3:
        return {"status": "incomplete", "interpretation": f"R25 {DATASET_SHORT_NAME} sliding-window diagnostic is incomplete."}
    best = min(rows, key=lambda row: row["val_loss"] if row["val_loss"] is not None else float("inf"))
    if best["method"] == "mono_inc":
        return {
            "status": f"limited_window_level_support_on_{DATASET_SHORT_NAME.lower().replace('-', '').replace(' ', '_')}",
            "interpretation": f"mono_inc is best in this {DATASET_SHORT_NAME} sliding-window diagnostic; write as limited window-level support only.",
        }
    if best["method"] == "uniform":
        return {
            "status": "uniform_strong_default_for_window_attention",
            "interpretation": f"uniform is best in this {DATASET_SHORT_NAME} sliding-window diagnostic; write as uniform remains a strong default for local-window sparse attention.",
        }
    return {
        "status": f"window_operator_sensitivity_evidence_on_{DATASET_SHORT_NAME.lower().replace('-', '').replace(' ', '_')}",
        "interpretation": f"mono_dec is best in this {DATASET_SHORT_NAME} sliding-window diagnostic; write as operator sensitivity evidence.",
    }


def plot_delta(rows: list[dict[str, Any]], out_dir: Path) -> None:
    methods = [row["method"] for row in rows]
    dval = [row.get("delta_val_vs_uniform") or 0.0 for row in rows]
    dppl = [row.get("delta_ppl_vs_uniform") or 0.0 for row in rows]
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 6.2), dpi=180)
    for ax, values, title, ylabel in [
        (axes[0], dval, "Validation loss improvement", "Delta Val = uniform - method"),
        (axes[1], dppl, "Test PPL improvement", "Delta PPL = uniform - method"),
    ]:
        bars = ax.bar(methods, values, color=[COLORS[m] for m in methods], alpha=0.88)
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.1)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ymin = min(min(values), 0.0)
        ymax = max(max(values), 0.0)
        span = max(ymax - ymin, 0.01 if ax is axes[0] else 0.1)
        ax.set_ylim(ymin - span * 0.25, ymax + span * 0.35)
        for bar, value in zip(bars, values):
            va = "bottom" if value >= 0 else "top"
            offset = span * (0.04 if value >= 0 else -0.04)
            label = f"{value:+.4f}" if ax is axes[0] else f"{value:+.3f}"
            ax.text(bar.get_x() + bar.get_width() / 2, value + offset, label, ha="center", va=va, fontsize=8)
    fig.suptitle("WikiText sliding-window diagnostic: improvement over uniform")
    fig.tight_layout()
    fig.savefig(out_dir / "sliding_wikitext_delta_vs_uniform.png")
    plt.close(fig)


def plot_windows(rows: list[dict[str, Any]], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 3.8), dpi=180)
    for row in rows:
        windows = row.get("window_sizes") or []
        if windows:
            ax.plot(range(1, len(windows) + 1), windows, marker="o", linewidth=1.5, label=row["method"], color=COLORS[row["method"]])
    ax.set_title("Layer-wise local window schedules")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Window size")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "sliding_wikitext_window_by_layer.png")
    plt.close(fig)


def write_markdown(rows: list[dict[str, Any]], decision_info: dict[str, str], out_path: Path) -> None:
    lines = [
        f"# {SUMMARY_TITLE}",
        "",
        f"Decision status: `{decision_info['status']}`",
        "",
        decision_info["interpretation"],
        "",
        "| Sparse platform | Sparse unit | Model | Dataset | Context | Method | Val Loss ↓ | Test PPL ↓ | Loss after active ↓ | Δ Val ↑ | Δ PPL ↑ | avg window | tok/s |",
        "|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {platform} | {unit} | {model} | {dataset} | {ctx} | {method} | {val} | {ppl} | {late} | {dval} | {dppl} | {avg} | {tok} |".format(
                platform=row["sparse_platform"],
                unit=row["sparse_unit"],
                model=row["model"],
                dataset=row["dataset"],
                ctx=fmt(row["context_length"], 0),
                method=row["method"],
                val=fmt(row["val_loss"], 4),
                ppl=fmt(row["test_ppl"], 3),
                late=fmt(row["val_loss_after_window_active"], 4),
                dval=fmt_signed(row.get("delta_val_vs_uniform"), 4),
                dppl=fmt_signed(row.get("delta_ppl_vs_uniform"), 3),
                avg=fmt(row["avg_window_size"], 0),
                tok=fmt(row["tok_per_s"], 1),
            )
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- Delta is defined as uniform - method; positive values indicate improvement over uniform.",
            "- This is lightweight cross-operator diagnostic evidence, not a full sliding-window benchmark or StreamingLLM reproduction.",
            "- WikiText official validation/test splits may provide fewer than the requested 393,216 tokens; actual token counts are recorded in JSON metadata.",
            "- The implementation uses evaluation-time attention masking and is not an optimized sparse runtime.",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    global SUMMARY_TITLE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=LOCAL_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--summary-prefix", default=SUMMARY_PREFIX)
    parser.add_argument("--summary-title", default=SUMMARY_TITLE)
    args = parser.parse_args()
    SUMMARY_TITLE = args.summary_title
    out_dir = args.output_dir or args.input_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    rows = collect(args.input_dir)
    decision_info = decision(rows)
    payload = {"decision": decision_info, "rows": rows}
    (out_dir / f"{args.summary_prefix}_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(rows, decision_info, out_dir / f"{args.summary_prefix}_summary.md")
    if rows:
        plot_delta(rows, fig_dir)
        plot_windows(rows, fig_dir)
    print(f"Wrote R25 summary to {out_dir}")


if __name__ == "__main__":
    main()
