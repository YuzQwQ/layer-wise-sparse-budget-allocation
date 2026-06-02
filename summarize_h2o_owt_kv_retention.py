"""Summarize R24 H2O-style token-level KV retention diagnostic on OpenWebText."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LOCAL_RESULTS_DIR = Path("results") / "round24-phase24-h2o-openwebtext-kv-retention-diagnostic"
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
        run_dir = input_dir / "formal" / f"r24_owt_{method}"
        summary = read_json(run_dir / "run_summary.json")
        profile = read_json(run_dir / "retention_profile.json")
        metrics = read_json(run_dir / "eval_metrics.json")
        if not summary:
            continue
        val_cache = (metrics or {}).get("val_cache") or {}
        test_cache = (metrics or {}).get("test_cache") or {}
        rows.append(
            {
                "sparse_platform": "H2O-style KV retention diagnostic",
                "sparse_unit": "KV tokens",
                "model": summary.get("model_name", "facebook/opt-350m"),
                "dataset": f"{summary.get('dataset', 'Skylion007/openwebtext')}/{summary.get('dataset_name', 'plain_text')}",
                "context_length": safe_float(summary.get("evaluation_context_length")),
                "method": method,
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
                "val_split": val_cache.get("split"),
                "test_split": test_cache.get("split"),
                "val_window_name": val_cache.get("window_name"),
                "test_window_name": test_cache.get("window_name"),
                "val_skip_tokens": val_cache.get("skip_tokens"),
                "test_skip_tokens": test_cache.get("skip_tokens"),
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
        return {"status": "incomplete", "interpretation": "R24 OpenWebText H2O-style diagnostic is incomplete."}
    best = min(rows, key=lambda row: row["val_loss"] if row["val_loss"] is not None else float("inf"))
    if best["method"] == "mono_inc":
        return {
            "status": "limited_cross_operator_support_on_openwebtext",
            "interpretation": "mono_inc is best in this OpenWebText H2O-style diagnostic; write as limited cross-operator support on OpenWebText only.",
        }
    return {
        "status": "operator_sensitivity_evidence_on_openwebtext",
        "interpretation": f"{best['method']} is best in this OpenWebText H2O-style diagnostic; write as operator sensitivity evidence on OpenWebText.",
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
    fig.suptitle("OpenWebText H2O-style KV retention diagnostic: improvement over uniform")
    fig.tight_layout()
    fig.savefig(out_dir / "h2o_owt_delta_vs_uniform.png")
    plt.close(fig)


def plot_budgets(rows: list[dict[str, Any]], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 3.8), dpi=180)
    for row in rows:
        budgets = row.get("layer_budgets") or []
        if budgets:
            ax.plot(range(1, len(budgets) + 1), budgets, marker="o", linewidth=1.5, label=row["method"], color=COLORS[row["method"]])
    ax.set_title("Layer-wise heavy-hitter budget schedules")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Heavy-hitter budget")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "h2o_owt_retention_budget_by_layer.png")
    plt.close(fig)


def write_markdown(rows: list[dict[str, Any]], decision_info: dict[str, str], out_path: Path) -> None:
    lines = [
        "# R24 H2O-style Token-level KV Retention Diagnostic on OpenWebText",
        "",
        f"Decision status: `{decision_info['status']}`",
        "",
        decision_info["interpretation"],
        "",
        "| Sparse platform | Sparse unit | Model | Dataset | Context | Method | Val Loss ↓ | Test PPL ↓ | Loss after cache full ↓ | Δ Val ↑ | Δ PPL ↑ | recent size | avg HH budget | avg total retained | tok/s |",
        "|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {platform} | {unit} | {model} | {dataset} | {ctx} | {method} | {val} | {ppl} | {late} | {dval} | {dppl} | {recent} | {hh} | {total} | {tok} |".format(
                platform=row["sparse_platform"],
                unit=row["sparse_unit"],
                model=row["model"],
                dataset=row["dataset"],
                ctx=fmt(row["context_length"], 0),
                method=row["method"],
                val=fmt(row["val_loss"], 4),
                ppl=fmt(row["test_ppl"], 3),
                late=fmt(row["val_loss_after_cache_full"], 4),
                dval=fmt_signed(row.get("delta_val_vs_uniform"), 4),
                dppl=fmt_signed(row.get("delta_ppl_vs_uniform"), 3),
                recent=fmt(row["recent_size"], 0),
                hh=fmt(row["avg_heavy_hitter_budget"], 0),
                total=fmt(row["avg_total_retained_budget"], 0),
                tok=fmt(row["tok_per_s"], 1),
            )
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- Delta is defined as uniform - method; positive values indicate improvement over uniform.",
            "- This is limited cross-operator diagnostic evidence, not a full H2O benchmark or H2O reproduction.",
            "- OpenWebText validation and test are deterministic windows from the same train split, with distinct skip tokens.",
            "- Formal validation/test token windows are required to contain exactly 393,216 tokens each.",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=LOCAL_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    out_dir = args.output_dir or args.input_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    rows = collect(args.input_dir)
    decision_info = decision(rows)
    payload = {"decision": decision_info, "rows": rows}
    (out_dir / "round24_h2o_owt_kv_retention_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(rows, decision_info, out_dir / "round24_h2o_owt_kv_retention_summary.md")
    if rows:
        plot_delta(rows, fig_dir)
        plot_budgets(rows, fig_dir)
    print(f"Wrote R24 summary to {out_dir}")


if __name__ == "__main__":
    main()
