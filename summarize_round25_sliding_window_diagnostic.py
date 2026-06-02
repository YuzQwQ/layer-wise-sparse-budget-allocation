"""Aggregate R25 sliding-window diagnostics across WikiText, C4, and OpenWebText."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("results")
DATASETS = {
    "WikiText-103": (
        ROOT / "round25-phase25-sliding-window-wikitext-diagnostic",
        "round25_sliding_window_wikitext_summary.json",
    ),
    "C4": (
        ROOT / "round25-phase25-sliding-window-c4-diagnostic",
        "round25_sliding_window_c4_summary.json",
    ),
    "OpenWebText": (
        ROOT / "round25-phase25-sliding-window-openwebtext-diagnostic",
        "round25_sliding_window_openwebtext_summary.json",
    ),
}
METHOD_ORDER = ["uniform", "mono_inc", "mono_dec"]
COLORS = {"uniform": "#6F7A83", "mono_inc": "#2FA17D", "mono_dec": "#D97A2B"}


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def fmt_signed(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):+.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def collect() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    decisions: dict[str, Any] = {}
    for dataset_name, (root, filename) in DATASETS.items():
        payload = read_json(root / filename)
        if not payload:
            decisions[dataset_name] = {"status": "missing"}
            continue
        decisions[dataset_name] = payload.get("decision", {})
        for row in payload.get("rows", []):
            enriched = dict(row)
            enriched["dataset_label"] = dataset_name
            rows.append(enriched)
    return rows, decisions


def plot_delta_by_dataset(rows: list[dict[str, Any]], out_dir: Path) -> None:
    datasets = list(DATASETS)
    x = list(range(len(datasets)))
    width = 0.24
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 6.4), dpi=180)
    for ax, key, title, ylabel, digits in [
        (axes[0], "delta_val_vs_uniform", "Validation loss improvement", "Delta Val = uniform - method", 4),
        (axes[1], "delta_ppl_vs_uniform", "Test PPL improvement", "Delta PPL = uniform - method", 3),
    ]:
        for idx, method in enumerate(METHOD_ORDER):
            values = []
            for dataset in datasets:
                row = next((item for item in rows if item.get("dataset_label") == dataset and item.get("method") == method), None)
                values.append(float(row.get(key, 0.0)) if row and row.get(key) is not None else 0.0)
            offsets = [pos + (idx - 1) * width for pos in x]
            bars = ax.bar(offsets, values, width=width, color=COLORS[method], alpha=0.88, label=method)
            for bar, value in zip(bars, values):
                if abs(value) < 1e-12:
                    continue
                va = "bottom" if value >= 0 else "top"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value,
                    f"{value:+.{digits}f}",
                    ha="center",
                    va=va,
                    fontsize=7,
                    rotation=90,
                )
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(datasets)
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False, ncols=3, loc="best")
    fig.suptitle("R25 sliding-window diagnostics: improvement over uniform")
    fig.tight_layout()
    fig.savefig(out_dir / "sliding_window_delta_vs_uniform_by_dataset.png")
    plt.close(fig)


def plot_budget(out_dir: Path) -> None:
    windows = {
        "uniform": [512] * 24,
        "mono_inc": [256] * 6 + [384] * 6 + [640] * 6 + [768] * 6,
        "mono_dec": [768] * 6 + [640] * 6 + [384] * 6 + [256] * 6,
    }
    fig, ax = plt.subplots(figsize=(8.0, 3.8), dpi=180)
    for method, values in windows.items():
        ax.plot(range(1, 25), values, marker="o", linewidth=1.5, label=method, color=COLORS[method])
    ax.set_title("Layer-wise local window schedules")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Window size")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "sliding_window_budget_by_layer.png")
    plt.close(fig)


def write_markdown(rows: list[dict[str, Any]], decisions: dict[str, Any], out_path: Path) -> None:
    lines = [
        "# R25 Sliding-window / Streaming-style Window-level Diagnostic",
        "",
        "This is lightweight cross-operator diagnostic evidence, not a full sliding-window benchmark or StreamingLLM reproduction.",
        "",
        "## Decisions",
        "",
    ]
    for dataset, decision in decisions.items():
        lines.append(f"- **{dataset}**: `{decision.get('status', 'missing')}`")
    lines.extend(
        [
            "",
            "## Results",
            "",
            "| Dataset | Sparse unit | Model | Context | Method | Val Loss ↓ | Test PPL ↓ | Loss after active ↓ | Δ Val ↑ | Δ PPL ↑ | Avg window | tok/s |",
            "|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for dataset in DATASETS:
        for method in METHOD_ORDER:
            row = next((item for item in rows if item.get("dataset_label") == dataset and item.get("method") == method), None)
            if not row:
                continue
            lines.append(
                "| {dataset} | {unit} | {model} | {ctx} | {method} | {val} | {ppl} | {late} | {dval} | {dppl} | {avg} | {tok} |".format(
                    dataset=dataset,
                    unit=row.get("sparse_unit", "local attention window"),
                    model=row.get("model", "facebook/opt-350m"),
                    ctx=fmt(row.get("context_length"), 0),
                    method=method,
                    val=fmt(row.get("val_loss"), 4),
                    ppl=fmt(row.get("test_ppl"), 3),
                    late=fmt(row.get("val_loss_after_window_active"), 4),
                    dval=fmt_signed(row.get("delta_val_vs_uniform"), 4),
                    dppl=fmt_signed(row.get("delta_ppl_vs_uniform"), 3),
                    avg=fmt(row.get("avg_window_size"), 0),
                    tok=fmt(row.get("tok_per_s"), 1),
                )
            )
    lines.extend(
        [
            "",
            "Notes:",
            "- Delta is defined as uniform - method; positive values indicate improvement over uniform.",
            "- Preferred direction should be interpreted as dataset/operator sensitivity, not universal optimality.",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "round25-phase25-sliding-window-diagnostic")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = args.output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    rows, decisions = collect()
    payload = {"decisions": decisions, "rows": rows}
    (args.output_dir / "round25_sliding_window_diagnostic_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_markdown(rows, decisions, args.output_dir / "round25_sliding_window_diagnostic_summary.md")
    if rows:
        plot_delta_by_dataset(rows, fig_dir)
        plot_budget(fig_dir)
    print(f"Wrote R25 aggregate summary to {args.output_dir}")


if __name__ == "__main__":
    main()
