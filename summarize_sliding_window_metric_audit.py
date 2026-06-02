"""Summarize R27 sliding-window metric consistency audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


LOCAL_RESULTS_DIR = Path("results") / "round27-phase27-sliding-window-metric-audit"
METHODS = ["uniform", "mono_inc", "mono_dec"]
DATASETS = ["c4", "owt"]
WINDOWS = ["A", "B", "C"]
METRICS = ["loss", "loss_after_512", "loss_after_768"]


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: float | None, digits: int = 4) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def fmt_signed(value: float | None, digits: int = 4) -> str:
    return "-" if value is None else f"{value:+.{digits}f}"


def collect(input_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for dataset in DATASETS:
        for window in WINDOWS:
            window_rows = []
            for method in METHODS:
                run_dir = input_dir / dataset / "formal" / f"window_{window}" / method
                summary = read_json(run_dir / "audit_summary.json")
                metrics = read_json(run_dir / "audit_metrics.json")
                profile = read_json(run_dir / "audit_profile.json")
                if not summary or not metrics:
                    continue
                val = metrics["validation"]
                row = {
                    "dataset_key": dataset,
                    "dataset": "C4" if dataset == "c4" else "OpenWebText",
                    "window_id": window,
                    "method": method,
                    "val_loss": val["loss"],
                    "val_loss_after_512": val["loss_after_512"],
                    "val_loss_after_768": val["loss_after_768"],
                    "tokens_all": val["tokens"],
                    "tokens_after_512": val["tokens_after_512"],
                    "tokens_after_768": val["tokens_after_768"],
                    "avg_window_size": (profile or {}).get("avg_window_size"),
                }
                window_rows.append(row)
            baseline = next((row for row in window_rows if row["method"] == "uniform"), None)
            if baseline:
                for row in window_rows:
                    row["delta_loss"] = baseline["val_loss"] - row["val_loss"]
                    row["delta_after_512"] = baseline["val_loss_after_512"] - row["val_loss_after_512"]
                    row["delta_after_768"] = baseline["val_loss_after_768"] - row["val_loss_after_768"]
            rows.extend(window_rows)
    return rows


def rankings(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, str]]]:
    out: dict[str, dict[str, dict[str, str]]] = {}
    for dataset in DATASETS:
        out[dataset] = {}
        for metric in METRICS:
            winners = []
            for window in WINDOWS:
                subset = [r for r in rows if r["dataset_key"] == dataset and r["window_id"] == window]
                if len(subset) == 3:
                    winners.append(min(subset, key=lambda r: r[f"val_{metric}" if metric != "loss" else "val_loss"])["method"])
            counts = {method: winners.count(method) for method in METHODS}
            out[dataset][metric] = {"winners": ",".join(winners), "counts": str(counts)}
    return out


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    agg = []
    for dataset in DATASETS:
        for method in METHODS:
            subset = [r for r in rows if r["dataset_key"] == dataset and r["method"] == method]
            if not subset:
                continue
            agg.append(
                {
                    "dataset_key": dataset,
                    "dataset": "C4" if dataset == "c4" else "OpenWebText",
                    "method": method,
                    "n": len(subset),
                    "val_loss_mean": mean(r["val_loss"] for r in subset),
                    "after512_mean": mean(r["val_loss_after_512"] for r in subset),
                    "after768_mean": mean(r["val_loss_after_768"] for r in subset),
                    "delta_loss_mean": mean(r["delta_loss"] for r in subset),
                    "delta_after512_mean": mean(r["delta_after_512"] for r in subset),
                    "delta_after768_mean": mean(r["delta_after_768"] for r in subset),
                    "tokens_all": subset[0]["tokens_all"],
                    "tokens_after_512": subset[0]["tokens_after_512"],
                    "tokens_after_768": subset[0]["tokens_after_768"],
                }
            )
    return agg


def write_markdown(rows: list[dict[str, Any]], agg: list[dict[str, Any]], rank: dict[str, Any], out_path: Path) -> None:
    lines = [
        "# R27 Sliding-window Metric Consistency Audit",
        "",
        "This audit recomputes all-token loss, after512, and after768 from the same logits/labels/forward path.",
        "",
        "## Mean across windows",
        "",
        "| Dataset | Method | Windows | Val all | Val after512 | Val after768 | Delta all | Delta after512 | Delta after768 | Tokens all/512/768 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in agg:
        lines.append(
            "| {dataset} | {method} | {n} | {all} | {a512} | {a768} | {dall} | {d512} | {d768} | {tok}/{tok512}/{tok768} |".format(
                dataset=row["dataset"],
                method=row["method"],
                n=row["n"],
                all=fmt(row["val_loss_mean"], 4),
                a512=fmt(row["after512_mean"], 4),
                a768=fmt(row["after768_mean"], 4),
                dall=fmt_signed(row["delta_loss_mean"], 4),
                d512=fmt_signed(row["delta_after512_mean"], 4),
                d768=fmt_signed(row["delta_after768_mean"], 4),
                tok=row["tokens_all"],
                tok512=row["tokens_after_512"],
                tok768=row["tokens_after_768"],
            )
        )
    lines.extend(["", "## Ranking audit", ""])
    for dataset, metric_info in rank.items():
        label = "C4" if dataset == "c4" else "OpenWebText"
        for metric, info in metric_info.items():
            lines.append(f"- **{label} / {metric}**: winners per window = `{info['winners']}`, counts = `{info['counts']}`")
    lines.extend(
        [
            "",
            "## Per-window deltas",
            "",
            "| Dataset | Window | Method | Delta all | Delta after512 | Delta after768 |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['window_id']} | {row['method']} | {fmt_signed(row.get('delta_loss'),4)} | {fmt_signed(row.get('delta_after_512'),4)} | {fmt_signed(row.get('delta_after_768'),4)} |"
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- Delta is uniform - method; positive values indicate improvement over uniform.",
            "- The previous R27 `loss_after_window_active` was not comparable across methods because uniform used 512 excluded tokens while mono_inc/mono_dec used 768.",
            "- This audit fixes the cutoff across all methods at 512 and 768.",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=LOCAL_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=LOCAL_RESULTS_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = collect(args.input_dir)
    agg = aggregate(rows)
    rank = rankings(rows)
    payload = {"rows": rows, "aggregates": agg, "rankings": rank}
    (args.output_dir / "round27_sliding_window_metric_audit_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(rows, agg, rank, args.output_dir / "round27_sliding_window_metric_audit_summary.md")
    print(f"Wrote audit summary to {args.output_dir}")


if __name__ == "__main__":
    main()
