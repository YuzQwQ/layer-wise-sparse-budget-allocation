"""Summarize R13 static-schedule direction validation."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ablation_remote_common import (
    ROUND8_LOCAL_RESULTS_DIR,
    ROUND10_LOCAL_RESULTS_DIR,
    ROUND11_LOCAL_RESULTS_DIR,
    ROUND12_LOCAL_RESULTS_DIR,
    ROUND13_LOCAL_RESULTS_DIR,
)


METHOD_ORDER = ["baseline", "mono_inc", "mono_dec"]
LABELS = {
    "baseline": "baseline",
    "mono_inc": "mono_inc",
    "mono_dec": "mono_dec",
}
COLORS = {
    "baseline": "#4C78A8",
    "mono_inc": "#54A24B",
    "mono_dec": "#E45756",
}
MARKERS = {
    "baseline": "o",
    "mono_inc": "s",
    "mono_dec": "^",
}


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def metric_value(row: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = row.get(key)
        if value is not None:
            return float(value)
    return None


def round_float(value: float | None, digits: int = 6) -> float | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return round(float(value), digits)


def stats(values: list[float | None]) -> tuple[float | None, float | None]:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None, None
    return mean(clean), pstdev(clean) if len(clean) > 1 else 0.0


def fmt(value: float | None, digits: int = 4) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def fmt_stat(avg: float | None, std: float | None, digits: int = 4) -> str:
    return "-" if avg is None else f"{avg:.{digits}f} +/- {0.0 if std is None else std:.{digits}f}"


def last_with_key(rows: list[dict[str, Any]], *keys: str) -> float | None:
    for row in reversed(rows):
        value = metric_value(row, *keys)
        if value is not None:
            return value
    return None


def extract_run(*, dataset: str, method: str, seed: int, run_name: str, run_dir: Path, source: str) -> dict[str, Any] | None:
    train_log = read_jsonl(run_dir / "train_log.jsonl")
    eval_log = read_jsonl(run_dir / "eval_log.jsonl")
    budget_log = read_jsonl(run_dir / "budget_diagnostics.jsonl")
    test_metrics = read_json(run_dir / "test_metrics.json") or {}
    summary = read_json(run_dir / "run_summary.json") or {}
    if not train_log and not eval_log and not test_metrics:
        return None

    last_eval = eval_log[-1] if eval_log else (summary.get("final_validation") or {})
    last_budget = budget_log[-1] if budget_log else (train_log[-1] if train_log else {})
    return {
        "dataset": dataset,
        "method": method,
        "seed": seed,
        "run_name": run_name,
        "source": source,
        "final_val_loss": metric_value(last_eval, "lm_loss", "loss"),
        "final_val_ppl": metric_value(last_eval, "ppl", "perplexity"),
        "test_loss": metric_value(test_metrics, "lm_loss", "loss"),
        "test_ppl": metric_value(test_metrics, "ppl", "perplexity"),
        "actual_k_mean": metric_value(last_budget, "actual_k_mean", "budget_mean"),
        "actual_k_p50": metric_value(last_budget, "actual_k_p50", "budget_p50"),
        "actual_k_p90": metric_value(last_budget, "actual_k_p90", "budget_p90"),
        "estimated_kv_access_mean": metric_value(last_budget, "estimated_kv_access_mean"),
        "tok_per_s": last_with_key(train_log, "tok_per_s"),
    }


def load_c4_runs(r10_dir: Path, r11_dir: Path, r12_dir: Path, r13_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    run_specs = []
    for seed in [42, 123, 3407]:
        run_specs.append(("baseline", seed, f"r11_m0_baseline_s{seed}", r11_dir / "exp", "final_baseline_reference"))
        run_specs.append(("mono_inc", seed, f"r10_m1_mono_inc_s{seed}", r10_dir / "exp", "longtrain_mono_inc_reference"))
        run_specs.append(("mono_dec", seed, f"r12b_m2_mono_dec_s{seed}", r12_dir / "exp", "r12_static_reference"))
    for seed in [2024, 2025]:
        run_specs.append(("baseline", seed, f"r12a_m0_baseline_s{seed}", r12_dir / "exp", "r12_extra_seed_reference"))
        run_specs.append(("mono_inc", seed, f"r12a_m1_mono_inc_s{seed}", r12_dir / "exp", "r12_extra_seed_reference"))
        run_specs.append(("mono_dec", seed, f"r13a_c4_m2_mono_dec_s{seed}", r13_dir / "exp", "r13_new_run"))

    for method, seed, run_name, exp_dir, source in run_specs:
        row = extract_run(
            dataset="c4",
            method=method,
            seed=seed,
            run_name=run_name,
            run_dir=exp_dir / run_name,
            source=source,
        )
        if row is not None:
            rows.append(row)
    return rows


def load_owt_runs(r8_dir: Path, r13_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in [42, 123, 3407]:
        for method, run_name, exp_dir, source in [
            ("baseline", f"r8a_owt_m0_baseline_s{seed}", r8_dir / "exp", "r8_openwebtext_reference"),
            ("mono_inc", f"r8a_owt_m1_mono_inc_s{seed}", r8_dir / "exp", "r8_openwebtext_reference"),
            ("mono_dec", f"r13b_owt_m2_mono_dec_s{seed}", r13_dir / "exp", "r13_new_run"),
        ]:
            row = extract_run(
                dataset="openwebtext",
                method=method,
                seed=seed,
                run_name=run_name,
                run_dir=exp_dir / run_name,
                source=source,
            )
            if row is not None:
                rows.append(row)
    return rows


def aggregate(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["method"])].append(row)

    metrics = [
        "final_val_loss",
        "final_val_ppl",
        "test_loss",
        "test_ppl",
        "actual_k_mean",
        "actual_k_p50",
        "actual_k_p90",
        "estimated_kv_access_mean",
        "tok_per_s",
    ]
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for dataset in ["c4", "openwebtext"]:
        out[dataset] = {}
        for method in METHOD_ORDER:
            items = grouped.get((dataset, method), [])
            entry: dict[str, Any] = {
                "label": LABELS[method],
                "count": len(items),
                "seeds": sorted(row["seed"] for row in items),
            }
            for metric in metrics:
                avg, std = stats([row.get(metric) for row in items])
                entry[f"{metric}_mean"] = round_float(avg)
                entry[f"{metric}_std"] = round_float(std)
            out[dataset][method] = entry
    return out


def add_deltas(summary: dict[str, dict[str, dict[str, Any]]]) -> None:
    for dataset, methods in summary.items():
        baseline_val = methods["baseline"].get("final_val_loss_mean")
        baseline_test = methods["baseline"].get("test_ppl_mean")
        mono_inc_val = methods["mono_inc"].get("final_val_loss_mean")
        mono_inc_test = methods["mono_inc"].get("test_ppl_mean")
        mono_dec_val = methods["mono_dec"].get("final_val_loss_mean")
        mono_dec_test = methods["mono_dec"].get("test_ppl_mean")
        for method, entry in methods.items():
            val = entry.get("final_val_loss_mean")
            test = entry.get("test_ppl_mean")
            entry["delta_val_vs_baseline"] = round_float(None if val is None or baseline_val is None else val - baseline_val)
            entry["delta_test_ppl_vs_baseline"] = round_float(None if test is None or baseline_test is None else test - baseline_test)
            entry["delta_val_vs_mono_inc"] = round_float(None if val is None or mono_inc_val is None else val - mono_inc_val)
            entry["delta_test_ppl_vs_mono_inc"] = round_float(None if test is None or mono_inc_test is None else test - mono_inc_test)
            entry["delta_val_vs_mono_dec"] = round_float(None if val is None or mono_dec_val is None else val - mono_dec_val)
            entry["delta_test_ppl_vs_mono_dec"] = round_float(None if test is None or mono_dec_test is None else test - mono_dec_test)


def grouped_by_method(rows: list[dict[str, Any]], dataset: str) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {method: [] for method in METHOD_ORDER}
    for row in rows:
        if row["dataset"] == dataset and row["method"] in out:
            out[row["method"]].append(row)
    for items in out.values():
        items.sort(key=lambda row: row["seed"])
    return out


def plot_final_metrics(rows: list[dict[str, Any]], dataset: str, out_path: Path, title: str) -> None:
    grouped = grouped_by_method(rows, dataset)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=180)
    metrics = [("final_val_loss", "Final validation loss"), ("test_ppl", "Final test PPL")]
    x_positions = range(len(METHOD_ORDER))
    for ax, (metric, ylabel) in zip(axes, metrics):
        means = []
        stds = []
        for method in METHOD_ORDER:
            avg, std = stats([row.get(metric) for row in grouped[method]])
            means.append(avg)
            stds.append(std)
        for idx, method in enumerate(METHOD_ORDER):
            if means[idx] is not None:
                ax.errorbar(
                    idx,
                    means[idx],
                    yerr=stds[idx],
                    fmt=MARKERS[method],
                    color=COLORS[method],
                    capsize=4,
                    markersize=7,
                    label=LABELS[method],
                    zorder=3,
                )
            values = [row.get(metric) for row in grouped[method] if row.get(metric) is not None]
            for j, value in enumerate(values):
                jitter = (j - (len(values) - 1) / 2) * 0.035
                ax.scatter(idx + jitter, value, s=22, color=COLORS[method], alpha=0.55, zorder=2)
        ax.set_xticks(list(x_positions))
        ax.set_xticklabels([LABELS[m] for m in METHOD_ORDER], rotation=15)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    axes[0].legend(frameon=False, loc="best")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_improvements(rows: list[dict[str, Any]], dataset: str, out_path: Path, title: str) -> None:
    grouped = grouped_by_method(rows, dataset)
    baseline_by_seed = {row["seed"]: row for row in grouped["baseline"]}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=180)
    panels = [("final_val_loss", "Validation loss improvement"), ("test_ppl", "Test PPL improvement")]
    seeds = sorted(baseline_by_seed)
    for ax, (metric, ylabel) in zip(axes, panels):
        for method in ["mono_inc", "mono_dec"]:
            xs = []
            ys = []
            by_seed = {row["seed"]: row for row in grouped[method]}
            for idx, seed in enumerate(seeds):
                base = baseline_by_seed.get(seed, {}).get(metric)
                current = by_seed.get(seed, {}).get(metric)
                if base is None or current is None:
                    continue
                xs.append(idx + (-0.08 if method == "mono_inc" else 0.08))
                ys.append(base - current)
            ax.scatter(xs, ys, color=COLORS[method], marker=MARKERS[method], s=45, label=LABELS[method], zorder=3)
            if ys:
                ax.axhline(mean(ys), color=COLORS[method], linestyle=":", linewidth=1.2, alpha=0.8)
        ax.axhline(0.0, color="#333333", linestyle="--", linewidth=1.2)
        ax.set_xticks(list(range(len(seeds))))
        ax.set_xticklabels([str(seed) for seed in seeds])
        ax.set_xlabel("seed")
        ax.set_ylabel(ylabel + " (baseline - method)")
        ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    axes[0].legend(frameon=False, loc="best")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_efficiency(rows: list[dict[str, Any]], dataset: str, out_path: Path, title: str) -> None:
    grouped = grouped_by_method(rows, dataset)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=180)
    metrics = [("estimated_kv_access_mean", "Estimated KV access"), ("tok_per_s", "Tokens / second")]
    for ax, (metric, ylabel) in zip(axes, metrics):
        values = []
        stds = []
        for method in METHOD_ORDER:
            avg, std = stats([row.get(metric) for row in grouped[method]])
            values.append(avg)
            stds.append(std)
        ax.bar(
            range(len(METHOD_ORDER)),
            [0 if value is None else value for value in values],
            yerr=[0 if value is None else stds[idx] for idx, value in enumerate(values)],
            color=[COLORS[m] for m in METHOD_ORDER],
            alpha=0.85,
            capsize=3,
        )
        ax.set_xticks(range(len(METHOD_ORDER)))
        ax.set_xticklabels([LABELS[m] for m in METHOD_ORDER], rotation=15)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def decision_text(summary: dict[str, dict[str, dict[str, Any]]]) -> dict[str, str]:
    out = {}
    c4 = summary["c4"]
    if any(c4[method].get("count", 0) < 5 for method in ["baseline", "mono_inc", "mono_dec"]):
        out["c4"] = "pending: C4 direction decision requires 5 seeds for baseline, mono_inc, and mono_dec."
    else:
        inc_val = c4["mono_inc"].get("final_val_loss_mean")
        dec_val = c4["mono_dec"].get("final_val_loss_mean")
        base_val = c4["baseline"].get("final_val_loss_mean")
        inc_test = c4["mono_inc"].get("test_ppl_mean")
        dec_test = c4["mono_dec"].get("test_ppl_mean")
        base_test = c4["baseline"].get("test_ppl_mean")
        if None in {inc_val, dec_val, base_val, inc_test, dec_test, base_test}:
            out["c4"] = "pending: missing one or more C4 5-seed groups."
        elif inc_val < base_val and inc_test < base_test and not (dec_val < base_val and dec_test < base_test):
            out["c4"] = "mono_inc remains the supported static direction; mono_dec does not retain a matched-compute improvement over baseline."
        elif dec_val < base_val and dec_test < base_test and not (inc_val < base_val and inc_test < base_test):
            out["c4"] = "mono_dec is the only monotonic schedule with a same-direction matched-compute signal; avoid a mono_inc-only claim."
        elif inc_val < base_val and dec_val < base_val and inc_test < base_test and dec_test < base_test:
            gap = abs(inc_val - dec_val)
            if gap < 0.005:
                out["c4"] = "monotonic static allocation has same-direction matched-compute signal; direction remains close."
            elif inc_val < dec_val:
                out["c4"] = "mono_inc is favored over mono_dec under the C4 5-seed setting."
            else:
                out["c4"] = "mono_dec is favored over mono_inc under the C4 5-seed setting; avoid a mono_inc-only final claim."
        else:
            out["c4"] = "static monotonic signal is not stable against the baseline under the C4 5-seed setting."

    owt = summary["openwebtext"]
    inc_val = owt["mono_inc"].get("final_val_loss_mean")
    dec_val = owt["mono_dec"].get("final_val_loss_mean")
    if inc_val is None or dec_val is None:
        out["openwebtext"] = "pending: missing OpenWebText mono_dec or reference results."
    elif inc_val < dec_val:
        out["openwebtext"] = "OpenWebText favors mono_inc over mono_dec, suggesting possible corpus sensitivity."
    elif dec_val < inc_val:
        out["openwebtext"] = "OpenWebText favors mono_dec or a non-increasing direction; monotonic direction is not settled."
    else:
        out["openwebtext"] = "OpenWebText shows no meaningful direction separation."
    return out


def write_markdown(summary: dict[str, dict[str, dict[str, Any]]], decisions: dict[str, str], out_path: Path) -> None:
    lines = ["# R13 Static Direction Summary", ""]
    for dataset, title in [("c4", "C4 5-seed static direction"), ("openwebtext", "OpenWebText direction check")]:
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| method | count | seeds | final val loss | test PPL | actual_k | KV access | tok/s | dVal vs baseline | dVal vs mono_inc |")
        lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|")
        for method in METHOD_ORDER:
            entry = summary[dataset][method]
            lines.append(
                "| {method} | {count} | {seeds} | {val} | {test} | {actual_k} | {kv} | {tok} | {d_base} | {d_inc} |".format(
                    method=LABELS[method],
                    count=entry["count"],
                    seeds=",".join(str(seed) for seed in entry["seeds"]),
                    val=fmt_stat(entry.get("final_val_loss_mean"), entry.get("final_val_loss_std")),
                    test=fmt_stat(entry.get("test_ppl_mean"), entry.get("test_ppl_std"), digits=3),
                    actual_k=fmt_stat(entry.get("actual_k_mean_mean"), entry.get("actual_k_mean_std"), digits=3),
                    kv=fmt_stat(entry.get("estimated_kv_access_mean_mean"), entry.get("estimated_kv_access_mean_std"), digits=2),
                    tok=fmt_stat(entry.get("tok_per_s_mean"), entry.get("tok_per_s_std"), digits=1),
                    d_base=fmt(entry.get("delta_val_vs_baseline")),
                    d_inc=fmt(entry.get("delta_val_vs_mono_inc")),
                )
            )
        lines.extend(["", f"Decision: {decisions[dataset]}", ""])
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round8-dir", type=Path, default=ROUND8_LOCAL_RESULTS_DIR)
    parser.add_argument("--round10-dir", type=Path, default=ROUND10_LOCAL_RESULTS_DIR)
    parser.add_argument("--round11-dir", type=Path, default=ROUND11_LOCAL_RESULTS_DIR)
    parser.add_argument("--round12-dir", type=Path, default=ROUND12_LOCAL_RESULTS_DIR)
    parser.add_argument("--round13-dir", type=Path, default=ROUND13_LOCAL_RESULTS_DIR)
    parser.add_argument("--out-dir", type=Path, default=ROUND13_LOCAL_RESULTS_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_c4_runs(args.round10_dir, args.round11_dir, args.round12_dir, args.round13_dir)
    rows.extend(load_owt_runs(args.round8_dir, args.round13_dir))
    summary = aggregate(rows)
    add_deltas(summary)
    decisions = decision_text(summary)

    payload = {
        "summary": summary,
        "decisions": decisions,
        "runs": rows,
    }
    (args.out_dir / "round13_static_direction_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_markdown(summary, decisions, args.out_dir / "round13_static_direction_summary.md")
    plot_final_metrics(rows, "c4", args.out_dir / "round13_c4_final_metrics.png", "C4 final matched-compute static direction")
    plot_improvements(rows, "c4", args.out_dir / "round13_c4_per_seed_improvement.png", "C4 per-seed improvement over baseline")
    plot_final_metrics(rows, "openwebtext", args.out_dir / "round13_openwebtext_final_metrics.png", "OpenWebText monotonic direction check")
    plot_efficiency(rows, "c4", args.out_dir / "round13_c4_efficiency.png", "C4 compute and throughput comparison")
    print(f"Wrote {args.out_dir / 'round13_static_direction_summary.md'}")


if __name__ == "__main__":
    main()
