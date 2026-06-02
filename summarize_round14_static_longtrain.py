"""Summarize R14 static-schedule 16384-step long training."""

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

from ablation_remote_common import ROUND14_LOCAL_RESULTS_DIR


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
RUN_PREFIX = {
    "baseline": "r14_m0_baseline",
    "mono_inc": "r14_m1_mono_inc",
    "mono_dec": "r14_m2_mono_dec",
}
SEEDS = [42, 123, 3407]


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


def value_at_step(rows: list[dict[str, Any]], step: int, *keys: str) -> float | None:
    best = None
    best_dist = None
    for row in rows:
        if "step" not in row:
            continue
        dist = abs(int(row["step"]) - step)
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best = row
    if best is None:
        return None
    return metric_value(best, *keys)


def rolling_mean(points: list[tuple[int, float]], window_steps: int = 100) -> list[tuple[int, float]]:
    out: list[tuple[int, float]] = []
    active: list[tuple[int, float]] = []
    for step, value in points:
        active.append((step, value))
        active = [(s, v) for s, v in active if step - s <= window_steps]
        out.append((step, sum(v for _, v in active) / len(active)))
    return out


def extract_run(method: str, seed: int, run_dir: Path) -> dict[str, Any] | None:
    train_log = read_jsonl(run_dir / "train_log.jsonl")
    eval_log = read_jsonl(run_dir / "eval_log.jsonl")
    budget_log = read_jsonl(run_dir / "budget_diagnostics.jsonl")
    test_metrics = read_json(run_dir / "test_metrics.json") or {}
    summary = read_json(run_dir / "run_summary.json") or {}
    if not train_log and not eval_log and not test_metrics:
        return None

    last_eval = eval_log[-1] if eval_log else (summary.get("final_validation") or {})
    last_budget = budget_log[-1] if budget_log else (train_log[-1] if train_log else {})
    train_points = []
    eval_points = []
    for row in train_log:
        loss = metric_value(row, "train_lm_loss", "lm_loss", "loss")
        if row.get("step") is not None and loss is not None:
            train_points.append((int(row["step"]), loss))
    for row in eval_log:
        loss = metric_value(row, "lm_loss", "loss")
        if row.get("step") is not None and loss is not None:
            eval_points.append((int(row["step"]), loss))

    return {
        "method": method,
        "seed": seed,
        "run_name": run_dir.name,
        "train_points": train_points,
        "eval_points": eval_points,
        "final_val_loss": metric_value(last_eval, "lm_loss", "loss"),
        "final_val_ppl": metric_value(last_eval, "ppl", "perplexity"),
        "val_at_8192": value_at_step(eval_log, 8192, "lm_loss", "loss"),
        "val_at_16384": value_at_step(eval_log, 16384, "lm_loss", "loss"),
        "test_loss": metric_value(test_metrics, "lm_loss", "loss"),
        "test_ppl": metric_value(test_metrics, "ppl", "perplexity"),
        "actual_k_mean": metric_value(last_budget, "actual_k_mean", "budget_mean"),
        "actual_k_p50": metric_value(last_budget, "actual_k_p50", "budget_p50"),
        "actual_k_p90": metric_value(last_budget, "actual_k_p90", "budget_p90"),
        "estimated_kv_access_mean": metric_value(last_budget, "estimated_kv_access_mean"),
        "estimated_sparse_attention_budget_ratio": metric_value(last_budget, "estimated_sparse_attention_budget_ratio"),
        "tok_per_s": last_with_key(train_log, "tok_per_s"),
        "cuda_max_memory_allocated_mb": last_with_key(train_log, "cuda_max_memory_allocated_mb"),
        "cuda_max_memory_reserved_mb": last_with_key(train_log, "cuda_max_memory_reserved_mb"),
    }


def load_runs(results_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    exp_dir = results_dir / "exp"
    for method in METHOD_ORDER:
        for seed in SEEDS:
            run_name = f"{RUN_PREFIX[method]}_s{seed}"
            row = extract_run(method, seed, exp_dir / run_name)
            if row is not None:
                rows.append(row)
    return rows


def aggregate(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)
    metrics = [
        "final_val_loss",
        "final_val_ppl",
        "val_at_8192",
        "val_at_16384",
        "test_loss",
        "test_ppl",
        "actual_k_mean",
        "actual_k_p50",
        "actual_k_p90",
        "estimated_kv_access_mean",
        "estimated_sparse_attention_budget_ratio",
        "tok_per_s",
        "cuda_max_memory_allocated_mb",
        "cuda_max_memory_reserved_mb",
    ]
    out: dict[str, dict[str, Any]] = {}
    for method in METHOD_ORDER:
        items = grouped.get(method, [])
        entry: dict[str, Any] = {
            "label": LABELS[method],
            "count": len(items),
            "seeds": sorted(row["seed"] for row in items),
        }
        for metric in metrics:
            avg, std = stats([row.get(metric) for row in items])
            entry[f"{metric}_mean"] = round_float(avg)
            entry[f"{metric}_std"] = round_float(std)
        if entry.get("val_at_8192_mean") is not None and entry.get("val_at_16384_mean") is not None:
            entry["post8192_val_delta"] = round_float(entry["val_at_16384_mean"] - entry["val_at_8192_mean"])
        out[method] = entry
    baseline = out["baseline"]
    for method, entry in out.items():
        for metric in ["final_val_loss", "test_ppl", "val_at_8192", "val_at_16384"]:
            base = baseline.get(f"{metric}_mean")
            current = entry.get(f"{metric}_mean")
            entry[f"delta_{metric}_vs_baseline"] = round_float(None if base is None or current is None else current - base)
    return out


def grouped_by_method(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out = {method: [] for method in METHOD_ORDER}
    for row in rows:
        out[row["method"]].append(row)
    for items in out.values():
        items.sort(key=lambda row: row["seed"])
    return out


def plot_final_metrics(rows: list[dict[str, Any]], out_path: Path) -> None:
    grouped = grouped_by_method(rows)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=180)
    for ax, (metric, ylabel) in zip(axes, [("final_val_loss", "Final validation loss"), ("test_ppl", "Final test PPL")]):
        for idx, method in enumerate(METHOD_ORDER):
            avg, std = stats([row.get(metric) for row in grouped[method]])
            if avg is not None:
                ax.errorbar(idx, avg, yerr=std, fmt=MARKERS[method], color=COLORS[method], capsize=4, markersize=7)
            values = [row.get(metric) for row in grouped[method] if row.get(metric) is not None]
            for j, value in enumerate(values):
                jitter = (j - (len(values) - 1) / 2) * 0.04
                ax.scatter(idx + jitter, value, color=COLORS[method], alpha=0.55, s=24)
        ax.set_xticks(range(len(METHOD_ORDER)))
        ax.set_xticklabels([LABELS[m] for m in METHOD_ORDER], rotation=15)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    fig.suptitle("R14 final matched-compute metrics")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_improvement_scatter(rows: list[dict[str, Any]], out_path: Path) -> None:
    grouped = grouped_by_method(rows)
    baseline = {row["seed"]: row for row in grouped["baseline"]}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=180)
    for ax, (metric, ylabel) in zip(axes, [("final_val_loss", "Validation loss improvement"), ("test_ppl", "Test PPL improvement")]):
        for method in ["mono_inc", "mono_dec"]:
            by_seed = {row["seed"]: row for row in grouped[method]}
            xs, ys = [], []
            for idx, seed in enumerate(SEEDS):
                base = baseline.get(seed, {}).get(metric)
                current = by_seed.get(seed, {}).get(metric)
                if base is None or current is None:
                    continue
                xs.append(idx + (-0.08 if method == "mono_inc" else 0.08))
                ys.append(base - current)
            ax.scatter(xs, ys, color=COLORS[method], marker=MARKERS[method], s=48, label=LABELS[method])
            if ys:
                ax.axhline(mean(ys), color=COLORS[method], linestyle=":", linewidth=1.2)
        ax.axhline(0.0, color="#333333", linestyle="--", linewidth=1.2)
        ax.set_xticks(range(len(SEEDS)))
        ax.set_xticklabels([str(seed) for seed in SEEDS])
        ax.set_xlabel("seed")
        ax.set_ylabel(f"{ylabel} (baseline - method)")
        ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    axes[0].legend(frameon=False)
    fig.suptitle("R14 per-seed improvement over baseline")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_validation_gap(rows: list[dict[str, Any]], out_path: Path) -> None:
    grouped = grouped_by_method(rows)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=180)
    has_lines = False
    for method in ["mono_inc", "mono_dec"]:
        seed_gaps: dict[int, list[tuple[int, float]]] = {}
        for seed in SEEDS:
            base_run = next((row for row in grouped["baseline"] if row["seed"] == seed), None)
            method_run = next((row for row in grouped[method] if row["seed"] == seed), None)
            if not base_run or not method_run:
                continue
            base_by_step = {step: value for step, value in base_run["eval_points"]}
            gaps = []
            for step, value in method_run["eval_points"]:
                if step in base_by_step:
                    gaps.append((step, value - base_by_step[step]))
            seed_gaps[seed] = gaps
        common_steps = sorted(set.intersection(*(set(step for step, _ in gaps) for gaps in seed_gaps.values()))) if seed_gaps else []
        mean_points = []
        for step in common_steps:
            values = []
            for gaps in seed_gaps.values():
                by_step = dict(gaps)
                values.append(by_step[step])
            mean_points.append((step, mean(values)))
        if mean_points:
            has_lines = True
            ax.plot(
                [step for step, _ in mean_points],
                [value for _, value in mean_points],
                color=COLORS[method],
                marker=MARKERS[method],
                markersize=3,
                linewidth=1.3,
                label=f"{LABELS[method]} - baseline",
            )
            final_step, final_gap = mean_points[-1]
            ax.annotate(f"{final_gap:+.4f}", xy=(final_step, final_gap), xytext=(5, 0), textcoords="offset points", color=COLORS[method])
    ax.axhline(0.0, color="#333333", linestyle="--", linewidth=1.2)
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("Validation loss gap to baseline (method - baseline)")
    ax.set_title("R14 validation gap to baseline")
    ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    if has_lines:
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_train_sanity(rows: list[dict[str, Any]], out_path: Path) -> None:
    grouped = grouped_by_method(rows)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=180)
    has_lines = False
    for method in METHOD_ORDER:
        all_points = []
        for row in grouped[method]:
            all_points.extend(row["train_points"])
        all_points.sort()
        smoothed = rolling_mean(all_points, window_steps=100)
        if smoothed:
            has_lines = True
            ax.plot([step for step, _ in smoothed], [value for _, value in smoothed], color=COLORS[method], linewidth=1.1, alpha=0.8, label=LABELS[method])
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("Training LM loss (rolling)")
    ax.set_title("R14 dense training loss sanity curve")
    ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    if has_lines:
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def decision_text(summary: dict[str, dict[str, Any]]) -> str:
    if any(summary[method].get("count", 0) < 3 for method in METHOD_ORDER):
        return "pending: R14 decision requires all 3 seeds for baseline, mono_inc, and mono_dec."
    base = summary["baseline"]
    inc = summary["mono_inc"]
    dec = summary["mono_dec"]
    inc_better = inc["final_val_loss_mean"] < base["final_val_loss_mean"] and inc["test_ppl_mean"] < base["test_ppl_mean"]
    dec_better_than_inc = dec["final_val_loss_mean"] < inc["final_val_loss_mean"] and dec["test_ppl_mean"] < inc["test_ppl_mean"]
    if inc_better and not dec_better_than_inc:
        return "mono_inc retains the matched-compute long-training signal and remains the supported static prior."
    if inc_better and dec_better_than_inc:
        return "static monotonic allocation remains useful, but direction is not settled because mono_dec exceeds mono_inc at 16384 steps."
    return "mono_inc does not retain a clear 16384-step advantage; keep the paper claim at the 8192-step empirical-signal level."


def write_markdown(summary: dict[str, dict[str, Any]], decision: str, out_path: Path) -> None:
    lines = [
        "# R14 Static Long-Training Summary",
        "",
        "| method | count | seeds | val@8192 | val@16384 | final val loss | test PPL | actual_k | KV access | tok/s | dVal final vs baseline | dTest PPL vs baseline |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHOD_ORDER:
        entry = summary[method]
        lines.append(
            "| {method} | {count} | {seeds} | {v8192} | {v16384} | {val} | {test} | {actual_k} | {kv} | {tok} | {dval} | {dtest} |".format(
                method=LABELS[method],
                count=entry["count"],
                seeds=",".join(str(seed) for seed in entry["seeds"]),
                v8192=fmt_stat(entry.get("val_at_8192_mean"), entry.get("val_at_8192_std")),
                v16384=fmt_stat(entry.get("val_at_16384_mean"), entry.get("val_at_16384_std")),
                val=fmt_stat(entry.get("final_val_loss_mean"), entry.get("final_val_loss_std")),
                test=fmt_stat(entry.get("test_ppl_mean"), entry.get("test_ppl_std"), digits=3),
                actual_k=fmt_stat(entry.get("actual_k_mean_mean"), entry.get("actual_k_mean_std"), digits=3),
                kv=fmt_stat(entry.get("estimated_kv_access_mean_mean"), entry.get("estimated_kv_access_mean_std"), digits=2),
                tok=fmt_stat(entry.get("tok_per_s_mean"), entry.get("tok_per_s_std"), digits=1),
                dval=fmt(entry.get("delta_final_val_loss_vs_baseline")),
                dtest=fmt(entry.get("delta_test_ppl_vs_baseline"), digits=3),
            )
        )
    lines.extend(["", f"Decision: {decision}", ""])
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROUND14_LOCAL_RESULTS_DIR)
    parser.add_argument("--out-dir", type=Path, default=ROUND14_LOCAL_RESULTS_DIR)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_runs(args.results_dir)
    summary = aggregate(rows)
    decision = decision_text(summary)
    payload = {"summary": summary, "decision": decision, "runs": rows}
    (args.out_dir / "round14_static_longtrain_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(summary, decision, args.out_dir / "round14_static_longtrain_summary.md")
    plot_final_metrics(rows, args.out_dir / "round14_final_metrics.png")
    plot_improvement_scatter(rows, args.out_dir / "round14_per_seed_improvement.png")
    plot_validation_gap(rows, args.out_dir / "round14_validation_gap_to_baseline.png")
    plot_train_sanity(rows, args.out_dir / "round14_dense_train_sanity.png")
    print(f"Wrote {args.out_dir / 'round14_static_longtrain_summary.md'}")


if __name__ == "__main__":
    main()
