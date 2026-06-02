"""Summarize R12 mono_inc reliability and static-budget evidence."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import matplotlib.pyplot as plt

from ablation_remote_common import (
    ROUND10_EXPERIMENTS,
    ROUND10_LOCAL_RESULTS_DIR,
    ROUND11_EXPERIMENTS,
    ROUND11_LOCAL_RESULTS_DIR,
    ROUND12_BASE_EXPERIMENTS,
    ROUND12_LOCAL_RESULTS_DIR,
)


METHOD_ORDER = ["baseline", "mono_inc", "mono_dec", "u_shape"]
LABELS = {
    "baseline": "baseline",
    "mono_inc": "mono_inc",
    "mono_dec": "mono_dec",
    "u_shape": "u_shape",
}
COLORS = {
    "baseline": "#4C78A8",
    "mono_inc": "#54A24B",
    "mono_dec": "#E45756",
    "u_shape": "#F58518",
}
MARKERS = {
    "baseline": "o",
    "mono_inc": "s",
    "mono_dec": "^",
    "u_shape": "D",
}


def round_float(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    if not math.isfinite(float(value)):
        return None
    return round(float(value), digits)


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


def stats(values: list[float | None]) -> tuple[float | None, float | None]:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None, None
    return mean(clean), pstdev(clean) if len(clean) > 1 else 0.0


def fmt(value: float | None, digits: int = 4) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def fmt_stat(avg: float | None, std: float | None, digits: int = 4) -> str:
    return "-" if avg is None else f"{avg:.{digits}f} +/- {0.0 if std is None else std:.{digits}f}"


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


def last_with_key(rows: list[dict[str, Any]], *keys: str) -> float | None:
    for row in reversed(rows):
        value = metric_value(row, *keys)
        if value is not None:
            return value
    return None


def rolling_mean(points: list[tuple[int, float]], window_steps: int = 100) -> list[tuple[int, float]]:
    out: list[tuple[int, float]] = []
    active: list[tuple[int, float]] = []
    for step, value in points:
        active.append((step, value))
        active = [(s, v) for s, v in active if step - s <= window_steps]
        out.append((step, sum(v for _, v in active) / len(active)))
    return out


def extract_run(*, method: str, run_name: str, seed: int, run_dir: Path, source: str) -> dict[str, Any] | None:
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
    for row in train_log:
        loss = metric_value(row, "train_lm_loss", "lm_loss", "loss")
        if row.get("step") is not None and loss is not None:
            train_points.append((int(row["step"]), loss))
    eval_points = []
    for row in eval_log:
        loss = metric_value(row, "lm_loss", "loss")
        if row.get("step") is not None and loss is not None:
            eval_points.append((int(row["step"]), loss))

    return {
        "run_name": run_name,
        "source": source,
        "method": method,
        "seed": seed,
        "train_points": train_points,
        "eval_points": eval_points,
        "final_val_loss": metric_value(last_eval, "lm_loss", "loss"),
        "final_val_ppl": metric_value(last_eval, "ppl", "perplexity"),
        "val_at_4096": value_at_step(eval_log, 4096, "lm_loss", "loss"),
        "val_at_8192": value_at_step(eval_log, 8192, "lm_loss", "loss"),
        "test_loss": metric_value(test_metrics, "lm_loss", "loss"),
        "test_ppl": metric_value(test_metrics, "ppl", "perplexity"),
        "actual_k_mean": metric_value(last_budget, "actual_k_mean", "budget_mean"),
        "actual_k_p50": metric_value(last_budget, "actual_k_p50", "budget_p50"),
        "actual_k_p90": metric_value(last_budget, "actual_k_p90", "budget_p90"),
        "estimated_kv_access_mean": metric_value(last_budget, "estimated_kv_access_mean"),
        "estimated_sparse_attention_budget_ratio": metric_value(last_budget, "estimated_sparse_attention_budget_ratio"),
        "budget_var": metric_value(last_budget, "actual_k_var", "budget_var"),
        "tok_per_s": last_with_key(train_log, "tok_per_s"),
        "cuda_max_memory_allocated_mb": last_with_key(train_log, "cuda_max_memory_allocated_mb"),
        "cuda_max_memory_reserved_mb": last_with_key(train_log, "cuda_max_memory_reserved_mb"),
    }


def method_from_key(key: str) -> str:
    if key == "m0_baseline":
        return "baseline"
    if key == "m1_mono_inc":
        return "mono_inc"
    if key == "m2_mono_dec":
        return "mono_dec"
    if key == "m3_u_shape":
        return "u_shape"
    raise ValueError(f"Unsupported R12 key: {key}")


def load_runs(r10_dir: Path, r11_dir: Path, r12_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    r10_exp = r10_dir / "exp"
    r11_exp = r11_dir / "exp"
    r12_exp = r12_dir / "exp"

    for spec in ROUND11_EXPERIMENTS:
        row = extract_run(
            method="baseline",
            run_name=spec.run_name,
            seed=spec.seed,
            run_dir=r11_exp / spec.run_name,
            source="final_c4_baseline_reference",
        )
        if row is not None:
            rows.append(row)

    for spec in ROUND10_EXPERIMENTS:
        if spec.key != "m1_mono_inc":
            continue
        row = extract_run(
            method="mono_inc",
            run_name=spec.run_name,
            seed=spec.seed,
            run_dir=r10_exp / spec.run_name,
            source="longtrain_mono_inc_reference",
        )
        if row is not None:
            rows.append(row)

    for spec in ROUND12_BASE_EXPERIMENTS:
        method = method_from_key(spec.key)
        row = extract_run(
            method=method,
            run_name=spec.run_name,
            seed=spec.seed,
            run_dir=r12_exp / spec.run_name,
            source="round12_new_run",
        )
        if row is not None:
            rows.append(row)
    return rows


def aggregate(runs: list[dict[str, Any]], methods: list[str] = METHOD_ORDER) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[run["method"]].append(run)

    metrics = [
        "final_val_loss",
        "final_val_ppl",
        "val_at_4096",
        "val_at_8192",
        "test_loss",
        "test_ppl",
        "actual_k_mean",
        "actual_k_p50",
        "actual_k_p90",
        "estimated_kv_access_mean",
        "estimated_sparse_attention_budget_ratio",
        "budget_var",
        "tok_per_s",
        "cuda_max_memory_allocated_mb",
        "cuda_max_memory_reserved_mb",
    ]
    out: dict[str, dict[str, Any]] = {}
    for method in methods:
        items = grouped.get(method, [])
        entry: dict[str, Any] = {
            "label": LABELS[method],
            "count": len(items),
            "seeds": sorted(run["seed"] for run in items),
        }
        for metric in metrics:
            avg, std = stats([run.get(metric) for run in items])
            entry[f"{metric}_mean"] = round_float(avg)
            entry[f"{metric}_std"] = round_float(std)
        out[method] = entry

    baseline = out.get("baseline", {})
    for method, entry in out.items():
        if method == "baseline":
            continue
        for metric in ["final_val_loss", "test_ppl", "actual_k_mean", "estimated_kv_access_mean", "tok_per_s"]:
            base_value = baseline.get(f"{metric}_mean")
            method_value = entry.get(f"{metric}_mean")
            entry[f"delta_{metric}_vs_baseline"] = (
                round_float(method_value - base_value) if method_value is not None and base_value is not None else None
            )
    return out


def subset_runs(runs: list[dict[str, Any]], methods: set[str], seeds: set[int] | None = None) -> list[dict[str, Any]]:
    return [
        run
        for run in runs
        if run["method"] in methods and (seeds is None or run["seed"] in seeds)
    ]


def plot_final_val_scatter(runs: list[dict[str, Any]], out_path: Path) -> None:
    methods = ["baseline", "mono_inc"]
    x_pos = {method: i for i, method in enumerate(methods)}
    plt.figure(figsize=(6.6, 4.2))
    seed_to_runs: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for run in runs:
        seed_to_runs[run["seed"]][run["method"]] = run

    for seed, pair in sorted(seed_to_runs.items()):
        if not all(method in pair and pair[method].get("final_val_loss") is not None for method in methods):
            continue
        xs = [x_pos[method] for method in methods]
        ys = [pair[method]["final_val_loss"] for method in methods]
        plt.plot(xs, ys, color="#B9B9B9", linewidth=0.8, alpha=0.7)
        for method, x, y in zip(methods, xs, ys):
            plt.scatter(x, y, color=COLORS[method], marker=MARKERS[method], s=42, zorder=3)
        plt.text(x_pos["mono_inc"] + 0.05, ys[-1], str(seed), fontsize=7, va="center", color="#555555")

    plt.xticks([x_pos[method] for method in methods], [LABELS[method] for method in methods])
    plt.ylabel("final validation LM loss")
    plt.title("Five-seed final validation comparison")
    plt.grid(axis="y", alpha=0.22)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_static_bars(summary: dict[str, dict[str, Any]], out_path: Path, metric: str, ylabel: str, title: str) -> None:
    methods = METHOD_ORDER
    means = [summary[method].get(f"{metric}_mean") for method in methods]
    stds = [summary[method].get(f"{metric}_std") or 0.0 for method in methods]
    plt.figure(figsize=(7.2, 4.2))
    xs = list(range(len(methods)))
    plt.bar(xs, [value if value is not None else 0.0 for value in means], yerr=stds, color=[COLORS[m] for m in methods], alpha=0.85, capsize=4)
    for x, value in zip(xs, means):
        if value is not None:
            plt.text(x, value, f"{value:.4f}", ha="center", va="bottom", fontsize=8)
    plt.xticks(xs, [LABELS[method] for method in methods])
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", alpha=0.22)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_dense_train_curves(runs: list[dict[str, Any]], out_path: Path) -> None:
    plt.figure(figsize=(8.0, 4.6))
    for method in METHOD_ORDER:
        method_runs = [run for run in runs if run["method"] == method and run.get("train_points")]
        if not method_runs:
            continue
        step_values: dict[int, list[float]] = defaultdict(list)
        for run in method_runs:
            for step, value in rolling_mean(run["train_points"], window_steps=100):
                step_values[step].append(value)
        points = [(step, mean(values)) for step, values in sorted(step_values.items()) if values]
        if not points:
            continue
        xs, ys = zip(*points)
        plt.plot(xs, ys, label=LABELS[method], color=COLORS[method], linewidth=1.2, alpha=0.9)
    plt.xlabel("optimizer step")
    plt.ylabel("train LM loss, rolling 100-step")
    plt.title("Dense training LM loss curves")
    plt.grid(alpha=0.22)
    plt.legend(frameon=False, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_validation_curves(runs: list[dict[str, Any]], out_path: Path) -> None:
    plt.figure(figsize=(8.0, 4.6))
    for method in METHOD_ORDER:
        method_runs = [run for run in runs if run["method"] == method and run.get("eval_points")]
        if not method_runs:
            continue
        common_steps = set(step for step, _ in method_runs[0]["eval_points"])
        for run in method_runs[1:]:
            common_steps &= set(step for step, _ in run["eval_points"])
        if not common_steps:
            continue
        step_to_means = []
        for step in sorted(common_steps):
            values = []
            for run in method_runs:
                value_map = dict(run["eval_points"])
                if step in value_map:
                    values.append(value_map[step])
            if values:
                step_to_means.append((step, mean(values)))
        if not step_to_means:
            continue
        xs, ys = zip(*step_to_means)
        plt.plot(xs, ys, label=LABELS[method], color=COLORS[method], marker=MARKERS[method], markersize=3.5, linewidth=1.1, alpha=0.9)
    plt.xlabel("optimizer step")
    plt.ylabel("validation LM loss")
    plt.title("Validation anchors on common steps")
    plt.grid(alpha=0.22)
    plt.legend(frameon=False, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def write_markdown(
    path: Path,
    *,
    five_seed_summary: dict[str, dict[str, Any]],
    static_summary: dict[str, dict[str, Any]],
    missing_runs: list[str],
) -> None:
    lines = [
        "# R12 Static Reliability Summary",
        "",
        "## Five-seed baseline vs mono_inc",
        "",
        "| method | count | seeds | final val loss | test ppl | actual_k | KV access | tok/s |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for method in ["baseline", "mono_inc"]:
        entry = five_seed_summary[method]
        lines.append(
            "| {method} | {count} | {seeds} | {val} | {test} | {k} | {kv} | {tok} |".format(
                method=LABELS[method],
                count=entry["count"],
                seeds=",".join(str(seed) for seed in entry["seeds"]),
                val=fmt_stat(entry.get("final_val_loss_mean"), entry.get("final_val_loss_std")),
                test=fmt_stat(entry.get("test_ppl_mean"), entry.get("test_ppl_std"), digits=3),
                k=fmt_stat(entry.get("actual_k_mean_mean"), entry.get("actual_k_mean_std"), digits=3),
                kv=fmt_stat(entry.get("estimated_kv_access_mean_mean"), entry.get("estimated_kv_access_mean_std"), digits=2),
                tok=fmt_stat(entry.get("tok_per_s_mean"), entry.get("tok_per_s_std"), digits=1),
            )
        )
    mono = five_seed_summary["mono_inc"]
    lines.extend(
        [
            "",
            "Deltas are mono_inc minus baseline; negative loss/PPL deltas are improvements.",
            "",
            f"- final validation loss improves by {fmt(abs(mono.get('delta_final_val_loss_vs_baseline')) if mono.get('delta_final_val_loss_vs_baseline') is not None and mono.get('delta_final_val_loss_vs_baseline') < 0 else mono.get('delta_final_val_loss_vs_baseline'))} if the delta is negative.",
            f"- final test PPL delta: {fmt(mono.get('delta_test_ppl_vs_baseline'), digits=3)}.",
            f"- estimated KV access delta: {fmt(mono.get('delta_estimated_kv_access_mean_vs_baseline'), digits=2)}.",
            "",
            "## Static schedule comparison",
            "",
            "| method | count | seeds | final val loss | test ppl | actual_k | KV access | tok/s |",
            "|---|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for method in METHOD_ORDER:
        entry = static_summary[method]
        lines.append(
            "| {method} | {count} | {seeds} | {val} | {test} | {k} | {kv} | {tok} |".format(
                method=LABELS[method],
                count=entry["count"],
                seeds=",".join(str(seed) for seed in entry["seeds"]),
                val=fmt_stat(entry.get("final_val_loss_mean"), entry.get("final_val_loss_std")),
                test=fmt_stat(entry.get("test_ppl_mean"), entry.get("test_ppl_std"), digits=3),
                k=fmt_stat(entry.get("actual_k_mean_mean"), entry.get("actual_k_mean_std"), digits=3),
                kv=fmt_stat(entry.get("estimated_kv_access_mean_mean"), entry.get("estimated_kv_access_mean_std"), digits=2),
                tok=fmt_stat(entry.get("tok_per_s_mean"), entry.get("tok_per_s_std"), digits=1),
            )
        )
    if missing_runs:
        lines.extend(["", "## Missing runs", ""])
        lines.extend(f"- {name}" for name in missing_runs)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def expected_run_names() -> list[str]:
    names = [spec.run_name for spec in ROUND12_BASE_EXPERIMENTS]
    return names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r10-dir", type=Path, default=ROUND10_LOCAL_RESULTS_DIR)
    parser.add_argument("--r11-dir", type=Path, default=ROUND11_LOCAL_RESULTS_DIR)
    parser.add_argument("--r12-dir", type=Path, default=ROUND12_LOCAL_RESULTS_DIR)
    parser.add_argument("--out-dir", type=Path, default=ROUND12_LOCAL_RESULTS_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = args.out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    runs = load_runs(args.r10_dir, args.r11_dir, args.r12_dir)
    new_run_names = {run["run_name"] for run in runs if run["source"] == "round12_new_run"}
    missing_runs = [name for name in expected_run_names() if name not in new_run_names]

    five_seed_runs = subset_runs(runs, {"baseline", "mono_inc"}, {42, 123, 3407, 2024, 2025})
    static_runs = subset_runs(runs, set(METHOD_ORDER), {42, 123, 3407})
    five_seed_summary = aggregate(five_seed_runs, ["baseline", "mono_inc"])
    static_summary = aggregate(static_runs, METHOD_ORDER)

    summary = {
        "five_seed_baseline_vs_mono_inc": five_seed_summary,
        "static_schedule_comparison": static_summary,
        "runs": [
            {
                key: value
                for key, value in run.items()
                if key not in {"train_points", "eval_points"}
            }
            for run in runs
        ],
        "missing_r12_runs": missing_runs,
        "notes": [
            "R12A extends baseline and mono_inc to seeds 2024/2025 and reuses same-setting references for seeds 42/123/3407.",
            "R12B adds mono_dec and u_shape for seeds 42/123/3407 to complete static schedule comparison.",
            "Validation curves use common logged steps only; no interpolation is applied.",
        ],
    }
    (args.out_dir / "round12_static_reliability_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_markdown(
        args.out_dir / "round12_static_reliability_summary.md",
        five_seed_summary=five_seed_summary,
        static_summary=static_summary,
        missing_runs=missing_runs,
    )

    if five_seed_runs:
        plot_final_val_scatter(five_seed_runs, fig_dir / "round12_five_seed_final_val_scatter.png")
    if static_runs:
        plot_static_bars(static_summary, fig_dir / "round12_static_final_val_loss.png", "final_val_loss", "final validation LM loss", "Static schedule final validation loss")
        plot_static_bars(static_summary, fig_dir / "round12_static_test_ppl.png", "test_ppl", "final test PPL", "Static schedule final test PPL")
        plot_dense_train_curves(static_runs, fig_dir / "round12_dense_train_lm_loss.png")
        plot_validation_curves(static_runs, fig_dir / "round12_validation_anchor_curves.png")

    print(f"Wrote {args.out_dir / 'round12_static_reliability_summary.json'}")
    print(f"Wrote {args.out_dir / 'round12_static_reliability_summary.md'}")
    if missing_runs:
        print(f"Missing R12 runs: {', '.join(missing_runs)}")


if __name__ == "__main__":
    main()
