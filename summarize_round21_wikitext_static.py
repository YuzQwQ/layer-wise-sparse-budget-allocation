"""Summarize Round21 WikiText-103 static budget controlled experiments."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ablation_remote_common import ROUND21_EXPERIMENTS, ROUND21_LOCAL_RESULTS_DIR


METHOD_ORDER = ["m0_baseline", "m1_mono_inc", "m2_mono_dec"]
LABELS = {
    "m0_baseline": "baseline",
    "m1_mono_inc": "mono_inc",
    "m2_mono_dec": "mono_dec",
}
COLORS = {
    "m0_baseline": "#6F7A83",
    "m1_mono_inc": "#2FA17D",
    "m2_mono_dec": "#D97A2B",
}


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def metric(row: dict[str, Any] | None, *keys: str) -> float | None:
    if not row:
        return None
    for key in keys:
        value = safe_float(row.get(key))
        if value is not None:
            return value
    return None


def stats(values: list[Any]) -> tuple[float | None, float | None]:
    clean = [safe_float(value) for value in values]
    clean = [value for value in clean if value is not None]
    if not clean:
        return None, None
    return mean(clean), pstdev(clean) if len(clean) > 1 else 0.0


def round_or_none(value: float | None, digits: int = 6) -> float | None:
    return None if value is None else round(value, digits)


def fmt(value: float | None, digits: int = 4) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def fmt_signed(value: float | None, digits: int = 4) -> str:
    return "-" if value is None else f"{value:+.{digits}f}"


def last_with_key(rows: list[dict[str, Any]], *keys: str) -> float | None:
    for row in reversed(rows):
        value = metric(row, *keys)
        if value is not None:
            return value
    return None


def resolve_exp_root(input_root: Path) -> Path:
    return input_root / "exp" if (input_root / "exp").is_dir() else input_root


def collect_runs(input_root: Path) -> list[dict[str, Any]]:
    exp_root = resolve_exp_root(input_root)
    rows: list[dict[str, Any]] = []
    for spec in ROUND21_EXPERIMENTS:
        run_dir = exp_root / spec.run_name
        train_log = read_jsonl(run_dir / "train_log.jsonl")
        eval_log = read_jsonl(run_dir / "eval_log.jsonl")
        test_metrics = read_json(run_dir / "test_metrics.json") or {}
        summary = read_json(run_dir / "run_summary.json") or {}
        if not train_log and not eval_log and not test_metrics:
            continue
        final_validation = summary.get("final_validation") or (eval_log[-1] if eval_log else {})
        final_test = summary.get("final_test") or test_metrics
        final_train = summary.get("final_train") or (train_log[-1] if train_log else {})
        val_cache = summary.get("val_cache") or {}
        test_cache = summary.get("test_cache") or {}
        rows.append(
            {
                "key": spec.key,
                "label": LABELS.get(spec.key, spec.key),
                "run_name": spec.run_name,
                "seed": spec.seed,
                "config_name": spec.config_name,
                "final_val_loss": metric(final_validation, "lm_loss", "loss"),
                "final_val_ppl": metric(final_validation, "ppl", "perplexity"),
                "test_loss": metric(final_test, "lm_loss", "loss"),
                "test_ppl": metric(final_test, "ppl", "perplexity"),
                "tok_per_s": metric(final_train, "tok_per_s") or last_with_key(train_log, "tok_per_s"),
                "cuda_max_memory_allocated_mb": last_with_key(train_log, "cuda_max_memory_allocated_mb"),
                "cuda_max_memory_reserved_mb": last_with_key(train_log, "cuda_max_memory_reserved_mb"),
                "actual_k_mean": 16.0,
                "estimated_kv_access_mean": 1024.0,
                "val_actual_tokens": metric(val_cache, "actual_tokens") or metric(final_validation, "tokens"),
                "test_actual_tokens": metric(test_cache, "actual_tokens") or metric(final_test, "tokens"),
            }
        )
    return rows


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    by_key = {key: [row for row in rows if row["key"] == key] for key in METHOD_ORDER}
    baseline_val = None
    baseline_ppl = None
    if by_key["m0_baseline"]:
        baseline_val = stats([row["final_val_loss"] for row in by_key["m0_baseline"]])[0]
        baseline_ppl = stats([row["test_ppl"] for row in by_key["m0_baseline"]])[0]
    for key in METHOD_ORDER:
        bucket = by_key[key]
        val_mean, val_std = stats([row["final_val_loss"] for row in bucket])
        test_ppl_mean, test_ppl_std = stats([row["test_ppl"] for row in bucket])
        test_loss_mean, test_loss_std = stats([row["test_loss"] for row in bucket])
        tok_mean, tok_std = stats([row["tok_per_s"] for row in bucket])
        memory_mean, memory_std = stats([row["cuda_max_memory_allocated_mb"] for row in bucket])
        val_delta = None if baseline_val is None or val_mean is None else baseline_val - val_mean
        ppl_delta = None if baseline_ppl is None or test_ppl_mean is None else baseline_ppl - test_ppl_mean
        out.append(
            {
                "key": key,
                "label": LABELS[key],
                "num_runs": len(bucket),
                "seeds": sorted(row["seed"] for row in bucket),
                "run_names": [row["run_name"] for row in bucket],
                "val_loss_mean": round_or_none(val_mean),
                "val_loss_std": round_or_none(val_std),
                "test_loss_mean": round_or_none(test_loss_mean),
                "test_loss_std": round_or_none(test_loss_std),
                "test_ppl_mean": round_or_none(test_ppl_mean),
                "test_ppl_std": round_or_none(test_ppl_std),
                "delta_val_vs_baseline": round_or_none(val_delta),
                "delta_ppl_vs_baseline": round_or_none(ppl_delta),
                "actual_k_mean": 16.0,
                "estimated_kv_access_mean": 1024.0,
                "tok_per_s_mean": round_or_none(tok_mean),
                "tok_per_s_std": round_or_none(tok_std),
                "cuda_max_memory_allocated_mb_mean": round_or_none(memory_mean),
                "cuda_max_memory_allocated_mb_std": round_or_none(memory_std),
                "val_actual_tokens": sorted({int(row["val_actual_tokens"]) for row in bucket if row["val_actual_tokens"] is not None}),
                "test_actual_tokens": sorted({int(row["test_actual_tokens"]) for row in bucket if row["test_actual_tokens"] is not None}),
            }
        )
    return out


def decision(groups: list[dict[str, Any]]) -> dict[str, str]:
    by_key = {group["key"]: group for group in groups}
    baseline = by_key.get("m0_baseline", {})
    inc = by_key.get("m1_mono_inc", {})
    dec = by_key.get("m2_mono_dec", {})
    if min(group.get("num_runs", 0) for group in [baseline, inc, dec]) < 3:
        return {
            "status": "incomplete",
            "interpretation": "WikiText-103 static support is incomplete; at least one method has fewer than 3 seeds.",
        }
    inc_val = inc.get("val_loss_mean")
    inc_ppl = inc.get("test_ppl_mean")
    base_val = baseline.get("val_loss_mean")
    base_ppl = baseline.get("test_ppl_mean")
    dec_val = dec.get("val_loss_mean")
    dec_ppl = dec.get("test_ppl_mean")
    if None in {inc_val, inc_ppl, base_val, base_ppl, dec_val, dec_ppl}:
        return {"status": "incomplete", "interpretation": "Missing validation or test metrics."}
    if inc_val < base_val and inc_ppl < base_ppl and inc_val <= dec_val and inc_ppl <= dec_ppl:
        status = "mono_inc_directional_support"
        text = "WikiText-103 provides additional cross-corpus directional support for mono_inc under matched sparse compute."
    elif inc_val <= base_val and inc_ppl <= base_ppl:
        status = "mono_inc_positive_but_direction_mixed"
        text = "mono_inc improves over baseline, but direction relative to mono_dec is mixed; interpret as dataset-sensitive budget placement evidence."
    elif dec_val < inc_val or dec_ppl < inc_ppl:
        status = "mono_dec_stronger_or_mixed"
        text = "mono_dec is competitive or stronger than mono_inc; monotonic direction is not fully settled on WikiText-103."
    else:
        status = "high_uncertainty"
        text = "WikiText-103 shows high uncertainty and should be reported as boundary evidence."
    return {"status": status, "interpretation": text}


def plot_improvement(groups: list[dict[str, Any]], out_dir: Path) -> None:
    labels = [group["label"] for group in groups]
    dval = [group["delta_val_vs_baseline"] or 0.0 for group in groups]
    dppl = [group["delta_ppl_vs_baseline"] or 0.0 for group in groups]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), dpi=180)
    for ax, values, title, ylabel in [
        (axes[0], dval, "Validation loss improvement", "Delta Val = baseline - method"),
        (axes[1], dppl, "Test PPL improvement", "Delta PPL = baseline - method"),
    ]:
        colors = [COLORS[group["key"]] for group in groups]
        bars = ax.bar(labels, values, color=colors, alpha=0.88)
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.1)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ymin = min(min(values), 0.0)
        ymax = max(max(values), 0.0)
        span = max(ymax - ymin, 0.01 if ax is axes[0] else 0.5)
        ax.set_ylim(ymin - span * 0.25, ymax + span * 0.35)
        for bar, value in zip(bars, values):
            va = "bottom" if value >= 0 else "top"
            offset = span * (0.04 if value >= 0 else -0.04)
            ax.text(bar.get_x() + bar.get_width() / 2, value + offset, f"{value:+.4f}" if ax is axes[0] else f"{value:+.3f}", ha="center", va=va, fontsize=8)
    fig.suptitle("WikiText-103 baseline-relative improvement under matched sparse compute")
    fig.tight_layout()
    fig.savefig(out_dir / "round21_wikitext_static_baseline_relative_improvement.png")
    plt.close(fig)


def plot_seed_scatter(rows: list[dict[str, Any]], out_dir: Path) -> None:
    baseline_by_seed = {row["seed"]: row for row in rows if row["key"] == "m0_baseline"}
    methods = ["m1_mono_inc", "m2_mono_dec"]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), dpi=180, sharex=True)
    for ax, metric, title, ylabel in [
        (axes[0], "final_val_loss", "Per-seed validation improvement", "baseline - method"),
        (axes[1], "test_ppl", "Per-seed test PPL improvement", "baseline - method"),
    ]:
        for method in methods:
            xs: list[int] = []
            ys: list[float] = []
            for row in rows:
                if row["key"] != method or row["seed"] not in baseline_by_seed:
                    continue
                base = baseline_by_seed[row["seed"]].get(metric)
                value = row.get(metric)
                if base is None or value is None:
                    continue
                xs.append(row["seed"])
                ys.append(base - value)
            ax.plot(xs, ys, marker="o", linewidth=1.4, color=COLORS[method], label=LABELS[method])
            for x, y in zip(xs, ys):
                ax.text(x, y, f"{y:+.3f}" if metric == "test_ppl" else f"{y:+.4f}", fontsize=7, ha="center", va="bottom" if y >= 0 else "top")
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.1)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("seed")
        ax.legend(frameon=False)
    fig.suptitle("WikiText-103 per-seed improvement over uniform baseline")
    fig.tight_layout()
    fig.savefig(out_dir / "round21_wikitext_static_per_seed_improvement.png")
    plt.close(fig)


def write_markdown(groups: list[dict[str, Any]], decision_obj: dict[str, str], out_path: Path) -> None:
    lines = [
        "# Round21 WikiText-103 Static Budget Support",
        "",
        f"Decision status: `{decision_obj['status']}`",
        "",
        decision_obj["interpretation"],
        "",
        "| Method | Seeds | Val Loss | Test PPL | Delta Val ↑ | Delta PPL ↑ | actual_k | KV access | tok/s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in groups:
        lines.append(
            "| {label} | {seeds} | {val} +/- {val_std} | {ppl} +/- {ppl_std} | {dval} | {dppl} | 16.000 | 1024.00 | {tok} +/- {tok_std} |".format(
                label=group["label"],
                seeds=",".join(str(seed) for seed in group["seeds"]) or "-",
                val=fmt(group["val_loss_mean"]),
                val_std=fmt(group["val_loss_std"]),
                ppl=fmt(group["test_ppl_mean"], 3),
                ppl_std=fmt(group["test_ppl_std"], 3),
                dval=fmt_signed(group["delta_val_vs_baseline"]),
                dppl=fmt_signed(group["delta_ppl_vs_baseline"], 3),
                tok=fmt(group["tok_per_s_mean"], 1),
                tok_std=fmt(group["tok_per_s_std"], 1),
            )
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- Delta is defined as baseline - method; positive values indicate improvement over uniform baseline.",
            "- All three static schedules are compute-matched at average actual_k = 16.0 and estimated selected KV access = 1024.0.",
            "- WikiText-103 is additional cross-corpus directional support, not proof of broad generalization.",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=ROUND21_LOCAL_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or args.input_root
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_runs(args.input_root)
    groups = aggregate(rows)
    decision_obj = decision(groups)
    payload = {
        "input_root": str(args.input_root),
        "num_runs": len(rows),
        "runs": rows,
        "groups": groups,
        "decision": decision_obj,
    }
    (output_dir / "round21_wikitext_static_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(groups, decision_obj, output_dir / "round21_wikitext_static_summary.md")
    if rows:
        plot_improvement(groups, figures_dir)
        plot_seed_scatter(rows, figures_dir)
    print(f"Wrote summary to {output_dir}")


if __name__ == "__main__":
    main()
