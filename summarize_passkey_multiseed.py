"""Summarize P1/R17 multi-seed forced-choice Passkey diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


METHOD_ORDER = ["baseline", "mono_inc", "mono_dec"]
METHOD_LABEL = {
    "baseline": "baseline",
    "mono_inc": "mono_inc",
    "mono_dec": "mono_dec",
}
CHANCE = 1.0 / 8.0


def infer_method(run_name: str) -> str:
    if "mono_inc" in run_name:
        return "mono_inc"
    if "mono_dec" in run_name:
        return "mono_dec"
    if "baseline" in run_name:
        return "baseline"
    raise ValueError(f"Cannot infer method from run_name={run_name!r}")


def infer_seed(run_name: str) -> int:
    marker = "_s"
    if marker not in run_name:
        raise ValueError(f"Cannot infer seed from run_name={run_name!r}")
    return int(run_name.rsplit(marker, 1)[1])


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return mean, var**0.5


def load_results(root: Path) -> list[dict]:
    rows = []
    for path in sorted((root / "passkey_formal").glob("*/passkey_summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        run_name = payload.get("run_name") or path.parent.name
        summary = payload["summary"]
        rows.append(
            {
                "run_name": run_name,
                "method": infer_method(run_name),
                "seed": infer_seed(run_name),
                "overall": float(summary["overall_accuracy"]),
                "by_length": {int(k): float(v["accuracy"]) for k, v in summary["by_length"].items()},
                "by_depth": {float(k): float(v["accuracy"]) for k, v in summary["by_depth"].items()},
                "num_samples": int(summary["num_samples"]),
                "path": str(path),
            }
        )
    return rows


def aggregate(rows: list[dict]) -> dict:
    result = {
        "chance_accuracy": CHANCE,
        "num_runs": len(rows),
        "methods": {},
    }
    for method in METHOD_ORDER:
        subset = [row for row in rows if row["method"] == method]
        overall_values = [row["overall"] for row in subset]
        overall_mean, overall_std = mean_std(overall_values)
        lengths = sorted({length for row in subset for length in row["by_length"]})
        depths = sorted({depth for row in subset for depth in row["by_depth"]})
        result["methods"][method] = {
            "seeds": sorted(row["seed"] for row in subset),
            "overall_mean": overall_mean,
            "overall_std": overall_std,
            "overall_by_seed": {str(row["seed"]): row["overall"] for row in sorted(subset, key=lambda x: x["seed"])},
            "by_length": {
                str(length): {
                    "mean": mean_std([row["by_length"][length] for row in subset if length in row["by_length"]])[0],
                    "std": mean_std([row["by_length"][length] for row in subset if length in row["by_length"]])[1],
                }
                for length in lengths
            },
            "by_depth": {
                str(depth): {
                    "mean": mean_std([row["by_depth"][depth] for row in subset if depth in row["by_depth"]])[0],
                    "std": mean_std([row["by_depth"][depth] for row in subset if depth in row["by_depth"]])[1],
                }
                for depth in depths
            },
        }
    return result


def write_markdown(path: Path, agg: dict) -> None:
    lines = [
        "# P1 Passkey Retrieval Multi-Seed Diagnostic",
        "",
        "Forced-choice synthetic Passkey Retrieval. Chance accuracy is 1/8 = 0.125.",
        "",
        "| Method | Seeds | Overall Acc mean +/- std | 4k | 8k | 16k |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in METHOD_ORDER:
        item = agg["methods"][method]
        by_len = item["by_length"]
        lines.append(
            "| {method} | {seeds} | {overall:.3f} +/- {std:.3f} | {l4:.3f} | {l8:.3f} | {l16:.3f} |".format(
                method=method,
                seeds=",".join(str(seed) for seed in item["seeds"]),
                overall=item["overall_mean"],
                std=item["overall_std"],
                l4=by_len.get("4096", {}).get("mean", float("nan")),
                l8=by_len.get("8192", {}).get("mean", float("nan")),
                l16=by_len.get("16384", {}).get("mean", float("nan")),
            )
        )
    lines.extend(
        [
            "",
            "Interpretation: this is a task-level directional diagnostic, not a definitive long-context benchmark.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_overall(root: Path, agg: dict) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    xs = list(range(len(METHOD_ORDER)))
    means = [agg["methods"][m]["overall_mean"] for m in METHOD_ORDER]
    stds = [agg["methods"][m]["overall_std"] for m in METHOD_ORDER]
    ax.bar(xs, means, yerr=stds, capsize=4, color=["#8da0cb", "#66c2a5", "#fc8d62"], alpha=0.85)
    for x, method in zip(xs, METHOD_ORDER):
        by_seed = agg["methods"][method]["overall_by_seed"]
        offsets = [-0.12, 0.0, 0.12]
        for offset, (_, value) in zip(offsets, sorted(by_seed.items(), key=lambda kv: int(kv[0]))):
            ax.scatter(x + offset, value, color="black", s=22, zorder=3)
    ax.axhline(CHANCE, color="black", linestyle="--", linewidth=1.2, label="chance = 0.125")
    ax.set_xticks(xs)
    ax.set_xticklabels([METHOD_LABEL[m] for m in METHOD_ORDER])
    ax.set_ylabel("Forced-choice accuracy")
    ax.set_title("Passkey Retrieval multi-seed diagnostic")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(root / "passkey_multiseed_overall_accuracy.png", dpi=180)
    plt.close(fig)


def plot_by_length(root: Path, agg: dict) -> None:
    lengths = [4096, 8192, 16384]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    width = 0.22
    x_base = list(range(len(lengths)))
    colors = {"baseline": "#8da0cb", "mono_inc": "#66c2a5", "mono_dec": "#fc8d62"}
    for idx, method in enumerate(METHOD_ORDER):
        offset = (idx - 1) * width
        values = [agg["methods"][method]["by_length"][str(length)]["mean"] for length in lengths]
        ax.bar([x + offset for x in x_base], values, width=width, label=method, color=colors[method], alpha=0.85)
    ax.axhline(CHANCE, color="black", linestyle="--", linewidth=1.2)
    ax.set_xticks(x_base)
    ax.set_xticklabels(["4k", "8k", "16k"])
    ax.set_ylabel("Forced-choice accuracy")
    ax.set_title("Passkey accuracy by context length")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(root / "passkey_multiseed_accuracy_by_length.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/round17-phase17-passkey-multiseed-diagnostic")
    args = parser.parse_args()
    root = Path(args.root)
    rows = load_results(root)
    if len(rows) != 9:
        raise SystemExit(f"Expected 9 passkey_summary.json files, found {len(rows)} under {root / 'passkey_formal'}")
    agg = aggregate(rows)
    (root / "passkey_multiseed_summary.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")
    write_markdown(root / "passkey_multiseed_summary.md", agg)
    plot_overall(root, agg)
    plot_by_length(root, agg)
    print(f"Wrote summary to {root}")


if __name__ == "__main__":
    main()
