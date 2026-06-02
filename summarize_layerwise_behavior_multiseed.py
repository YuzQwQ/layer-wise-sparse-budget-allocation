"""Summarize R18 multi-seed layer-wise sparse context access diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


METHODS = ["baseline", "mono_inc", "mono_dec"]
GROUPS = ["layers_1_6", "layers_7_12", "layers_13_18", "layers_19_24"]
METRICS = [
    "mean_selected_distance_blocks",
    "far_context_ratio",
    "selection_distance_entropy_norm",
    "selected_count_mean",
    "eligible_selected_count_mean",
]


def split_method_seed(name: str) -> tuple[str, int]:
    method, seed_text = name.rsplit("_s", 1)
    return method, int(seed_text)


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return mean, var**0.5


def load_runs(root: Path) -> list[dict]:
    runs = []
    for path in sorted((root / "analysis_formal").glob("*/layerwise_behavior_summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        name = path.parent.name
        method, seed = split_method_seed(name)
        group_summary = payload["methods"][name]["group_summary"]
        runs.append({"name": name, "method": method, "seed": seed, "group_summary": group_summary, "path": str(path)})
    return runs


def aggregate(runs: list[dict]) -> dict:
    out = {"num_runs": len(runs), "methods": {}, "runs": runs}
    for method in METHODS:
        subset = [run for run in runs if run["method"] == method]
        out["methods"][method] = {"seeds": sorted(run["seed"] for run in subset), "groups": {}}
        for group in GROUPS:
            out["methods"][method]["groups"][group] = {}
            for metric in METRICS:
                values = [float(run["group_summary"][group][metric]) for run in subset if run["group_summary"][group].get(metric) is not None]
                mean, std = mean_std(values)
                out["methods"][method]["groups"][group][metric] = {"mean": mean, "std": std}
    return out


def write_markdown(path: Path, agg: dict) -> None:
    lines = [
        "# R18 Layer-wise Sparse Context Access Multi-Seed Diagnostic",
        "",
        "Behavioral diagnostic only; not a causal mechanism proof.",
        "",
        "| Method | Group | Mean distance blocks | Far ratio | Norm. entropy | Selected count | Eligible selected count |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        for group in GROUPS:
            stats = agg["methods"][method]["groups"][group]
            lines.append(
                "| {method} | {group} | {dist:.3f} +/- {dist_s:.3f} | {far:.3f} +/- {far_s:.3f} | {ent:.3f} +/- {ent_s:.3f} | {sel:.3f} +/- {sel_s:.3f} | {elig:.3f} +/- {elig_s:.3f} |".format(
                    method=method,
                    group=group,
                    dist=stats["mean_selected_distance_blocks"]["mean"],
                    dist_s=stats["mean_selected_distance_blocks"]["std"],
                    far=stats["far_context_ratio"]["mean"],
                    far_s=stats["far_context_ratio"]["std"],
                    ent=stats["selection_distance_entropy_norm"]["mean"],
                    ent_s=stats["selection_distance_entropy_norm"]["std"],
                    sel=stats["selected_count_mean"]["mean"],
                    sel_s=stats["selected_count_mean"]["std"],
                    elig=stats["eligible_selected_count_mean"]["mean"],
                    elig_s=stats["eligible_selected_count_mean"]["std"],
                )
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_group_metric(root: Path, agg: dict, metric: str, title: str, ylabel: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    width = 0.24
    x_base = list(range(len(GROUPS)))
    colors = {"baseline": "#8da0cb", "mono_inc": "#66c2a5", "mono_dec": "#fc8d62"}
    for idx, method in enumerate(METHODS):
        offset = (idx - 1) * width
        means = [agg["methods"][method]["groups"][group][metric]["mean"] for group in GROUPS]
        stds = [agg["methods"][method]["groups"][group][metric]["std"] for group in GROUPS]
        ax.bar([x + offset for x in x_base], means, yerr=stds, width=width, capsize=3, label=method, color=colors[method], alpha=0.85)
    ax.set_xticks(x_base)
    ax.set_xticklabels(["1-6", "7-12", "13-18", "19-24"])
    ax.set_xlabel("Layer group")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(root / filename, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/round18-phase18-layerwise-behavior-multiseed-c4")
    args = parser.parse_args()
    root = Path(args.root)
    runs = load_runs(root)
    if len(runs) != 9:
        raise SystemExit(f"Expected 9 layerwise summaries, found {len(runs)} under {root / 'analysis_formal'}")
    agg = aggregate(runs)
    (root / "layerwise_behavior_multiseed_summary.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")
    write_markdown(root / "layerwise_behavior_multiseed_summary.md", agg)
    plot_group_metric(root, agg, "mean_selected_distance_blocks", "Mean selected block distance by layer group", "Distance (blocks)", "layerwise_multiseed_mean_distance.png")
    plot_group_metric(root, agg, "far_context_ratio", "Far-context selection ratio by layer group", "Ratio (distance >= 8 blocks)", "layerwise_multiseed_far_ratio.png")
    plot_group_metric(root, agg, "selection_distance_entropy_norm", "Selection distance entropy by layer group", "Normalized entropy", "layerwise_multiseed_entropy.png")
    print(f"Wrote multi-seed layer-wise summary to {root}")


if __name__ == "__main__":
    main()
