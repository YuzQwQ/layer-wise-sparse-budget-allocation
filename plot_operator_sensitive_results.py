"""Plot R21 WikiText and R22+ cross-operator diagnostic results.

The figures use the paper-wide delta convention:
    Delta = uniform - method
so positive values indicate improvement over the uniform baseline within the
same platform/dataset/budget setting.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output" / "figures" / "operator_sensitive_results"

R21 = ROOT / "results" / "round21-phase21-wikitext-static-support" / "round21_wikitext_static_summary.json"
R22 = ROOT / "results" / "round22-phase22-h2o-kv-retention-diagnostic" / "round22_h2o_kv_retention_summary.json"
R25_WIKI = ROOT / "results" / "round25-phase25-sliding-window-wikitext-diagnostic" / "round25_sliding_window_wikitext_summary.json"
R26 = ROOT / "results" / "round26-phase26-h2o-window-robustness-diagnostic" / "round26_h2o_window_robustness_summary.json"
R27_AUDIT = ROOT / "results" / "round27-phase27-sliding-window-metric-audit" / "round27_sliding_window_metric_audit_summary.json"


COLORS = {
    "uniform": "#5B6472",
    "baseline": "#5B6472",
    "mono_inc": "#2F6F9F",
    "mono_dec": "#B66A3C",
}


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def method_label(method: str) -> str:
    return {"baseline": "uniform", "uniform": "uniform", "mono_inc": "mono_inc", "mono_dec": "mono_dec"}[method]


def nice_axes(ax):
    ax.grid(axis="y", color="#E6E8EB", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.75)


def annotate_bars(ax, bars, fmt="{:+.3f}", pad_frac=0.025):
    y0, y1 = ax.get_ylim()
    pad = (y1 - y0) * pad_frac
    for b in bars:
        h = b.get_height()
        x = b.get_x() + b.get_width() / 2
        va = "bottom" if h >= 0 else "top"
        y = h + pad if h >= 0 else h - pad
        ax.text(x, y, fmt.format(h), ha="center", va=va, fontsize=8)


def set_symmetric_ylim(ax, values, margin=0.18, floor=0.01):
    vmax = max([abs(v) for v in values] + [floor])
    ax.set_ylim(-vmax * (1 + margin), vmax * (1 + margin))


def plot_nsa_wikitext():
    data = load(R21)
    groups = {g["label"]: g for g in data["groups"]}
    methods = ["baseline", "mono_inc", "mono_dec"]
    labels = [method_label(m) for m in methods]
    val = [0.0, groups["mono_inc"]["delta_val_vs_baseline"], groups["mono_dec"]["delta_val_vs_baseline"]]
    ppl = [0.0, groups["mono_inc"]["delta_ppl_vs_baseline"], groups["mono_dec"]["delta_ppl_vs_baseline"]]

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 7.4))
    for ax, vals, title, ylabel, fmt in [
        (axes[0], val, "Validation loss improvement", "Delta Val = uniform - method (higher is better)", "{:+.4f}"),
        (axes[1], ppl, "Test PPL improvement", "Delta PPL = uniform - method (higher is better)", "{:+.3f}"),
    ]:
        x = np.arange(len(methods))
        bars = ax.bar(x, vals, color=[COLORS[m] for m in methods], width=0.62)
        ax.set_xticks(x, labels)
        ax.set_title(title, fontsize=12, weight="bold")
        ax.set_ylabel(ylabel, fontsize=9)
        nice_axes(ax)
        set_symmetric_ylim(ax, vals)
        annotate_bars(ax, bars, fmt=fmt)
    fig.suptitle("NSA WikiText-103 static schedules under matched selected-block compute", fontsize=13, weight="bold")
    fig.subplots_adjust(top=0.90, bottom=0.09, hspace=0.52)
    fig.savefig(OUT / "fig01_nsa_wikitext_static_delta.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_nsa_three_corpus_delta():
    datasets = ["C4-16384", "OpenWebText-4096", "WikiText-103"]
    values = {
        "C4-16384": {"mono_inc": (0.0074, None), "mono_dec": (-0.0140, None)},
        "OpenWebText-4096": {"mono_inc": (0.0158, None), "mono_dec": (0.0090, None)},
        "WikiText-103": {"mono_inc": (-0.0221, None), "mono_dec": (0.0037, None)},
    }
    fig, axes = plt.subplots(2, 1, figsize=(8.4, 7.6))
    metrics = [
        ("Delta Val", {"C4-16384": {"mono_inc": 0.0074, "mono_dec": -0.0140}, "OpenWebText-4096": {"mono_inc": 0.0158, "mono_dec": 0.0090}, "WikiText-103": {"mono_inc": -0.0221, "mono_dec": 0.0037}}, "{:+.4f}"),
        ("Delta PPL", {"C4-16384": {"mono_inc": 0.402, "mono_dec": -0.783}, "OpenWebText-4096": {"mono_inc": 2.120, "mono_dec": 1.257}, "WikiText-103": {"mono_inc": -1.360, "mono_dec": 0.337}}, "{:+.3f}"),
    ]
    for ax, (ylabel, data, fmt) in zip(axes, metrics):
        x = np.arange(len(datasets))
        width = 0.34
        all_vals = []
        for j, method in enumerate(["mono_inc", "mono_dec"]):
            vals = [data[d][method] for d in datasets]
            all_vals.extend(vals)
            bars = ax.bar(
                x + (j - 0.5) * width,
                vals,
                width,
                color=COLORS[method],
                label=method,
                edgecolor="white",
                linewidth=0.8,
            )
            annotate_bars(ax, bars, fmt=fmt, pad_frac=0.04)
        ax.set_xticks(x, datasets)
        ax.set_ylabel(f"{ylabel} vs uniform\n(higher is better)")
        ax.set_title(f"NSA selected-block {ylabel} under matched compute", fontsize=12, weight="bold")
        nice_axes(ax)
        set_symmetric_ylim(ax, all_vals, margin=0.28, floor=0.01 if ylabel == "Delta Val" else 0.5)
        ax.legend(frameon=False, ncol=2, loc="upper right")
    fig.suptitle("NSA selected-block static schedules across corpora", fontsize=13, weight="bold", y=0.98)
    fig.subplots_adjust(top=0.90, bottom=0.10, hspace=0.52)
    fig.savefig(OUT / "figS1_nsa_three_corpus_static_delta.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def h2o_rows_combined():
    rows = []
    # WikiText single official split from R22.
    r22 = load(R22)
    for row in r22["rows"]:
        rows.append(
            {
                "dataset": "WikiText-103",
                "method": row["method"],
                "delta_val": row["delta_val_vs_uniform"],
                "delta_ppl": row["delta_ppl_vs_uniform"],
                "delta_val_std": 0.0,
                "delta_ppl_std": 0.0,
                "windows": 1,
            }
        )
    # C4/OWT multi-window from R26.
    r26 = load(R26)
    for row in r26["aggregates"]:
        rows.append(
            {
                "dataset": "C4" if row["dataset_key"] == "c4" else "OpenWebText",
                "method": row["method"],
                "delta_val": row["delta_val_mean"],
                "delta_ppl": row["delta_ppl_mean"],
                "delta_val_std": row.get("delta_val_std", 0.0) or 0.0,
                "delta_ppl_std": row.get("delta_ppl_std", 0.0) or 0.0,
                "windows": row["num_windows"],
            }
        )
    return rows


def plot_grouped_delta(rows, title, out_name, methods=("mono_inc", "mono_dec")):
    datasets = ["WikiText-103", "C4", "OpenWebText"]
    fig, axes = plt.subplots(2, 1, figsize=(8.6, 8.0))
    for ax, metric, std_metric, ylabel, subtitle, fmt in [
        (axes[0], "delta_val", "delta_val_std", "Delta Val = uniform - method", "Validation loss improvement", "{:+.4f}"),
        (axes[1], "delta_ppl", "delta_ppl_std", "Delta PPL = uniform - method", "Test PPL improvement", "{:+.2f}"),
    ]:
        x = np.arange(len(datasets))
        width = 0.34
        all_vals = []
        for j, m in enumerate(methods):
            vals = []
            errs = []
            for d in datasets:
                item = next((r for r in rows if r["dataset"] == d and r["method"] == m), None)
                vals.append(item[metric] if item else np.nan)
                errs.append(item[std_metric] if item else 0.0)
            all_vals.extend([v for v in vals if not np.isnan(v)])
            offset = (j - (len(methods) - 1) / 2) * width
            bars = ax.bar(
                x + offset,
                vals,
                width,
                yerr=errs,
                capsize=3,
                color=COLORS[m],
                label=m,
                alpha=0.92,
                edgecolor="white",
                linewidth=0.7,
            )
            annotate_bars(ax, bars, fmt=fmt, pad_frac=0.035)
        ax.set_xticks(x, [f"{d}\n({1 if d == 'WikiText-103' else 3} window{'s' if d != 'WikiText-103' else ''})" for d in datasets])
        ax.set_title(subtitle, fontsize=12, weight="bold")
        ax.set_ylabel(ylabel + " (higher is better)", fontsize=9)
        nice_axes(ax)
        set_symmetric_ylim(ax, all_vals, margin=0.28)
        ax.legend(frameon=False, ncol=2)
    fig.suptitle(title, fontsize=13, weight="bold")
    fig.subplots_adjust(top=0.90, bottom=0.12, hspace=0.55)
    fig.savefig(OUT / out_name, dpi=220, bbox_inches="tight")
    plt.close(fig)


def sliding_rows_combined():
    rows = []
    r25 = load(R25_WIKI)
    for row in r25["rows"]:
        rows.append(
            {
                "dataset": "WikiText-103",
                "method": row["method"],
                "delta_val": row["delta_val_vs_uniform"],
                "delta_ppl": row["delta_ppl_vs_uniform"],
                "delta_val_std": 0.0,
                "delta_ppl_std": 0.0,
                "delta_after512": np.nan,
                "delta_after768": np.nan,
                "windows": 1,
            }
        )
    r27 = load(R27_AUDIT)
    # Need PPL from robustness for C4/OWT; audit only has val-loss variants.
    r27_robust = load(ROOT / "results" / "round27-phase27-sliding-window-robustness-diagnostic" / "round27_sliding_window_robustness_summary.json")
    ppl_map = {(r["dataset_key"], r["method"]): (r["delta_ppl_mean"], r.get("delta_ppl_std", 0.0) or 0.0) for r in r27_robust["aggregates"]}
    for row in r27["aggregates"]:
        dname = "C4" if row["dataset_key"] == "c4" else "OpenWebText"
        dppl, dppl_std = ppl_map[(row["dataset_key"], row["method"])]
        rows.append(
            {
                "dataset": dname,
                "method": row["method"],
                "delta_val": row["delta_loss_mean"],
                "delta_ppl": dppl,
                "delta_val_std": 0.0,
                "delta_ppl_std": dppl_std,
                "delta_after512": row["delta_after512_mean"],
                "delta_after768": row["delta_after768_mean"],
                "windows": row["n"],
            }
        )
    return rows


def plot_sliding_audit_cutoffs():
    rows = [r for r in sliding_rows_combined() if r["dataset"] in ("C4", "OpenWebText") and r["method"] in ("mono_inc", "mono_dec")]
    datasets = ["C4", "OpenWebText"]
    metrics = [("delta_val", "All-token"), ("delta_after512", "After512"), ("delta_after768", "After768")]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), sharey=True, constrained_layout=True)
    for ax, dataset in zip(axes, datasets):
        x = np.arange(len(metrics))
        width = 0.34
        vals_all = []
        for j, method in enumerate(["mono_inc", "mono_dec"]):
            vals = [next(r for r in rows if r["dataset"] == dataset and r["method"] == method)[k] for k, _ in metrics]
            vals_all.extend(vals)
            bars = ax.bar(x + (j - 0.5) * width, vals, width, label=method, color=COLORS[method], edgecolor="white")
            annotate_bars(ax, bars, fmt="{:+.3f}", pad_frac=0.04)
        ax.set_title(dataset, fontsize=12, weight="bold")
        ax.set_xticks(x, [name for _, name in metrics])
        ax.set_ylabel("Delta Val = uniform - method (higher is better)")
        nice_axes(ax)
        set_symmetric_ylim(ax, vals_all, margin=0.32)
        ax.legend(frameon=False)
    fig.suptitle("Sliding-window audit: fixed-cutoff deltas confirm mono_dec advantage", fontsize=13, weight="bold")
    fig.savefig(OUT / "fig04_sliding_window_audit_cutoffs.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_h2o_per_window_scatter():
    r26 = load(R26)
    rows = [r for r in r26["rows"] if r["method"] in ("mono_inc", "mono_dec")]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True, constrained_layout=True)
    for ax, dataset_key, dataset_name in zip(axes, ["c4", "owt"], ["C4", "OpenWebText"]):
        xs = {"mono_inc": 0, "mono_dec": 1}
        for method in ["mono_inc", "mono_dec"]:
            vals = [r["delta_val_vs_uniform"] for r in rows if r["dataset_key"] == dataset_key and r["method"] == method]
            jitter = np.linspace(-0.045, 0.045, len(vals))
            ax.scatter(np.full(len(vals), xs[method]) + jitter, vals, s=55, color=COLORS[method], label=method if dataset_key == "c4" else None, zorder=3)
            ax.plot([xs[method] - 0.08, xs[method] + 0.08], [np.mean(vals), np.mean(vals)], color="black", linewidth=1.4)
        ax.set_xticks([0, 1], ["mono_inc", "mono_dec"])
        ax.set_title(dataset_name, fontsize=12, weight="bold")
        ax.set_ylabel("Delta Val = uniform - method")
        nice_axes(ax)
        set_symmetric_ylim(ax, [r["delta_val_vs_uniform"] for r in rows if r["dataset_key"] == dataset_key], margin=0.35)
    fig.suptitle("H2O-style token retention: per-window validation deltas", fontsize=13, weight="bold")
    axes[0].legend(frameon=False)
    fig.savefig(OUT / "fig05_h2o_per_window_delta_scatter.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_effect_matrix():
    entries = []
    for platform, rows in [("NSA block", [
        ("WikiText-103", "mono_inc", -0.0221), ("WikiText-103", "mono_dec", 0.0037),
        ("C4", "mono_inc", 0.0074), ("C4", "mono_dec", -0.0140),
        ("OpenWebText", "mono_inc", 0.0158), ("OpenWebText", "mono_dec", 0.0090),
    ])]:
        for dataset, method, val in rows:
            entries.append((platform, dataset, method, val))
    for r in h2o_rows_combined():
        if r["method"] != "uniform":
            entries.append(("H2O token", r["dataset"], r["method"], r["delta_val"]))
    for r in sliding_rows_combined():
        if r["method"] != "uniform":
            entries.append(("Sliding window", r["dataset"], r["method"], r["delta_val"]))
    row_labels = []
    for platform in ["NSA block", "H2O token", "Sliding window"]:
        for dataset in ["WikiText-103", "C4", "OpenWebText"]:
            row_labels.append((platform, dataset))
    data = np.full((len(row_labels), 2), np.nan)
    for i, (platform, dataset) in enumerate(row_labels):
        for j, method in enumerate(["mono_inc", "mono_dec"]):
            matches = [v for p, d, m, v in entries if p == platform and d == dataset and m == method]
            if matches:
                data[i, j] = matches[0]
    fig, ax = plt.subplots(figsize=(7.6, 7.2), constrained_layout=True)
    vmax = np.nanmax(np.abs(data))
    im = ax.imshow(data, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks([0, 1], ["mono_inc", "mono_dec"])
    ax.set_yticks(np.arange(len(row_labels)), [f"{p}\n{d}" for p, d in row_labels])
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if not np.isnan(data[i, j]):
                ax.text(j, i, f"{data[i,j]:+.3f}", ha="center", va="center", fontsize=8, color="black")
    ax.set_title("Operator-sensitive budget-placement effect matrix (Delta Val)", fontsize=13, weight="bold")
    cb = fig.colorbar(im, ax=ax, shrink=0.82)
    cb.set_label("Delta Val = uniform - method (positive is better)")
    fig.savefig(OUT / "fig06_operator_sensitive_delta_matrix.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_operator_sensitive_grouped_bars():
    """Main summary figure: three panels, one per sparse platform."""
    panels = [
        {
            "title": "A. NSA selected-block attention",
            "note": "training-time controlled platform",
            "datasets": ["C4", "OpenWebText", "WikiText-103"],
            "values": {
                "C4": {"mono_inc": (0.0074, None), "mono_dec": (-0.0140, None)},
                "OpenWebText": {"mono_inc": (0.0158, None), "mono_dec": (0.0090, None)},
                "WikiText-103": {"mono_inc": (-0.0221, None), "mono_dec": (0.0037, None)},
            },
        },
        {
            "title": "B. H2O-style token-level KV retention",
            "note": "inference-time diagnostic, OPT-350M",
            "datasets": ["C4", "OpenWebText", "WikiText-103"],
            "values": {
                "C4": {"mono_inc": (-0.0110, 0.0008), "mono_dec": (-0.0046, 0.0002)},
                "OpenWebText": {"mono_inc": (-0.0245, 0.0010), "mono_dec": (-0.0007, 0.0022)},
                "WikiText-103": {"mono_inc": (-0.0157, None), "mono_dec": (-0.0015, None)},
            },
        },
        {
            "title": "C. Sliding-window local-window attention",
            "note": "inference-time diagnostic, OPT-350M",
            "datasets": ["C4", "OpenWebText", "WikiText-103"],
            "values": {
                "C4": {"mono_inc": (-0.0129, 0.0098), "mono_dec": (0.2996, 0.0049)},
                "OpenWebText": {"mono_inc": (-0.0799, 0.0108), "mono_dec": (0.3273, 0.0112)},
                "WikiText-103": {"mono_inc": (0.0281, None), "mono_dec": (0.2413, None)},
            },
        },
    ]
    methods = ["mono_inc", "mono_dec"]
    fig, axes = plt.subplots(3, 1, figsize=(8.8, 10.8))
    for ax, panel in zip(axes, panels):
        datasets = panel["datasets"]
        x = np.arange(len(datasets))
        width = 0.34
        all_vals = []
        for j, method in enumerate(methods):
            vals = [panel["values"][d][method][0] for d in datasets]
            errs = [panel["values"][d][method][1] or 0.0 for d in datasets]
            all_vals.extend(vals)
            bars = ax.bar(
                x + (j - 0.5) * width,
                vals,
                width,
                yerr=errs,
                capsize=3,
                color=COLORS[method],
                label=method,
                edgecolor="white",
                linewidth=0.8,
                alpha=0.95,
            )
            annotate_bars(ax, bars, fmt="{:+.3f}", pad_frac=0.045)
        # Mark the best schedule in each dataset group if it improves over uniform.
        for i, dataset in enumerate(datasets):
            best_method = max(methods, key=lambda m: panel["values"][dataset][m][0])
            best_value = panel["values"][dataset][best_method][0]
            if best_value > 0:
                j = methods.index(best_method)
                xpos = i + (j - 0.5) * width
                ax.text(xpos, best_value + (max(abs(v) for v in all_vals) * 0.16), "*", ha="center", va="bottom", fontsize=13, weight="bold")
        ax.set_xticks(x, datasets)
        ax.set_ylabel("Delta Val vs uniform\n(higher is better)", fontsize=9)
        ax.set_title(f"{panel['title']}  -  {panel['note']}", fontsize=12, weight="bold")
        nice_axes(ax)
        set_symmetric_ylim(ax, all_vals, margin=0.34, floor=0.02)
        ax.legend(frameon=False, ncol=2, loc="upper right")
    fig.suptitle("Operator-sensitive budget-placement effects across sparse units and datasets", fontsize=14, weight="bold", y=0.985)
    fig.text(
        0.5,
        0.015,
        "Delta Val = ValLoss(uniform) - ValLoss(method); positive values mean better than uniform. Each panel uses its own y-axis scale.",
        ha="center",
        fontsize=8.8,
    )
    fig.subplots_adjust(top=0.94, bottom=0.07, hspace=0.58)
    fig.savefig(OUT / "fig_main_operator_sensitive_grouped_bars.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_effect_matrix_v2():
    """Appendix-style qualitative heatmap with platform separators and N.A. support."""
    row_labels = [
        ("NSA block", "C4"),
        ("NSA block", "OpenWebText"),
        ("NSA block", "WikiText-103"),
        ("H2O token", "C4"),
        ("H2O token", "OpenWebText"),
        ("H2O token", "WikiText-103"),
        ("Sliding window", "C4"),
        ("Sliding window", "OpenWebText"),
        ("Sliding window", "WikiText-103"),
    ]
    values = {
        ("NSA block", "C4", "mono_inc"): 0.0074,
        ("NSA block", "C4", "mono_dec"): -0.0140,
        ("NSA block", "OpenWebText", "mono_inc"): 0.0158,
        ("NSA block", "OpenWebText", "mono_dec"): 0.0090,
        ("NSA block", "WikiText-103", "mono_inc"): -0.0221,
        ("NSA block", "WikiText-103", "mono_dec"): 0.0037,
        ("H2O token", "C4", "mono_inc"): -0.0110,
        ("H2O token", "C4", "mono_dec"): -0.0046,
        ("H2O token", "OpenWebText", "mono_inc"): -0.0245,
        ("H2O token", "OpenWebText", "mono_dec"): -0.0007,
        ("H2O token", "WikiText-103", "mono_inc"): -0.0157,
        ("H2O token", "WikiText-103", "mono_dec"): -0.0015,
        ("Sliding window", "C4", "mono_inc"): -0.0129,
        ("Sliding window", "C4", "mono_dec"): 0.2996,
        ("Sliding window", "OpenWebText", "mono_inc"): -0.0799,
        ("Sliding window", "OpenWebText", "mono_dec"): 0.3273,
        ("Sliding window", "WikiText-103", "mono_inc"): 0.0281,
        ("Sliding window", "WikiText-103", "mono_dec"): 0.2413,
    }
    data = np.full((len(row_labels), 2), np.nan)
    for i, (platform, dataset) in enumerate(row_labels):
        for j, method in enumerate(["mono_inc", "mono_dec"]):
            data[i, j] = values.get((platform, dataset, method), np.nan)
    fig, ax = plt.subplots(figsize=(7.6, 7.2), constrained_layout=True)
    vmax = np.nanmax(np.abs(data))
    cmap = plt.cm.RdBu.copy()
    cmap.set_bad("#F2F2F2")
    im = ax.imshow(np.ma.masked_invalid(data), cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks([0, 1], ["mono_inc", "mono_dec"])
    ax.set_yticks(np.arange(len(row_labels)), [f"{p} - {d}" for p, d in row_labels])
    for sep in [2.5, 5.5]:
        ax.axhline(sep, color="black", linewidth=1.4)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.isnan(data[i, j]):
                ax.text(j, i, "N.A.", ha="center", va="center", fontsize=8, color="#555")
            else:
                ax.text(j, i, f"{data[i,j]:+.3f}", ha="center", va="center", fontsize=8, color="black")
    ax.set_title("Budget-placement effect matrix across sparse units", fontsize=13, weight="bold")
    cb = fig.colorbar(im, ax=ax, shrink=0.82)
    cb.set_label("Delta Val vs uniform; positive is better")
    fig.savefig(OUT / "figS_operator_sensitive_delta_matrix_grouped.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def write_best_schedule_table():
    rows = [
        ("NSA block", "C4", "mono_inc", "+0.007", "C4-16384 long-training main comparison"),
        ("NSA block", "OpenWebText", "mono_inc", "+0.016", "cross-corpus support"),
        ("NSA block", "WikiText-103", "mono_dec", "+0.004", "mixed direction on third corpus"),
        ("H2O token", "C4", "uniform", "+0.000", "uniform strong default"),
        ("H2O token", "OpenWebText", "uniform ~= mono_dec", "+0.000 / -0.001", "near-tie; mono_dec PPL slightly better"),
        ("H2O token", "WikiText-103", "uniform", "+0.000", "uniform strongest"),
        ("Sliding window", "C4", "mono_dec", "+0.300", "mono_dec wins 3/3 windows"),
        ("Sliding window", "OpenWebText", "mono_dec", "+0.327", "mono_dec wins 3/3 windows"),
        ("Sliding window", "WikiText-103", "mono_dec", "+0.241", "single official split diagnostic"),
    ]
    md = [
        "| Sparse unit / platform | Dataset | Best schedule | Delta Val | Interpretation |",
        "|---|---|---|---:|---|",
    ]
    csv = ["platform,dataset,best_schedule,delta_val,interpretation"]
    for r in rows:
        md.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} |")
        csv.append(",".join('"' + str(x).replace('"', '""') + '"' for x in r))
    (OUT / "best_schedule_summary_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (OUT / "best_schedule_summary_table.csv").write_text("\n".join(csv) + "\n", encoding="utf-8")


def plot_nsa_behavioral_diagnostics():
    groups = ["layers 1-6", "layers 7-12", "layers 13-18", "layers 19-24"]
    x = np.arange(len(groups))
    data = {
        "baseline": {
            "distance": [8.490, 7.938, 8.113, 8.041],
            "distance_std": [0.128, 0.201, 0.199, 0.133],
            "far_ratio": [0.454, 0.434, 0.440, 0.435],
            "far_ratio_std": [0.003, 0.008, 0.009, 0.007],
            "far_mass": [0.140513, 0.079938, 0.121913, 0.101962],
            "far_mass_std": [0.020605, 0.007926, 0.038131, 0.027836],
        },
        "mono_inc": {
            "distance": [6.061, 6.683, 8.960, 9.732],
            "distance_std": [0.040, 0.391, 0.090, 0.032],
            "far_ratio": [0.266, 0.332, 0.497, 0.539],
            "far_ratio_std": [0.009, 0.016, 0.001, 0.001],
            "far_mass": [0.131032, 0.066483, 0.085201, 0.115267],
            "far_mass_std": [0.019005, 0.008983, 0.026542, 0.037497],
        },
        "mono_dec": {
            "distance": [9.871, 8.903, 6.688, 5.282],
            "distance_std": [0.031, 0.100, 0.090, 0.353],
            "far_ratio": [0.543, 0.496, 0.328, 0.208],
            "far_ratio_std": [0.000, 0.002, 0.009, 0.025],
            "far_mass": [0.165404, 0.082867, 0.073583, 0.078939],
            "far_mass_std": [0.005241, 0.010192, 0.014926, 0.022671],
        },
    }
    fig, axes = plt.subplots(3, 1, figsize=(8.8, 10.0), sharex=True)
    panels = [
        ("distance", "distance_std", "A. Mean selected-block distance", "Mean selected distance (blocks)"),
        ("far_ratio", "far_ratio_std", "B. Far selected-block ratio", "Far selected ratio (distance >= 8 blocks)"),
        ("far_mass", "far_mass_std", "C. Far selected-attention mass", "Attention mass on far retained blocks"),
    ]
    markers = {"baseline": "o", "mono_inc": "s", "mono_dec": "^"}
    for ax, (key, std_key, title, ylabel) in zip(axes, panels):
        for method in ["baseline", "mono_inc", "mono_dec"]:
            vals = np.array(data[method][key])
            errs = np.array(data[method][std_key])
            ax.errorbar(
                x,
                vals,
                yerr=errs,
                marker=markers[method],
                linewidth=2.0,
                markersize=5.5,
                capsize=3,
                label=method,
                color=COLORS.get(method, "#5B6472"),
            )
            for xi, yi in zip(x, vals):
                ax.text(xi, yi, f"{yi:.3f}" if key != "distance" else f"{yi:.2f}", ha="center", va="bottom", fontsize=7.5)
        ax.set_title(title, fontsize=12, weight="bold")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(axis="y", color="#E6E8EB", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_axisbelow(True)
    axes[-1].set_xticks(x, groups)
    axes[0].legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.20))
    fig.suptitle("NSA behavioral diagnostics for layer-wise sparse context access", fontsize=14, weight="bold", y=0.985)
    fig.text(
        0.5,
        0.012,
        "Behavioral diagnostic only: these trends describe selected-block access patterns and attention mass within NSA, not causal mechanisms.",
        ha="center",
        fontsize=8.5,
    )
    fig.subplots_adjust(top=0.91, bottom=0.08, hspace=0.38)
    fig.savefig(OUT / "fig_nsa_behavioral_diagnostics_3panel.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_dynamic_budget_exploration():
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6))

    # Panel A: short-/long-training diagnostic values from Table 7.
    stages = ["Mid-training\n(4096 steps)", "Long-training\n(8192 steps)"]
    methods = ["mono_inc", "tool_aux 3e-4", "strict_clamped"]
    vals = {
        "mono_inc": [4.9088, 4.4111],
        "tool_aux 3e-4": [4.8597, 4.4222],
        "strict_clamped": [4.8520, 4.4166],
    }
    colors = {"mono_inc": COLORS["mono_inc"], "tool_aux 3e-4": "#7A6AAE", "strict_clamped": "#4C9A72"}
    x = np.arange(len(stages))
    width = 0.23
    for j, method in enumerate(methods):
        y = vals[method]
        bars = axes[0].bar(x + (j - 1) * width, y, width, label=method, color=colors[method], edgecolor="white", linewidth=0.8)
        for b in bars:
            h = b.get_height()
            axes[0].text(b.get_x() + b.get_width() / 2, h + 0.006, f"{h:.4f}", ha="center", va="bottom", fontsize=7.5, rotation=0)
    axes[0].set_xticks(x, stages)
    axes[0].set_ylabel("Validation loss (lower is better)")
    axes[0].set_title("A. Stage-dependent validation behavior", fontsize=12, weight="bold")
    axes[0].grid(axis="y", color="#E6E8EB", linewidth=0.8)
    axes[0].set_axisbelow(True)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    axes[0].set_ylim(4.35, 4.96)

    # Panel B: mechanism diagnostics as deltas relative to mono_inc in their settings.
    # Positive means lower loss than the corresponding mono_inc reference.
    mech_labels = ["tool_aux\nmid", "strict\nmid", "tool_aux\nlong", "strict\nlong", "random\nreplay", "shuffled"]
    mech_vals = [
        4.9088 - 4.8597,
        4.9088 - 4.8520,
        4.4111 - 4.4222,
        4.4111 - 4.4166,
        4.9088 - 4.9159,
        4.9088 - 4.8682,
    ]
    mech_colors = ["#7A6AAE", "#4C9A72", "#7A6AAE", "#4C9A72", "#C56E5A", "#D69B3A"]
    bars = axes[1].bar(np.arange(len(mech_labels)), mech_vals, color=mech_colors, edgecolor="white", linewidth=0.8)
    axes[1].axhline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.75)
    for b, v in zip(bars, mech_vals):
        axes[1].text(b.get_x() + b.get_width() / 2, v + (0.004 if v >= 0 else -0.004), f"{v:+.4f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=7.5)
    axes[1].set_xticks(np.arange(len(mech_labels)), mech_labels)
    axes[1].set_ylabel("Delta Val vs matched mono_inc reference\n(positive means lower loss)")
    axes[1].set_title("B. Mechanism diagnostics are mixed", fontsize=12, weight="bold")
    axes[1].grid(axis="y", color="#E6E8EB", linewidth=0.8)
    axes[1].set_axisbelow(True)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    axes[1].set_ylim(-0.025, 0.070)

    fig.suptitle("Dynamic budget exploration: short-training signal, long-training instability", fontsize=13, weight="bold")
    fig.text(
        0.5,
        -0.02,
        "Diagnostic figure only. Values summarize representative validation losses; dynamic variants are not treated as final methods.",
        ha="center",
        fontsize=8.5,
    )
    fig.subplots_adjust(top=0.84, bottom=0.24, wspace=0.28)
    fig.savefig(OUT / "fig_dynamic_budget_exploration_diagnostic.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 12,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
        }
    )
    plot_nsa_wikitext()
    plot_nsa_three_corpus_delta()
    plot_grouped_delta(h2o_rows_combined(), "H2O-style KV retention: delta over uniform", "fig02_h2o_kv_retention_delta.png")
    plot_grouped_delta(sliding_rows_combined(), "Sliding-window attention: delta over uniform", "fig03_sliding_window_delta.png")
    plot_sliding_audit_cutoffs()
    plot_h2o_per_window_scatter()
    plot_effect_matrix()
    plot_operator_sensitive_grouped_bars()
    plot_effect_matrix_v2()
    plot_nsa_behavioral_diagnostics()
    plot_dynamic_budget_exploration()
    write_best_schedule_table()
    print(f"Wrote figures to {OUT}")
    for p in sorted(OUT.glob("*.png")):
        print(p)


if __name__ == "__main__":
    main()
