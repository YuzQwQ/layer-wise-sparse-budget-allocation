
import json
import math
import statistics as st
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
R14 = ROOT / "results" / "round14-phase14-static-longtrain-c4" / "exp"
R20 = ROOT / "results" / "round20-phase20-static-longtrain-extra-seeds-c4" / "exp"
R19 = ROOT / "results" / "round19-phase19-selected-attention-mass-c4" / "formal" / "selected_attention_mass_summary.json"
OUT = ROOT / "results" / "round20-phase20-static-longtrain-extra-seeds-c4" / "paper_update"
OUT.mkdir(parents=True, exist_ok=True)

METHOD_LABEL = {
    "baseline": "Uniform baseline",
    "mono_inc": "mono_inc",
    "mono_dec": "mono_dec",
}
COLOR = {
    "baseline": "#5B6770",
    "mono_inc": "#1B9E77",
    "mono_dec": "#D95F02",
}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_run(run_dir: Path, method: str, seed: int, source: str):
    rs = load_json(run_dir / "run_summary.json")
    tm = load_json(run_dir / "test_metrics.json")
    train = rs.get("final_train", {})
    val = rs["final_validation"]
    test = rs["final_test"]
    return {
        "method": method,
        "seed": seed,
        "source": source,
        "run_name": run_dir.name,
        "val_loss": float(val["loss"]),
        "val_ppl": float(val["ppl"]),
        "test_loss": float(test["loss"]),
        "test_ppl": float(test["ppl"]),
        "actual_k_mean": float(train.get("actual_k_mean", 16.0)),
        "estimated_kv_access_mean": float(train.get("estimated_kv_access_mean", 1024.0)),
        "tok_per_s": float(train.get("tok_per_s", 0.0)),
    }


def collect_runs():
    rows = []
    for method, prefix in [("baseline", "r14_m0_baseline"), ("mono_inc", "r14_m1_mono_inc"), ("mono_dec", "r14_m2_mono_dec")]:
        for seed in [42, 123, 3407]:
            d = R14 / f"{prefix}_s{seed}"
            if d.exists():
                rows.append(load_run(d, method, seed, "r14"))
    for method, prefix in [("baseline", "r20_m0_baseline"), ("mono_inc", "r20_m1_mono_inc")]:
        for seed in [2024, 2025]:
            d = R20 / f"{prefix}_s{seed}"
            if d.exists():
                rows.append(load_run(d, method, seed, "r20"))
    return rows


def mean_std(vals):
    vals = list(vals)
    if not vals:
        return None, None
    return sum(vals) / len(vals), (st.stdev(vals) if len(vals) > 1 else 0.0)


def summarize_static(rows):
    by = defaultdict(list)
    for r in rows:
        by[r["method"]].append(r)
    summary = {}
    for method, items in by.items():
        summary[method] = {
            "seeds": sorted([r["seed"] for r in items]),
            "n": len(items),
        }
        for key in ["val_loss", "test_ppl", "actual_k_mean", "estimated_kv_access_mean", "tok_per_s"]:
            m, s = mean_std([r[key] for r in items])
            summary[method][f"{key}_mean"] = m
            summary[method][f"{key}_std"] = s
    base = summary["baseline"]
    for method, s in summary.items():
        s["delta_val_improvement"] = base["val_loss_mean"] - s["val_loss_mean"]
        s["delta_ppl_improvement"] = base["test_ppl_mean"] - s["test_ppl_mean"]
    return summary


def plot_paired_improvement(rows):
    by_seed = defaultdict(dict)
    for r in rows:
        by_seed[r["seed"]][r["method"]] = r
    seeds = sorted(seed for seed, d in by_seed.items() if "baseline" in d and "mono_inc" in d)
    val_imp = [by_seed[s]["baseline"]["val_loss"] - by_seed[s]["mono_inc"]["val_loss"] for s in seeds]
    ppl_imp = [by_seed[s]["baseline"]["test_ppl"] - by_seed[s]["mono_inc"]["test_ppl"] for s in seeds]
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 7.2), sharex=True)
    for ax, vals, ylabel, title in [
        (axes[0], val_imp, "Validation improvement\n(baseline - mono_inc)", "Final validation loss improvement"),
        (axes[1], ppl_imp, "Test PPL improvement\n(baseline - mono_inc)", "Final test PPL improvement"),
    ]:
        x = np.arange(len(seeds))
        lo = min(vals + [0.0])
        hi = max(vals + [0.0])
        pad = max((hi - lo) * 0.18, 0.006 if abs(hi - lo) < 1 else 0.35)
        ax.set_ylim(lo - pad, hi + pad)
        ax.axhline(0, color="black", linestyle="--", linewidth=1.2)
        ax.scatter(x, vals, s=58, color=COLOR["mono_inc"], zorder=3)
        ax.plot(x, vals, color=COLOR["mono_inc"], linewidth=1.2, alpha=0.65)
        mean_val = sum(vals) / len(vals)
        ax.axhline(mean_val, color=COLOR["mono_inc"], linestyle=":", linewidth=1.5, label=f"mean={mean_val:+.4f}" if "loss" in title else f"mean={mean_val:+.3f}")
        for xi, yi in zip(x, vals):
            fmt = f"{yi:+.4f}" if abs(yi) < 1 else f"{yi:+.3f}"
            va = "bottom" if yi >= 0 else "top"
            offset = pad * 0.16
            ax.text(xi, yi + (offset if yi >= 0 else -offset), fmt, ha="center", va=va, fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False, loc="best", fontsize=8)
    axes[1].set_xticks(np.arange(len(seeds)))
    axes[1].set_xticklabels([str(s) for s in seeds])
    axes[1].set_xlabel("Seed")
    fig.suptitle("Per-seed paired improvement under matched compute (C4-16384, 5 seeds)", y=0.995, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "c4_16384_paired_seed_improvement.png", dpi=220)
    plt.close(fig)


def plot_final_static_summary(summary):
    methods = ["baseline", "mono_inc", "mono_dec"]
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 7.8), sharex=True)
    for ax, metric, err, ylabel, title in [
        (axes[0], "val_loss_mean", "val_loss_std", "Final validation loss", "C4-16384 final validation loss"),
        (axes[1], "test_ppl_mean", "test_ppl_std", "Final test PPL", "C4-16384 final test PPL"),
    ]:
        x = np.arange(len(methods))
        vals = [summary[m][metric] for m in methods]
        errs = [summary[m][err] for m in methods]
        ax.errorbar(x, vals, yerr=errs, fmt="o", color="#333333", ecolor="#777777", capsize=4, markersize=7)
        for xi, m, v in zip(x, methods, vals):
            ax.scatter([xi], [v], color=COLOR[m], s=64, zorder=3)
            ax.text(xi, v, f" {v:.4f}" if metric.startswith("val") else f" {v:.3f}", va="center", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=0.25)
    labels = [f"{METHOD_LABEL[m]}\n(n={summary[m]['n']})" for m in methods]
    axes[1].set_xticks(np.arange(len(methods)))
    axes[1].set_xticklabels(labels)
    fig.suptitle("C4-16384 static allocation comparison under matched compute", y=0.995, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "c4_16384_static_final_metrics_updated.png", dpi=220)
    plt.close(fig)


def plot_p2plus():
    data = load_json(R19)
    agg = data["aggregate"]["methods"]
    groups = ["layers_1_6", "layers_7_12", "layers_13_18", "layers_19_24"]
    group_labels = ["1-6", "7-12", "13-18", "19-24"]
    methods = ["baseline", "mono_inc", "mono_dec"]

    def grouped_bar(metric_key, ylabel, title, path, ylim=None):
        x = np.arange(len(groups))
        width = 0.24
        fig, ax = plt.subplots(figsize=(8.2, 4.6))
        for i, method in enumerate(methods):
            vals = [agg[method]["groups"][g][f"{metric_key}_mean"] for g in groups]
            errs = [agg[method]["groups"][g][f"{metric_key}_std"] for g in groups]
            pos = x + (i - 1) * width
            bars = ax.bar(pos, vals, width=width, color=COLOR[method], alpha=0.82, label=METHOD_LABEL[method], yerr=errs, capsize=3, linewidth=0)
            for b, v in zip(bars, vals):
                ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.006, f"{v:.3f}", ha="center", va="bottom", fontsize=7, rotation=0)
        ax.set_xticks(x)
        ax.set_xticklabels(group_labels)
        ax.set_xlabel("Layer group")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(axis="y", alpha=0.22)
        ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.16))
        fig.tight_layout()
        fig.savefig(OUT / path, dpi=220)
        plt.close(fig)

    grouped_bar("far_attention_mass_ratio", "Far-attention mass ratio\n(distance >= 8 blocks)", "Selected attention mass assigned to far context by layer group", "p2plus_far_attention_mass_by_layer_group.png", (0, 0.24))
    grouped_bar("selected_attention_entropy_norm", "Normalized selected-attention entropy", "Selected attention entropy by layer group", "p2plus_selected_attention_entropy_by_layer_group.png", (0, 0.62))

    buckets = ["0", "1", "2_3", "4_7", "8_15", "16_plus"]
    bucket_labels = ["0", "1", "2-3", "4-7", "8-15", "16+"]
    # Focus on high layers for distance bucket mass, because it directly supports depth-aligned access explanation.
    high_group = "layers_19_24"
    x = np.arange(len(buckets))
    width = 0.24
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for i, method in enumerate(methods):
        vals = [agg[method]["groups"][high_group]["distance_bucket_mass"][b]["mean"] for b in buckets]
        pos = x + (i - 1) * width
        ax.bar(pos, vals, width=width, color=COLOR[method], alpha=0.82, label=METHOD_LABEL[method])
    ax.set_xticks(x)
    ax.set_xticklabels(bucket_labels)
    ax.set_xlabel("Selected block distance bucket")
    ax.set_ylabel("Attention mass")
    ax.set_title("High-layer selected attention mass by distance bucket (layers 19-24)")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    fig.tight_layout()
    fig.savefig(OUT / "p2plus_high_layer_attention_mass_distance_bucket.png", dpi=220)
    plt.close(fig)


def write_outputs(rows, summary):
    p2 = load_json(R19)
    payload = {
        "static_c4_16384": {"runs": rows, "summary": summary},
        "p2plus_selected_attention_mass": p2,
    }
    (OUT / "p2plus_p3_update_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = []
    lines.append("# P2-plus + P3 Update Summary")
    lines.append("")
    lines.append("## C4-16384 static main-result update")
    lines.append("")
    lines.append("| Method | Seeds | Val Loss | Test PPL | Delta Val ? | Delta PPL ? | actual_k | KV access | tok/s |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for method in ["baseline", "mono_inc", "mono_dec"]:
        s = summary[method]
        lines.append(
            f"| {method} | {s['n']} | {s['val_loss_mean']:.4f} ? {s['val_loss_std']:.4f} | "
            f"{s['test_ppl_mean']:.3f} ? {s['test_ppl_std']:.3f} | {s['delta_val_improvement']:+.4f} | "
            f"{s['delta_ppl_improvement']:+.3f} | {s['actual_k_mean_mean']:.1f} | {s['estimated_kv_access_mean_mean']:.1f} | {s['tok_per_s_mean']:.0f} |"
        )
    lines.append("")
    lines.append("Note: baseline and mono_inc use 5 seeds; mono_dec remains a 3-seed reverse-direction control.")
    lines.append("")
    lines.append("## P2-plus selected attention probability behavior")
    lines.append("")
    p2agg = p2["aggregate"]["methods"]
    lines.append("| Method | Low far mass (1-6) | High far mass (19-24) | Low entropy norm | High entropy norm |")
    lines.append("|---|---:|---:|---:|---:|")
    for method in ["baseline", "mono_inc", "mono_dec"]:
        low = p2agg[method]["groups"]["layers_1_6"]
        high = p2agg[method]["groups"]["layers_19_24"]
        lines.append(
            f"| {method} | {low['far_attention_mass_ratio_mean']:.3f} ? {low['far_attention_mass_ratio_std']:.3f} | "
            f"{high['far_attention_mass_ratio_mean']:.3f} ? {high['far_attention_mass_ratio_std']:.3f} | "
            f"{low['selected_attention_entropy_norm_mean']:.3f} | {high['selected_attention_entropy_norm_mean']:.3f} |"
        )
    lines.append("")
    lines.append("Interpretation: P2-plus supports a probability-level behavioral diagnostic: mono_inc assigns slightly more selected-attention mass to far context in high layers, while mono_dec suppresses high-layer far-context mass and shifts broader access toward lower layers. This remains diagnostic evidence, not a causal proof.")
    (OUT / "p2plus_p3_update_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    rows = collect_runs()
    summary = summarize_static(rows)
    write_outputs(rows, summary)
    plot_paired_improvement(rows)
    plot_final_static_summary(summary)
    plot_p2plus()
    print(f"Wrote {OUT}")

if __name__ == "__main__":
    main()
