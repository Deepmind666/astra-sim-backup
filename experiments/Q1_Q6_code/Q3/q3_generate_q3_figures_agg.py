import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _load_seed_json(base_dir: str, seed: str) -> Dict:
    path = os.path.join(base_dir, seed, "task_drift_summary.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _mean_ci(values: List[float], ci_pairs: List[List[float]]) -> Tuple[float, float, float]:
    mean_val = float(np.mean(values))
    lows = [c[0] for c in ci_pairs]
    highs = [c[1] for c in ci_pairs]
    mean_low = float(np.mean(lows))
    mean_high = float(np.mean(highs))
    err_low = max(mean_val - mean_low, 0.0)
    err_high = max(mean_high - mean_val, 0.0)
    return mean_val, err_low, err_high


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    seeds = ["seed11", "seed22", "seed33"]
    out_dir = os.path.join(base_dir, "figures_v2")
    os.makedirs(out_dir, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    seed_data = [_load_seed_json(base_dir, s) for s in seeds]
    tasks = sorted(seed_data[0]["tasks"].keys())

    # -------- Switch rate + Run mean (CI) --------
    switch_means = []
    switch_errs = []
    run_means = []
    run_errs = []
    for task in tasks:
        switch_vals = [d["tasks"][task]["switch_rate"] for d in seed_data]
        switch_ci = [d["tasks"][task]["switch_rate_ci_95"] for d in seed_data]
        mean_val, err_low, err_high = _mean_ci(switch_vals, switch_ci)
        switch_means.append(mean_val)
        switch_errs.append([err_low, err_high])

        run_vals = [d["tasks"][task]["run_mean"] for d in seed_data]
        run_ci = [d["tasks"][task]["run_mean_ci_95"] for d in seed_data]
        mean_val, err_low, err_high = _mean_ci(run_vals, run_ci)
        run_means.append(mean_val)
        run_errs.append([err_low, err_high])

    x = np.arange(len(tasks))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].bar(x, switch_means, color="#4C78A8", alpha=0.85)
    axes[0].errorbar(x, switch_means, yerr=np.array(switch_errs).T, fmt="none", ecolor="black", capsize=3)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(tasks, rotation=25, ha="right")
    axes[0].set_ylabel("switch_rate")
    axes[0].set_title("Switch Rate (mean ± 95% CI)")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(x, run_means, color="#59A14F", alpha=0.85)
    axes[1].errorbar(x, run_means, yerr=np.array(run_errs).T, fmt="none", ecolor="black", capsize=3)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(tasks, rotation=25, ha="right")
    axes[1].set_ylabel("run_mean")
    axes[1].set_title("Run Mean (mean ± 95% CI)")
    axes[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "q3_switch_run_ci.png"), dpi=200)
    fig.savefig(os.path.join(out_dir, "q3_switch_run_ci.pdf"))
    plt.close(fig)

    # -------- Run length ratio (length=1) --------
    run1_means = []
    run1_errs = []
    for task in tasks:
        ratios = []
        for d in seed_data:
            r = d["tasks"][task]["run_length_hist_ratio"]
            ratios.append(r.get("1", 0.0))
        ratios = np.array(ratios)
        run1_means.append(float(np.mean(ratios)))
        run1_errs.append(float(np.std(ratios)))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x, run1_means, color="#F28E2B", alpha=0.85)
    ax.errorbar(x, run1_means, yerr=run1_errs, fmt="none", ecolor="black", capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=25, ha="right")
    ax.set_ylabel("ratio of run_length=1")
    ax.set_title("Run Length = 1 Ratio (mean ± std across seeds)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "q3_runlen_ratio.png"), dpi=200)
    fig.savefig(os.path.join(out_dir, "q3_runlen_ratio.pdf"))
    plt.close(fig)

    # -------- Run length vs geometric baseline (small multiples) --------
    bins = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", ">10"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharey=True)
    axes = axes.flatten()
    for idx, task in enumerate(tasks):
        # observed ratio (mean across seeds)
        ratios = []
        run_means_task = []
        for d in seed_data:
            r = d["tasks"][task]["run_length_hist_ratio"]
            ratios.append([r.get(b, 0.0) for b in bins])
            run_means_task.append(d["tasks"][task]["run_mean"])
        ratios = np.array(ratios)
        mean_ratio = ratios.mean(axis=0)
        # geometric expected with p = 1/mean
        mean_run = float(np.mean(run_means_task))
        p = 1.0 / max(mean_run, 1e-6)
        geom = [p * (1 - p) ** (k - 1) for k in range(1, 11)]
        geom.append((1 - sum(geom)))
        ax = axes[idx]
        ax.plot(bins, mean_ratio, marker="o", label="observed")
        ax.plot(bins, geom, marker="s", linestyle="--", label="geometric")
        ax.set_title(task)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("ratio")
    axes[3].set_ylabel("ratio")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle("Run Length vs Geometric Baseline (mean across seeds)", y=0.98)
    fig.tight_layout(rect=[0, 0, 0.96, 0.95])
    fig.savefig(os.path.join(out_dir, "q3_runlen_geom_fit.png"), dpi=200)
    fig.savefig(os.path.join(out_dir, "q3_runlen_geom_fit.pdf"))
    plt.close(fig)

    # -------- Window drift JSD/TV with CI --------
    jsd_means = []
    jsd_errs = []
    jsd_shuffle_means = []
    tv_means = []
    tv_errs = []
    tv_shuffle_means = []
    jsd_b_means = []
    for task in tasks:
        jsd_vals = [d["tasks"][task]["window_jsd_mean"] for d in seed_data]
        jsd_ci = [d["tasks"][task]["window_jsd_ci_95"] for d in seed_data]
        mean_val, err_low, err_high = _mean_ci(jsd_vals, jsd_ci)
        jsd_means.append(mean_val)
        jsd_errs.append([err_low, err_high])
        jsd_shuffle_means.append(float(np.mean([d["tasks"][task]["window_jsd_mean_shuffle"] for d in seed_data])))

        tv_vals = [d["tasks"][task]["window_tv_mean"] for d in seed_data]
        tv_mean = float(np.mean(tv_vals))
        tv_std = float(np.std(tv_vals))
        tv_means.append(tv_mean)
        tv_errs.append([tv_std, tv_std])
        tv_shuffle_means.append(float(np.mean([d["tasks"][task]["window_tv_mean_shuffle"] for d in seed_data])))

        jsd_b_means.append(float(np.mean([d["tasks"][task]["window_jsd_mean_method_b"] for d in seed_data])))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].errorbar(
        x,
        jsd_means,
        yerr=np.array(jsd_errs).T,
        fmt="o",
        capsize=4,
        color="#d62728",
        label="JSD(A) mean ± CI",
    )
    axes[0].plot(x, jsd_b_means, "s--", color="#9467bd", label="JSD(B) mean")
    axes[0].plot(x, jsd_shuffle_means, "k--", label="shuffle mean")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(tasks, rotation=25, ha="right")
    axes[0].set_ylabel("window JSD")
    axes[0].set_title("Window Drift (JSD) with CI")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=9)

    axes[1].errorbar(
        x,
        tv_means,
        yerr=np.array(tv_errs).T,
        fmt="o",
        capsize=4,
        color="#ff7f0e",
        label="TV mean ± std",
    )
    axes[1].plot(x, tv_shuffle_means, "k--", label="shuffle mean")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(tasks, rotation=25, ha="right")
    axes[1].set_ylabel("window TV")
    axes[1].set_title("Window Drift (TV) with CI")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "q3_window_drift_ci.png"), dpi=200)
    fig.savefig(os.path.join(out_dir, "q3_window_drift_ci.pdf"))
    plt.close(fig)

    # -------- Window size sensitivity (JSD) --------
    window_sizes = ["64", "128", "256"]
    fig, ax = plt.subplots(figsize=(10, 5))
    for task in tasks:
        vals = []
        for w in window_sizes:
            ws_vals = []
            for d in seed_data:
                ws = d["tasks"][task]["window_stats"].get(w, {})
                ws_vals.append(ws.get("window_jsd_mean", 0.0))
            vals.append(float(np.mean(ws_vals)))
        ax.plot(window_sizes, vals, marker="o", label=task)
    ax.set_xlabel("window size")
    ax.set_ylabel("window JSD (mean across seeds)")
    ax.set_title("Window Size Sensitivity (JSD)")
    ax.grid(alpha=0.3)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "q3_window_sensitivity.png"), dpi=200)
    fig.savefig(os.path.join(out_dir, "q3_window_sensitivity.pdf"))
    plt.close(fig)

    # -------- Summary 3-panel (clean) --------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].bar(x, switch_means, color="#4C78A8", alpha=0.85)
    axes[0].errorbar(x, switch_means, yerr=np.array(switch_errs).T, fmt="none", ecolor="black", capsize=3)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(tasks, rotation=25, ha="right")
    axes[0].set_title("Switch Rate")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(x, run_means, color="#59A14F", alpha=0.85)
    axes[1].errorbar(x, run_means, yerr=np.array(run_errs).T, fmt="none", ecolor="black", capsize=3)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(tasks, rotation=25, ha="right")
    axes[1].set_title("Run Mean")
    axes[1].grid(axis="y", alpha=0.3)

    axes[2].bar(x, jsd_means, color="#B279A2", alpha=0.85, label="JSD (A)")
    axes[2].errorbar(x, jsd_means, yerr=np.array(jsd_errs).T, fmt="none", ecolor="black", capsize=3)
    axes[2].plot(x, jsd_b_means, "s--", color="#9467bd", label="JSD (B)")
    axes[2].plot(x, jsd_shuffle_means, "k--", label="Shuffle")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(tasks, rotation=25, ha="right")
    axes[2].set_title("Window Drift (JSD)")
    axes[2].grid(axis="y", alpha=0.3)
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "q3_summary_main.png"), dpi=200)
    fig.savefig(os.path.join(out_dir, "q3_summary_main.pdf"))
    plt.close(fig)

    print("Saved figures to:", out_dir)


if __name__ == "__main__":
    main()
