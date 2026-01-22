#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q6 图表绘制脚本（与 analyze_q6_load_concentration.py 配套）

用途：
- 读取 q6_results.json
- 生成论文风格的多子图拼接（4x3）

输出示例：
- n_eff_grid.svg：任务 x (soft/hard) 的 n_eff 主图
- gini_grid.svg：任务 x (soft/hard) 的 Gini / Max-Share
- topn_coverage_grid.svg：Top-N 覆盖率曲线
- lorenz_grid.svg：Lorenz 曲线
- shuffle_vs_real_soft.svg / shuffle_vs_real_hard.svg：置乱基线对比
"""

import argparse
import json
import os
from typing import Dict, Any, List

import numpy as np
import matplotlib.pyplot as plt


# ===========================================================================
# 工具函数
# ===========================================================================

def load_results(json_path: str) -> Dict[str, Any]:
    """读取 q6_results.json"""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: str) -> None:
    """确保输出目录存在"""
    os.makedirs(path, exist_ok=True)


def default_task_order(tasks: List[str]) -> List[str]:
    """
    固定的任务顺序（方便对齐多图）：
    wiki / gsm8k / humaneval / cmrc2018 / piqa / winogrande
    """
    preferred = ["wiki", "gsm8k", "humaneval", "cmrc2018", "piqa", "winogrande"]
    ordered = [t for t in preferred if t in tasks]
    for t in tasks:
        if t not in ordered:
            ordered.append(t)
    return ordered


def panel_position(task_idx: int, mode: str) -> tuple:
    """
    将 6 个任务映射到 4x3 网格：
    - 前两行：soft
    - 后两行：hard
    每行 3 列
    """
    base_row = 0 if mode == "soft" else 2
    row = base_row + (0 if task_idx < 3 else 1)
    col = task_idx % 3
    return row, col


def _set_panel_title(ax, task: str, mode: str) -> None:
    """设置子图标题"""
    ax.set_title(f"{task} ({mode})", fontsize=9)


# ===========================================================================
# 绘图：n_eff 主图（4x3）
# ===========================================================================

def plot_n_eff_grid(results: Dict[str, Any], output_dir: str) -> None:
    """
    每个子图画两根柱：
    - Entropy n_eff
    - Simpson n_eff
    soft/hard 分开显示
    """
    e12 = results["evidence"]["E1_E2_metrics"]["tasks"]
    tasks = default_task_order(list(e12.keys()))

    # 先找到全局最大值，统一 y 轴
    max_val = 0.0
    for task in tasks:
        for mode in ["soft", "hard"]:
            if mode not in e12[task]:
                continue
            if mode == "soft":
                stats = e12[task][mode]["sample_level"]["renorm_60"]
            else:
                stats = e12[task][mode]["sample_level"]
            max_val = max(max_val, stats["entropy"]["ci_high"], stats["simpson"]["ci_high"])

    fig, axs = plt.subplots(4, 3, figsize=(12, 10))
    for task_idx, task in enumerate(tasks):
        for mode in ["soft", "hard"]:
            row, col = panel_position(task_idx, mode)
            ax = axs[row, col]

            if mode == "soft":
                stats = e12[task][mode]["sample_level"]["renorm_60"]
            else:
                stats = e12[task][mode]["sample_level"]

            # 两个柱：entropy / simpson
            means = [stats["entropy"]["mean"], stats["simpson"]["mean"]]
            errs = [
                [means[0] - stats["entropy"]["ci_low"], stats["entropy"]["ci_high"] - means[0]],
                [means[1] - stats["simpson"]["ci_low"], stats["simpson"]["ci_high"] - means[1]],
            ]

            ax.bar([0, 1], means, color=["#4C78A8", "#F58518"], width=0.6)
            ax.errorbar([0, 1], means, yerr=np.array(errs).T, fmt="none", ecolor="#333333", capsize=3)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["entropy", "simpson"], fontsize=8)
            ax.set_ylim(0, max_val * 1.05)
            ax.grid(alpha=0.2)
            _set_panel_title(ax, task, mode)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "n_eff_grid.svg"))
    plt.close(fig)


# ===========================================================================
# 绘图：Gini / Max-Share（4x3）
# ===========================================================================

def plot_gini_grid(results: Dict[str, Any], output_dir: str) -> None:
    """
    每个子图画两根柱：
    - Gini
    - Max-Share
    """
    e12 = results["evidence"]["E1_E2_metrics"]["tasks"]
    tasks = default_task_order(list(e12.keys()))

    fig, axs = plt.subplots(4, 3, figsize=(12, 10))
    for task_idx, task in enumerate(tasks):
        for mode in ["soft", "hard"]:
            row, col = panel_position(task_idx, mode)
            ax = axs[row, col]

            conc = e12[task][mode]["concentration"]
            means = [conc["gini"], conc["max_share"]]

            ax.bar([0, 1], means, color=["#54A24B", "#E45756"], width=0.6)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["gini", "max"], fontsize=8)
            ax.set_ylim(0, 1.0)
            ax.grid(alpha=0.2)
            _set_panel_title(ax, task, mode)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "gini_grid.svg"))
    plt.close(fig)


# ===========================================================================
# 绘图：Top-N 覆盖率（4x3）
# ===========================================================================

def plot_topn_grid(results: Dict[str, Any], output_dir: str) -> None:
    """每个子图画 Top-N 覆盖率曲线（含置乱基线）"""
    e12 = results["evidence"]["E1_E2_metrics"]["tasks"]
    e5 = results["evidence"].get("E5_label_shuffle", {}).get("tasks", {})
    tasks = default_task_order(list(e12.keys()))

    fig, axs = plt.subplots(4, 3, figsize=(12, 10))
    for task_idx, task in enumerate(tasks):
        for mode in ["soft", "hard"]:
            row, col = panel_position(task_idx, mode)
            ax = axs[row, col]

            topn = e12[task][mode]["concentration"]["topn"]
            x = [int(k.split("_")[1]) for k in topn.keys()]
            y = [topn[k] for k in topn.keys()]

            # 真实曲线
            ax.plot(x, y, marker="o", linewidth=1.5, label="real")

            # 置乱基线（如果存在）
            if task in e5 and mode in e5[task] and "topn" in e5[task][mode]:
                y_shuf = [e5[task][mode]["topn"][f"top_{n}"]["mean"] for n in x]
                ax.plot(x, y_shuf, marker="x", linewidth=1.2, linestyle="--", label="shuffle")
            ax.set_xticks(x)
            ax.set_ylim(0, 1.0)
            ax.grid(alpha=0.2)
            _set_panel_title(ax, task, mode)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "topn_coverage_grid.svg"))
    plt.close(fig)


# ===========================================================================
# 绘图：Lorenz 曲线（4x3）
# ===========================================================================

def plot_lorenz_grid(results: Dict[str, Any], output_dir: str) -> None:
    """每个子图画 Lorenz 曲线（含置乱基线）"""
    e12 = results["evidence"]["E1_E2_metrics"]["tasks"]
    e5 = results["evidence"].get("E5_label_shuffle", {}).get("tasks", {})
    tasks = default_task_order(list(e12.keys()))

    fig, axs = plt.subplots(4, 3, figsize=(12, 10))
    for task_idx, task in enumerate(tasks):
        for mode in ["soft", "hard"]:
            row, col = panel_position(task_idx, mode)
            ax = axs[row, col]

            lorenz = e12[task][mode]["concentration"]["lorenz"]
            x = lorenz["x"]
            y = lorenz["y"]

            # 对角线 = 完全均匀
            ax.plot([0, 1], [0, 1], linestyle="--", color="#999999", linewidth=1)
            ax.plot(x, y, linewidth=1.5, label="real")

            # 置乱基线（如果存在）
            if task in e5 and mode in e5[task] and "lorenz" in e5[task][mode]:
                x_shuf = e5[task][mode]["lorenz"]["x"]
                y_shuf = e5[task][mode]["lorenz"]["y"]
                ax.plot(x_shuf, y_shuf, linestyle="--", linewidth=1.2, label="shuffle")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.2)
            _set_panel_title(ax, task, mode)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "lorenz_grid.svg"))
    plt.close(fig)


# ===========================================================================
# 绘图：置乱基线对比（soft / hard 分开）
# ===========================================================================

def plot_shuffle_vs_real(results: Dict[str, Any], output_dir: str, mode: str) -> None:
    """对比真实 vs 置乱基线（n_eff 熵口径）"""
    e12 = results["evidence"]["E1_E2_metrics"]["tasks"]
    e5 = results["evidence"]["E5_label_shuffle"]["tasks"]
    tasks = default_task_order(list(e12.keys()))

    real_means = []
    real_errs = []
    shuf_means = []
    shuf_errs = []

    for task in tasks:
        if mode == "soft":
            real_stats = e12[task][mode]["sample_level"]["renorm_60"]["entropy"]
        else:
            real_stats = e12[task][mode]["sample_level"]["entropy"]
        shuf_stats = e5[task][mode]["entropy"]

        real_means.append(real_stats["mean"])
        real_errs.append([real_stats["mean"] - real_stats["ci_low"], real_stats["ci_high"] - real_stats["mean"]])

        shuf_means.append(shuf_stats["mean"])
        shuf_errs.append([shuf_stats["mean"] - shuf_stats["ci_low"], shuf_stats["ci_high"] - shuf_stats["mean"]])

    x = np.arange(len(tasks))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - width / 2, real_means, width, label="real", color="#4C78A8")
    ax.bar(x + width / 2, shuf_means, width, label="shuffle", color="#F58518")
    ax.errorbar(x - width / 2, real_means, yerr=np.array(real_errs).T, fmt="none", ecolor="#333", capsize=3)
    ax.errorbar(x + width / 2, shuf_means, yerr=np.array(shuf_errs).T, fmt="none", ecolor="#333", capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=30, ha="right")
    ax.set_ylabel("n_eff (entropy)")
    ax.set_title(f"Shuffle vs Real ({mode})")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"shuffle_vs_real_{mode}.svg"))
    plt.close(fig)


# ===========================================================================
# 绘图：Gini 的置乱基线对比
# ===========================================================================

def plot_gini_shuffle_vs_real(results: Dict[str, Any], output_dir: str, mode: str) -> None:
    """对比真实 vs 置乱基线（Gini）"""
    e12 = results["evidence"]["E1_E2_metrics"]["tasks"]
    e5 = results["evidence"].get("E5_label_shuffle", {}).get("tasks", {})
    tasks = default_task_order(list(e12.keys()))

    real_means = []
    shuf_means = []
    shuf_errs = []

    for task in tasks:
        real_means.append(e12[task][mode]["concentration"]["gini"])
        if task in e5 and mode in e5[task] and "gini" in e5[task][mode]:
            stats = e5[task][mode]["gini"]
            shuf_means.append(stats["mean"])
            shuf_errs.append([stats["mean"] - stats["ci_low"], stats["ci_high"] - stats["mean"]])
        else:
            shuf_means.append(0.0)
            shuf_errs.append([0.0, 0.0])

    x = np.arange(len(tasks))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - width / 2, real_means, width, label="real", color="#54A24B")
    ax.bar(x + width / 2, shuf_means, width, label="shuffle", color="#E45756")
    ax.errorbar(x + width / 2, shuf_means, yerr=np.array(shuf_errs).T,
                fmt="none", ecolor="#333", capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=30, ha="right")
    ax.set_ylabel("Gini")
    ax.set_ylim(0, 1.0)
    ax.set_title(f"Gini: Shuffle vs Real ({mode})")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"gini_shuffle_vs_real_{mode}.svg"))
    plt.close(fig)


# ===========================================================================
# 绘图：长度敏感性（segment_len）
# ===========================================================================

def plot_segment_len_sensitivity(results: Dict[str, Any], output_dir: str, mode: str) -> None:
    """
    画 segment_len 对 n_eff 的敏感性曲线（每个任务一条线）。
    - soft：使用 sample_level renorm_60 entropy 的均值
    - hard：使用 sample_level entropy 的均值
    """
    sensitivity = results.get("sensitivity", {}).get("segment_len", {})
    if not sensitivity:
        return

    # 只取有 metrics 的长度
    seg_lens = []
    for k, v in sensitivity.items():
        if "metrics" in v:
            try:
                seg_lens.append(int(k))
            except ValueError:
                continue
    if not seg_lens:
        return

    seg_lens = sorted(seg_lens)
    # 取第一组的任务列表
    first_metrics = sensitivity[str(seg_lens[0])]["metrics"]["tasks"]
    tasks = default_task_order(list(first_metrics.keys()))

    fig, ax = plt.subplots(figsize=(10, 5))
    for task in tasks:
        y_vals = []
        for seg in seg_lens:
            metrics = sensitivity[str(seg)]["metrics"]["tasks"][task]
            if mode == "soft":
                y = metrics["soft"]["sample_level"]["renorm_60"]["entropy"]["mean"]
            else:
                y = metrics["hard"]["sample_level"]["entropy"]["mean"]
            y_vals.append(y)
        ax.plot(seg_lens, y_vals, marker="o", linewidth=1.5, label=task)

    ax.set_xlabel("segment_len")
    ax.set_ylabel("n_eff (entropy)")
    ax.set_title(f"Segment Length Sensitivity ({mode})")
    ax.grid(alpha=0.2)
    ax.legend(ncol=2, fontsize=8)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"segment_len_sensitivity_{mode}.svg"))
    plt.close(fig)
# ===========================================================================
# 主函数
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Q6 图表绘制")
    parser.add_argument("--input_json", type=str, required=True, help="q6_results.json 路径")
    parser.add_argument("--output_dir", type=str, required=True, help="图表输出目录")
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    results = load_results(args.input_json)

    plot_n_eff_grid(results, args.output_dir)
    plot_gini_grid(results, args.output_dir)
    plot_topn_grid(results, args.output_dir)
    plot_lorenz_grid(results, args.output_dir)
    plot_shuffle_vs_real(results, args.output_dir, "soft")
    plot_shuffle_vs_real(results, args.output_dir, "hard")
    plot_gini_shuffle_vs_real(results, args.output_dir, "soft")
    plot_gini_shuffle_vs_real(results, args.output_dir, "hard")
    plot_segment_len_sensitivity(results, args.output_dir, "soft")
    plot_segment_len_sensitivity(results, args.output_dir, "hard")


if __name__ == "__main__":
    main()
