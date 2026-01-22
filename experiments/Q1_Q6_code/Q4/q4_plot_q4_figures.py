#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q4 绘图脚本：生成“论文级”拼图式大图（每类图 12 个子图）

输出示例：
- position_drift_grid.svg  （位置漂移曲线 4x3）
- boundary_jsd_grid.svg    （边界对比 4x3）
- entropy_grid.svg         （熵曲线 4x3）

注：所有图为 SVG，适合论文/报告排版。
"""

import argparse
import json
import os
from typing import Dict, List

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# 1. 论文级样式设置（尽量“干净、统一、可读”）
# ---------------------------------------------------------------------------
def set_paper_style() -> None:
    """
    设置论文级的绘图风格。
    如果 Times New Roman 不存在，会自动回退到 STIX/DejaVu Serif。
    """
    mpl.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.5,
    })


# ---------------------------------------------------------------------------
# 2. 读取所有数据集的 q4_summary.json
# ---------------------------------------------------------------------------
def load_summaries(input_root: str, datasets: List[str]) -> Dict[str, Dict]:
    """
    从 input_root/dataset/q4_summary.json 读取所有 summary。
    如果 datasets 为空，则自动扫描所有子目录。
    """
    summaries = {}

    if not datasets:
        # 自动扫描子目录
        for name in os.listdir(input_root):
            sub = os.path.join(input_root, name)
            if not os.path.isdir(sub):
                continue
            cand = os.path.join(sub, "q4_summary.json")
            if os.path.isfile(cand):
                datasets.append(name)

    for d in datasets:
        path = os.path.join(input_root, d, "q4_summary.json")
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            summaries[d] = json.load(f)

    return summaries


# ---------------------------------------------------------------------------
# 3. 工具函数：自动识别“窗口大小列表”
# ---------------------------------------------------------------------------
def collect_window_sizes(summaries: Dict[str, Dict]) -> List[str]:
    """
    自动收集所有数据集中出现过的窗口大小（例如 16/32/64 或 64/128/256）。

    设计动机：
    - 我们的实验可能会用“短窗口”或“长窗口”两套参数。
    - 如果绘图脚本写死窗口列表，就会出现“图几乎为空”的误解。
    - 所以这里改成：从 q4_summary.json 里动态提取窗口大小。

    返回：
    - 字符串列表（例如 ["16", "32", "64"]），按数值从小到大排序。
    """
    sizes = set()

    # 遍历所有数据集，搜集 position->soft/hard 下的窗口 key
    for data in summaries.values():
        position = data.get("position", {})
        for dist_key in ("soft", "hard"):
            dist_data = position.get(dist_key, {})
            for key in dist_data.keys():
                try:
                    sizes.add(int(str(key)))
                except (TypeError, ValueError):
                    # 如果 key 不是数字（理论上不应发生），就忽略
                    continue

    # 如果没有任何窗口大小，就回退到默认值（兼容旧数据）
    if not sizes:
        return ["64", "128", "256"]

    return [str(x) for x in sorted(sizes)]


# ---------------------------------------------------------------------------
# 4. 工具函数：构造“4x3 排列”的子图顺序
# ---------------------------------------------------------------------------
def build_panel_order(datasets: List[str]) -> List[Dict[str, str]]:
    """
    构造 12 个子图的顺序，目标是“每个数据集的 soft/hard 紧挨着”。

    设计理由（给不会写代码的人也能懂）：
    - 我们要画 6 个数据集 × 2 种口径 = 12 个子图。
    - 论文里更舒服的排版是 4 列 × 3 行，而不是又长又细的 2×6。
    - 所以我们按顺序排：数据集1(soft), 数据集1(hard), 数据集2(soft), 数据集2(hard)...
      这样一行正好放两个数据集（4个子图），一共 3 行放完 6 个数据集。
    """
    panels: List[Dict[str, str]] = []
    for d in datasets:
        # 先放 soft，再放 hard，保证成对出现，阅读更直观
        panels.append({"dataset": d, "mode": "soft"})
        panels.append({"dataset": d, "mode": "hard"})
    return panels


# ---------------------------------------------------------------------------
# 5. 绘图：位置漂移曲线（4x3 = 12 子图）
# ---------------------------------------------------------------------------
def plot_position_drift_grid(summaries: Dict[str, Dict], output_dir: str) -> None:
    """
    每个数据集一个“soft + hard”成对子图：
    - 4列×3行，总共 12 个子图
    - 每个数据集两个子图紧挨着（soft 在左，hard 在右）
    - 每个子图画“当前数据中实际存在的窗口大小”曲线 + 虚线基线
    """
    set_paper_style()

    # 固定数据集顺序，方便对比
    order = ["wiki", "gsm8k", "humaneval", "cmrc2018", "piqa", "winogrande"]
    # 如果没有就用现有 keys
    datasets = [d for d in order if d in summaries] or list(summaries.keys())

    # 自动识别窗口大小（避免“空白图”）
    window_sizes = collect_window_sizes(summaries)

    # 颜色（根据窗口大小动态分配）
    # 说明：颜色是“按窗口排序”分配的，而不是固定某个数字
    color_pool = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd", "#8c564b"]
    colors = {}
    for i, w in enumerate(window_sizes):
        colors[w] = color_pool[i % len(color_pool)]

    # 4列×3行，总计 12 个子图
    n_rows, n_cols = 3, 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 8), sharey=True, sharex=True)

    panels = build_panel_order(datasets)

    for idx in range(n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        if idx >= len(panels):
            # 如果数据集不足 6 个，把多余子图隐藏掉
            ax.axis("off")
            continue

        panel = panels[idx]
        d = panel["dataset"]
        mode = panel["mode"]  # "soft" or "hard"

        data = summaries[d]
        pos_block = data["position"][mode]

        # 画曲线：每个窗口一条线，虚线是置乱基线
        for w in window_sizes:
            if w not in pos_block:
                continue
            curve = pos_block[w]["mean_curve"]
            base = pos_block[w]["baseline_mean_curve"]
            x = np.linspace(0, 1, num=len(curve))
            ax.plot(x, curve, color=colors[w], label=f"W{w}")
            if base:
                ax.plot(x, base, color=colors[w], linestyle="--", alpha=0.5)

        ax.set_title(f"{d} ({mode})")
        ax.grid(alpha=0.2)

        # 只给最左列加 y 轴标签，避免满屏重复
        if col == 0:
            ax.set_ylabel("JSD")

        # 只给最底行加 x 轴标签，避免满屏重复
        if row == n_rows - 1:
            ax.set_xlabel("Relative Position (0→1)")

    # 统一图例（放在图外）
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(3, len(window_sizes)),
               frameon=False, bbox_to_anchor=(0.5, 1.02))

    os.makedirs(output_dir, exist_ok=True)
    out_svg = os.path.join(output_dir, "position_drift_grid.svg")
    fig.tight_layout()
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. 绘图：边界 JSD 对比（4x3 = 12 子图）
# ---------------------------------------------------------------------------
def plot_boundary_grid(summaries: Dict[str, Dict], output_dir: str) -> None:
    """
    每个数据集两个子图（soft/hard 成对），画“真实边界 vs 随机边界”的均值对比。
    使用 4列×3行排版，保证每个数据集软硬口径并排，阅读更直观。
    """
    set_paper_style()

    order = ["wiki", "gsm8k", "humaneval", "cmrc2018", "piqa", "winogrande"]
    datasets = [d for d in order if d in summaries] or list(summaries.keys())

    # 4列×3行，总计 12 个子图
    n_rows, n_cols = 3, 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 7), sharey=True)

    panels = build_panel_order(datasets)

    for idx in range(n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        if idx >= len(panels):
            ax.axis("off")
            continue

        panel = panels[idx]
        d = panel["dataset"]
        mode = panel["mode"]

        data = summaries[d]["boundary"]
        source = data.get("source", "unknown")
        block = data[mode]

        real = block["real_mean"]
        rand = block["random_mean"]
        real_std = block["real_std"]
        rand_std = block["random_std"]
        n_real = block["n_real"]
        n_rand = block["n_random"]

        if n_real == 0 and n_rand == 0:
            # 重要：这里不是“画图失败”，而是“边界检测没有命中”
            # 为了避免“空白图”，我们显式写清楚原因
            ax.set_facecolor("#f7f7f7")
            ax.text(0.5, 0.62, "No boundary hits", ha="center", va="center", fontsize=9)
            ax.text(0.5, 0.42, "n_real=0, n_rand=0", ha="center", va="center", fontsize=8)
            ax.set_title(f"{d} ({mode})")
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            # 有边界时再画柱状图 + 误差条
            color_main = "#1f77b4" if mode == "soft" else "#ff7f0e"
            ax.bar([0, 1], [real, rand], yerr=[real_std, rand_std],
                   color=[color_main, "#cccccc"], capsize=3)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Real", "Random"])
            ax.set_title(f"{d} ({mode})")
            ax.grid(axis="y", alpha=0.2)
            # 在图内标注样本数 + 边界来源，避免口径混淆
            ymax = max(real + real_std, rand + rand_std, 1e-6)
            ax.text(0.5, ymax * 1.05, f"n_real={n_real}, n_rand={n_rand}",
                    ha="center", va="bottom", fontsize=8)
            if source in ("pseudo", "mixed"):
                ax.text(0.5, ymax * 1.18, f"source={source}", ha="center", va="bottom", fontsize=8)

        # 只给最左列加 y 轴标签，避免满屏重复
        if col == 0:
            ax.set_ylabel("JSD")

    os.makedirs(output_dir, exist_ok=True)
    out_svg = os.path.join(output_dir, "boundary_jsd_grid.svg")
    fig.tight_layout()
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 7. 绘图：熵曲线（4x3 = 12 子图）
# ---------------------------------------------------------------------------
def plot_entropy_grid(summaries: Dict[str, Dict], output_dir: str) -> None:
    """
    每个数据集两个子图（soft/hard 成对）：
    - 4列×3行，总共 12 个子图
    - 每个数据集两个子图紧挨着，方便看“软/硬口径”的差异
    """
    set_paper_style()

    order = ["wiki", "gsm8k", "humaneval", "cmrc2018", "piqa", "winogrande"]
    datasets = [d for d in order if d in summaries] or list(summaries.keys())

    # 自动识别窗口大小（与位置漂移一致）
    window_sizes = collect_window_sizes(summaries)

    # 颜色池与位置漂移保持一致
    color_pool = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd", "#8c564b"]
    colors = {}
    for i, w in enumerate(window_sizes):
        colors[w] = color_pool[i % len(color_pool)]

    # 4列×3行，总计 12 个子图
    n_rows, n_cols = 3, 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 8), sharey=True, sharex=True)

    panels = build_panel_order(datasets)

    for idx in range(n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        if idx >= len(panels):
            ax.axis("off")
            continue

        panel = panels[idx]
        d = panel["dataset"]
        mode = panel["mode"]

        ent_block = summaries[d]["entropy"][mode]
        for w in window_sizes:
            if w not in ent_block:
                continue
            curve = ent_block[w]["mean_curve"]
            x = np.linspace(0, 1, num=len(curve))
            ax.plot(x, curve, color=colors[w], label=f"W{w}")

        ax.set_title(f"{d} ({mode})")
        ax.grid(alpha=0.2)

        # 只给最左列加 y 轴标签，避免满屏重复
        if col == 0:
            ax.set_ylabel("Entropy")

        # 只给最底行加 x 轴标签，避免满屏重复
        if row == n_rows - 1:
            ax.set_xlabel("Relative Position (0→1)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(3, len(window_sizes)),
               frameon=False, bbox_to_anchor=(0.5, 1.02))

    os.makedirs(output_dir, exist_ok=True)
    out_svg = os.path.join(output_dir, "entropy_grid.svg")
    fig.tight_layout()
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. 命令行入口
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Q4 论文级拼图绘图脚本")
    parser.add_argument("--input_root", type=str, required=True, help="包含各数据集结果的根目录")
    parser.add_argument("--output_dir", type=str, required=True, help="输出图表目录")
    parser.add_argument("--datasets", type=str, default="", help="数据集列表（逗号分隔，可空）")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    summaries = load_summaries(args.input_root, datasets)

    # 三类大拼图
    plot_position_drift_grid(summaries, args.output_dir)
    plot_boundary_grid(summaries, args.output_dir)
    plot_entropy_grid(summaries, args.output_dir)


if __name__ == "__main__":
    main()
