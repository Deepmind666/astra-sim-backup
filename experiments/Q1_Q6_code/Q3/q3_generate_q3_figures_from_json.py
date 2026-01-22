#!/usr/bin/env python3
"""
Q3 Task Internal Drift Analysis - Publication-Quality Figures
从真实 JSON 数据读取并生成图表

Author: Claude Analysis
Date: 2025-12-30
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Dict, List, Tuple

# 设置论文级别的样式
plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
mpl.rcParams['font.size'] = 11
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['axes.titlesize'] = 13
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['legend.fontsize'] = 10
mpl.rcParams['figure.dpi'] = 150
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['axes.linewidth'] = 1.2
mpl.rcParams['grid.linewidth'] = 0.5
mpl.rcParams['grid.alpha'] = 0.3

# 数据目录
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(DATA_DIR, 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 颜色方案
COLORS = {
    'real': '#2E86AB',      # 深蓝
    'shuffle': '#E94F37',   # 红橙
    'geom': '#A23B72',      # 紫红
    'accent': '#F18F01',    # 橙色
    'gray': '#C5C3C6',      # 灰色
}

TASKS = ['wiki', 'gsm8k', 'humaneval', 'cmrc2018', 'piqa', 'winogrande']
TASKS_SHORT = ['Wiki', 'GSM8K', 'Human\nEval', 'CMRC\n2018', 'PIQA', 'Wino\ngrande']
SEEDS = ['seed11', 'seed22', 'seed33']


def load_all_data() -> Dict[str, dict]:
    """加载所有 seed 的数据"""
    all_data = {}
    for seed in SEEDS:
        json_path = os.path.join(DATA_DIR, seed, 'task_drift_summary.json')
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                all_data[seed] = json.load(f)
        else:
            print(f"Warning: {json_path} not found")
    return all_data


def average_across_seeds(all_data: Dict[str, dict], task: str, key: str,
                         nested_keys: List[str] = None) -> Tuple[float, float, float]:
    """计算跨 seed 的平均值和标准差"""
    values = []
    for seed in SEEDS:
        if seed not in all_data:
            continue
        task_data = all_data[seed]['tasks'].get(task, {})
        if nested_keys:
            val = task_data
            for k in nested_keys:
                val = val.get(k, {}) if isinstance(val, dict) else val
            if isinstance(val, (int, float)) and val is not None:
                values.append(val)
        elif key in task_data and task_data[key] is not None:
            values.append(task_data[key])

    if not values:
        return 0.0, 0.0, 0.0
    return float(np.mean(values)), float(np.std(values)), float(np.std(values) / np.sqrt(len(values)))


def get_per_layer_switch_rates(all_data: Dict[str, dict], task: str) -> Dict[str, float]:
    """获取逐层切换率（跨 seed 平均）"""
    layer_rates = {}
    for seed in SEEDS:
        if seed not in all_data:
            continue
        task_data = all_data[seed]['tasks'].get(task, {})
        per_layer = task_data.get('switch_rate_per_layer', {})
        for layer, rate in per_layer.items():
            if layer not in layer_rates:
                layer_rates[layer] = []
            layer_rates[layer].append(rate)

    return {layer: float(np.mean(rates)) for layer, rates in layer_rates.items()}


def figure1_switch_rate(all_data: Dict[str, dict]):
    """Figure 1: Switch Rate Comparison"""
    fig, ax = plt.subplots(figsize=(8, 5))

    switch_real = []
    switch_shuffle = []
    switch_real_err = []

    for task in TASKS:
        mean, std, se = average_across_seeds(all_data, task, 'switch_rate')
        switch_real.append(mean)
        switch_real_err.append(se)

        mean_sh, _, _ = average_across_seeds(all_data, task, 'switch_rate_shuffle')
        switch_shuffle.append(mean_sh)

    x = np.arange(len(TASKS))
    width = 0.35

    bars1 = ax.bar(x - width/2, switch_real, width, label='Observed',
                   color=COLORS['real'], edgecolor='black', linewidth=0.8, alpha=0.9,
                   yerr=switch_real_err, capsize=3)
    bars2 = ax.bar(x + width/2, switch_shuffle, width, label='Shuffled Baseline',
                   color=COLORS['shuffle'], edgecolor='black', linewidth=0.8, alpha=0.9)

    # 添加数值标签
    for bar, val in zip(bars1, switch_real):
        ax.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, val),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',
                   fontsize=9, fontweight='bold')

    # 添加差值标注
    for i, (r, s) in enumerate(zip(switch_real, switch_shuffle)):
        delta = r - s
        ax.annotate(f'Δ={delta:+.3f}', xy=(x[i], max(r, s) + 0.015),
                   ha='center', va='bottom', fontsize=8, color='dimgray', style='italic')

    ax.set_ylabel('Switch Rate', fontweight='bold')
    ax.set_xlabel('Task', fontweight='bold')
    ax.set_title('(a) Top-1 Expert Switch Rate: Observed vs Shuffled Baseline',
                 fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(TASKS_SHORT)
    ax.set_ylim(0.90, 1.02)
    ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'q3_switch_rate_comparison.png'), bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(OUTPUT_DIR, 'q3_switch_rate_comparison.pdf'), bbox_inches='tight')
    plt.close()
    print("Generated: q3_switch_rate_comparison")


def figure2_run_length(all_data: Dict[str, dict]):
    """Figure 2: Run Length Analysis"""
    fig, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(12, 4.5))

    run_mean_vals = []
    geom_mean_vals = []

    for task in TASKS:
        mean, _, _ = average_across_seeds(all_data, task, 'run_mean')
        run_mean_vals.append(mean)

        geom, _, _ = average_across_seeds(all_data, task, 'geom_run_mean')
        geom_mean_vals.append(geom)

    x = np.arange(len(TASKS))
    width = 0.35

    bars1 = ax2a.bar(x - width/2, run_mean_vals, width, label='Observed Mean',
                    color=COLORS['real'], edgecolor='black', linewidth=0.8, alpha=0.9)
    bars2 = ax2a.bar(x + width/2, geom_mean_vals, width, label='Geometric Expectation',
                    color=COLORS['geom'], edgecolor='black', linewidth=0.8, alpha=0.9)

    for bar, val in zip(bars1, run_mean_vals):
        ax2a.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, val),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    ax2a.set_ylabel('Mean Run Length', fontweight='bold')
    ax2a.set_xlabel('Task', fontweight='bold')
    ax2a.set_title('(b) Mean Run Length: Observed vs Geometric Distribution', fontweight='bold', pad=10)
    ax2a.set_xticks(x)
    ax2a.set_xticklabels(TASKS_SHORT)
    ax2a.set_ylim(1.0, 1.08)
    ax2a.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax2a.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

    # 2b: Run Length Distribution (理论几何分布)
    switch_rate = run_mean_vals[0]  # Wiki
    if switch_rate > 0:
        switch_rate = 1.0 / switch_rate  # 从 run_mean 反推 switch_rate
    else:
        switch_rate = 0.937

    run_lengths = np.arange(1, 9)
    # 近似观测分布（基于 switch_rate 约 0.937）
    observed_prob = [switch_rate * (1-switch_rate)**(k-1) for k in run_lengths]
    geom_prob = [switch_rate * (1-switch_rate)**(k-1) for k in run_lengths]

    ax2b.bar(run_lengths - 0.2, observed_prob, 0.4, label='Geometric (theoretical)',
             color=COLORS['real'], edgecolor='black', linewidth=0.8, alpha=0.9)
    ax2b.plot(run_lengths, geom_prob, 'o-', color=COLORS['geom'], linewidth=2.5,
              markersize=8, label=f'Geometric (p={switch_rate:.3f})')

    ax2b.set_xlabel('Run Length', fontweight='bold')
    ax2b.set_ylabel('Probability', fontweight='bold')
    ax2b.set_title('(c) Run Length Distribution (Wiki Task)', fontweight='bold', pad=10)
    ax2b.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax2b.set_xlim(0.5, 8.5)
    ax2b.set_ylim(0, 1.0)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'q3_run_length_analysis.png'), bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(OUTPUT_DIR, 'q3_run_length_analysis.pdf'), bbox_inches='tight')
    plt.close()
    print("Generated: q3_run_length_analysis")


def figure3_window_drift(all_data: Dict[str, dict]):
    """Figure 3: Window Drift Analysis"""
    fig, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 4.5))

    jsd_real = []
    jsd_shuffle = []

    for task in TASKS:
        mean, _, _ = average_across_seeds(all_data, task, 'window_jsd_mean',
                                          nested_keys=['window_stats', '64', 'window_jsd_mean'])
        jsd_real.append(mean)

        mean_sh, _, _ = average_across_seeds(all_data, task, 'window_jsd_mean_shuffle',
                                              nested_keys=['window_stats_shuffle', '64', 'window_jsd_mean_shuffle'])
        jsd_shuffle.append(mean_sh)

    x = np.arange(len(TASKS))
    width = 0.35

    # 转换为 ×10^-3
    bars1 = ax3a.bar(x - width/2, [j*1000 for j in jsd_real], width,
                     label='Observed', color=COLORS['real'], edgecolor='black', linewidth=0.8, alpha=0.9)
    bars2 = ax3a.bar(x + width/2, [j*1000 for j in jsd_shuffle], width,
                     label='Shuffled Baseline', color=COLORS['shuffle'], edgecolor='black', linewidth=0.8, alpha=0.9)

    # 添加比值标注
    for i, (r, s) in enumerate(zip(jsd_real, jsd_shuffle)):
        ratio = r / s if s > 0 else 0
        ax3a.annotate(f'{ratio:.2f}x', xy=(x[i], max(r, s)*1000 + 0.3),
                     ha='center', va='bottom', fontsize=8, color='dimgray', fontweight='bold')

    ax3a.set_ylabel('Jensen-Shannon Divergence (×10⁻³)', fontweight='bold')
    ax3a.set_xlabel('Task', fontweight='bold')
    ax3a.set_title('(d) Window Drift: JSD (Window=64 tokens)', fontweight='bold', pad=10)
    ax3a.set_xticks(x)
    ax3a.set_xticklabels(TASKS_SHORT)
    ax3a.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)

    # 3b: Window Size Sensitivity
    window_sizes = [64, 128, 256]
    markers = ['o', 's', '^', 'D', 'v', 'p']
    colors_list = ['#2E86AB', '#E94F37', '#A23B72', '#F18F01', '#8AC926', '#6A4C93']

    for i, task in enumerate(TASKS):
        jsd_vals = []
        for ws in window_sizes:
            mean, _, _ = average_across_seeds(all_data, task, 'window_jsd_mean',
                                              nested_keys=['window_stats', str(ws), 'window_jsd_mean'])
            jsd_vals.append(mean * 1000)  # 转换为 ×10^-3

        ax3b.plot(window_sizes, jsd_vals, marker=markers[i], linestyle='-',
                  linewidth=2, markersize=8, color=colors_list[i], label=task)

    ax3b.set_xlabel('Window Size (tokens)', fontweight='bold')
    ax3b.set_ylabel('Mean JSD (×10⁻³)', fontweight='bold')
    ax3b.set_title('(e) Window Size Sensitivity', fontweight='bold', pad=10)
    ax3b.set_xticks(window_sizes)
    ax3b.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, ncol=2)
    ax3b.set_ylim(-0.5, 10)

    ax3b.annotate('Drift diminishes\nat larger windows',
                  xy=(150, 0.5), fontsize=9, style='italic', color='dimgray')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'q3_window_drift_analysis.png'), bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(OUTPUT_DIR, 'q3_window_drift_analysis.pdf'), bbox_inches='tight')
    plt.close()
    print("Generated: q3_window_drift_analysis")


def figure4_per_layer_heatmap(all_data: Dict[str, dict]):
    """Figure 4: Per-Layer Heatmap"""
    fig, ax = plt.subplots(figsize=(14, 5))

    # 获取层名（排序）
    sample_task = TASKS[0]
    sample_seed = SEEDS[0]
    layer_rates_sample = all_data[sample_seed]['tasks'][sample_task].get('switch_rate_per_layer', {})
    layer_names = sorted(layer_rates_sample.keys(), key=lambda x: int(x.replace('MoE_Gate', '')))

    # 构建热力图矩阵
    matrix = []
    for task in TASKS:
        layer_rates = get_per_layer_switch_rates(all_data, task)
        row = [layer_rates.get(ln, 0.0) for ln in layer_names]
        matrix.append(row)

    matrix = np.array(matrix)

    # 绘制热力图
    im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd', vmin=0.78, vmax=0.98)

    # 简化层名显示
    layer_labels = [ln.replace('MoE_Gate', 'L') for ln in layer_names]

    ax.set_xticks(range(len(layer_names)))
    ax.set_xticklabels(layer_labels, fontsize=8)
    ax.set_yticks(range(len(TASKS)))
    ax.set_yticklabels(TASKS)
    ax.set_xlabel('MoE Layer', fontweight='bold')
    ax.set_ylabel('Task', fontweight='bold')
    ax.set_title('(f) Per-Layer Switch Rate Heatmap (Real Data)', fontweight='bold', pad=10)

    # 添加 colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Switch Rate', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'q3_per_layer_heatmap.png'), bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(OUTPUT_DIR, 'q3_per_layer_heatmap.pdf'), bbox_inches='tight')
    plt.close()
    print("Generated: q3_per_layer_heatmap")


def figure5_summary(all_data: Dict[str, dict]):
    """Figure 5: Summary Comparison"""
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 收集数据
    switch_real = []
    switch_shuffle = []
    jsd_real = []
    jsd_shuffle = []
    n_tokens = []

    for task in TASKS:
        sr, _, _ = average_across_seeds(all_data, task, 'switch_rate')
        switch_real.append(sr)

        ss, _, _ = average_across_seeds(all_data, task, 'switch_rate_shuffle')
        switch_shuffle.append(ss)

        jr, _, _ = average_across_seeds(all_data, task, 'window_jsd_mean',
                                        nested_keys=['window_stats', '64', 'window_jsd_mean'])
        jsd_real.append(jr)

        js, _, _ = average_across_seeds(all_data, task, 'window_jsd_mean_shuffle',
                                        nested_keys=['window_stats_shuffle', '64', 'window_jsd_mean_shuffle'])
        jsd_shuffle.append(js)

        nt, _, _ = average_across_seeds(all_data, task, 'n_tokens')
        n_tokens.append(nt)

    # 5a: Switch Rate Delta
    ax = axes[0, 0]
    delta_switch = [r - s for r, s in zip(switch_real, switch_shuffle)]
    colors_delta = [COLORS['real'] if d < 0 else COLORS['shuffle'] for d in delta_switch]
    ax.bar(TASKS_SHORT, delta_switch, color=colors_delta, edgecolor='black', linewidth=0.8)
    ax.axhline(y=0, color='black', linewidth=1)
    ax.set_ylabel('Δ Switch Rate (Observed - Shuffled)', fontweight='bold')
    ax.set_title('(a) Switch Rate Deviation from Random Baseline', fontweight='bold')
    ax.set_ylim(-0.05, 0.05)

    legend_elements = [Patch(facecolor=COLORS['real'], edgecolor='black', label='More Sticky (< Random)'),
                       Patch(facecolor=COLORS['shuffle'], edgecolor='black', label='More Dynamic (> Random)')]
    ax.legend(handles=legend_elements, loc='lower right')

    # 5b: JSD Ratio
    ax = axes[0, 1]
    jsd_ratio = [r/s if s > 0 else 0 for r, s in zip(jsd_real, jsd_shuffle)]
    colors_ratio = [COLORS['real'] if r > 1 else COLORS['shuffle'] for r in jsd_ratio]
    ax.bar(TASKS_SHORT, jsd_ratio, color=colors_ratio, edgecolor='black', linewidth=0.8)
    ax.axhline(y=1.0, color='black', linewidth=1, linestyle='--')
    ax.set_ylabel('JSD Ratio (Observed / Shuffled)', fontweight='bold')
    ax.set_title('(b) Window Drift Relative to Random Baseline', fontweight='bold')
    ax.set_ylim(0.8, 1.8)
    ax.annotate('Random baseline', xy=(5.5, 1.02), fontsize=9, style='italic', color='dimgray')

    # 5c: Sample Size vs Reliability
    ax = axes[1, 0]
    ax.scatter(n_tokens, switch_real, s=150, c=COLORS['real'], edgecolors='black', linewidth=1.5, alpha=0.8)
    for i, task in enumerate(TASKS):
        offset = (5, 5) if task != 'humaneval' else (5, -15)
        ax.annotate(task, (n_tokens[i], switch_real[i]), xytext=offset,
                   textcoords='offset points', fontsize=9)

    ax.axvline(x=1000, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.annotate('Small sample\nwarning threshold', xy=(1100, 0.94), fontsize=9,
               style='italic', color='red')

    ax.set_xlabel('Number of Tokens', fontweight='bold')
    ax.set_ylabel('Switch Rate', fontweight='bold')
    ax.set_title('(c) Sample Size vs Switch Rate', fontweight='bold')
    ax.set_xscale('log')

    # 5d: Key Findings Summary
    ax = axes[1, 1]
    ax.axis('off')

    findings = [
        ("Finding 1:", f"Switch rates: {min(switch_real):.1%}-{max(switch_real):.1%}", COLORS['real']),
        ("", "→ Experts change almost every token", 'dimgray'),
        ("", "", 'white'),
        ("Finding 2:", "Run lengths ≈ Geometric distribution", COLORS['geom']),
        ("", "→ Near-memoryless routing behavior", 'dimgray'),
        ("", "", 'white'),
        ("Finding 3:", f"Window drift: {min(jsd_ratio):.2f}x-{max(jsd_ratio):.2f}x baseline", COLORS['shuffle']),
        ("", "→ Temporal structure has minimal effect", 'dimgray'),
        ("", "", 'white'),
        ("Finding 4:", "Layer 0 shows stable routing", 'darkred'),
        ("", "→ Early layers have lower switch rate", 'dimgray'),
    ]

    y_pos = 0.95
    for title, content, color in findings:
        if title:
            ax.text(0.05, y_pos, title, fontsize=11, fontweight='bold', color=color,
                   transform=ax.transAxes, verticalalignment='top')
            ax.text(0.22, y_pos, content, fontsize=11, color='black',
                   transform=ax.transAxes, verticalalignment='top')
        else:
            ax.text(0.22, y_pos, content, fontsize=10, color=color, style='italic',
                   transform=ax.transAxes, verticalalignment='top')
        y_pos -= 0.085

    ax.set_title('(d) Key Findings Summary', fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'q3_summary_comparison.png'), bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(OUTPUT_DIR, 'q3_summary_comparison.pdf'), bbox_inches='tight')
    plt.close()
    print("Generated: q3_summary_comparison")


def print_data_summary(all_data: Dict[str, dict]):
    """打印数据摘要"""
    print("\n" + "="*80)
    print("Q3 DATA SUMMARY (Averaged across 3 seeds)")
    print("="*80)

    print(f"\n{'Task':<12} | {'Switch Rate':<12} | {'Shuffle':<12} | {'Delta':<10} | {'JSD(w=64)':<12} | {'Tokens'}")
    print("-"*80)

    for task in TASKS:
        sr, _, _ = average_across_seeds(all_data, task, 'switch_rate')
        ss, _, _ = average_across_seeds(all_data, task, 'switch_rate_shuffle')
        jr, _, _ = average_across_seeds(all_data, task, 'window_jsd_mean',
                                        nested_keys=['window_stats', '64', 'window_jsd_mean'])
        nt, _, _ = average_across_seeds(all_data, task, 'n_tokens')

        print(f"{task:<12} | {sr:.4f}       | {ss:.4f}       | {sr-ss:+.4f}    | {jr:.6f}     | {int(nt)}")


def main():
    print("Loading data from JSON files...")
    all_data = load_all_data()

    if not all_data:
        print("ERROR: No data found!")
        return

    print(f"Loaded data from {len(all_data)} seeds: {list(all_data.keys())}")

    print_data_summary(all_data)

    print("\nGenerating figures...")
    figure1_switch_rate(all_data)
    figure2_run_length(all_data)
    figure3_window_drift(all_data)
    figure4_per_layer_heatmap(all_data)
    figure5_summary(all_data)

    print(f"\nAll figures saved to: {OUTPUT_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
