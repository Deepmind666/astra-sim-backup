#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q5 路由置信度分析 - 精美图表生成脚本

生成5个证据的可视化图表，用于GitHub issue和论文
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# 颜色方案
COLORS = {
    'gsm8k': '#E74C3C',      # 红色 - 数学
    'wiki': '#3498DB',       # 蓝色 - 百科
    'cmrc2018': '#27AE60',   # 绿色 - 中文阅读
    'piqa': '#9B59B6',       # 紫色 - 常识问答
    'winogrande': '#F39C12', # 橙色 - 代词消歧
    'humaneval': '#1ABC9C',  # 青色 - 代码
}

TASK_LABELS = {
    'gsm8k': 'GSM8K',
    'wiki': 'WikiText',
    'cmrc2018': 'CMRC2018',
    'piqa': 'PIQA',
    'winogrande': 'Winogrande',
    'humaneval': 'HumanEval',
}

# 带描述的标签（用于需要详细说明的地方）
TASK_LABELS_FULL = {
    'gsm8k': 'GSM8K (Math)',
    'wiki': 'WikiText (LM)',
    'cmrc2018': 'CMRC2018 (Chinese RC)',
    'piqa': 'PIQA (Commonsense)',
    'winogrande': 'Winogrande (Coreference)',
    'humaneval': 'HumanEval (Code)',
}


def load_results(json_path: str) -> dict:
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_evidence1_task_comparison(results: dict, output_dir: Path):
    """证据1：任务间置信度对比 - 多面板图"""
    e1 = results['evidence']['E1_task_comparison']
    tasks = e1['ranking']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 数据准备
    top1_means = [e1['tasks'][t]['top1_prob']['mean'] for t in tasks]
    margin_means = [e1['tasks'][t]['margin']['mean'] for t in tasks]
    entropy_means = [e1['tasks'][t]['entropy']['mean'] for t in tasks]

    x = np.arange(len(tasks))
    colors = [COLORS[t] for t in tasks]

    # Panel A: Top-1 Probability (去掉误差线)
    ax1 = axes[0]
    bars1 = ax1.bar(x, top1_means, color=colors,
                    alpha=0.8, edgecolor='black', linewidth=0.5)
    ax1.set_ylabel('Top-1 Probability', fontsize=12, fontweight='bold')
    ax1.set_title('(A) Routing Confidence', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([TASK_LABELS[t] for t in tasks], fontsize=9, rotation=30, ha='right')
    ax1.axhline(y=1/60, color='gray', linestyle='--', linewidth=1.5, label='Random (1/60)')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_ylim(0, 0.18)
    ax1.grid(axis='y', alpha=0.3)
    # 添加数值标签
    for i, v in enumerate(top1_means):
        ax1.text(i, v + 0.005, f'{v:.3f}', ha='center', fontsize=8)

    # Panel B: Margin
    ax2 = axes[1]
    ax2.bar(x, margin_means, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax2.set_ylabel('Margin (Top1 - Top2)', fontsize=12, fontweight='bold')
    ax2.set_title('(B) Decision Boundary Clarity', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([TASK_LABELS[t] for t in tasks], fontsize=9, rotation=30, ha='right')
    ax2.set_ylim(0, 0.09)
    ax2.grid(axis='y', alpha=0.3)
    for i, v in enumerate(margin_means):
        ax2.text(i, v + 0.003, f'{v:.3f}', ha='center', fontsize=8)

    # Panel C: Normalized Entropy
    ax3 = axes[2]
    ax3.bar(x, entropy_means, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax3.set_ylabel('Normalized Entropy', fontsize=12, fontweight='bold')
    ax3.set_title('(C) Routing Uncertainty', fontsize=13, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([TASK_LABELS[t] for t in tasks], fontsize=9, rotation=30, ha='right')
    ax3.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5, label='Max Entropy')
    ax3.legend(loc='lower right', fontsize=9)
    ax3.set_ylim(0.82, 0.96)
    ax3.grid(axis='y', alpha=0.3)
    for i, v in enumerate(entropy_means):
        ax3.text(i, v + 0.005, f'{v:.3f}', ha='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / 'E1_task_comparison.png')
    plt.savefig(output_dir / 'E1_task_comparison.svg')
    plt.close()
    print("[OK] E1_task_comparison saved")


def plot_evidence2_layer_consistency(results: dict, output_dir: Path):
    """证据2：多层一致性热力图"""
    e2 = results['evidence']['E2_layer_consistency']

    # 提取相关系数矩阵
    layers = [0, 4, 8, 12, 16, 20]
    n = len(layers)
    corr_matrix = np.ones((n, n))

    for key, val in e2['layer_correlations'].items():
        # 解析 layer0_vs_layer4 格式
        parts = key.replace('layer', '').split('_vs_')
        i, j = int(parts[0]), int(parts[1])
        idx_i, idx_j = layers.index(i), layers.index(j)
        corr_matrix[idx_i, idx_j] = val
        corr_matrix[idx_j, idx_i] = val

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr_matrix, cmap='RdYlBu_r', vmin=0, vmax=1)

    # 添加数值标注
    for i in range(n):
        for j in range(n):
            color = 'white' if corr_matrix[i, j] > 0.5 else 'black'
            ax.text(j, i, f'{corr_matrix[i, j]:.2f}', ha='center', va='center',
                   fontsize=11, fontweight='bold', color=color)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f'Layer {l}' for l in layers], fontsize=11)
    ax.set_yticklabels([f'Layer {l}' for l in layers], fontsize=11)
    ax.set_title('Inter-Layer Confidence Correlation\n(avg_corr=0.515, Moderate Consistency)',
                 fontsize=13, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Pearson Correlation', fontsize=11)

    plt.tight_layout()
    plt.savefig(output_dir / 'E2_layer_consistency.png')
    plt.savefig(output_dir / 'E2_layer_consistency.svg')
    plt.close()
    print("[OK] E2_layer_consistency saved")


def plot_evidence3_confidence_vs_switch(results: dict, output_dir: Path):
    """证据3：置信度vs切换率"""
    e3 = results['evidence']['E3_confidence_vs_switch']

    bins_order = ['very_low', 'low', 'medium', 'high']
    bin_labels = ['Very Low\n(<0.05)', 'Low\n(0.05-0.10)', 'Medium\n(0.10-0.20)', 'High\n(>0.20)']

    confidences = [e3['bins'][b]['avg_confidence'] for b in bins_order]
    switch_rates = [e3['bins'][b]['switch_rate'] for b in bins_order]
    n_tokens = [e3['bins'][b]['n_tokens'] for b in bins_order]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(bins_order))
    width = 0.35

    # 双Y轴
    ax2 = ax.twinx()

    bars1 = ax.bar(x - width/2, confidences, width, label='Avg Confidence',
                   color='#3498DB', alpha=0.8, edgecolor='black')
    bars2 = ax2.bar(x + width/2, switch_rates, width, label='Switch Rate',
                    color='#E74C3C', alpha=0.8, edgecolor='black')

    ax.set_xlabel('Confidence Bin', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Confidence', fontsize=12, fontweight='bold', color='#3498DB')
    ax2.set_ylabel('Layer Switch Rate', fontsize=12, fontweight='bold', color='#E74C3C')

    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, fontsize=10)
    ax.tick_params(axis='y', labelcolor='#3498DB')
    ax2.tick_params(axis='y', labelcolor='#E74C3C')

    ax2.set_ylim(0.96, 1.0)

    # 添加数据标签
    for i, (conf, sw, n) in enumerate(zip(confidences, switch_rates, n_tokens)):
        ax.text(i - width/2, conf + 0.01, f'{conf:.3f}', ha='center', fontsize=9)
        ax2.text(i + width/2, sw + 0.002, f'{sw:.1%}', ha='center', fontsize=9)
        ax.text(i, -0.02, f'n={n:,}', ha='center', fontsize=8, color='gray',
                transform=ax.get_xaxis_transform())

    ax.set_title('Confidence vs Layer Switch Rate\n(Counter-intuitive: Higher confidence → Higher switch rate)',
                 fontsize=13, fontweight='bold')

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='lower right', fontsize=10)

    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'E3_confidence_vs_switch.png')
    plt.savefig(output_dir / 'E3_confidence_vs_switch.svg')
    plt.close()
    print("[OK] E3_confidence_vs_switch saved")


def plot_evidence4_position_effect(results: dict, output_dir: Path):
    """证据4：位置效应（各任务）"""
    e4 = results['evidence']['E4_position_effect']

    tasks = list(e4['tasks'].keys())
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()

    positions = ['first_25pct', 'middle_50pct', 'last_25pct']
    pos_labels = ['First 25%', 'Middle 50%', 'Last 25%']

    for idx, task in enumerate(tasks):
        ax = axes[idx]
        task_data = e4['tasks'][task]

        means = [task_data[p]['mean'] for p in positions]

        x = np.arange(len(positions))
        # 去掉误差线，只画均值柱状图
        bars = ax.bar(x, means, color=COLORS[task], alpha=0.8,
                     edgecolor='black', linewidth=0.5)

        ax.set_ylabel('Entropy', fontsize=10)
        # 使用英文标题避免乱码
        trend = task_data.get('trend', 'No significant effect')
        ax.set_title(f'{TASK_LABELS_FULL[task]}\n({trend})',
                    fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(pos_labels, fontsize=9)
        ax.set_ylim(0.8, 1.0)
        ax.grid(axis='y', alpha=0.3)

        # 标注差值
        diff = task_data.get('entropy_diff_last_minus_first', 0)
        ax.text(0.95, 0.95, f'Delta={diff:+.4f}', transform=ax.transAxes,
               ha='right', va='top', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # 添加数值标签
        for i, v in enumerate(means):
            ax.text(i, v + 0.005, f'{v:.3f}', ha='center', fontsize=8)

    plt.suptitle('Position Effect on Routing Entropy\n(No significant position effect detected)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'E4_position_effect.png')
    plt.savefig(output_dir / 'E4_position_effect.svg')
    plt.close()
    print("[OK] E4_position_effect saved")


def plot_evidence5_low_confidence_tokens(results: dict, output_dir: Path):
    """证据5：低置信度token类型分布"""
    e5 = results['evidence']['E5_low_confidence_tokens']

    # 全局分布饼图
    global_dist = e5['global']['type_distribution_high_entropy']

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左图：全局分布
    ax1 = axes[0]
    types = list(global_dist.keys())
    values = list(global_dist.values())

    # 排序
    sorted_pairs = sorted(zip(types, values), key=lambda x: x[1], reverse=True)
    types, values = zip(*sorted_pairs)

    type_colors = {
        'word': '#3498DB',
        'short_word': '#9B59B6',
        'number': '#E74C3C',
        'punctuation': '#F39C12',
        'other': '#95A5A6',
        'single_char': '#1ABC9C',
        'whitespace': '#BDC3C7',
    }
    colors = [type_colors.get(t, '#7F8C8D') for t in types]

    wedges, texts, autotexts = ax1.pie(values, labels=types, autopct='%1.1f%%',
                                        colors=colors, startangle=90,
                                        explode=[0.02]*len(types))
    ax1.set_title(f'Global Distribution of High-Entropy Tokens\n(n={e5["global"]["high_entropy_tokens"]:,}, '
                  f'ratio={e5["global"]["high_entropy_ratio"]:.1%})',
                  fontsize=12, fontweight='bold')

    # 右图：各任务高熵比例对比
    ax2 = axes[1]
    tasks = list(e5['tasks'].keys())
    ratios = [e5['tasks'][t]['high_entropy_ratio'] for t in tasks]

    x = np.arange(len(tasks))
    colors_task = [COLORS[t] for t in tasks]
    bars = ax2.bar(x, ratios, color=colors_task, alpha=0.8, edgecolor='black')

    ax2.set_ylabel('High Entropy Token Ratio', fontsize=12, fontweight='bold')
    ax2.set_title('Proportion of Low-Confidence Tokens by Task', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([TASK_LABELS[t] for t in tasks], fontsize=9, rotation=30, ha='right')
    ax2.set_ylim(0, 0.6)
    ax2.grid(axis='y', alpha=0.3)

    # 添加数值标签
    for i, (task, ratio) in enumerate(zip(tasks, ratios)):
        n = e5['tasks'][task]['high_entropy_tokens']
        ax2.text(i, ratio + 0.01, f'{ratio:.1%}\n({n:,})', ha='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / 'E5_low_confidence_tokens.png')
    plt.savefig(output_dir / 'E5_low_confidence_tokens.svg')
    plt.close()
    print("[OK] E5_low_confidence_tokens saved")


def plot_summary_dashboard(results: dict, output_dir: Path):
    """汇总仪表板"""
    fig = plt.figure(figsize=(16, 12))

    # 创建网格布局
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    e1 = results['evidence']['E1_task_comparison']
    e2 = results['evidence']['E2_layer_consistency']
    e3 = results['evidence']['E3_confidence_vs_switch']
    e5 = results['evidence']['E5_low_confidence_tokens']

    # Panel 1: Task Ranking (横跨两列)
    ax1 = fig.add_subplot(gs[0, :2])
    tasks = e1['ranking']
    top1_means = [e1['tasks'][t]['top1_prob']['mean'] for t in tasks]
    colors = [COLORS[t] for t in tasks]

    bars = ax1.barh(range(len(tasks)), top1_means, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_yticks(range(len(tasks)))
    ax1.set_yticklabels([TASK_LABELS[t].replace('\n', ' ') for t in tasks], fontsize=10)
    ax1.set_xlabel('Top-1 Probability', fontsize=11)
    ax1.set_title('E1: Task Confidence Ranking', fontsize=12, fontweight='bold')
    ax1.axvline(x=1/60, color='gray', linestyle='--', label='Random')
    ax1.invert_yaxis()

    for i, v in enumerate(top1_means):
        ax1.text(v + 0.002, i, f'{v:.3f}', va='center', fontsize=9)

    # Panel 2: Key Statistics - 使用表格形式
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')

    # 创建表格数据
    table_data = [
        ['E1', 'Task Ranking', 'CMRC > GSM8K'],
        ['E2', 'Layer Corr.', '0.515 (Mod.)'],
        ['E3', 'Switch Rate', '> 96% (all)'],
        ['E4', 'Position', 'No effect'],
        ['E5', 'Token Types', 'Uniform'],
    ]

    table = ax2.table(cellText=table_data,
                      colLabels=['Evidence', 'Metric', 'Finding'],
                      loc='center',
                      cellLoc='center',
                      colColours=['#3498DB', '#3498DB', '#3498DB'])

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)

    # 设置表头样式
    for i in range(3):
        table[(0, i)].set_text_props(weight='bold', color='white')

    ax2.set_title('Q5 Summary\n(4.4M tokens, 34.5% high entropy)',
                  fontsize=11, fontweight='bold', pad=10)

    # Panel 3: Layer Correlation Heatmap
    ax3 = fig.add_subplot(gs[1, 0])
    layers = [0, 4, 8, 12, 16, 20]
    n = len(layers)
    corr_matrix = np.ones((n, n))
    for key, val in e2['layer_correlations'].items():
        parts = key.replace('layer', '').split('_vs_')
        i, j = int(parts[0]), int(parts[1])
        idx_i, idx_j = layers.index(i), layers.index(j)
        corr_matrix[idx_i, idx_j] = val
        corr_matrix[idx_j, idx_i] = val

    im = ax3.imshow(corr_matrix, cmap='RdYlBu_r', vmin=0, vmax=1)
    ax3.set_xticks(range(n))
    ax3.set_yticks(range(n))
    ax3.set_xticklabels([f'L{l}' for l in layers], fontsize=8)
    ax3.set_yticklabels([f'L{l}' for l in layers], fontsize=8)
    ax3.set_title('E2: Layer Correlation', fontsize=11, fontweight='bold')

    # Panel 4: Confidence vs Switch
    ax4 = fig.add_subplot(gs[1, 1])
    bins_order = ['very_low', 'low', 'medium', 'high']
    switch_rates = [e3['bins'][b]['switch_rate'] for b in bins_order]
    x = np.arange(len(bins_order))
    ax4.bar(x, switch_rates, color=['#3498DB', '#9B59B6', '#F39C12', '#E74C3C'],
            alpha=0.8, edgecolor='black')
    ax4.set_xticks(x)
    ax4.set_xticklabels(['V.Low', 'Low', 'Med', 'High'], fontsize=9)
    ax4.set_ylabel('Switch Rate', fontsize=10)
    ax4.set_ylim(0.96, 1.0)
    ax4.set_title('E3: Confidence → Switch Rate', fontsize=11, fontweight='bold')

    for i, v in enumerate(switch_rates):
        ax4.text(i, v + 0.002, f'{v:.1%}', ha='center', fontsize=8)

    # Panel 5: High Entropy Ratio by Task
    ax5 = fig.add_subplot(gs[1, 2])
    tasks = list(e5['tasks'].keys())
    ratios = [e5['tasks'][t]['high_entropy_ratio'] for t in tasks]
    colors = [COLORS[t] for t in tasks]
    ax5.bar(range(len(tasks)), ratios, color=colors, alpha=0.8, edgecolor='black')
    ax5.set_xticks(range(len(tasks)))
    ax5.set_xticklabels([t[:4] for t in tasks], fontsize=8, rotation=45)
    ax5.set_ylabel('High Entropy Ratio', fontsize=10)
    ax5.set_title('E5: Low-Confidence Token Ratio', fontsize=11, fontweight='bold')

    # Panel 6: Token Type Distribution (横跨整行)
    ax6 = fig.add_subplot(gs[2, :])
    global_dist = e5['global']['type_distribution_high_entropy']
    types = list(global_dist.keys())
    values = list(global_dist.values())
    sorted_pairs = sorted(zip(types, values), key=lambda x: x[1], reverse=True)
    types, values = zip(*sorted_pairs)

    type_colors = ['#3498DB', '#9B59B6', '#E74C3C', '#F39C12', '#95A5A6', '#1ABC9C', '#BDC3C7']
    ax6.bar(range(len(types)), values, color=type_colors[:len(types)], alpha=0.8, edgecolor='black')
    ax6.set_xticks(range(len(types)))
    ax6.set_xticklabels([t.replace('_', '\n') for t in types], fontsize=10)
    ax6.set_ylabel('Proportion', fontsize=11)
    ax6.set_title('E5: Global Distribution of High-Entropy Token Types', fontsize=12, fontweight='bold')

    for i, v in enumerate(values):
        ax6.text(i, v + 0.01, f'{v:.1%}', ha='center', fontsize=9)

    plt.suptitle('Q5: Routing Confidence Analysis - Complete Dashboard',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.savefig(output_dir / 'Q5_summary_dashboard.png')
    plt.savefig(output_dir / 'Q5_summary_dashboard.svg')
    plt.close()
    print("[OK] Q5_summary_dashboard saved")


def main():
    script_dir = Path(__file__).parent
    results_path = script_dir / 'q5_full_results.json'
    output_dir = script_dir / 'figures'
    output_dir.mkdir(exist_ok=True)

    print(f"Loading results from: {results_path}")
    results = load_results(results_path)

    print(f"Generating figures to: {output_dir}")

    plot_evidence1_task_comparison(results, output_dir)
    plot_evidence2_layer_consistency(results, output_dir)
    plot_evidence3_confidence_vs_switch(results, output_dir)
    plot_evidence4_position_effect(results, output_dir)
    plot_evidence5_low_confidence_tokens(results, output_dir)
    plot_summary_dashboard(results, output_dir)

    print("\n[SUCCESS] All figures generated successfully!")
    print(f"   Output directory: {output_dir}")


if __name__ == '__main__':
    main()
