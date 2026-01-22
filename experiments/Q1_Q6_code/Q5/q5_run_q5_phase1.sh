#!/usr/bin/env bash
# =============================================================================
# Q5 Phase 1: 路由置信度无标签分析
# =============================================================================
# 功能：分析六个数据集的路由置信度特征（无需 is_correct 标签）
# 指标：Top-1 概率、Margin、路由熵
# 输出：q5_confidence_results.json + 任务间对比
# =============================================================================

set -euo pipefail

echo "[INFO] Q5 Phase 1 开始 $(date)"

# --- 环境配置 ---
cd /home/zys/li_kangrui/SimAI-zyx-expert-load/experiments/moe_measure

# Conda 激活
CONDA_BASE="$(conda info --base 2>/dev/null || true)"
if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  source "$CONDA_BASE/etc/profile.d/conda.sh"
else
  source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || \
  source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
fi
conda activate qwen-moe

# --- 路径配置 ---
DATA_BASE=/home/zys/li_kangrui/SimAI-zyx-expert-load/experiments/moe_measure/results
OUT_ROOT=/home/zys/li_kangrui/SimAI-zyx-expert-load/experiments/moe_measure/2025-01-10_q5_phase1

# 六个数据集路径
GSM="$DATA_BASE/Qwen1.5-MoE-A2.7B-gsm8k-train-all-gate-state"
WIKI="$DATA_BASE/Qwen1.5-MoE-A2.7B-wikitext2_raw-train-all-gate-state"
CMRC="$DATA_BASE/Qwen1.5-MoE-A2.7B-cmrc2018-train-all-gate-state"
PIQA="$DATA_BASE/Qwen1.5-MoE-A2.7B-piqa-train-all-gate-state"
WINO="$DATA_BASE/Qwen1.5-MoE-A2.7B-winogrande_xl-train-all-gate-state"
HUM="$DATA_BASE/Qwen1.5-MoE-A2.7B-humaneval-test-500-gate-state"

# 检查数据目录
echo "[INFO] 检查数据目录..."
for name in GSM WIKI CMRC PIQA WINO HUM; do
  dir="${!name}"
  if [ ! -d "$dir" ]; then
    echo "[WARN] 目录不存在: $name -> $dir"
    # 尝试 find 查找
    alt=$(find "$DATA_BASE" -maxdepth 1 -type d -name "*$(echo $name | tr '[:upper:]' '[:lower:]')*" 2>/dev/null | head -n 1)
    if [ -n "$alt" ] && [ -d "$alt" ]; then
      echo "[INFO] 找到替代: $alt"
      eval "$name=\"$alt\""
    fi
  else
    echo "[OK] $name -> $dir"
  fi
done

# 创建输出目录
mkdir -p "$OUT_ROOT" "$OUT_ROOT/logs"

# --- 运行 Q5 Phase 1 ---
echo "[INFO] 开始运行 Q5 Phase 1 分析..."

python analyze_q5_confidence.py \
  --task gsm8k="$GSM" \
  --task wiki="$WIKI" \
  --task cmrc2018="$CMRC" \
  --task piqa="$PIQA" \
  --task winogrande="$WINO" \
  --task humaneval="$HUM" \
  --output_dir "$OUT_ROOT" \
  --n_experts 60 \
  --layer 0 \
  --phase 1 \
  --max_files -1 \
  --max_samples 0 \
  --seed 42

echo "[INFO] Q5 Phase 1 完成 $(date)"
echo "[INFO] 结果目录: $OUT_ROOT"
echo "[INFO] 主要输出: $OUT_ROOT/q5_confidence_results.json"
