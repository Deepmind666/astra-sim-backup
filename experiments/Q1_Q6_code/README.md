# Q1-Q6 代码运行指南

本文档提供了Q1-Q6所有问题的详细运行指令,确保可以在zys服务器上复现实验结果。

重要说明：
以本目录 `Q1_Q6_code` 为唯一根目录。
本地重跑所需的数据集请统一放在 `datasets/`，详细见 `docs/README_新目录运行说明.md`。

---

## 目录
- [环境准备](#环境准备)
- [数据路径](#数据路径)
- [Q1: 专家使用频率统计](#q1-专家使用频率统计)
- [Q2: 任务专属性分析](#q2-任务专属性分析)
- [Q3: 任务内路由模式分析](#q3-任务内路由模式分析)
- [Q4: 长上下文位置漂移分析](#q4-长上下文位置漂移分析)
- [Q5: 路由置信度特征分析](#q5-路由置信度特征分析)
- [Q6: 负载集中度分析](#q6-负载集中度分析)

---

## 环境准备

### Python环境
```bash
# 激活虚拟环境
source /home/zys/li_kangrui/SimAI-zyx-expert-load/.venv/bin/activate

# 或者使用conda环境
conda activate moe_analysis
```

### 依赖包
```bash
pip install numpy scipy matplotlib
```

---

## 数据路径

### 服务器数据目录
```bash
# 基础路径
BASE_DIR=/home/zys/li_kangrui/SimAI-zyx-expert-load/data

# 各任务数据路径
GSMS8K_DATA=${BASE_DIR}/gsm8k
PIQA_DATA=${BASE_DIR}/piqa
CMRC2018_DATA=${BASE_DIR}/cmrc2018
HUMANEVAL_DATA=${BASE_DIR}/humaneval
WIKI_DATA=${BASE_DIR}/wiki
WINOGRANDE_DATA=${BASE_DIR}/winogrande
```

---

## Q1: 专家使用频率统计

### 核心问题
统计每个专家在不同任务中的使用频率,验证专家是否被均匀使用。

### 运行命令
```bash
cd /home/zys/li_kangrui/SimAI-zyx-expert-load/experiments/moe_measure

python analyze_q1_expert_usage.py \
    --task gsm8k=${BASE_DIR}/gsm8k \
    --task piqa=${BASE_DIR}/piqa \
    --task cmrc2018=${BASE_DIR}/cmrc2018 \
    --task humaneval=${BASE_DIR}/humaneval \
    --task wiki=${BASE_DIR}/wiki \
    --task winogrande=${BASE_DIR}/winogrande \
    --output_dir ./output_q1 \
    --n_experts 60 \
    --layer 0 \
    --max_files -1 \
    --seed 42
```

### 参数说明
- `--task`: 任务名称和数据路径,格式为 `task_name=path`,可多次指定
- `--output_dir`: 输出目录
- `--n_experts`: 专家数量,默认60
- `--layer`: 分析的层编号,默认0
- `--max_files`: 最多加载的NPZ文件数,-1表示全部加载
- `--seed`: 随机种子,保证可复现

### 输出文件
- `q1_results.json`: 完整统计结果
- `plots/q1_expert_usage_heatmap.png`: 专家使用热力图
- `plots/q1_expert_usage_distribution.png`: 专家使用分布图

---

## Q2: 任务专属性分析

### 核心问题
分析不同任务是否有专属的专家,以及专家对任务的偏好程度。

### 第一步: 单个种子分析
```bash
cd /home/zys/li_kangrui/SimAI-zyx-expert-load/experiments/moe_measure

# 运行seed11
python analyze_task_specialization_v1.py \
    --task gsm8k=${BASE_DIR}/gsm8k \
    --task piqa=${BASE_DIR}/piqa \
    --task cmrc2018=${BASE_DIR}/cmrc2018 \
    --task humaneval=${BASE_DIR}/humaneval \
    --task wiki=${BASE_DIR}/wiki \
    --task winogrande=${BASE_DIR}/winogrande \
    --output_dir ./output_q2/seed11 \
    --n_experts 60 \
    --max_files -1 \
    --sample_size 20000 \
    --seed 11 \
    --alpha 1.0 \
    --topn 10 \
    --min_support 0.002 \
    --k_values 1,2,5,10,20 \
    --shuffle_runs 20 \
    --matched_shuffle \
    --bootstrap_runs 200

# 运行seed22
python analyze_task_specialization_v1.py \
    --task gsm8k=${BASE_DIR}/gsm8k \
    --task piqa=${BASE_DIR}/piqa \
    --task cmrc2018=${BASE_DIR}/cmrc2018 \
    --task humaneval=${BASE_DIR}/humaneval \
    --task wiki=${BASE_DIR}/wiki \
    --task winogrande=${BASE_DIR}/winogrande \
    --output_dir ./output_q2/seed22 \
    --seed 22 \
    # ... 其他参数同上

# 运行seed33
python analyze_task_specialization_v1.py \
    --task gsm8k=${BASE_DIR}/gsm8k \
    --task piqa=${BASE_DIR}/piqa \
    --task cmrc2018=${BASE_DIR}/cmrc2018 \
    --task humaneval=${BASE_DIR}/humaneval \
    --task wiki=${BASE_DIR}/wiki \
    --task winogrande=${BASE_DIR}/winogrande \
    --output_dir ./output_q2/seed33 \
    --seed 33 \
    # ... 其他参数同上
```

### 第二步: 聚合多个种子的结果
```bash
cd /home/zys/li_kangrui/SimAI_Cluster_Experiment/2025-12-28_task_specialization_v2b/analysis

python aggregate_top_experts.py
```

**注意**: 该脚本会自动从 `seed11/`, `seed22/`, `seed33/` 目录读取结果并聚合。

### 参数说明
- `--sample_size`: 每个任务采样的token数量,默认20000
- `--alpha`: Laplace平滑参数,默认1.0
- `--topn`: 提取的Top专家数量,默认10
- `--min_support`: 最小支持度阈值,默认0.002
- `--k_values`: Top-k覆盖率的k值列表
- `--shuffle_runs`: 置乱基线的运行次数,默认20
- `--matched_shuffle`: 启用匹配置乱(控制token类型/长度/熵)
- `--bootstrap_runs`: Bootstrap置信区间的运行次数,默认200

### 输出文件
**单个种子**:
- `task_specialization_summary.json`: 完整分析结果
- `plots/coverage_hard.png`: Hard口径的Top-k覆盖率曲线
- `plots/coverage_soft.png`: Soft口径的Top-k覆盖率曲线
- `plots/tss.png`: 任务专属性得分(TSS)柱状图

**聚合结果**:
- `top_experts_soft_score1_delta_mean.json`: Soft口径的Top专家列表
- `top_experts_hard_score1_delta_mean.json`: Hard口径的Top专家列表
- `top_experts_summary.md`: Markdown格式的汇总报告
- `top_experts_soft_bar.png`: Soft口径的柱状图
- `top_experts_hard_bar.png`: Hard口径的柱状图

---

## Q3: 任务内路由模式分析

### 核心问题
分析任务内部的路由动态,判断是否存在"粘滞性"或"阶段切换"现象。

### 运行命令
```bash
cd /home/zys/li_kangrui/SimAI-zyx-expert-load/experiments/moe_measure

python analyze_task_drift_v2.py \
    --task gsm8k=${BASE_DIR}/gsm8k \
    --task piqa=${BASE_DIR}/piqa \
    --task cmrc2018=${BASE_DIR}/cmrc2018 \
    --task humaneval=${BASE_DIR}/humaneval \
    --task wiki=${BASE_DIR}/wiki \
    --task winogrande=${BASE_DIR}/winogrande \
    --output_dir ./output_q3 \
    --n_experts 60 \
    --max_files -1 \
    --sample_size 20000 \
    --sample_start 0 \
    --window_sizes 64,128,256 \
    --window_stride 0 \
    --bootstrap_runs 200 \
    --seed 42 \
    --shuffle_baseline \
    --normalize_per_layer
```

### 参数说明
- `--window_sizes`: 窗口大小列表,逗号分隔,默认"64,128,256"
- `--window_stride`: 窗口步长,0表示等于窗口大小(不重叠)
- `--bootstrap_runs`: Bootstrap置信区间的运行次数,默认200
- `--skip_p0`: 跳过P0数据结构检查
- `--skip_per_layer`: 跳过逐层分析
- `--shuffle_baseline`: 启用置乱基线
- `--normalize_per_layer`: 对每层概率先归一化

### 输出文件
- `p0_data_structure_check.json`: P0数据结构检查结果
- `task_drift_summary.json`: 完整分析结果
- `plots/q3_switch_rate_comparison.png`: 切换率对比图
- `plots/q3_run_length_analysis.png`: Run Length分布图
- `plots/q3_window_drift_analysis.png`: 窗口漂移曲线
- `plots/q3_per_layer_heatmap.png`: 逐层切换率热力图

---

## Q4: 长上下文位置漂移分析

### 核心问题
在长序列上验证MoE路由分布是否随位置漂移,以及在语义阶段边界是否出现跳变。

### 运行命令
```bash
cd /home/zys/li_kangrui/SimAI_Cluster_Experiment/experiments/moe_measure

python analyze_q4_long_context.py \
    --data_dir ${BASE_DIR}/gsm8k \
    --output_dir ./output_q4/gsm8k \
    --window_sizes 64,128,256 \
    --min_tokens 512 \
    --n_experts 60 \
    --layer 0 \
    --shuffle_runs 100 \
    --max_files -1 \
    --max_samples -1 \
    --seed 42

# 对其他任务重复运行
python analyze_q4_long_context.py \
    --data_dir ${BASE_DIR}/piqa \
    --output_dir ./output_q4/piqa \
    --window_sizes 64,128,256 \
    --min_tokens 512 \
    --seed 42
```

### 参数说明
- `--data_dir`: 单个任务的数据目录
- `--window_sizes`: 窗口大小列表,逗号分隔
- `--min_tokens`: 最小token数量,过滤短序列
- `--layer`: 分析的层编号,默认0
- `--shuffle_runs`: 随机位置基线的采样次数,默认100
- `--max_files`: 最多加载的NPZ文件数,-1表示全部
- `--max_samples`: 最多分析的样本数,-1表示全部

### 输出文件
- `q4_summary.json`: 完整分析结果

---

## Q5: 路由置信度特征分析

### 核心问题
分析MoE路由的置信度特征,探究专家选择的确定性。

### 运行命令
```bash
cd /home/zys/li_kangrui/SimAI_Cluster_Experiment/experiments/moe_measure

python analyze_q5_full.py \
    --task gsm8k=${BASE_DIR}/gsm8k \
    --task piqa=${BASE_DIR}/piqa \
    --task cmrc2018=${BASE_DIR}/cmrc2018 \
    --task humaneval=${BASE_DIR}/humaneval \
    --task wiki=${BASE_DIR}/wiki \
    --task winogrande=${BASE_DIR}/winogrande \
    --output_dir ./output_q5 \
    --layer 0 \
    --max_files -1 \
    --seed 42
```

### 参数说明
- `--task`: 任务名称和数据路径,格式为 `task_name=path`,可多次指定
- `--layer`: 分析的层编号,默认0
- `--max_files`: 最多加载的NPZ文件数,-1表示全部

### 输出文件
- `q5_full_results.json`: 完整分析结果
- `plots/`: 所有可视化图表

---

## Q6: 负载集中度分析

### 核心问题
分析不同任务的专家负载集中度,判断负载是否均衡。

### 运行命令
```bash
cd /home/zys/li_kangrui/SimAI_Cluster_Experiment/experiments/moe_measure

python analyze_q6_load_concentration.py \
    --task gsm8k=${BASE_DIR}/gsm8k \
    --task piqa=${BASE_DIR}/piqa \
    --task cmrc2018=${BASE_DIR}/cmrc2018 \
    --task humaneval=${BASE_DIR}/humaneval \
    --task wiki=${BASE_DIR}/wiki \
    --task winogrande=${BASE_DIR}/winogrande \
    --output_dir ./output_q6 \
    --layer 0 \
    --max_files -1 \
    --seed 42
```

### 参数说明
- `--task`: 任务名称和数据路径,格式为 `task_name=path`,可多次指定
- `--layer`: 分析的层编号,默认0
- `--max_files`: 最多加载的NPZ文件数,-1表示全部

### 输出文件
- `q6_results.json`: 完整分析结果

---

## 附录文档（docs/）

当前仅保留新目录口径相关文档：

- `docs/README_新目录运行说明.md`：新目录结构、数据集放置与运行口径  
- `docs/README_采集数据全流程.md`：Stage1 数据采集全流程（含配置修改与输出检查）

---

## 常见问题

### 1. 数据路径错误
**问题**: `FileNotFoundError: No NPZ files found`

**解决**: 检查数据路径是否正确,确保NPZ文件存在
```bash
ls ${BASE_DIR}/gsm8k/*.npz | head -5
```

### 2. 内存不足
**问题**: `MemoryError` 或 `Killed`

**解决**: 减少加载的文件数量或采样大小
```bash
--max_files 100 \
--sample_size 10000
```

### 3. 缺少依赖包
**问题**: `ModuleNotFoundError: No module named 'scipy'`

**解决**: 安装缺失的依赖包
```bash
pip install scipy matplotlib
```

---

## 批量运行脚本

### 运行所有问题
```bash
#!/bin/bash
# run_all_q1_q6.sh

BASE_DIR=/home/zys/li_kangrui/SimAI-zyx-expert-load/data
OUTPUT_BASE=./output_all

# Q1
python analyze_q1_expert_usage.py \
    --task gsm8k=${BASE_DIR}/gsm8k \
    --task piqa=${BASE_DIR}/piqa \
    --output_dir ${OUTPUT_BASE}/q1 \
    --seed 42

# Q2 (需要运行3个种子)
for seed in 11 22 33; do
    python analyze_task_specialization_v1.py \
        --task gsm8k=${BASE_DIR}/gsm8k \
        --task piqa=${BASE_DIR}/piqa \
        --output_dir ${OUTPUT_BASE}/q2/seed${seed} \
        --seed ${seed}
done

# Q3
python analyze_task_drift_v2.py \
    --task gsm8k=${BASE_DIR}/gsm8k \
    --task piqa=${BASE_DIR}/piqa \
    --output_dir ${OUTPUT_BASE}/q3 \
    --seed 42

# Q4 (每个任务单独运行)
for task in gsm8k piqa cmrc2018 humaneval wiki winogrande; do
    python analyze_q4_long_context.py \
        --data_dir ${BASE_DIR}/${task} \
        --output_dir ${OUTPUT_BASE}/q4/${task} \
        --seed 42
done

# Q5
python analyze_q5_full.py \
    --task gsm8k=${BASE_DIR}/gsm8k \
    --task piqa=${BASE_DIR}/piqa \
    --output_dir ${OUTPUT_BASE}/q5 \
    --seed 42

# Q6
python analyze_q6_load_concentration.py \
    --task gsm8k=${BASE_DIR}/gsm8k \
    --task piqa=${BASE_DIR}/piqa \
    --output_dir ${OUTPUT_BASE}/q6 \
    --seed 42

echo "所有分析完成!"
```

---

**文档创建时间**: 2026-01-19
**作者**: Claude (基于李康锐的原始代码)
**最后更新**: 2026-01-19
