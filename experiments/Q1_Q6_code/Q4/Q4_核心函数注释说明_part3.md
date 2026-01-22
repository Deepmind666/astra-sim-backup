# Q4: 长上下文位置漂移分析 - 核心函数注释说明 (Part 3)

## 文件对齐
- **对应脚本**: `q4_analyze_q4_long_context.py`
- **行号说明**: 以当前脚本为准，本文行号仅作理解辅助

## 6. 主分析函数

### 函数: `analyze_q4()`
**位置**: 第424-798行

**目的**: 执行完整的 Q4 分析流程

**主要步骤**:
1. P0 数据结构检查
2. 位置漂移分析
3. 阶段切换分析
4. 熵曲线分析
5. 输出 JSON 结果

---

### 工作流程详解

#### 第一部分: 初始化和任务类型推断
```python
def analyze_q4(data_dir, output_dir, window_sizes, ...):
    # 1. 初始化随机数生成器
    rng = np.random.RandomState(seed)

    # 2. 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 3. 从目录名推断任务类型
    data_dir_name = os.path.basename(data_dir).lower()
    if "gsm8k" in data_dir_name:
        task_type = "gsm8k"
    elif "humaneval" in data_dir_name:
        task_type = "humaneval"
    # ... 其他任务类型
```

**任务类型**: 用于选择合适的边界检测规则

---

#### 第二部分: 数据收集容器初始化
```python
# 位置漂移曲线
position_curves = {"soft": {}, "hard": {}}

# 置乱基线曲线
baseline_curves = {"soft": {}, "hard": {}}

# 熵曲线
entropy_curves = {"soft": {}, "hard": {}}

# 边界统计
boundary_stats = {
    "soft": {"real": [], "random": []},
    "hard": {"real": [], "random": []},
    "n_boundaries": 0,
    "n_real_boundaries": 0,
    "n_pseudo_boundaries": 0,
}
```

---

#### 第三部分: 遍历所有样本
```python
for npz_name in npz_files:
    # 1. 加载样本
    samples = _iter_npz_samples(npz_path, layer)

    for sample in samples:
        # 2. 提取数据
        probs = sample["probs"]
        indices = sample["indices"]
        tokens_str = sample["tokens_str"]
        n_tokens = probs.shape[0]

        # 3. 过滤短序列
        if n_tokens < min_tokens:
            continue

        # 4. P0 检查
        token_lengths.append(n_tokens)
        prob_sums.append(probs.sum(axis=1).mean())

        # 5. 位置漂移分析
        for window_size in window_sizes:
            # 5.1 划分窗口
            windows = split_windows(n_tokens, window_size)

            # 5.2 计算每个窗口的分布
            dists_soft = []
            dists_hard = []
            for start, end in windows:
                dist_soft = build_soft_distribution(
                    probs[start:end], indices[start:end], n_experts
                )
                dist_hard = build_hard_distribution(
                    probs[start:end], indices[start:end], n_experts
                )
                dists_soft.append(dist_soft)
                dists_hard.append(dist_hard)

            # 5.3 计算相邻窗口的 JSD
            jsd_curve_soft = []
            for i in range(len(dists_soft) - 1):
                jsd_val = jsd(dists_soft[i], dists_soft[i+1])
                jsd_curve_soft.append(jsd_val)

            # 5.4 保存曲线
            position_curves["soft"][window_size].append(jsd_curve_soft)

        # 6. 边界分析
        boundaries = detect_boundaries(tokens_str, task_type)
        if not boundaries:
            boundaries = build_pseudo_boundaries(n_tokens)

        # 计算边界处的 JSD
        for b in boundaries:
            left_dist = build_soft_distribution(...)
            right_dist = build_soft_distribution(...)
            boundary_jsd = jsd(left_dist, right_dist)
            boundary_stats["soft"]["real"].append(boundary_jsd)

        # 7. 随机位置基线
        for _ in range(shuffle_runs):
            random_pos = rng.randint(margin, n_tokens - margin)
            left_dist = build_soft_distribution(...)
            right_dist = build_soft_distribution(...)
            random_jsd = jsd(left_dist, right_dist)
            boundary_stats["soft"]["random"].append(random_jsd)
```

---

#### 第四部分: 结果汇总和输出
```python
# 1. 计算曲线的均值和置信区间
for window_size in window_sizes:
    curves = position_curves["soft"][window_size]
    # 重采样到统一长度
    resampled = [resample_curve(c, 50) for c in curves]
    # 计算均值和 CI
    mean, ci = mean_ci(resampled)

# 2. 构建输出字典
summary = {
    "meta": {...},
    "p0_check": {
        "n_samples": total_samples,
        "avg_token_length": np.mean(token_lengths),
        "avg_prob_sum": np.mean(prob_sums),
        "has_tokens_str": has_tokens_str,
    },
    "position_drift": {
        "soft": {...},
        "hard": {...},
    },
    "boundary_effect": {
        "soft": {
            "real_mean": np.mean(boundary_stats["soft"]["real"]),
            "random_mean": np.mean(boundary_stats["soft"]["random"]),
        },
        "hard": {...},
    },
}

# 3. 保存 JSON
with open(os.path.join(output_dir, "q4_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
```

---

## 7. 核心指标

### 指标1: 位置漂移曲线
**定义**: 相邻窗口的 JSD 随位置的变化

**计算方法**:
1. 将序列划分为多个窗口
2. 计算每个窗口的专家分布
3. 计算相邻窗口的 JSD
4. 绘制 JSD 曲线

**判定标准**:
- 曲线平稳 → 无位置漂移
- 曲线上升/下降 → 存在位置漂移

---

### 指标2: 边界效应
**定义**: 边界处的 JSD vs 随机位置的 JSD

**计算方法**:
1. 检测语义边界(或使用伪边界)
2. 计算边界处左右窗口的 JSD
3. 随机采样多个位置,计算其左右窗口的 JSD
4. 比较两者的均值

**判定标准**:
- 边界 JSD > 随机 JSD → 存在阶段切换
- 边界 JSD ≈ 随机 JSD → 无阶段切换

---

### 指标3: 熵曲线
**定义**: 专家分布的熵随位置的变化

**用途**: 观察不确定性是否随位置变化
- 熵高 → 分布均匀,不确定性大
- 熵低 → 分布集中,不确定性小

---

## 8. 实验结论

### 结论1: 位置漂移弱
**观察**: 位置漂移曲线基本平稳,无明显上升或下降趋势

**解释**: 专家分布不随序列位置发生系统性变化

**含义**: 模型在处理长序列时,专家选择不受位置影响

---

### 结论2: 无阶段切换
**观察**: 边界处的 JSD ≈ 随机位置的 JSD

**解释**: 语义边界(如问题→答案)并未导致专家分布跳变

**含义**: 专家选择与语义结构无关

---

### 结论3: 熵稳定
**观察**: 熵曲线基本平稳,无明显波动

**解释**: 专家分布的不确定性不随位置变化

**含义**: 模型在整个序列中保持一致的专家选择策略

---

## 9. 与其他问题的关联

### Q3: 任务内路由模式
- Q3 关注时序粘滞性(相邻 token)
- Q4 关注位置效应(序列前部 vs 后部)
- **共同结论**: 路由是无记忆的,不受时序或位置影响

### Q2: 任务专属性
- Q2 发现任务间有专属专家
- Q4 发现任务内无位置漂移
- **结合**: 专家分工体现在任务级别,而非位置级别

---

## 10. 代码运行示例

```bash
python analyze_q4_long_context.py \
    --data_dir /path/to/gsm8k \
    --output_dir ./output_q4 \
    --window_sizes 64,128,256 \
    --min_tokens 512 \
    --layer 0 \
    --shuffle_runs 100 \
    --seed 42
```

**输出文件**:
- `q4_summary.json`: 主分析结果

---

**文档创建时间**: 2026-01-19
**作者**: Claude (基于李康锐的原始代码)
**说明**: 这是 Part 3,包含主分析函数和实验结论
