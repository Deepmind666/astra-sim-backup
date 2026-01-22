# Q4: 长上下文位置漂移分析 - 核心函数注释说明 (Part 1)

## 文件信息
- **对应脚本**: `q4_analyze_q4_long_context.py`（计算与统计）+ `q4_plot_q4_figures.py`（画图汇总）
- **行号说明**: 以当前脚本为准，本文行号仅作理解辅助
- **核心问题**: 在长序列上验证 MoE 路由分布是否随位置漂移,以及在语义阶段边界是否出现跳变

## 核心问题

### 问题描述
Q4 关注的是**长上下文场景**下的专家路由模式:
1. **位置漂移**: 专家分布是否随着 token 位置(序列位置)发生系统性变化?
2. **阶段切换**: 在语义边界(如问题→答案)处,专家分布是否发生跳变?

### 与 Q3 的区别
- **Q3**: 关注任务内部的**时序粘滞性**(相邻 token 是否倾向选同一专家)
- **Q4**: 关注**位置效应**(序列前部 vs 后部的专家分布差异)

### 实验场景
- **长序列任务**: 如长文档阅读理解、长代码生成
- **窗口分析**: 将长序列划分为多个窗口,观察窗口间的分布变化

---

## 核心概念

### 1. 位置漂移 (Position Drift)
**定义**: 专家分布随序列位置的系统性变化

**示例**:
```
序列前部(0-512):   专家分布 [0.1, 0.2, 0.3, ...]
序列中部(512-1024): 专家分布 [0.15, 0.18, 0.25, ...]
序列后部(1024-1536): 专家分布 [0.2, 0.15, 0.2, ...]
```

**判定标准**:
- 如果前部、中部、后部的专家分布显著不同 → 存在位置漂移
- 如果分布基本一致 → 无位置漂移

### 2. 阶段切换 (Phase Transition)
**定义**: 在语义边界处,专家分布发生跳变

**示例**:
```
问题部分: 专家分布 [0.1, 0.3, 0.2, ...]
答案部分: 专家分布 [0.3, 0.1, 0.25, ...]  ← 分布跳变
```

**判定标准**:
- 如果边界处的分布差异 > 随机位置的差异 → 存在阶段切换
- 如果差异相当 → 无阶段切换

### 3. 软分布 vs 硬分布
**软分布 (Soft Distribution)**:
- 累加所有 Top-K 专家的概率
- 包含 Other 桶(未被 Top-K 捕获的概率质量)
- 维度: 61 (60 专家 + 1 Other)

**硬分布 (Hard Distribution)**:
- 只统计 Top-1 专家的出现频次
- 不包含 Other 桶
- 维度: 60

---

## 核心函数详解

### 1. 数学工具函数

#### 函数: `_safe_normalize()`
**位置**: 第35-44行

**目的**: 将向量归一化为概率分布(和为1)

**算法**:
```python
def _safe_normalize(vec: np.ndarray) -> np.ndarray:
    total = float(np.sum(vec))
    if total <= 0:
        # 返回均匀分布,防止除零
        return np.ones_like(vec) / float(len(vec))
    return vec / total
```

**边界情况处理**:
- 如果向量总和为 0 → 返回均匀分布
- 避免除零错误

---

#### 函数: `jsd()`
**位置**: 第47-65行

**定义**: Jensen-Shannon Divergence (JSD) 距离

**公式**:
```
JSD(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
其中 M = 0.5 * (P + Q)
```

**性质**:
- 取值范围: [0, 1] (使用 log2 时)
- 对称性: JSD(P||Q) = JSD(Q||P)
- 有界性: 比 KL 散度更稳定

**实现细节**:
```python
def jsd(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    # 1. 裁剪小概率,避免 log(0)
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)

    # 2. 归一化
    p = _safe_normalize(p)
    q = _safe_normalize(q)

    # 3. 计算中间分布 M
    m = 0.5 * (p + q)

    # 4. 计算 KL 散度
    kl_pm = np.sum(p * (np.log2(p) - np.log2(m)))
    kl_qm = np.sum(q * (np.log2(q) - np.log2(m)))

    # 5. 返回 JSD
    return 0.5 * (kl_pm + kl_qm)
```

---

#### 函数: `tv()`
**位置**: 第68-74行

**定义**: Total Variation (TV) 距离

**公式**:
```
TV(P, Q) = 0.5 * Σ|P - Q|
```

**性质**:
- 取值范围: [0, 1]
- 对称性: TV(P, Q) = TV(Q, P)
- 直观解释: 两个分布的最大差异概率

**实现**:
```python
def tv(p: np.ndarray, q: np.ndarray) -> float:
    p = _safe_normalize(p)
    q = _safe_normalize(q)
    return 0.5 * float(np.sum(np.abs(p - q)))
```

---

#### 函数: `entropy()`
**位置**: 第77-84行

**定义**: 分布熵 (Shannon Entropy)

**公式**:
```
H(P) = -Σ P * log2(P)
```

**用途**: 观察不确定性是否随位置变化
- 熵高 → 分布均匀,不确定性大
- 熵低 → 分布集中,不确定性小

**实现**:
```python
def entropy(p: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1.0)
    p = _safe_normalize(p)
    return float(-np.sum(p * np.log2(p)))
```

---

### 2. 数据读取函数

#### 函数: `_iter_npz_samples()`
**位置**: 第90-132行

**目的**: 从 NPZ 文件中提取所有样本

**兼容两种格式**:
1. **常见格式**: 一个 NPZ = 一个样本(数组为 2D)
2. **旧格式**: 一个 NPZ = 多个样本(数组 dtype=object)

**算法流程**:
```python
def _iter_npz_samples(npz_path: str, layer: int) -> List[Dict]:
    # 1. 加载 NPZ 文件
    data = np.load(npz_path, allow_pickle=True)

    # 2. 构建键名
    prob_key = f"layers/MoE_Gate{layer}_out_probs"
    idx_key = f"layers/MoE_Gate{layer}_out_indices"

    # 3. 检查键是否存在
    if prob_key not in data or idx_key not in data:
        return []

    # 4. 提取数据
    probs = data[prob_key]
    indices = data[idx_key]
    tokens_str = data.get("tokens_str", None)

    # 5. 判断格式并提取样本
    samples = []
    if probs.dtype == object:
        # 旧格式: 多样本
        for i in range(len(probs)):
            sample = {
                "probs": probs[i],
                "indices": indices[i],
                "tokens_str": tokens_str[i] if tokens_str else None
            }
            samples.append(sample)
    else:
        # 常见格式: 单样本
        sample = {
            "probs": probs,
            "indices": indices,
            "tokens_str": tokens_str
        }
        samples.append(sample)

    return samples
```

**返回值**: 样本列表,每个样本包含:
- `probs`: Top-K 概率数组,形状 [n_tokens, k]
- `indices`: Top-K 专家索引数组,形状 [n_tokens, k]
- `tokens_str`: token 文本列表(可选)

---

### 3. 分布构建函数

#### 函数: `build_soft_distribution()`
**位置**: 第138-170行

**目的**: 构建"软分布"(概率累加 + Other 桶)

**维度**: n_experts + 1 (60 专家 + 1 Other)

**算法流程**:
```python
def build_soft_distribution(
    probs_window: np.ndarray,  # 形状 [window_size, k]
    idx_window: np.ndarray,    # 形状 [window_size, k]
    n_experts: int,            # 60
) -> np.ndarray:
    # 1. 初始化 61 维分布数组
    full = np.zeros(n_experts + 1, dtype=np.float64)

    # 2. 遍历窗口内每个 token
    for t in range(probs_window.shape[0]):
        p = probs_window[t]    # Top-K 概率
        idx = idx_window[t]    # Top-K 索引

        # 3. 计算 Other 桶的概率
        prob_sum = float(np.sum(p))
        residual = max(0.0, 1.0 - prob_sum)

        # 4. 累加 Top-K 专家的概率
        for k in range(len(p)):
            expert_id = int(idx[k])
            if 0 <= expert_id < n_experts:
                full[expert_id] += float(p[k])

        # 5. 累加 Other 桶的概率
        full[-1] += residual

    # 6. 归一化为平均分布
    return _safe_normalize(full)
```

**关键点**:
- **Other 桶**: 存储未被 Top-K 捕获的概率质量
- **归一化**: 除以窗口大小,得到平均分布

---

#### 函数: `build_hard_distribution()`
**位置**: 第173-193行

**目的**: 构建"硬分布"(Top-1 频次)

**维度**: n_experts (60)

**算法流程**:
```python
def build_hard_distribution(
    probs_window: np.ndarray,
    idx_window: np.ndarray,
    n_experts: int,
) -> np.ndarray:
    # 1. 初始化 60 维计数数组
    counts = np.zeros(n_experts, dtype=np.float64)

    # 2. 遍历窗口内每个 token
    for t in range(probs_window.shape[0]):
        p = probs_window[t]
        idx = idx_window[t]

        # 3. 找到 Top-1 专家
        top1_k = int(np.argmax(p))  # 概率最大的位置
        expert_id = int(idx[top1_k])

        # 4. 累加 Top-1 专家的出现次数
        if 0 <= expert_id < n_experts:
            counts[expert_id] += 1.0

    # 5. 归一化为频率分布
    return _safe_normalize(counts)
```

**关键点**:
- **Top-1 提取**: 使用 `np.argmax(p)` 找到概率最大的位置
- **不假设排序**: 不假设 Top-K 已按概率降序排列

---

**文档创建时间**: 2026-01-19
**作者**: Claude (基于李康锐的原始代码)
**说明**: 这是 Part 1,包含基础函数和分布构建函数的详细注释
