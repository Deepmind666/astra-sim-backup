# Q4: 长上下文位置漂移分析 - 核心函数注释说明 (Part 2)

## 文件对齐
- **对应脚本**: `q4_analyze_q4_long_context.py`
- **行号说明**: 以当前脚本为准，本文行号仅作理解辅助

## 4. 窗口划分与曲线处理

### 函数: `split_windows()`
**位置**: 第199-210行

**目的**: 将 token 序列切分成不重叠的窗口

**示例**:
```
T=500, window_size=128
→ 3个窗口: [0,128), [128,256), [256,384)
→ 剩余116个token被丢弃
```

**算法**:
```python
def split_windows(n_tokens: int, window_size: int):
    n_windows = n_tokens // window_size
    windows = []
    for i in range(n_windows):
        start = i * window_size
        end = (i + 1) * window_size
        windows.append((start, end))
    return windows
```

---

### 函数: `resample_curve()`
**位置**: 第213-226行

**目的**: 将不同长度的曲线统一重采样到固定长度

**用途**: 便于跨样本平均(不同样本的窗口数量可能不同)

**方法**: 线性插值,横坐标统一映射到 [0, 1]

**算法**:
```python
def resample_curve(curve: List[float], target_len: int):
    # 1. 边界情况处理
    if len(curve) == 0:
        return [0.0] * target_len
    if len(curve) == 1:
        return [curve[0]] * target_len

    # 2. 线性插值
    x_old = np.linspace(0.0, 1.0, num=len(curve))
    x_new = np.linspace(0.0, 1.0, num=target_len)
    y_new = np.interp(x_new, x_old, curve)
    return y_new.tolist()
```

**示例**:
```
原始曲线: [0.1, 0.2, 0.15] (3个点)
目标长度: 5
重采样后: [0.1, 0.15, 0.2, 0.175, 0.15]
```

---

### 函数: `mean_ci()`
**位置**: 第229-239行

**目的**: 计算曲线的均值与 95% 置信区间

**公式**:
```
CI = 1.96 * SEM
SEM = std / sqrt(n)
```

**算法**:
```python
def mean_ci(curves: List[List[float]]):
    arr = np.array(curves)
    mean = np.mean(arr, axis=0)
    sem = np.std(arr, axis=0, ddof=1) / math.sqrt(arr.shape[0])
    ci = 1.96 * sem
    return mean.tolist(), ci.tolist()
```

---

## 5. 边界检测

### 函数: `_normalize_token_text()`
**位置**: 第245-272行

**目的**: 将 token 转换为更接近人类可读的文本

**问题背景**:
- tokens_str 通常是分词后的片段(BPE/SentencePiece)
- 直接用正则匹配容易失败(如 "Answer:" 被拆开)
- 需要先做"弱解码",还原空格和换行符

**处理规则**:
```python
def _normalize_token_text(token: str):
    text = str(token)

    # SentencePiece: "▁" → 空格
    if text.startswith("▁"):
        text = " " + text[1:]

    # GPT2 BPE: "Ġ" → 空格
    if text.startswith("Ġ"):
        text = " " + text[1:]

    # 残留的 "▁" → 空格
    text = text.replace("▁", " ")

    # GPT2 BPE: "Ċ" → 换行
    text = text.replace("Ċ", "\n")

    return text
```

**示例**:
```
输入: "▁Answer", "Ġ:", "▁is"
输出: " Answer", " :", " is"
拼接: " Answer : is"
```

---

### 函数: `_build_text_with_offsets()`
**位置**: 第275-296行

**目的**: 将 token 列表拼接成长文本,并记录每个 token 的起始位置

**返回值**:
- `full_text`: 拼接后的完整文本
- `start_offsets`: 每个 token 在 full_text 中的起始字符位置

**用途**:
1. 在 full_text 上运行正则表达式,得到字符位置
2. 将字符位置映射回 token 索引

**算法**:
```python
def _build_text_with_offsets(tokens: List[str]):
    pieces = []
    offsets = []
    cursor = 0

    for tok in tokens:
        piece = _normalize_token_text(tok)
        offsets.append(cursor)
        pieces.append(piece)
        cursor += len(piece)

    return "".join(pieces), offsets
```

**示例**:
```
tokens: ["▁Answer", ":", "▁42"]
pieces: [" Answer", ":", " 42"]
offsets: [0, 7, 8]
full_text: " Answer: 42"
```

---

### 函数: `_charpos_to_token_index()`
**位置**: 第299-312行

**目的**: 将字符位置映射回 token 索引

**方法**: 二分查找,找到最后一个 <= char_pos 的索引

**算法**:
```python
def _charpos_to_token_index(start_offsets: List[int], char_pos: int):
    # 使用 bisect_right 找到插入位置
    idx = bisect.bisect_right(start_offsets, char_pos) - 1

    # 边界处理
    if idx < 0:
        return 0
    if idx >= len(start_offsets):
        return len(start_offsets) - 1

    return idx
```

**示例**:
```
start_offsets: [0, 7, 8]
char_pos: 5 → token_idx: 0 (在 "Answer" 内)
char_pos: 7 → token_idx: 1 (在 ":" 的起始)
char_pos: 9 → token_idx: 2 (在 "42" 内)
```

---

### 函数: `build_pseudo_boundaries()`
**位置**: 第315-350行

**目的**: 构造"伪边界"(基于位置)

**使用场景**: 当真实语义边界无法命中时的兜底方案

**重要说明**:
- 伪边界只能说明"位置效应",不能证明"语义阶段切换"
- 这是"弱证据/兜底方案",不是最终结论

**算法**:
```python
def build_pseudo_boundaries(n_tokens: int):
    if n_tokens <= 0:
        return []

    # 1. 动态 margin(避免边界太靠近两端)
    margin = max(5, min(20, n_tokens // 10))

    # 2. 将序列粗分为3段(前/中/后)
    cand1 = int(n_tokens * (1.0 / 3.0))
    cand2 = int(n_tokens * (2.0 / 3.0))
    candidates = [cand1, cand2]

    # 3. 过滤掉太靠近两端的边界
    valid = [b for b in candidates if margin <= b <= n_tokens - margin]

    # 4. 如果都不合格,退化为中点边界
    if not valid:
        mid = n_tokens // 2
        if margin <= mid <= n_tokens - margin:
            return [mid]
        return []

    return valid
```

**示例**:
```
n_tokens=300, margin=20
cand1=100, cand2=200
valid=[100, 200] (都在 [20, 280] 范围内)
```

---

**文档创建时间**: 2026-01-19
**作者**: Claude (基于李康锐的原始代码)
**说明**: 这是 Part 2,包含窗口划分、曲线处理和边界检测函数
