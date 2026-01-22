#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q4：长上下文位置漂移 / 阶段切换分析

这个脚本的目标是：在“长序列”上验证 MoE 路由分布是否随位置漂移，
以及在“语义阶段边界”是否出现跳变。输出为 JSON，供后续绘图脚本聚合。

注意：本脚本面向不会写代码的同事，因此注释极其详细。
"""

import argparse
import bisect
import json
import logging
import math
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# 1. 日志设置
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# ---------------------------------------------------------------------------
# 2. 数学工具：JSD / TV / 熵
# ---------------------------------------------------------------------------
def _safe_normalize(vec: np.ndarray) -> np.ndarray:
    """
    把向量归一化为概率分布（和为1）。
    如果向量总和为 0，就返回均匀分布（避免除零）。
    """
    total = float(np.sum(vec))
    if total <= 0:
        # 返回均匀分布，防止 NaN
        return np.ones_like(vec, dtype=np.float64) / float(len(vec))
    return vec.astype(np.float64) / total


def jsd(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """
    Jensen-Shannon Divergence（基于 log2，范围约在 [0,1]）。
    输入 p, q 必须是“概率分布”（和为 1）。
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    # 小概率裁剪，避免 log(0)
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = _safe_normalize(p)
    q = _safe_normalize(q)
    m = 0.5 * (p + q)

    # KL 散度（log2）
    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.sum(a * (np.log2(a) - np.log2(b))))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def tv(p: np.ndarray, q: np.ndarray) -> float:
    """
    Total Variation 距离，范围 [0,1]。
    """
    p = _safe_normalize(p)
    q = _safe_normalize(q)
    return 0.5 * float(np.sum(np.abs(p - q)))


def entropy(p: np.ndarray, eps: float = 1e-12) -> float:
    """
    分布熵（log2）。用于观察不确定性是否随位置变化。
    """
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, eps, 1.0)
    p = _safe_normalize(p)
    return float(-np.sum(p * np.log2(p)))


# ---------------------------------------------------------------------------
# 3. 数据读取：NPZ -> 样本列表
# ---------------------------------------------------------------------------
def _iter_npz_samples(npz_path: str, layer: int) -> List[Dict[str, np.ndarray]]:
    """
    从一个 NPZ 里提取所有样本。

    兼容两种格式：
    1) 常见格式：一个 NPZ = 一个样本（数组为 2D）。
    2) 旧格式：一个 NPZ = 多个样本（数组 dtype=object）。
    """
    data = np.load(npz_path, allow_pickle=True)
    prob_key = f"layers/MoE_Gate{layer}_out_probs"
    idx_key = f"layers/MoE_Gate{layer}_out_indices"

    if prob_key not in data or idx_key not in data:
        logging.warning(f"[SKIP] 缺少关键键: {npz_path}")
        return []

    probs = data[prob_key]
    indices = data[idx_key]
    tokens_str = data.get("tokens_str", None)

    samples = []

    # 旧格式：object array -> 多样本
    if probs.dtype == object:
        n_samples = len(probs)
        for i in range(n_samples):
            sample = {
                "probs": probs[i],
                "indices": indices[i],
                "tokens_str": tokens_str[i] if tokens_str is not None else None
            }
            samples.append(sample)
    else:
        # 常见格式：单样本
        sample = {
            "probs": probs,
            "indices": indices,
            "tokens_str": tokens_str if tokens_str is not None else None
        }
        samples.append(sample)

    return samples


# ---------------------------------------------------------------------------
# 4. 分布构建：软分布（含 Other）+ 硬分布（Top-1 频次）
# ---------------------------------------------------------------------------
def build_soft_distribution(
    probs_window: np.ndarray,
    idx_window: np.ndarray,
    n_experts: int,
) -> np.ndarray:
    """
    构建“软分布”（概率累加 + Other 桶）。
    - 维度：n_experts + 1
    - Other 桶 = 1 - sum(Top-6) 的剩余概率
    """
    # 初始化：60 专家 + 1 Other
    full = np.zeros(n_experts + 1, dtype=np.float64)

    # 遍历窗口内每个 token
    for t in range(probs_window.shape[0]):
        p = probs_window[t]
        idx = idx_window[t]

        # 安全处理：防止 top-k 概率和稍大于 1
        prob_sum = float(np.sum(p))
        residual = max(0.0, 1.0 - prob_sum)

        # 把 top-k 概率累加到对应专家
        for k in range(len(p)):
            expert_id = int(idx[k])
            if 0 <= expert_id < n_experts:
                full[expert_id] += float(p[k])

        # 把剩余概率累加到 Other
        full[-1] += residual

    # 归一化成“平均分布”
    return _safe_normalize(full)


def build_hard_distribution(
    probs_window: np.ndarray,
    idx_window: np.ndarray,
    n_experts: int,
) -> np.ndarray:
    """
    构建“硬分布”（Top-1 频次）。
    - 维度：n_experts
    - Top-1 取概率最大的位置（避免假设 top-k 已排序）
    """
    counts = np.zeros(n_experts, dtype=np.float64)

    for t in range(probs_window.shape[0]):
        p = probs_window[t]
        idx = idx_window[t]
        top1_k = int(np.argmax(p))
        expert_id = int(idx[top1_k])
        if 0 <= expert_id < n_experts:
            counts[expert_id] += 1.0

    return _safe_normalize(counts)


# ---------------------------------------------------------------------------
# 5. 窗口划分与曲线重采样
# ---------------------------------------------------------------------------
def split_windows(n_tokens: int, window_size: int) -> List[Tuple[int, int]]:
    """
    把 token 序列切成不重叠窗口。
    例如 T=500, L=128 -> 3 个窗口（前 384 token）
    """
    n_windows = n_tokens // window_size
    windows = []
    for i in range(n_windows):
        start = i * window_size
        end = (i + 1) * window_size
        windows.append((start, end))
    return windows


def resample_curve(curve: List[float], target_len: int) -> List[float]:
    """
    把不同长度的曲线统一重采样到固定长度，便于跨样本平均。
    使用线性插值，横坐标统一映射到 [0, 1]。
    """
    if len(curve) == 0:
        return [0.0] * target_len
    if len(curve) == 1:
        return [curve[0]] * target_len

    x_old = np.linspace(0.0, 1.0, num=len(curve))
    x_new = np.linspace(0.0, 1.0, num=target_len)
    y_new = np.interp(x_new, x_old, np.array(curve, dtype=np.float64))
    return y_new.tolist()


def mean_ci(curves: List[List[float]]) -> Tuple[List[float], List[float]]:
    """
    计算曲线的均值与 95% 置信区间（1.96 * SEM）。
    """
    if not curves:
        return [], []
    arr = np.array(curves, dtype=np.float64)
    mean = np.mean(arr, axis=0)
    sem = np.std(arr, axis=0, ddof=1) / math.sqrt(arr.shape[0]) if arr.shape[0] > 1 else np.zeros_like(mean)
    ci = 1.96 * sem
    return mean.tolist(), ci.tolist()


# ---------------------------------------------------------------------------
# 6. 边界检测（弱规则，但更“可命中”）
# ---------------------------------------------------------------------------
def _normalize_token_text(token: str) -> str:
    """
    把单个 token 转成“更接近人类可读文本”的片段。

    说明（非常重要）：
    - tokens_str 里通常是分词后的片段（BPE / SentencePiece），
      直接用正则在 token 上匹配，很容易失败（例如“Answer:”被拆开）。
    - 所以我们先做一个“弱解码”，把常见的空格/换行符号还原，
      让正则能在“拼回的文本”里命中。

    这个函数不是严格解码，只是为了“能命中边界信号”。
    """
    text = str(token)

    # SentencePiece：前导 "▁" 表示空格
    if text.startswith("▁"):
        text = " " + text[1:]
    # GPT2 BPE：前导 "Ġ" 表示空格
    if text.startswith("Ġ"):
        text = " " + text[1:]

    # 把残留的 "▁" 也当作空格处理
    text = text.replace("▁", " ")

    # GPT2 BPE：Ċ 表示换行（可能连续出现）
    text = text.replace("Ċ", "\n")

    return text


def _build_text_with_offsets(tokens: List[str]) -> Tuple[str, List[int]]:
    """
    把 token 列表拼成一段长文本，同时返回“每个 token 在长文本中的起始位置”。

    返回：
    - full_text：拼接后的完整文本
    - start_offsets：长度等于 token 数量，
      start_offsets[i] 表示第 i 个 token 在 full_text 中的起始字符位置

    用途：
    - 我们在 full_text 上跑正则，得到字符位置
    - 再把字符位置映射回 token 索引
    """
    pieces = []
    offsets = []
    cursor = 0
    for tok in tokens:
        piece = _normalize_token_text(tok)
        offsets.append(cursor)
        pieces.append(piece)
        cursor += len(piece)
    return "".join(pieces), offsets


def _charpos_to_token_index(start_offsets: List[int], char_pos: int) -> int:
    """
    把“字符位置”映射回“token 索引”。

    思路：
    - start_offsets 是升序列表
    - 找到最后一个 <= char_pos 的索引
    """
    idx = bisect.bisect_right(start_offsets, char_pos) - 1
    if idx < 0:
        return 0
    if idx >= len(start_offsets):
        return len(start_offsets) - 1
    return idx


def build_pseudo_boundaries(n_tokens: int) -> List[int]:
    """
    构造“伪边界”（position-based）：
    - 当真实语义边界无法命中时，用“前/中/后”位置切分作替代证据。

    重要口径（必须写清楚）：
    - 伪边界只能说明“位置效应”，不能证明“语义阶段切换”。
    - 所以它是“弱证据 / 兜底方案”，不是最终结论。

    返回：
    - token 索引列表（例如 [n/3, 2n/3]）
    - 如果序列太短，则可能返回空
    """
    if n_tokens <= 0:
        return []

    # 动态 margin：短序列放宽，长序列更严格
    # 目的：避免边界太靠近两端，导致“左/右段”几乎没有信息
    margin = max(5, min(20, n_tokens // 10))

    # 目标：把序列粗分成 3 段（前/中/后）
    cand1 = int(n_tokens * (1.0 / 3.0))
    cand2 = int(n_tokens * (2.0 / 3.0))
    candidates = [cand1, cand2]

    # 过滤掉太靠近两端的边界
    valid = [b for b in candidates if margin <= b <= n_tokens - margin]

    # 如果 2 个都不合格，退化为“中点边界”
    if not valid:
        mid = n_tokens // 2
        if margin <= mid <= n_tokens - margin:
            valid = [mid]

    # 去重 + 排序
    valid = sorted(set(valid))
    return valid


def detect_boundaries(tokens: Optional[np.ndarray], task_type: str) -> List[int]:
    """
    用简单规则检测语义边界的位置（返回 token 索引）。
    这是“弱规则”，用于触发阶段对比，不能当作硬标签。
    """
    if tokens is None:
        return []

    # 把 token 转成字符串列表（防止 numpy object）
    toks = [str(t) for t in tokens]

    patterns = []
    if task_type == "gsm8k":
        patterns = [r"####", r"Answer:", r"答案", r"Final Answer"]
    elif task_type == "humaneval":
        patterns = [r"\bdef\b", r"\breturn\b"]
    elif task_type == "cmrc2018":
        patterns = [r"问题", r"答案", r"问题：", r"答案：", r"问题:", r"答案:"]
    else:
        # 无明确阶段的任务，直接返回空
        return []

    boundary_indices = set()

    # ---------------------------------------------------------------------
    # 方案A：先把 tokens 拼成文本，再用正则查找
    # 好处：能命中“跨 token 的关键词”（比如 "Answer:" 被拆开）
    # ---------------------------------------------------------------------
    try:
        full_text, offsets = _build_text_with_offsets(toks)
        if full_text:
            for pat in patterns:
                for m in re.finditer(pat, full_text, flags=re.IGNORECASE):
                    idx = _charpos_to_token_index(offsets, m.start())
                    boundary_indices.add(int(idx))
    except Exception:
        # 极端情况：拼接失败就退回 token 级匹配
        pass

    # ---------------------------------------------------------------------
    # 方案B：token 级兜底（避免方案A完全失败时边界全空）
    # ---------------------------------------------------------------------
    if not boundary_indices:
        for i, tok in enumerate(toks):
            for pat in patterns:
                if re.search(pat, tok):
                    boundary_indices.add(i)
                    break

    # ---------------------------------------------------------------------
    # 去重 + 排序，并做“邻近合并”
    # 说明：
    # - 同一个边界关键词可能被切成多个 token，
    #   会产生一串很近的 index，这会重复计算边界。
    # - 这里简单合并：相距 < 5 的视为同一边界，只保留第一个。
    # ---------------------------------------------------------------------
    boundary_indices = sorted(boundary_indices)
    merged = []
    last = -999
    for idx in boundary_indices:
        if idx - last >= 5:
            merged.append(idx)
            last = idx

    return merged


# ---------------------------------------------------------------------------
# 7. 主分析函数
# ---------------------------------------------------------------------------
def analyze_q4(
    data_dir: str,
    output_dir: str,
    window_sizes: List[int],
    min_tokens: int,
    n_experts: int,
    layer: int,
    shuffle_runs: int,
    max_files: int,
    max_samples: int,
    seed: int,
) -> None:
    """
    入口：执行 P0 检查 + 位置漂移 + 阶段切换 + 熵曲线。
    最终输出 q4_summary.json。
    """
    rng = np.random.RandomState(seed)

    os.makedirs(output_dir, exist_ok=True)

    # --- P0：数据结构检查统计 ---
    token_lengths = []
    prob_sums = []
    has_tokens_str = True
    multi_sample_detected = False
    total_samples = 0
    used_samples = 0

    # 任务类型从目录名推断
    data_dir_name = os.path.basename(os.path.normpath(data_dir)).lower()
    if "gsm8k" in data_dir_name:
        task_type = "gsm8k"
    elif "humaneval" in data_dir_name:
        task_type = "humaneval"
    elif "cmrc" in data_dir_name:
        task_type = "cmrc2018"
    elif "wiki" in data_dir_name:
        task_type = "wiki"
    elif "piqa" in data_dir_name:
        task_type = "piqa"
    elif "winogrande" in data_dir_name:
        task_type = "winogrande"
    else:
        task_type = "unknown"

    # --- 用于收集曲线 ---
    # position_curves[dist_type][window_size] = list of curves
    position_curves = {"soft": {}, "hard": {}}
    baseline_curves = {"soft": {}, "hard": {}}
    entropy_curves = {"soft": {}, "hard": {}}
    last_jsd_values = {"soft": {}, "hard": {}}
    baseline_last_jsd = {"soft": {}, "hard": {}}

    # 边界统计（真实 vs 随机）
    boundary_stats = {
        "soft": {"real": [], "random": []},
        "hard": {"real": [], "random": []},
        "n_boundaries": 0,
        "n_real_boundaries": 0,
        "n_pseudo_boundaries": 0,
    }

    # 读取所有 npz
    npz_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".npz")])
    if max_files > 0:
        npz_files = npz_files[:max_files]

    logging.info(f"[INFO] 找到 {len(npz_files)} 个 npz 文件")

    for npz_name in npz_files:
        npz_path = os.path.join(data_dir, npz_name)

        samples = _iter_npz_samples(npz_path, layer)
        if len(samples) > 1:
            multi_sample_detected = True

        for sample in samples:
            total_samples += 1

            probs = sample["probs"]
            indices = sample["indices"]
            tokens_str = sample["tokens_str"]

            # tokens 长度
            n_tokens = probs.shape[0]
            token_lengths.append(int(n_tokens))

            # tokens_str 是否存在
            if tokens_str is None:
                has_tokens_str = False

            # 统计 Top-6 概率和（P0 检查）
            prob_sum_mean = float(np.mean(np.sum(probs, axis=1)))
            prob_sums.append(prob_sum_mean)

            # 长度过滤（位置漂移至少要两窗）
            if n_tokens < min_tokens:
                continue

            # 样本数量限制
            if max_samples > 0 and used_samples >= max_samples:
                continue

            used_samples += 1

            # 对每个窗口大小做位置漂移
            for w in window_sizes:
                windows = split_windows(n_tokens, w)
                if len(windows) < 2:
                    continue

                # 准备窗口分布列表
                soft_windows = []
                hard_windows = []
                ent_soft = []
                ent_hard = []

                for (s, e) in windows:
                    probs_w = probs[s:e]
                    idx_w = indices[s:e]
                    soft_dist = build_soft_distribution(probs_w, idx_w, n_experts)
                    hard_dist = build_hard_distribution(probs_w, idx_w, n_experts)
                    soft_windows.append(soft_dist)
                    hard_windows.append(hard_dist)
                    ent_soft.append(entropy(soft_dist))
                    ent_hard.append(entropy(hard_dist))

                # 真实曲线：首窗对其它窗
                soft_curve = [jsd(soft_windows[0], x) for x in soft_windows]
                hard_curve = [jsd(hard_windows[0], x) for x in hard_windows]

                # 重采样到固定长度（20点）
                soft_curve_rs = resample_curve(soft_curve, 20)
                hard_curve_rs = resample_curve(hard_curve, 20)

                # 保存
                position_curves["soft"].setdefault(w, []).append(soft_curve_rs)
                position_curves["hard"].setdefault(w, []).append(hard_curve_rs)
                last_jsd_values["soft"].setdefault(w, []).append(float(soft_curve[-1]))
                last_jsd_values["hard"].setdefault(w, []).append(float(hard_curve[-1]))

                # 熵曲线
                entropy_curves["soft"].setdefault(w, []).append(resample_curve(ent_soft, 20))
                entropy_curves["hard"].setdefault(w, []).append(resample_curve(ent_hard, 20))

                # 置乱基线：打乱窗口顺序
                if shuffle_runs > 0:
                    soft_shuffle_curves = []
                    hard_shuffle_curves = []
                    for _ in range(shuffle_runs):
                        perm = rng.permutation(len(soft_windows))
                        soft_shuffled = [soft_windows[i] for i in perm]
                        hard_shuffled = [hard_windows[i] for i in perm]
                        soft_c = [jsd(soft_shuffled[0], x) for x in soft_shuffled]
                        hard_c = [jsd(hard_shuffled[0], x) for x in hard_shuffled]
                        soft_shuffle_curves.append(resample_curve(soft_c, 20))
                        hard_shuffle_curves.append(resample_curve(hard_c, 20))
                        baseline_last_jsd["soft"].setdefault(w, []).append(float(soft_c[-1]))
                        baseline_last_jsd["hard"].setdefault(w, []).append(float(hard_c[-1]))

                    # 把 shuffle 曲线也保存下来，后面做均值+CI
                    baseline_curves["soft"].setdefault(w, []).extend(soft_shuffle_curves)
                    baseline_curves["hard"].setdefault(w, []).extend(hard_shuffle_curves)

            # 阶段边界分析（真实边界优先；没有则使用伪边界）
            boundaries: List[int] = []
            boundary_source = "none"

            # 真实边界：只能在 tokens_str 存在时尝试
            if tokens_str is not None:
                boundaries = detect_boundaries(tokens_str, task_type)
                # 真实边界的安全过滤：太靠近两端的边界直接丢弃
                boundaries = [b for b in boundaries if 20 <= b <= n_tokens - 20]
                if boundaries:
                    boundary_source = "real"

            # 如果真实边界为空，使用“伪边界”兜底
            if not boundaries:
                boundaries = build_pseudo_boundaries(n_tokens)
                if boundaries:
                    boundary_source = "pseudo"

            # 有边界就计算 JSD
            if boundaries:
                for b in boundaries:
                    probs_left = probs[:b]
                    probs_right = probs[b:]
                    idx_left = indices[:b]
                    idx_right = indices[b:]

                    soft_left = build_soft_distribution(probs_left, idx_left, n_experts)
                    soft_right = build_soft_distribution(probs_right, idx_right, n_experts)
                    hard_left = build_hard_distribution(probs_left, idx_left, n_experts)
                    hard_right = build_hard_distribution(probs_right, idx_right, n_experts)

                    boundary_stats["soft"]["real"].append(jsd(soft_left, soft_right))
                    boundary_stats["hard"]["real"].append(jsd(hard_left, hard_right))

                # 随机边界基线（无论是真边界还是伪边界，都需要随机对照）
                for _ in range(max(1, shuffle_runs)):
                    # 随机边界也要保证不靠近两端
                    rand_margin = max(5, min(20, n_tokens // 10))
                    if n_tokens <= rand_margin * 2:
                        break
                    rand_b = rng.randint(rand_margin, n_tokens - rand_margin)
                    probs_left = probs[:rand_b]
                    probs_right = probs[rand_b:]
                    idx_left = indices[:rand_b]
                    idx_right = indices[rand_b:]

                    soft_left = build_soft_distribution(probs_left, idx_left, n_experts)
                    soft_right = build_soft_distribution(probs_right, idx_right, n_experts)
                    hard_left = build_hard_distribution(probs_left, idx_left, n_experts)
                    hard_right = build_hard_distribution(probs_right, idx_right, n_experts)

                    boundary_stats["soft"]["random"].append(jsd(soft_left, soft_right))
                    boundary_stats["hard"]["random"].append(jsd(hard_left, hard_right))

                # 统计边界来源
                boundary_stats["n_boundaries"] += len(boundaries)
                if boundary_source == "real":
                    boundary_stats["n_real_boundaries"] += len(boundaries)
                elif boundary_source == "pseudo":
                    boundary_stats["n_pseudo_boundaries"] += len(boundaries)

    # -----------------------------------------------------------------------
    # 8. 汇总统计
    # -----------------------------------------------------------------------
    summary = {
        "meta": {
            "data_dir": data_dir,
            "task_type": task_type,
            "n_experts": n_experts,
            "layer": layer,
            "window_sizes": window_sizes,
            "min_tokens": min_tokens,
            "shuffle_runs": shuffle_runs,
            "seed": seed,
        },
        "p0": {
            "total_samples": total_samples,
            "used_samples": used_samples,
            "token_length_mean": float(np.mean(token_lengths)) if token_lengths else 0.0,
            "token_length_median": float(np.median(token_lengths)) if token_lengths else 0.0,
            "token_length_min": int(np.min(token_lengths)) if token_lengths else 0,
            "token_length_max": int(np.max(token_lengths)) if token_lengths else 0,
            "prob_sum_mean": float(np.mean(prob_sums)) if prob_sums else 0.0,
            "has_tokens_str": bool(has_tokens_str),
            "multi_sample_detected": bool(multi_sample_detected),
        },
        "position": {},
        "entropy": {},
        "boundary": {},
    }

    # 位置漂移曲线汇总
    for dist_type in ["soft", "hard"]:
        summary["position"][dist_type] = {}
        summary["entropy"][dist_type] = {}
        for w in window_sizes:
            mean_curve, ci_curve = mean_ci(position_curves[dist_type].get(w, []))
            base_mean, base_ci = mean_ci(baseline_curves[dist_type].get(w, []))
            ent_mean, ent_ci = mean_ci(entropy_curves[dist_type].get(w, []))

            last_vals = last_jsd_values[dist_type].get(w, [])
            base_last_vals = baseline_last_jsd[dist_type].get(w, [])

            summary["position"][dist_type][str(w)] = {
                "mean_curve": mean_curve,
                "ci_curve": ci_curve,
                "baseline_mean_curve": base_mean,
                "baseline_ci_curve": base_ci,
                "mean_last": float(np.mean(last_vals)) if last_vals else 0.0,
                "n_samples": len(last_vals),
                "baseline_mean_last": float(np.mean(base_last_vals)) if base_last_vals else 0.0,
            }

            summary["entropy"][dist_type][str(w)] = {
                "mean_curve": ent_mean,
                "ci_curve": ent_ci,
                "n_samples": len(entropy_curves[dist_type].get(w, [])),
            }

    # 边界统计汇总
    def _mean_std(x: List[float]) -> Tuple[float, float]:
        if not x:
            return 0.0, 0.0
        return float(np.mean(x)), float(np.std(x, ddof=1)) if len(x) > 1 else 0.0

    real_mean_s, real_std_s = _mean_std(boundary_stats["soft"]["real"])
    rand_mean_s, rand_std_s = _mean_std(boundary_stats["soft"]["random"])
    real_mean_h, real_std_h = _mean_std(boundary_stats["hard"]["real"])
    rand_mean_h, rand_std_h = _mean_std(boundary_stats["hard"]["random"])

    # 边界来源统计（用于区分真实/伪边界）
    if boundary_stats["n_real_boundaries"] > 0 and boundary_stats["n_pseudo_boundaries"] > 0:
        boundary_source = "mixed"
    elif boundary_stats["n_real_boundaries"] > 0:
        boundary_source = "real"
    elif boundary_stats["n_pseudo_boundaries"] > 0:
        boundary_source = "pseudo"
    else:
        boundary_source = "none"

    summary["boundary"] = {
        "soft": {
            "real_mean": real_mean_s,
            "real_std": real_std_s,
            "random_mean": rand_mean_s,
            "random_std": rand_std_s,
            "n_real": len(boundary_stats["soft"]["real"]),
            "n_random": len(boundary_stats["soft"]["random"]),
        },
        "hard": {
            "real_mean": real_mean_h,
            "real_std": real_std_h,
            "random_mean": rand_mean_h,
            "random_std": rand_std_h,
            "n_real": len(boundary_stats["hard"]["real"]),
            "n_random": len(boundary_stats["hard"]["random"]),
        },
        "n_boundaries": boundary_stats["n_boundaries"],
        "n_real_boundaries": boundary_stats["n_real_boundaries"],
        "n_pseudo_boundaries": boundary_stats["n_pseudo_boundaries"],
        "source": boundary_source,
    }

    # 保存结果
    out_path = os.path.join(output_dir, "q4_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logging.info(f"[DONE] 写出结果: {out_path}")


# ---------------------------------------------------------------------------
# 8. 命令行入口
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Q4 长上下文位置漂移 / 阶段切换分析"
    )
    parser.add_argument("--data_dir", type=str, required=True, help="单个数据集的 NPZ 目录")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--window_sizes", type=str, default="64,128,256", help="窗口大小列表，如 64,128,256")
    parser.add_argument("--min_tokens", type=int, default=500, help="最小 token 数（短样本过滤）")
    parser.add_argument("--n_experts", type=int, default=60, help="专家数（必须与采集一致）")
    parser.add_argument("--layer", type=int, default=0, help="使用哪一层的路由数据")
    parser.add_argument("--shuffle_runs", type=int, default=10, help="置乱基线次数")
    parser.add_argument("--max_files", type=int, default=-1, help="最多处理多少个 npz 文件")
    parser.add_argument("--max_samples", type=int, default=-1, help="最多处理多少个样本（过滤后）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    window_sizes = [int(x.strip()) for x in args.window_sizes.split(",") if x.strip()]

    analyze_q4(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        window_sizes=window_sizes,
        min_tokens=args.min_tokens,
        n_experts=args.n_experts,
        layer=args.layer,
        shuffle_runs=args.shuffle_runs,
        max_files=args.max_files,
        max_samples=args.max_samples,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
