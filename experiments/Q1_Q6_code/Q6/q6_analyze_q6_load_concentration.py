#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q6 负载集中度分析脚本（强化版）

核心目标：
1) 比较不同任务的“有效专家数 n_eff”，判断负载是否更集中/更分散
2) 用 Gini / Top-N 覆盖率 / Lorenz 曲线做互证
3) 做长度控制 + 置乱基线，避免把样本长度差当成任务差

非常重要的口径说明（写给非专业同学）：
- 本脚本只讨论“负载集中/偏斜”，不直接推断“泛化能力强/弱”。
- soft 口径只来自 Top-(K+2) 概率，仍然是“下界估计”。
- hard 口径只看 Top-1 专家频次，是最保守的集中度估计。

输出：
- q6_results.json：完整结果（含所有证据）
- figures/：可视化图表（由 plot_q6_figures.py 读取生成）
"""

import argparse
import json
import logging
import math
import os
import random
from typing import Dict, List, Any, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ===========================================================================
# 全局常量（默认值）
# ===========================================================================
N_EXPERTS = 60  # Qwen1.5-MoE-A2.7B 固定 60 专家
TOP_N_LIST = [1, 3, 5, 10, 20]  # Top-N 覆盖率的 N 值

# ===========================================================================
# 随机种子（保证可复现）
# ===========================================================================

def set_random_seed(seed: int) -> None:
    """
    设置随机种子。
    这样同样的输入与参数，每次结果一致，方便对比和复现。
    """
    random.seed(seed)
    np.random.seed(seed)


# ===========================================================================
# 数学工具（n_eff / Gini / Top-N / Lorenz 等）
# ===========================================================================

def safe_normalize(vec: np.ndarray) -> np.ndarray:
    """
    安全归一化：
    - 如果总和为 0，则返回均匀分布
    - 否则返回归一化后的概率分布
    """
    total = float(np.sum(vec))
    if total <= 0:
        return np.ones_like(vec, dtype=np.float64) / float(len(vec))
    return vec.astype(np.float64) / total


def n_eff_entropy(dist: np.ndarray, eps: float = 1e-12) -> float:
    """
    熵口径的有效专家数：n_eff = exp(H)
    H 是自然对数熵：H = -sum(p_i * ln(p_i))

    注意：
    - n_eff 代表“等效均匀分布的专家数”
    - 这个值越大，说明分布越分散
    """
    p = safe_normalize(dist)
    p = np.clip(p, eps, 1.0)
    h = float(-np.sum(p * np.log(p)))
    return float(np.exp(h))


def n_eff_simpson(dist: np.ndarray, eps: float = 1e-12) -> float:
    """
    Simpson 口径的有效专家数：n_eff = 1 / sum(p_i^2)

    注意：
    - 与 exp(H) 不等价，只在某些分布下接近
    - 这里作为“并列口径”与稳健性对照
    """
    p = safe_normalize(dist)
    p = np.clip(p, eps, 1.0)
    return float(1.0 / np.sum(p ** 2))


def gini_coefficient(dist: np.ndarray) -> float:
    """
    Gini 系数：衡量分布是否不均匀
    - 0 表示完全均匀
    - 1 表示完全集中
    """
    values = np.asarray(dist, dtype=np.float64)
    if values.size == 0 or np.sum(values) == 0:
        return 0.0
    values = np.sort(values)
    n = len(values)
    cumsum = np.cumsum(values)
    return float((2 * np.sum((np.arange(1, n + 1) * values)) - (n + 1) * cumsum[-1]) / (n * cumsum[-1]))


def top_n_coverage(dist: np.ndarray, n: int) -> float:
    """
    Top-N 覆盖率：最大的 N 个专家的概率质量之和
    - 越大说明越集中
    """
    values = np.asarray(dist, dtype=np.float64)
    if values.size == 0 or np.sum(values) == 0:
        return 0.0
    sorted_vals = np.sort(values)[::-1]
    return float(np.sum(sorted_vals[:n]) / np.sum(values))


def lorenz_curve(dist: np.ndarray) -> Tuple[List[float], List[float]]:
    """
    Lorenz 曲线：
    - 横轴：专家累计占比
    - 纵轴：负载累计占比
    """
    values = safe_normalize(dist)
    values = np.sort(values)
    cum = np.cumsum(values)
    x = np.linspace(0.0, 1.0, len(values) + 1)
    y = np.concatenate(([0.0], cum))
    return x.tolist(), y.tolist()


def bootstrap_mean_ci(values: List[float], rng: np.random.Generator,
                      n_bootstrap: int = 1000, ci: float = 0.95,
                      max_points: int = 20000) -> Dict[str, float]:
    """
    Bootstrap 置信区间（均值）：
    - 先抽样再 bootstrap，避免超大样本导致计算极慢
    """
    if not values:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "std": 0.0, "n": 0}

    arr = np.array(values, dtype=np.float64)
    if len(arr) > max_points:
        arr = rng.choice(arr, size=max_points, replace=False)

    means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means.append(np.mean(sample))

    alpha = (1.0 - ci) / 2.0
    ci_low = float(np.percentile(means, alpha * 100))
    ci_high = float(np.percentile(means, (1.0 - alpha) * 100))

    return {
        "mean": float(np.mean(arr)),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "std": float(np.std(arr)),
        "n": int(len(arr)),
    }


def cohens_d(group1: List[float], group2: List[float]) -> float:
    """
    Cohen's d：简单效应量（仅供参考，不作为硬阈值）
    """
    if not group1 or not group2:
        return 0.0
    n1, n2 = len(group1), len(group2)
    m1, m2 = np.mean(group1), np.mean(group2)
    s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    pooled = math.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    if pooled < 1e-10:
        return 0.0
    return float((m1 - m2) / pooled)


# ===========================================================================
# 数据加载与样本拆分
# ===========================================================================

def _ensure_2d(arr: np.ndarray) -> np.ndarray:
    """
    确保数组为二维（n_tokens, top_k）
    """
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr


def split_samples_from_npz(probs_raw: np.ndarray, indices_raw: np.ndarray) -> List[Dict[str, Any]]:
    """
    把 NPZ 里的 object 数组拆成“样本列表”。
    - 如果是 object array，每个元素就是一个样本
    - 如果不是 object array，则整个数组就是一个样本
    """
    samples = []

    # case 1: object array（多样本）
    if probs_raw.dtype == object or indices_raw.dtype == object:
        for p, idx in zip(probs_raw, indices_raw):
            if p is None or idx is None:
                continue
            p = _ensure_2d(p)
            idx = _ensure_2d(idx)
            if p.shape[0] == 0:
                continue
            samples.append({
                "probs": p.astype(np.float64),
                "indices": idx.astype(np.int64),
                "length": int(p.shape[0]),
            })
        return samples

    # case 2: 单样本（整块数组）
    probs = _ensure_2d(probs_raw)
    indices = _ensure_2d(indices_raw)
    if probs.shape[0] == 0:
        return []

    samples.append({
        "probs": probs.astype(np.float64),
        "indices": indices.astype(np.int64),
        "length": int(probs.shape[0]),
    })
    return samples


def load_npz_layer(npz_path: str, layer: int) -> List[Dict[str, Any]]:
    """
    读取单个 NPZ 文件的指定层输出。
    返回该文件内的“样本列表”。
    """
    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception:
        return []

    prob_key = f"layers/MoE_Gate{layer}_out_probs"
    idx_key = f"layers/MoE_Gate{layer}_out_indices"

    if prob_key not in data or idx_key not in data:
        return []

    return split_samples_from_npz(data[prob_key], data[idx_key])


def load_task_samples(data_dir: str, layer: int, max_files: int, min_tokens: int) -> List[Dict[str, Any]]:
    """
    加载一个任务目录下的所有样本。
    - max_files = -1 表示不限制文件数
    - min_tokens 过滤太短的样本
    """
    if not os.path.isdir(data_dir):
        logging.warning(f"目录不存在: {data_dir}")
        return []

    npz_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".npz")])
    if max_files > 0:
        npz_files = npz_files[:max_files]

    all_samples = []
    total_tokens = 0
    for fname in npz_files:
        npz_path = os.path.join(data_dir, fname)
        samples = load_npz_layer(npz_path, layer)
        for s in samples:
            if s["length"] >= min_tokens:
                all_samples.append(s)
                total_tokens += s["length"]

    logging.info(f"  加载 {len(all_samples)} 个样本, 共 {total_tokens} tokens from {data_dir}")
    return all_samples


# ===========================================================================
# 长度控制 + 等量抽样
# ===========================================================================

def build_length_controlled_samples(task_samples: Dict[str, List[Dict[str, Any]]],
                                    segment_len: int,
                                    seed: int) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    """
    把每个任务的样本切成“等长度片段”，再做等量抽样。

    设计目的：
    - 避免“任务平均长度不同”导致的统计偏差
    - 所有任务在样本数和长度上都公平对齐
    """
    rng = np.random.default_rng(seed)
    stats = {"segment_len": segment_len, "tasks": {}}

    # 先对每个任务做“长度过滤 + 随机切片”
    temp_segments = {}
    for task, samples in task_samples.items():
        segments = []
        for s in samples:
            if s["length"] < segment_len:
                continue
            # 在样本内随机取一个连续片段
            start = int(rng.integers(0, s["length"] - segment_len + 1))
            end = start + segment_len
            segments.append({
                "probs": s["probs"][start:end],
                "indices": s["indices"][start:end],
                "length": segment_len,
            })

        temp_segments[task] = segments
        stats["tasks"][task] = {
            "raw_samples": len(samples),
            "raw_tokens": int(sum([x["length"] for x in samples])),
            "kept_samples": len(segments),
            "kept_tokens": int(len(segments) * segment_len),
        }

    # 找到最小样本数，用于等量抽样
    valid_counts = [len(v) for v in temp_segments.values() if len(v) > 0]
    min_count = min(valid_counts) if valid_counts else 0
    stats["min_samples_per_task"] = int(min_count)

    # 按最小样本数等量抽样
    controlled = {}
    for task, segments in temp_segments.items():
        if len(segments) == 0 or min_count == 0:
            controlled[task] = []
            stats["tasks"][task]["equalized_samples"] = 0
            continue
        if len(segments) > min_count:
            chosen_idx = rng.choice(len(segments), size=min_count, replace=False)
            controlled[task] = [segments[i] for i in chosen_idx]
        else:
            controlled[task] = segments
        stats["tasks"][task]["equalized_samples"] = len(controlled[task])
        stats["tasks"][task]["equalized_tokens"] = len(controlled[task]) * segment_len

    return controlled, stats


# ===========================================================================
# 分布构建（soft/hard）
# ===========================================================================

def build_distributions_for_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    从一个样本里构建：
    - soft 分布（60 专家 + Other）
    - soft 分布（仅 60 专家，重归一化）
    - hard 分布（Top-1 频次）
    """
    probs = sample["probs"]
    indices = sample["indices"]
    n_tokens = sample["length"]

    # ===== soft: 累计每个专家的概率质量 =====
    soft_sum = np.zeros(N_EXPERTS, dtype=np.float64)

    flat_idx = indices.reshape(-1)
    flat_probs = probs.reshape(-1)
    mask = (flat_idx >= 0) & (flat_idx < N_EXPERTS)
    np.add.at(soft_sum, flat_idx[mask], flat_probs[mask])

    # Other 桶的概率质量（每个 token 1 - sum(topk)）
    sum_p = np.sum(probs, axis=1)
    other_sum = float(np.sum(np.maximum(0.0, 1.0 - sum_p)))

    # soft_full: 60 专家 + Other
    soft_full = np.concatenate([soft_sum, np.array([other_sum])], axis=0)
    soft_full_dist = safe_normalize(soft_full)

    # soft_renorm: 只看 60 专家，再重归一化
    soft_renorm_dist = safe_normalize(soft_sum)

    # ===== hard: Top-1 专家频次 =====
    top1 = indices[:, 0] if indices.ndim > 1 else indices
    hard_counts = np.zeros(N_EXPERTS, dtype=np.float64)
    mask1 = (top1 >= 0) & (top1 < N_EXPERTS)
    np.add.at(hard_counts, top1[mask1], 1)
    hard_dist = safe_normalize(hard_counts)

    # 记录 Other 的平均质量（便于分析截断影响）
    other_mass_mean = other_sum / n_tokens if n_tokens > 0 else 0.0

    return {
        "soft_full_dist": soft_full_dist,
        "soft_renorm_dist": soft_renorm_dist,
        "hard_dist": hard_dist,
        "other_mass_mean": other_mass_mean,
        "soft_sum": soft_sum,
        "other_sum": other_sum,
        "hard_counts": hard_counts,
    }


def token_level_n_eff_from_topk(probs: np.ndarray) -> Dict[str, List[float]]:
    """
    直接从 Top-K 概率计算 token 级 n_eff：
    - renorm_60：只用 Top-K 概率，重归一化
    - full_61：Top-K + Other 桶（下界估计）

    注意：token 级 hard 口径恒为 1，没有意义，所以这里不算 hard。
    """
    eps = 1e-12
    sum_p = np.sum(probs, axis=1)

    # renorm：只看 top-k，重归一化（下界）
    sum_safe = np.where(sum_p > eps, sum_p, 1.0)
    p_renorm = probs / sum_safe[:, None]
    p_renorm = np.clip(p_renorm, eps, 1.0)

    h_renorm = -np.sum(p_renorm * np.log(p_renorm), axis=1)
    n_eff_entropy_renorm = np.exp(h_renorm)
    n_eff_simpson_renorm = 1.0 / np.sum(p_renorm ** 2, axis=1)

    # full：top-k + Other（一个桶）
    other = np.maximum(0.0, 1.0 - sum_p)
    total = sum_p + other
    total = np.where(total > eps, total, 1.0)
    p_full = probs / total[:, None]
    other_full = other / total
    p_full = np.clip(p_full, eps, 1.0)
    other_full = np.clip(other_full, eps, 1.0)

    h_full = -np.sum(p_full * np.log(p_full), axis=1) - other_full * np.log(other_full)
    n_eff_entropy_full = np.exp(h_full)
    n_eff_simpson_full = 1.0 / (np.sum(p_full ** 2, axis=1) + other_full ** 2)

    return {
        "renorm_entropy": n_eff_entropy_renorm.tolist(),
        "renorm_simpson": n_eff_simpson_renorm.tolist(),
        "full_entropy": n_eff_entropy_full.tolist(),
        "full_simpson": n_eff_simpson_full.tolist(),
        "other_mass": other.tolist(),
    }


# ===========================================================================
# 核心分析：E1 n_eff / E2 Gini+TopN / E3 长度控制 / E4 层间一致性 / E5 置乱基线
# ===========================================================================

def analyze_task_metrics(task_samples: Dict[str, List[Dict[str, Any]]],
                         rng: np.random.Generator,
                         n_bootstrap: int,
                         bootstrap_max_points: int) -> Dict[str, Any]:
    """
    对任务做核心指标分析：
    - n_eff（token级 / sample级 / task级）
    - Gini / Top-N / Lorenz
    """
    results = {
        "tasks": {},
        "ranking": {"soft_token": [], "soft_sample": [], "hard_sample": []},
    }

    for task, samples in task_samples.items():
        if not samples:
            continue

        # ===== 1) token 级 n_eff =====
        token_renorm_entropy = []
        token_renorm_simpson = []
        token_full_entropy = []
        token_full_simpson = []
        other_mass_list = []

        # ===== 2) sample 级分布 =====
        sample_soft_entropy = []
        sample_soft_simpson = []
        sample_soft_full_entropy = []
        sample_soft_full_simpson = []
        sample_hard_entropy = []
        sample_hard_simpson = []

        # ===== 3) task 级累积分布 =====
        task_soft_sum = np.zeros(N_EXPERTS, dtype=np.float64)
        task_other_sum = 0.0
        task_hard_counts = np.zeros(N_EXPERTS, dtype=np.float64)

        for s in samples:
            probs = s["probs"]
            indices = s["indices"]

            # token 级 n_eff（soft）
            token_stats = token_level_n_eff_from_topk(probs)
            token_renorm_entropy.extend(token_stats["renorm_entropy"])
            token_renorm_simpson.extend(token_stats["renorm_simpson"])
            token_full_entropy.extend(token_stats["full_entropy"])
            token_full_simpson.extend(token_stats["full_simpson"])
            other_mass_list.extend(token_stats["other_mass"])

            # sample 分布
            dist_info = build_distributions_for_sample(s)

            # sample 级 n_eff
            sample_soft_entropy.append(n_eff_entropy(dist_info["soft_renorm_dist"]))
            sample_soft_simpson.append(n_eff_simpson(dist_info["soft_renorm_dist"]))
            sample_soft_full_entropy.append(n_eff_entropy(dist_info["soft_full_dist"]))
            sample_soft_full_simpson.append(n_eff_simpson(dist_info["soft_full_dist"]))
            sample_hard_entropy.append(n_eff_entropy(dist_info["hard_dist"]))
            sample_hard_simpson.append(n_eff_simpson(dist_info["hard_dist"]))

            # task 级累积
            task_soft_sum += dist_info["soft_sum"]
            task_other_sum += dist_info["other_sum"]
            task_hard_counts += dist_info["hard_counts"]

        # ===== 汇总任务级分布 =====
        task_soft_full = np.concatenate([task_soft_sum, np.array([task_other_sum])], axis=0)
        task_soft_full_dist = safe_normalize(task_soft_full)
        task_soft_renorm_dist = safe_normalize(task_soft_sum)
        task_hard_dist = safe_normalize(task_hard_counts)

        # ===== 任务级指标 =====
        task_soft_entropy = n_eff_entropy(task_soft_renorm_dist)
        task_soft_simpson = n_eff_simpson(task_soft_renorm_dist)
        task_soft_full_entropy = n_eff_entropy(task_soft_full_dist)
        task_soft_full_simpson = n_eff_simpson(task_soft_full_dist)
        task_hard_entropy = n_eff_entropy(task_hard_dist)
        task_hard_simpson = n_eff_simpson(task_hard_dist)

        # ===== E2 指标（Gini / Top-N / Lorenz） =====
        def summarize_concentration(dist: np.ndarray) -> Dict[str, Any]:
            return {
                "gini": gini_coefficient(dist),
                "max_share": float(np.max(dist)) if dist.size > 0 else 0.0,
                "topn": {f"top_{n}": top_n_coverage(dist, n) for n in TOP_N_LIST},
                "lorenz": {"x": lorenz_curve(dist)[0], "y": lorenz_curve(dist)[1]},
            }

        results["tasks"][task] = {
            "soft": {
                "token_level": {
                    "renorm_60": {
                        "entropy": bootstrap_mean_ci(token_renorm_entropy, rng, n_bootstrap, max_points=bootstrap_max_points),
                        "simpson": bootstrap_mean_ci(token_renorm_simpson, rng, n_bootstrap, max_points=bootstrap_max_points),
                    },
                    "full_61": {
                        "entropy": bootstrap_mean_ci(token_full_entropy, rng, n_bootstrap, max_points=bootstrap_max_points),
                        "simpson": bootstrap_mean_ci(token_full_simpson, rng, n_bootstrap, max_points=bootstrap_max_points),
                    },
                    "other_mass": {
                        "mean": float(np.mean(other_mass_list)) if other_mass_list else 0.0,
                        "std": float(np.std(other_mass_list)) if other_mass_list else 0.0,
                    }
                },
                "sample_level": {
                    "renorm_60": {
                        "entropy": bootstrap_mean_ci(sample_soft_entropy, rng, n_bootstrap),
                        "simpson": bootstrap_mean_ci(sample_soft_simpson, rng, n_bootstrap),
                    },
                    "full_61": {
                        "entropy": bootstrap_mean_ci(sample_soft_full_entropy, rng, n_bootstrap),
                        "simpson": bootstrap_mean_ci(sample_soft_full_simpson, rng, n_bootstrap),
                    },
                },
                "task_level": {
                    "renorm_60": {"entropy": task_soft_entropy, "simpson": task_soft_simpson},
                    "full_61": {"entropy": task_soft_full_entropy, "simpson": task_soft_full_simpson},
                },
                "concentration": summarize_concentration(task_soft_renorm_dist),
            },
            "hard": {
                "sample_level": {
                    "entropy": bootstrap_mean_ci(sample_hard_entropy, rng, n_bootstrap),
                    "simpson": bootstrap_mean_ci(sample_hard_simpson, rng, n_bootstrap),
                },
                "task_level": {"entropy": task_hard_entropy, "simpson": task_hard_simpson},
                "concentration": summarize_concentration(task_hard_dist),
            }
        }

    # ===== 任务排名（只做参考） =====
    tasks = list(results["tasks"].keys())
    results["ranking"]["soft_token"] = sorted(
        tasks,
        key=lambda t: results["tasks"][t]["soft"]["token_level"]["renorm_60"]["entropy"]["mean"],
        reverse=True,
    )
    results["ranking"]["soft_sample"] = sorted(
        tasks,
        key=lambda t: results["tasks"][t]["soft"]["sample_level"]["renorm_60"]["entropy"]["mean"],
        reverse=True,
    )
    results["ranking"]["hard_sample"] = sorted(
        tasks,
        key=lambda t: results["tasks"][t]["hard"]["sample_level"]["entropy"]["mean"],
        reverse=True,
    )

    return results


def analyze_layer_consistency(task_paths: Dict[str, str],
                              layers: List[int],
                              max_files: int,
                              min_tokens: int) -> Dict[str, Any]:
    """
    分层分析：看 n_eff 在不同层是否稳定。
    这里只做“soft 口径 + token级”用于快速观察。
    """
    results = {"layers": layers, "tasks": {}}

    for task, task_path in task_paths.items():
        layer_stats = {}
        for layer in layers:
            samples = load_task_samples(task_path, layer, max_files, min_tokens)
            if not samples:
                continue
            all_token_n_eff = []
            for s in samples:
                token_stats = token_level_n_eff_from_topk(s["probs"])
                all_token_n_eff.extend(token_stats["renorm_entropy"])
            if all_token_n_eff:
                layer_stats[layer] = {
                    "mean": float(np.mean(all_token_n_eff)),
                    "std": float(np.std(all_token_n_eff)),
                    "n": len(all_token_n_eff),
                }

        if layer_stats:
            means = [layer_stats[l]["mean"] for l in layer_stats]
            cv = float(np.std(means) / np.mean(means)) if np.mean(means) > 0 else 0.0
            results["tasks"][task] = {
                "layers": layer_stats,
                "cross_layer_cv": cv,
                "is_stable": cv < 0.1,
            }

    return results


def analyze_label_shuffle(sample_bank: Dict[str, List[Dict[str, Any]]],
                           n_shuffles: int,
                           seed: int) -> Dict[str, Any]:
    """
    任务标签置乱基线：
    - 保持每个任务样本数一致
    - 仅打乱样本所属任务
    """
    rng = np.random.default_rng(seed)

    tasks = list(sample_bank.keys())
    task_counts = {t: len(sample_bank[t]) for t in tasks}
    total_samples = sum(task_counts.values())

    # 把样本扁平化
    all_samples = []
    for t in tasks:
        all_samples.extend(sample_bank[t])

    # 结果容器（先收集原始列表，最后再做统计）
    results = {"n_shuffles": n_shuffles, "tasks": {}}
    accum = {}
    for t in tasks:
        accum[t] = {
            "soft": {
                "entropy": [], "simpson": [],
                "gini": [], "max_share": [],
                "topn": {n: [] for n in TOP_N_LIST},
                "sum_dist": np.zeros(N_EXPERTS, dtype=np.float64),
                "count": 0,
            },
            "hard": {
                "entropy": [], "simpson": [],
                "gini": [], "max_share": [],
                "topn": {n: [] for n in TOP_N_LIST},
                "sum_dist": np.zeros(N_EXPERTS, dtype=np.float64),
                "count": 0,
            }
        }

    # 开始置乱
    for _ in range(n_shuffles):
        perm = rng.permutation(total_samples)
        cursor = 0
        for t in tasks:
            cnt = task_counts[t]
            assigned = [all_samples[i] for i in perm[cursor:cursor + cnt]]
            cursor += cnt

            # 聚合 soft / hard 分布
            soft_sum = np.zeros(N_EXPERTS, dtype=np.float64)
            hard_sum = np.zeros(N_EXPERTS, dtype=np.float64)
            for s in assigned:
                soft_sum += s["soft_renorm_dist"]
                hard_sum += s["hard_dist"]

            soft_dist = safe_normalize(soft_sum)
            hard_dist = safe_normalize(hard_sum)

            # === 记录 soft 口径 ===
            accum[t]["soft"]["entropy"].append(n_eff_entropy(soft_dist))
            accum[t]["soft"]["simpson"].append(n_eff_simpson(soft_dist))
            accum[t]["soft"]["gini"].append(gini_coefficient(soft_dist))
            accum[t]["soft"]["max_share"].append(float(np.max(soft_dist)))
            for n in TOP_N_LIST:
                accum[t]["soft"]["topn"][n].append(top_n_coverage(soft_dist, n))
            accum[t]["soft"]["sum_dist"] += soft_dist
            accum[t]["soft"]["count"] += 1

            # === 记录 hard 口径 ===
            accum[t]["hard"]["entropy"].append(n_eff_entropy(hard_dist))
            accum[t]["hard"]["simpson"].append(n_eff_simpson(hard_dist))
            accum[t]["hard"]["gini"].append(gini_coefficient(hard_dist))
            accum[t]["hard"]["max_share"].append(float(np.max(hard_dist)))
            for n in TOP_N_LIST:
                accum[t]["hard"]["topn"][n].append(top_n_coverage(hard_dist, n))
            accum[t]["hard"]["sum_dist"] += hard_dist
            accum[t]["hard"]["count"] += 1

    # 汇总为均值 + CI
    for t in tasks:
        results["tasks"][t] = {"soft": {}, "hard": {}}
        for mode in ["soft", "hard"]:
            # n_eff 统计
            results["tasks"][t][mode]["entropy"] = bootstrap_mean_ci(
                accum[t][mode]["entropy"], rng, n_bootstrap=1000
            )
            results["tasks"][t][mode]["simpson"] = bootstrap_mean_ci(
                accum[t][mode]["simpson"], rng, n_bootstrap=1000
            )
            # Gini / Max-Share
            results["tasks"][t][mode]["gini"] = bootstrap_mean_ci(
                accum[t][mode]["gini"], rng, n_bootstrap=1000
            )
            results["tasks"][t][mode]["max_share"] = bootstrap_mean_ci(
                accum[t][mode]["max_share"], rng, n_bootstrap=1000
            )
            # Top-N 覆盖率
            results["tasks"][t][mode]["topn"] = {}
            for n in TOP_N_LIST:
                results["tasks"][t][mode]["topn"][f"top_{n}"] = bootstrap_mean_ci(
                    accum[t][mode]["topn"][n], rng, n_bootstrap=1000
                )
            # Lorenz（使用均值分布）
            if accum[t][mode]["count"] > 0:
                mean_dist = accum[t][mode]["sum_dist"] / float(accum[t][mode]["count"])
            else:
                mean_dist = np.ones(N_EXPERTS, dtype=np.float64) / float(N_EXPERTS)
            x, y = lorenz_curve(mean_dist)
            results["tasks"][t][mode]["lorenz"] = {"x": x, "y": y}

    return results


# ===========================================================================
# 主程序
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Q6 负载集中度分析（强化版）")
    parser.add_argument("--task", action="append", required=True,
                        help="任务配置：name=path，可以重复多次")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--layer", type=int, default=0, help="主分析层")
    parser.add_argument("--layers_consistency", type=str, default="0,4,8,12,16,20",
                        help="层间一致性分析的层列表")
    parser.add_argument("--max_files", type=int, default=-1, help="每任务最多文件数")
    parser.add_argument("--min_tokens", type=int, default=32, help="最小样本长度过滤")
    parser.add_argument("--segment_len", type=int, default=100, help="长度控制的片段长度")
    parser.add_argument("--segment_lens", type=str, default="",
                        help="长度敏感性分析列表，如 \"64,128,256\"（可选）")
    parser.add_argument("--n_bootstrap", type=int, default=1000, help="bootstrap次数")
    parser.add_argument("--bootstrap_max_points", type=int, default=20000, help="bootstrap最多采样点数")
    parser.add_argument("--n_shuffles", type=int, default=200, help="置乱基线次数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    args = parser.parse_args()
    set_random_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # 解析任务配置
    task_map = {}
    for spec in args.task:
        if "=" in spec:
            name, path = spec.split("=", 1)
            task_map[name.strip()] = path.strip()

    if not task_map:
        logging.error("未提供有效任务配置")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "figures"), exist_ok=True)

    logging.info("=== 加载数据 ===")
    raw_task_samples = {}
    for task, path in task_map.items():
        samples = load_task_samples(path, args.layer, args.max_files, args.min_tokens)
        raw_task_samples[task] = samples

    # 解析长度敏感性列表
    segment_lens = [args.segment_len]
    if args.segment_lens.strip():
        parsed = []
        for x in args.segment_lens.split(","):
            x = x.strip()
            if not x:
                continue
            try:
                parsed.append(int(x))
            except ValueError:
                continue
        # 去重并保持顺序
        for v in parsed:
            if v not in segment_lens:
                segment_lens.append(v)

    # 长度控制 + 等量抽样（主口径：segment_len）
    logging.info("=== 长度控制 + 等量抽样（主口径） ===")
    controlled_samples, length_stats = build_length_controlled_samples(
        raw_task_samples, args.segment_len, args.seed
    )

    # 如果有任务没有样本，直接退出
    if any(len(v) == 0 for v in controlled_samples.values()):
        logging.error("有任务样本数为 0，请检查 segment_len / min_tokens")
        return

    # 准备 sample_bank（用于置乱基线）
    sample_bank = {}
    for task, samples in controlled_samples.items():
        bank = []
        for s in samples:
            dist_info = build_distributions_for_sample(s)
            bank.append({
                "soft_renorm_dist": dist_info["soft_renorm_dist"],
                "hard_dist": dist_info["hard_dist"],
            })
        sample_bank[task] = bank

    # 证据1 + 证据2（主口径）
    logging.info("=== 证据1/2：n_eff + Gini/TopN/Lorenz ===")
    e12_results = analyze_task_metrics(controlled_samples, rng, args.n_bootstrap, args.bootstrap_max_points)

    # 证据3：长度控制统计
    logging.info("=== 证据3：长度控制统计 ===")
    e3_results = length_stats

    # 证据4：层间一致性
    logging.info("=== 证据4：层间一致性 ===")
    layers = [int(x) for x in args.layers_consistency.split(",") if x.strip() != ""]
    e4_results = analyze_layer_consistency(task_map, layers, args.max_files, args.min_tokens)

    # 证据5：任务标签置乱
    logging.info("=== 证据5：任务标签置乱基线 ===")
    e5_results = analyze_label_shuffle(sample_bank, args.n_shuffles, args.seed)

    # 长度敏感性（可选，多 segment_len）
    sensitivity = {}
    if len(segment_lens) > 1:
        logging.info("=== 长度敏感性分析 ===")
        for seg_len in segment_lens:
            logging.info(f"  segment_len = {seg_len}")
            # 不同长度用不同随机种子，避免采样位置完全一致导致偏差
            seg_seed = int(args.seed + seg_len)
            seg_rng = np.random.default_rng(seg_seed)
            seg_samples, seg_stats = build_length_controlled_samples(
                raw_task_samples, seg_len, seg_seed
            )
            if any(len(v) == 0 for v in seg_samples.values()):
                sensitivity[str(seg_len)] = {
                    "error": "有任务样本数为 0，请降低 segment_len 或检查数据",
                    "length_control": seg_stats,
                }
                continue
            seg_metrics = analyze_task_metrics(seg_samples, seg_rng, args.n_bootstrap, args.bootstrap_max_points)
            sensitivity[str(seg_len)] = {
                "length_control": seg_stats,
                "metrics": seg_metrics,
            }

    # 汇总结果
    results = {
        "config": {
            "layer": args.layer,
            "segment_len": args.segment_len,
            "segment_lens": segment_lens,
            "min_tokens": args.min_tokens,
            "n_bootstrap": args.n_bootstrap,
            "n_shuffles": args.n_shuffles,
            "seed": args.seed,
            "tasks": list(task_map.keys()),
        },
        "evidence": {
            "E1_E2_metrics": e12_results,
            "E3_length_control": e3_results,
            "E4_layer_consistency": e4_results,
            "E5_label_shuffle": e5_results,
        },
        "sensitivity": {
            "segment_len": sensitivity
        }
    }

    # 保存结果
    output_path = os.path.join(args.output_dir, "q6_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logging.info(f"结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
