#!/usr/bin/env python3
"""
================================================================================
问题3：任务内路由模式分析 (Task Internal Drift Analysis)
================================================================================

【核心问题】
任务内部的路由动态是否存在"粘滞/阶段切换"现象？
- 粘滞：连续多个 token 倾向选择相同专家
- 阶段切换：在语义边界处专家分布发生跳变

【主要指标】
1. 切换率 (Switch Rate): 相邻 token 切换专家的比例
   - 公式: Switch_Rate = Σ 1[Top1_t ≠ Top1_{t+1}] / (N-1)
   - 高切换率 → 粘滞性弱

2. Run Length: 连续选择同一专家的段长度
   - 若符合几何分布 → 近似无记忆
   - 几何分布期望: E[Run] = 1/p (p=切换率)

3. 窗口漂移 (Window Drift): 滑动窗口间的分布差异
   - 使用 JSD/TV 度量
   - 漂移弱 → 无明显阶段切换

【实验结论】
- 6个任务切换率都很高（94%-97%）
- Run Length 与几何分布高度吻合
- 窗口漂移很弱，接近置乱基线
- 结论：任务内部路由近似无记忆，不存在粘滞或阶段切换

【版本】v2 - 完善版
【功能模块】
  - P0 数据结构检查（概率质量、tokens_str）
  - 带 Other 桶的 JSD/TV 计算
  - 切换率 + run length + 几何分布理论基线
  - 多窗口敏感性分析 (64/128/256)
  - Bootstrap CI
  - 逐层分析

服务器路径：/home/zys/li_kangrui/SimAI-zyx-expert-load/experiments/moe_measure/
================================================================================
"""

import os
import glob
import json
import argparse
import logging
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


# ============ 常量定义 ============
# 【边界标记词典】
# 用于检测任务内部的语义边界（如问题→答案的切换点）
# 不同任务有不同的边界标记模式

DEFAULT_MARKERS = {
    "gsm8k": [r"####", r"Answer", r"answer", r"答案", r"最终", r"Therefore", r"So"],  # 数学推理：答案标记
    "piqa": [r"A)", r"B)", r"(A)", r"(B)", r"A.", r"B."],  # 常识问答：选项标记
    "cmrc2018": [r"问题", r"问:", r"答案", r"答:", r"【问题】", r"【答案】"],  # 中文阅读理解
    "humaneval": [r"def ", r"return", r":\n", r"\n\n"],  # 代码生成：函数定义/返回
    "wiki": [],  # 百科文本：无明确边界
    "wikitext2": [],
    "winogrande": [],  # 代词消歧：无明确边界
}


# ============ 数据加载 ============

def list_npz_files(data_dir: str, max_files: int = -1) -> List[str]:
    """列出 NPZ 文件"""
    npz_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if max_files > 0:
        npz_files = npz_files[:max_files]
    return npz_files


def get_layer_names(npz_path: str) -> List[str]:
    """从 NPZ 文件提取层名"""
    with np.load(npz_path, allow_pickle=True) as data:
        layer_names = []
        for k in data.keys():
            if k.startswith("layers/") and k.endswith("_out_probs"):
                layer_name = k.split("/")[1].replace("_out_probs", "")
                layer_names.append(layer_name)
        return sorted(layer_names)


def check_npz_structure(npz_path: str) -> Dict[str, Any]:
    """P0: 检查单个 NPZ 文件的数据结构"""
    result = {
        "file": npz_path,
        "has_tokens_str": False,
        "keys": [],
        "n_tokens": 0,
        "n_layers": 0,
        "topk": 0,
        "prob_sum_stats": {},
        "sample_probs": None,
    }

    with np.load(npz_path, allow_pickle=True) as data:
        result["keys"] = list(data.keys())
        result["has_tokens_str"] = "tokens_str" in data

        # 找层名
        layer_names = []
        for k in data.keys():
            if k.startswith("layers/") and k.endswith("_out_probs"):
                layer_name = k.split("/")[1].replace("_out_probs", "")
                layer_names.append(layer_name)
        layer_names = sorted(layer_names)
        result["n_layers"] = len(layer_names)

        if not layer_names:
            return result

        # 检查第一层的概率
        first_layer = layer_names[0]
        probs = data[f"layers/{first_layer}_out_probs"]

        # 处理 object 数组
        if probs.dtype == object:
            # 可能是变长数组
            all_probs = []
            for p in probs:
                if isinstance(p, np.ndarray):
                    all_probs.append(p)
            if all_probs:
                probs = np.vstack(all_probs) if all_probs[0].ndim == 1 else np.concatenate(all_probs, axis=0)

        result["n_tokens"] = probs.shape[0]
        result["topk"] = probs.shape[1] if probs.ndim > 1 else 1

        # 计算概率和统计
        if probs.ndim == 1:
            prob_sums = probs
        else:
            prob_sums = probs.sum(axis=-1)

        result["prob_sum_stats"] = {
            "min": float(prob_sums.min()),
            "max": float(prob_sums.max()),
            "mean": float(prob_sums.mean()),
            "std": float(prob_sums.std()),
            "less_than_1_ratio": float((prob_sums < 0.999).mean()),
        }
        result["sample_probs"] = probs[:5].tolist() if probs.shape[0] >= 5 else probs.tolist()

    return result


def run_p0_check(data_dirs: Dict[str, str], output_dir: str) -> Dict[str, Any]:
    """P0: 对所有任务进行数据结构检查"""
    p0_results = {"tasks": {}, "summary": {}}

    for task_name, data_dir in data_dirs.items():
        logging.info(f"P0 checking task: {task_name}")
        npz_files = list_npz_files(data_dir, max_files=3)  # 只检查前3个文件

        if not npz_files:
            p0_results["tasks"][task_name] = {"error": "No NPZ files found"}
            continue

        task_checks = []
        for f in npz_files:
            check = check_npz_structure(f)
            task_checks.append(check)

        # 汇总
        has_tokens_str = all(c.get("has_tokens_str", False) for c in task_checks)
        prob_means = [c["prob_sum_stats"].get("mean", 0) for c in task_checks if c["prob_sum_stats"]]
        prob_less_than_1 = [c["prob_sum_stats"].get("less_than_1_ratio", 0) for c in task_checks if c["prob_sum_stats"]]

        p0_results["tasks"][task_name] = {
            "n_files_checked": len(task_checks),
            "has_tokens_str": has_tokens_str,
            "n_layers": task_checks[0].get("n_layers", 0) if task_checks else 0,
            "topk": task_checks[0].get("topk", 0) if task_checks else 0,
            "prob_sum_mean": float(np.mean(prob_means)) if prob_means else 0,
            "prob_less_than_1_ratio": float(np.mean(prob_less_than_1)) if prob_less_than_1 else 0,
            "details": task_checks,
        }

    # 全局汇总
    all_has_tokens = all(t.get("has_tokens_str", False) for t in p0_results["tasks"].values() if "error" not in t)
    all_prob_means = [t.get("prob_sum_mean", 0) for t in p0_results["tasks"].values() if "error" not in t]

    p0_results["summary"] = {
        "all_have_tokens_str": all_has_tokens,
        "avg_prob_sum": float(np.mean(all_prob_means)) if all_prob_means else 0,
        "need_other_bucket": float(np.mean(all_prob_means)) < 0.99 if all_prob_means else True,
        "recommendation": "使用 Other 桶" if float(np.mean(all_prob_means)) < 0.99 else "Top-6 已接近归一化，Other 桶影响小",
    }

    # 保存
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "p0_data_structure_check.json"), "w", encoding="utf-8") as f:
        json.dump(p0_results, f, ensure_ascii=False, indent=2, default=_json_default)

    logging.info(f"P0 check complete. Need Other bucket: {p0_results['summary']['need_other_bucket']}")
    return p0_results


# ============ 数据加载（完整版） ============

def ensure_2d_array(arr: np.ndarray, pad_value: float, dtype: np.dtype) -> np.ndarray:
    """确保数组是 2D"""
    if not isinstance(arr, np.ndarray):
        return np.asarray(arr, dtype=dtype)
    if arr.dtype == object or (arr.ndim == 1 and arr.size > 0 and isinstance(arr[0], (list, np.ndarray))):
        elems = [np.asarray(x) for x in arr]
        if elems and all(e.ndim == 2 for e in elems):
            max_cols = max(e.shape[1] for e in elems)
            padded = []
            for e in elems:
                e = e.astype(dtype)
                if e.shape[1] < max_cols:
                    pad = np.full((e.shape[0], max_cols - e.shape[1]), pad_value, dtype=dtype)
                    e = np.concatenate([e, pad], axis=1)
                padded.append(e)
            return np.concatenate(padded, axis=0)
        max_len = max(e.shape[0] for e in elems) if elems else 0
        out = np.full((len(elems), max_len), pad_value, dtype=dtype)
        for i, e in enumerate(elems):
            e = e.astype(dtype)
            out[i, : e.shape[0]] = e
        return out
    if arr.ndim == 1:
        return arr.reshape(-1, 1).astype(dtype)
    return arr.astype(dtype)


def load_task_data(
    data_dir: str,
    max_files: int = -1,
    sample_start: int = 0,
    sample_size: int = -1,
) -> Tuple[List[str], Dict[str, np.ndarray], Dict[str, np.ndarray], List[Tuple[int, int]]]:
    """
    加载任务数据
    返回: (tokens_str, probs_by_layer, inds_by_layer, segments)
    """
    npz_files = list_npz_files(data_dir, max_files)
    if not npz_files:
        raise FileNotFoundError(f"No NPZ files found in {data_dir}")

    layer_names = get_layer_names(npz_files[0])
    probs_by_layer: Dict[str, List[np.ndarray]] = {ln: [] for ln in layer_names}
    inds_by_layer: Dict[str, List[np.ndarray]] = {ln: [] for ln in layer_names}
    tokens_str: List[str] = []
    segments: List[Tuple[int, int]] = []
    cursor = 0

    logging.info(f"Loading {len(npz_files)} NPZ files from {data_dir}")

    for f in npz_files:
        with np.load(f, allow_pickle=True) as data:
            # tokens
            if "tokens_str" in data:
                toks = list(data["tokens_str"])
            else:
                first_layer = layer_names[0]
                probs_raw = data[f"layers/{first_layer}_out_probs"]
                if probs_raw.dtype == object:
                    token_count = sum(len(np.asarray(x)) if np.asarray(x).ndim == 1 else np.asarray(x).shape[0] for x in probs_raw)
                else:
                    token_count = probs_raw.shape[0]
                toks = [""] * token_count

            tokens_str.extend(toks)
            seg_len = len(toks)
            segments.append((cursor, cursor + seg_len))
            cursor += seg_len

            # 每层的 probs 和 indices
            for ln in layer_names:
                probs = ensure_2d_array(data[f"layers/{ln}_out_probs"], 0.0, np.float64)
                inds = ensure_2d_array(data[f"layers/{ln}_out_indices"], -1, np.int64)
                probs_by_layer[ln].append(probs)
                inds_by_layer[ln].append(inds)

    # 拼接
    probs_concat = {ln: np.concatenate(probs_by_layer[ln], axis=0) for ln in layer_names}
    inds_concat = {ln: np.concatenate(inds_by_layer[ln], axis=0) for ln in layer_names}

    # 采样
    total_tokens = len(tokens_str)
    end_idx = total_tokens if sample_size <= 0 else min(total_tokens, sample_start + sample_size)
    start_idx = min(sample_start, end_idx)

    if start_idx > 0 or end_idx < total_tokens:
        tokens_str = tokens_str[start_idx:end_idx]
        probs_concat = {ln: probs_concat[ln][start_idx:end_idx] for ln in layer_names}
        inds_concat = {ln: inds_concat[ln][start_idx:end_idx] for ln in layer_names}
        # 调整 segments
        new_segments = []
        for s, e in segments:
            if e <= start_idx or s >= end_idx:
                continue
            new_s = max(s, start_idx) - start_idx
            new_e = min(e, end_idx) - start_idx
            new_segments.append((new_s, new_e))
        segments = new_segments

    logging.info(f"Loaded {len(tokens_str)} tokens, {len(segments)} segments, {len(layer_names)} layers")
    return tokens_str, probs_concat, inds_concat, segments


# ============ 辅助函数 ============

def _normalize_row(probs: np.ndarray) -> np.ndarray:
    """对单行概率归一化（学习 v1 的做法）"""
    s = probs.sum()
    if s > 0:
        return probs / s
    return probs


# ============ 分布构建（带 Other 桶） ============

def build_sparse_distribution_with_other(
    topk_ids: np.ndarray,
    topk_probs: np.ndarray,
    n_experts: int = 60,
    normalize_first: bool = True,
) -> np.ndarray:
    """
    构建带 Other 桶的 61 维分布
    topk_ids: shape (k,)
    topk_probs: shape (k,)
    normalize_first: 是否先对 topk_probs 归一化（确保每层贡献相等）
    返回: shape (n_experts + 1,)，最后一维是 Other
    """
    dist = np.zeros(n_experts + 1, dtype=np.float64)

    # 可选：先归一化，确保每层贡献权重相等
    if normalize_first:
        probs_to_use = _normalize_row(topk_probs)
        # 归一化后，Other 桶为 0（因为概率和已经是 1）
        other_mass = 0.0
    else:
        probs_to_use = topk_probs
        other_mass = max(0.0, 1.0 - topk_probs.sum())

    for idx, prob in zip(topk_ids, probs_to_use):
        if idx >= 0 and idx < n_experts:
            dist[idx] = prob
    dist[-1] = other_mass
    return dist


def build_token_distribution_with_other(
    layer_names: List[str],
    probs_by_layer: Dict[str, np.ndarray],
    inds_by_layer: Dict[str, np.ndarray],
    token_idx: int,
    n_experts: int = 60,
    normalize_per_layer: bool = True,
) -> np.ndarray:
    """
    构建单个 token 的平均分布（跨层平均，带 Other 桶）
    normalize_per_layer: 是否对每层概率先归一化（v1 的做法，确保每层贡献相等）
    返回: shape (n_experts + 1,)
    """
    dist = np.zeros(n_experts + 1, dtype=np.float64)
    for ln in layer_names:
        probs = probs_by_layer[ln][token_idx]
        inds = inds_by_layer[ln][token_idx]
        layer_dist = build_sparse_distribution_with_other(
            inds, probs, n_experts, normalize_first=normalize_per_layer
        )
        dist += layer_dist
    dist /= len(layer_names)
    return dist


def build_window_distribution_with_other(
    layer_names: List[str],
    probs_by_layer: Dict[str, np.ndarray],
    inds_by_layer: Dict[str, np.ndarray],
    start: int,
    end: int,
    n_experts: int = 60,
) -> np.ndarray:
    """
    构建窗口内的聚合分布（带 Other 桶）
    方案 A: 对窗口内所有 token 的分布求平均
    """
    dist = np.zeros(n_experts + 1, dtype=np.float64)
    n_tokens = end - start

    for t in range(start, end):
        token_dist = build_token_distribution_with_other(
            layer_names, probs_by_layer, inds_by_layer, t, n_experts
        )
        dist += token_dist

    if n_tokens > 0:
        dist /= n_tokens
    return dist


def build_window_distribution_top1_freq(
    layer_names: List[str],
    probs_by_layer: Dict[str, np.ndarray],
    inds_by_layer: Dict[str, np.ndarray],
    start: int,
    end: int,
    n_experts: int = 60,
) -> np.ndarray:
    """
    构建窗口内的 Top-1 频率分布（方案 B）
    统计窗口内每层 Top-1 专家的出现频率
    返回: shape (n_experts,)，归一化后的频率分布
    """
    freq = np.zeros(n_experts, dtype=np.float64)
    count = 0

    for t in range(start, end):
        for ln in layer_names:
            probs = probs_by_layer[ln][t]
            inds = inds_by_layer[ln][t]
            # Top-1 是概率最大的
            if len(probs) > 0:
                top1_idx = inds[np.argmax(probs)]
                if 0 <= top1_idx < n_experts:
                    freq[top1_idx] += 1
                    count += 1

    if count > 0:
        freq /= count
    return freq


# ============ 统计指标 ============

def jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon Divergence"""
    eps = 1e-12
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    # 确保归一化
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log2(p / m))
    kl_qm = np.sum(q * np.log2(q / m))
    return float(0.5 * (kl_pm + kl_qm))


def tv(p: np.ndarray, q: np.ndarray) -> float:
    """Total Variation Distance"""
    p = p / p.sum() if p.sum() > 0 else p
    q = q / q.sum() if q.sum() > 0 else q
    return float(0.5 * np.abs(p - q).sum())


def get_token_top1(
    layer_names: List[str],
    probs_by_layer: Dict[str, np.ndarray],
    inds_by_layer: Dict[str, np.ndarray],
    n_tokens: int,
) -> np.ndarray:
    """
    获取每个 token 的"综合 Top1"专家（跨层聚合后的 argmax）
    返回: shape (n_tokens,)
    """
    top1s = np.zeros(n_tokens, dtype=np.int64)

    for t in range(n_tokens):
        # 聚合跨层的概率分布
        agg = np.zeros(60, dtype=np.float64)
        for ln in layer_names:
            probs = probs_by_layer[ln][t]
            inds = inds_by_layer[ln][t]
            for idx, prob in zip(inds, probs):
                if 0 <= idx < 60:
                    agg[idx] += prob
        top1s[t] = np.argmax(agg)

    return top1s


def get_layer_top1(
    probs: np.ndarray,
    inds: np.ndarray,
) -> np.ndarray:
    """
    获取单层的 Top1 专家序列
    probs: shape (n_tokens, k)
    inds: shape (n_tokens, k)
    返回: shape (n_tokens,)
    """
    n_tokens = probs.shape[0]
    top1s = np.zeros(n_tokens, dtype=np.int64)
    for t in range(n_tokens):
        top1_pos = np.argmax(probs[t])
        top1s[t] = inds[t, top1_pos]
    return top1s


def compute_switch_rate(top1: np.ndarray, segments: List[Tuple[int, int]]) -> Tuple[float, int]:
    """
    【核心指标1】计算切换率 (Switch Rate)

    公式: Switch_Rate = Σ 1[Top1_t ≠ Top1_{t+1}] / (N-1)

    含义:
        - 切换率高（如 0.95）→ 几乎每个 token 都在切换专家，粘滞性弱
        - 切换率低（如 0.50）→ 连续 token 倾向选同一专家，粘滞性强

    参数:
        top1: 每个 token 的 Top1 专家 ID，shape (n_tokens,)
        segments: 样本边界列表，避免跨样本计算切换

    返回:
        (切换率, 有效 token 对数)

    注意: 不跨样本边界计算，因为不同样本之间的切换没有意义
    """
    switches = 0  # 切换次数
    total = 0     # 总 token 对数
    for s, e in segments:
        if e - s < 2:  # 样本太短，跳过
            continue
        # 比较相邻 token 的 Top1 是否不同
        diff = top1[s + 1:e] != top1[s:e - 1]
        switches += int(diff.sum())
        total += int(diff.size)
    if total == 0:
        return 0.0, 0
    return float(switches / total), total


def compute_run_lengths(top1: np.ndarray, segments: List[Tuple[int, int]]) -> List[int]:
    """
    【核心指标2】计算 Run Length 分布

    定义: Run Length = 连续选择同一专家的 token 数量

    示例:
        Top1 序列: [3, 3, 3, 5, 5, 2, 2, 2, 2]
        Run Lengths: [3, 2, 4]  (专家3连续3次, 专家5连续2次, 专家2连续4次)

    理论基线:
        如果路由是"无记忆"的（每次独立选择），Run Length 应服从几何分布
        几何分布: P(Run=k) = (1-p)^(k-1) * p，其中 p = 切换率
        期望: E[Run] = 1/p

    参数:
        top1: 每个 token 的 Top1 专家 ID
        segments: 样本边界列表

    返回:
        所有 run length 的列表
    """
    lengths = []
    for s, e in segments:
        if e - s <= 0:
            continue
        seq = top1[s:e]
        run = 1  # 当前 run 的长度
        for i in range(1, len(seq)):
            if seq[i] == seq[i - 1]:
                run += 1  # 同一专家，继续累加
            else:
                lengths.append(run)  # 专家切换，记录当前 run
                run = 1  # 重新开始计数
        lengths.append(run)  # 记录最后一个 run
    return lengths


def geometric_baseline_mean(switch_rate: float) -> float:
    """
    【理论基线】几何分布的期望 Run Length

    公式: E[Run] = 1/p，其中 p = 切换率

    用途:
        如果实测 run_mean ≈ 1/switch_rate，说明路由近似无记忆
        如果实测 run_mean > 1/switch_rate，说明存在粘滞性
    """
    if switch_rate <= 0 or switch_rate >= 1:
        return float('inf') if switch_rate <= 0 else 1.0
    return 1.0 / switch_rate


def geometric_pmf(k: int, p: float) -> float:
    """几何分布 PMF: P(X=k) = (1-p)^(k-1) * p"""
    if p <= 0 or p > 1:
        return 0.0
    return ((1 - p) ** (k - 1)) * p


def ks_test_geometric(run_lengths: List[int], switch_rate: float) -> Dict[str, float]:
    """
    【统计检验】KS 检验：Run Length 是否符合几何分布

    原理:
        Kolmogorov-Smirnov 检验比较实测分布与理论分布的 CDF
        D = max|F_实测(x) - F_理论(x)|

    假设检验:
        H0（原假设）: Run Length 服从几何分布（无记忆）
        H1（备择假设）: Run Length 不服从几何分布

    判定:
        p_value > 0.05 → 不能拒绝 H0 → 支持"无记忆"
        p_value < 0.05 → 拒绝 H0 → 存在记忆/粘滞

    返回:
        {
            "ks_statistic": KS 统计量 D,
            "p_value": p 值,
            "reject_null": 是否拒绝原假设
        }
    """
    if not run_lengths or switch_rate <= 0 or switch_rate >= 1:
        return {"ks_statistic": None, "p_value": None, "reject_null": None}

    from scipy.stats import kstest

    # 几何分布的 CDF: F(k) = P(X <= k) = 1 - (1-p)^k
    def geom_cdf(k):
        return 1 - (1 - switch_rate) ** k

    # 执行 KS 检验
    ks_stat, p_val = kstest(run_lengths, geom_cdf)

    return {
        "ks_statistic": float(ks_stat),
        "p_value": float(p_val),
        "reject_null": bool(p_val < 0.05),
    }


# ============ 窗口漂移分析 ============
# 【核心指标3】通过滑动窗口检测专家分布是否随位置变化

def compute_window_drift(
    layer_names: List[str],
    probs_by_layer: Dict[str, np.ndarray],
    inds_by_layer: Dict[str, np.ndarray],
    segments: List[Tuple[int, int]],
    window_size: int,
    stride: int,
    n_experts: int = 60,
    use_other_bucket: bool = True,
) -> Tuple[List[float], List[float], List[np.ndarray]]:
    """
    【核心指标3】计算窗口间的 JSD/TV 漂移

    原理:
        将 token 序列划分为滑动窗口，计算相邻窗口的分布差异
        JSD/TV 大 → 分布变化剧烈（阶段切换）
        JSD/TV 小 → 分布稳定（无阶段切换）

    参数:
        window_size: 窗口大小（如 64/128/256）
        stride: 滑动步长
        use_other_bucket: 是否使用 Other 桶（处理概率和 < 1 的情况）

    返回:
        (jsd_list, tv_list, distributions)
    """
    distributions = []

    # Step 1: 为每个窗口构建专家分布
    for s, e in segments:
        pos = s
        while pos + window_size <= e:
            if use_other_bucket:
                dist = build_window_distribution_with_other(
                    layer_names, probs_by_layer, inds_by_layer, pos, pos + window_size, n_experts
                )
            else:
                dist = build_window_distribution_top1_freq(
                    layer_names, probs_by_layer, inds_by_layer, pos, pos + window_size, n_experts
                )
            distributions.append(dist)
            pos += stride

    # Step 2: 计算相邻窗口的 JSD/TV
    jsd_vals = []
    tv_vals = []
    for i in range(len(distributions) - 1):
        jsd_vals.append(jsd(distributions[i], distributions[i + 1]))
        tv_vals.append(tv(distributions[i], distributions[i + 1]))

    return jsd_vals, tv_vals, distributions


# ============ 边界分析 ============

def detect_boundaries(tokens: List[str], task_name: str) -> List[int]:
    """检测任务结构边界（基于规则）"""
    markers = DEFAULT_MARKERS.get(task_name.lower(), [])
    if not markers:
        return []

    boundaries = []
    for i, tok in enumerate(tokens):
        tok_str = str(tok)
        for m in markers:
            if m and m in tok_str:
                boundaries.append(i)
                break
    return boundaries


def detect_boundaries_data_driven(
    jsd_values: List[float],
    window_size: int,
    n_top: int = 10,
    min_distance: int = 32,
    threshold_percentile: float = 90.0,
) -> List[int]:
    """
    数据驱动边界检测：使用漂移曲线峰值作为边界候选
    当 tokens_str 缺失时的备选方案

    Args:
        jsd_values: 相邻窗口的 JSD 序列
        window_size: 窗口大小（用于计算实际 token 位置）
        n_top: 返回 top N 个峰值
        min_distance: 峰值间最小距离（避免聚集）
        threshold_percentile: JSD 阈值百分位数

    Returns:
        边界位置列表（token 索引）
    """
    if not jsd_values:
        return []

    jsd_arr = np.array(jsd_values)
    threshold = np.percentile(jsd_arr, threshold_percentile)

    # 找所有超过阈值的峰值
    peaks = []
    for i in range(1, len(jsd_arr) - 1):
        if jsd_arr[i] > jsd_arr[i-1] and jsd_arr[i] > jsd_arr[i+1] and jsd_arr[i] >= threshold:
            peaks.append((i, jsd_arr[i]))

    # 按 JSD 降序排序
    peaks.sort(key=lambda x: x[1], reverse=True)

    # 去除距离太近的峰值
    selected = []
    for idx, _val in peaks:
        if len(selected) >= n_top:
            break
        # 检查与已选峰值的距离
        too_close = False
        for sel_idx in selected:
            if abs(idx - sel_idx) < min_distance // window_size:
                too_close = True
                break
        if not too_close:
            selected.append(idx)

    # 转换为 token 位置（窗口中心）
    boundaries = [idx * window_size + window_size // 2 for idx in selected]
    return sorted(boundaries)


def compute_boundary_effect(
    layer_names: List[str],
    probs_by_layer: Dict[str, np.ndarray],
    inds_by_layer: Dict[str, np.ndarray],
    segments: List[Tuple[int, int]],
    boundaries: List[int],
    window_size: int,
    n_experts: int,
    rng: np.random.Generator,
    n_random: int = 100,
) -> Dict[str, Any]:
    """计算边界处的漂移 vs 随机位置的漂移"""

    # 筛选有效边界（边界两侧都有足够空间）
    valid_boundaries = []
    for b in boundaries:
        for s, e in segments:
            if b - window_size >= s and b + window_size <= e:
                valid_boundaries.append(b)
                break

    if not valid_boundaries:
        return {
            "n_boundaries": 0,
            "boundary_jsd": [],
            "boundary_tv": [],
            "random_jsd": [],
            "random_tv": [],
        }

    # 计算边界处的漂移
    boundary_jsd = []
    boundary_tv = []
    for b in valid_boundaries:
        left_dist = build_window_distribution_with_other(
            layer_names, probs_by_layer, inds_by_layer, b - window_size, b, n_experts
        )
        right_dist = build_window_distribution_with_other(
            layer_names, probs_by_layer, inds_by_layer, b, b + window_size, n_experts
        )
        boundary_jsd.append(jsd(left_dist, right_dist))
        boundary_tv.append(tv(left_dist, right_dist))

    # 计算随机位置的漂移
    random_jsd = []
    random_tv = []
    for _ in range(n_random):
        # 随机选一个 segment
        valid_segs = [(s, e) for s, e in segments if e - s >= 2 * window_size]
        if not valid_segs:
            break
        seg = valid_segs[rng.integers(0, len(valid_segs))]
        s, e = seg
        # 随机选一个位置
        b = rng.integers(s + window_size, e - window_size + 1)
        left_dist = build_window_distribution_with_other(
            layer_names, probs_by_layer, inds_by_layer, b - window_size, b, n_experts
        )
        right_dist = build_window_distribution_with_other(
            layer_names, probs_by_layer, inds_by_layer, b, b + window_size, n_experts
        )
        random_jsd.append(jsd(left_dist, right_dist))
        random_tv.append(tv(left_dist, right_dist))

    return {
        "n_boundaries": len(valid_boundaries),
        "boundary_jsd": boundary_jsd,
        "boundary_tv": boundary_tv,
        "random_jsd": random_jsd,
        "random_tv": random_tv,
    }


# ============ 置乱基线 ============
# 【基线设计】通过打乱 token 顺序来估计"随机水平"
# 如果实测指标 ≈ 置乱基线，说明无时序效应

def shuffle_within_segments(
    n_tokens: int,
    segments: List[Tuple[int, int]],
    rng: np.random.Generator,
) -> np.ndarray:
    """
    【置乱基线】在每个 segment 内部打乱 token 顺序

    原理:
        - 保持边际分布不变（每个专家的总使用量不变）
        - 只破坏时序关系（相邻 token 的依赖被打破）

    用途:
        如果实测切换率 ≈ 置乱切换率，说明路由本身就是随机的
        如果实测切换率 < 置乱切换率，说明存在粘滞性
    """
    idxs = np.arange(n_tokens)
    for s, e in segments:
        if e - s <= 1:
            continue
        # 在 segment 内部随机打乱
        idxs[s:e] = idxs[s:e][rng.permutation(e - s)]
    return idxs


def compute_shuffled_metrics(
    layer_names: List[str],
    probs_by_layer: Dict[str, np.ndarray],
    inds_by_layer: Dict[str, np.ndarray],
    segments: List[Tuple[int, int]],
    window_size: int,
    stride: int,
    n_experts: int,
    rng: np.random.Generator,
) -> Dict[str, float]:
    """
    【置乱基线】计算置乱后的各项指标

    流程:
        1. 生成置乱索引（segment 内部打乱）
        2. 按置乱索引重排数据
        3. 计算置乱后的切换率和窗口漂移

    返回:
        {
            "switch_rate_shuffle": 置乱后的切换率,
            "window_jsd_mean_shuffle": 置乱后的窗口 JSD 均值,
            "window_tv_mean_shuffle": 置乱后的窗口 TV 均值,
        }

    判定逻辑:
        实测 ≈ 置乱 → 无时序效应
        实测 < 置乱 → 存在粘滞性
        实测 > 置乱 → 异常（需检查）
    """
    n_tokens = probs_by_layer[layer_names[0]].shape[0]
    shuffle_idx = shuffle_within_segments(n_tokens, segments, rng)

    # 置乱数据
    probs_shuffled = {ln: probs_by_layer[ln][shuffle_idx] for ln in layer_names}
    inds_shuffled = {ln: inds_by_layer[ln][shuffle_idx] for ln in layer_names}

    # 计算切换率
    top1_shuffled = get_token_top1(layer_names, probs_shuffled, inds_shuffled, n_tokens)
    switch_shuf, _ = compute_switch_rate(top1_shuffled, segments)

    # 计算窗口漂移
    jsd_shuf, tv_shuf, _ = compute_window_drift(
        layer_names, probs_shuffled, inds_shuffled, segments,
        window_size, stride, n_experts, use_other_bucket=True
    )

    return {
        "switch_rate_shuffle": switch_shuf,
        "window_jsd_mean_shuffle": float(np.mean(jsd_shuf)) if jsd_shuf else 0.0,
        "window_tv_mean_shuffle": float(np.mean(tv_shuf)) if tv_shuf else 0.0,
    }


# ============ Bootstrap CI ============
# 【统计方法】Bootstrap 用于估计统计量的置信区间，不需要假设数据分布

def bootstrap_ci(
    values: List[float],
    n_bootstrap: int = 200,
    ci: float = 0.95,
    rng: np.random.Generator = None,
) -> Tuple[float, float, float]:
    """
    【Bootstrap CI】计算通用的 Bootstrap 置信区间

    算法步骤:
        1. 从原始 n 个样本中有放回地抽取 n 个样本
        2. 计算该重采样的统计量（如均值）
        3. 重复 B 次（如200次），得到 B 个统计量
        4. 取第 2.5% 和 97.5% 分位数作为 95% 置信区间

    参数:
        values: 原始数据列表
        n_bootstrap: 重采样次数，默认 200
        ci: 置信水平，默认 0.95

    返回:
        (均值, 置信区间下界, 置信区间上界)
    """
    if not values:
        return 0.0, 0.0, 0.0

    if rng is None:
        rng = np.random.default_rng()

    values = np.array(values)
    n = len(values)

    boot_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=n, replace=True)
        boot_means.append(sample.mean())

    boot_means = np.array(boot_means)
    alpha = 1 - ci
    lower = np.percentile(boot_means, 100 * alpha / 2)
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2))
    mean = values.mean()

    return float(mean), float(lower), float(upper)


def bootstrap_switch_rate_ci(
    top1: np.ndarray,
    segments: List[Tuple[int, int]],
    n_bootstrap: int = 200,
    ci: float = 0.95,
    rng: np.random.Generator = None,
) -> Tuple[float, float, float]:
    """
    【Bootstrap CI】对切换率做 Bootstrap 置信区间

    关键点:
        重采样单位是 segment，而非单个值
        这样能保持样本内部的依赖结构

    为什么按 segment 重采样:
        - 同一样本内的 token 有依赖关系
        - 如果按单个切换事件重采样，会破坏这种依赖
        - 按 segment 重采样保持了样本的完整性

    返回:
        (均值, 置信区间下界, 置信区间上界)
    """
    if not segments or rng is None:
        rng = np.random.default_rng()

    # 计算每个 segment 的切换率
    seg_switch_rates = []
    for s, e in segments:
        if e - s < 2:
            continue
        diff = top1[s + 1:e] != top1[s:e - 1]
        seg_switch_rates.append(float(diff.mean()))

    if not seg_switch_rates:
        return 0.0, 0.0, 0.0

    seg_switch_rates = np.array(seg_switch_rates)
    n = len(seg_switch_rates)

    boot_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(seg_switch_rates, size=n, replace=True)
        boot_means.append(sample.mean())

    boot_means = np.array(boot_means)
    alpha = 1 - ci
    lower = np.percentile(boot_means, 100 * alpha / 2)
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2))
    mean = seg_switch_rates.mean()

    return float(mean), float(lower), float(upper)


def bootstrap_run_mean_ci(
    run_lengths: List[int],
    n_bootstrap: int = 200,
    ci: float = 0.95,
    rng: np.random.Generator = None,
) -> Tuple[float, float, float]:
    """
    对 run_mean 做 Bootstrap CI（按 run length 重采样）
    """
    if not run_lengths:
        return 0.0, 0.0, 0.0

    if rng is None:
        rng = np.random.default_rng()

    run_lengths = np.array(run_lengths)
    n = len(run_lengths)

    boot_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(run_lengths, size=n, replace=True)
        boot_means.append(sample.mean())

    boot_means = np.array(boot_means)
    alpha = 1 - ci
    lower = np.percentile(boot_means, 100 * alpha / 2)
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2))
    mean = run_lengths.mean()

    return float(mean), float(lower), float(upper)


# ============ 逐层分析 ============

def analyze_per_layer(
    layer_names: List[str],
    probs_by_layer: Dict[str, np.ndarray],
    inds_by_layer: Dict[str, np.ndarray],
    segments: List[Tuple[int, int]],
) -> Dict[str, Dict[str, float]]:
    """逐层分析切换率"""
    results = {}
    for ln in layer_names:
        top1 = get_layer_top1(probs_by_layer[ln], inds_by_layer[ln])
        switch, total = compute_switch_rate(top1, segments)
        runs = compute_run_lengths(top1, segments)
        results[ln] = {
            "switch_rate": switch,
            "run_mean": float(np.mean(runs)) if runs else 0.0,
            "run_median": float(np.median(runs)) if runs else 0.0,
        }
    return results


# ============ 可视化 ============

def plot_switch_rate_comparison(
    tasks: List[str],
    real_values: List[float],
    shuffle_values: List[float],
    output_path: str,
):
    """切换率对比柱状图"""
    x = np.arange(len(tasks))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, real_values, width, label='Real', color='steelblue')
    bars2 = ax.bar(x + width/2, shuffle_values, width, label='Shuffled', color='coral')

    ax.set_ylabel('Switch Rate')
    ax.set_title('Top1 Expert Switch Rate: Real vs Shuffled Baseline')
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=30, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_run_length_distribution(
    task_name: str,
    run_lengths: List[int],
    switch_rate: float,
    output_path: str,
):
    """Run length 分布 + 几何分布基线"""
    if not run_lengths:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # 实际分布
    counter = Counter(run_lengths)
    max_len = min(max(run_lengths), 50)  # 限制显示范围
    x = list(range(1, max_len + 1))
    y = [counter.get(i, 0) / len(run_lengths) for i in x]

    ax.bar(x, y, alpha=0.7, label='Observed', color='steelblue')

    # 几何分布基线
    if 0 < switch_rate < 1:
        geom_y = [geometric_pmf(k, switch_rate) for k in x]
        ax.plot(x, geom_y, 'r-', linewidth=2, label=f'Geometric (p={switch_rate:.3f})')

    ax.set_xlabel('Run Length')
    ax.set_ylabel('Probability')
    ax.set_title(f'{task_name}: Run Length Distribution')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_window_drift_curve(
    task_name: str,
    jsd_values: List[float],
    shuffle_jsd_mean: float,
    output_path: str,
):
    """窗口漂移曲线"""
    if not jsd_values:
        return

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(jsd_values, 'b-', alpha=0.7, linewidth=0.8, label='Window JSD')
    ax.axhline(y=np.mean(jsd_values), color='blue', linestyle='--', label=f'Mean: {np.mean(jsd_values):.4f}')
    ax.axhline(y=shuffle_jsd_mean, color='red', linestyle='--', label=f'Shuffle baseline: {shuffle_jsd_mean:.4f}')

    ax.set_xlabel('Window Index')
    ax.set_ylabel('JSD')
    ax.set_title(f'{task_name}: Window Drift Curve')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_boundary_comparison(
    tasks: List[str],
    boundary_means: List[float],
    random_means: List[float],
    output_path: str,
):
    """边界 vs 随机位置的漂移对比"""
    x = np.arange(len(tasks))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, boundary_means, width, label='Boundary', color='steelblue')
    bars2 = ax.bar(x + width/2, random_means, width, label='Random', color='coral')

    ax.set_ylabel('JSD')
    ax.set_title('Boundary vs Random Position Drift (JSD)')
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=30, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_window_sensitivity(
    tasks: List[str],
    results_by_window: Dict[int, Dict[str, float]],
    output_path: str,
):
    """窗口大小敏感性分析"""
    window_sizes = sorted(results_by_window.keys())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # JSD
    for task in tasks:
        jsd_vals = [results_by_window[w].get(task, {}).get("window_jsd_mean", 0) for w in window_sizes]
        axes[0].plot(window_sizes, jsd_vals, 'o-', label=task)
    axes[0].set_xlabel('Window Size')
    axes[0].set_ylabel('Mean JSD')
    axes[0].set_title('Window Size Sensitivity (JSD)')
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    # Switch Rate（不受窗口大小影响，但展示以对比）
    for task in tasks:
        switch_vals = [results_by_window[w].get(task, {}).get("switch_rate", 0) for w in window_sizes]
        axes[1].plot(window_sizes, switch_vals, 'o-', label=task)
    axes[1].set_xlabel('Window Size')
    axes[1].set_ylabel('Switch Rate')
    axes[1].set_title('Switch Rate (should be constant)')
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_per_layer_heatmap(
    tasks: List[str],
    layer_results: Dict[str, Dict[str, Dict[str, float]]],
    output_path: str,
):
    """逐层切换率热力图"""
    # 获取所有层名（假设各任务相同）
    first_task = tasks[0]
    layer_names = sorted(layer_results[first_task].keys())

    # 构建矩阵
    matrix = np.zeros((len(tasks), len(layer_names)))
    for i, task in enumerate(tasks):
        for j, ln in enumerate(layer_names):
            matrix[i, j] = layer_results[task].get(ln, {}).get("switch_rate", 0)

    fig, ax = plt.subplots(figsize=(16, 8))
    im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd')

    ax.set_xticks(range(len(layer_names)))
    ax.set_xticklabels([ln.replace("moe_layer_", "L") for ln in layer_names], rotation=90, fontsize=8)
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels(tasks)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Task')
    ax.set_title('Per-Layer Switch Rate Heatmap')

    plt.colorbar(im, ax=ax, label='Switch Rate')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _bar_plot_simple(
    tasks: List[str],
    values: List[float],
    ylabel: str,
    title: str,
    output_path: str,
):
    """简单柱状图"""
    x = np.arange(len(tasks))
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(x, values, color='steelblue')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=30, ha='right')
    ax.grid(axis='y', alpha=0.3)

    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _bar_plot_with_baseline(
    tasks: List[str],
    values: List[float],
    baseline: List[float],
    ylabel: str,
    title: str,
    output_path: str,
):
    """带基线的双柱状图"""
    x = np.arange(len(tasks))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, values, width, label='Real', color='steelblue')
    bars2 = ax.bar(x + width/2, baseline, width, label='Baseline', color='coral')

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=30, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# ============ 主函数 ============

def main():
    parser = argparse.ArgumentParser(description="Task Internal Drift Analysis (Problem 3) - v2")
    parser.add_argument("--task", action="append", required=True, help="task_name=path (可多次指定)")
    parser.add_argument("--output_dir", required=True, help="输出目录")
    parser.add_argument("--n_experts", type=int, default=60, help="专家数量")
    parser.add_argument("--max_files", type=int, default=-1, help="每任务最多加载的 NPZ 文件数")
    parser.add_argument("--sample_size", type=int, default=20000, help="采样 token 数")
    parser.add_argument("--sample_start", type=int, default=0, help="采样起始位置")
    parser.add_argument("--window_sizes", type=str, default="64,128,256", help="窗口大小列表，逗号分隔")
    parser.add_argument("--window_stride", type=int, default=0, help="窗口步长，0 表示等于窗口大小")
    parser.add_argument("--bootstrap_runs", type=int, default=200, help="Bootstrap 次数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--skip_p0", action="store_true", help="跳过 P0 检查")
    parser.add_argument("--skip_per_layer", action="store_true", help="跳过逐层分析")
    parser.add_argument("--shuffle_baseline", action="store_true", help="启用置乱基线（学习 v1）")
    parser.add_argument("--normalize_per_layer", action="store_true", default=True, help="对每层概率先归一化（学习 v1）")
    args = parser.parse_args()

    # 解析任务
    tasks = {}
    for item in args.task:
        if "=" not in item:
            raise ValueError(f"Invalid task spec: {item}")
        name, path = item.split("=", 1)
        tasks[name.strip()] = path.strip()

    # 解析窗口大小
    window_sizes = [int(w.strip()) for w in args.window_sizes.split(",")]

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    plot_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # P0 检查
    if not args.skip_p0:
        logging.info("Running P0 data structure check...")
        p0_results = run_p0_check(tasks, args.output_dir)

    # 主分析
    summary = {
        "meta": vars(args),
        "tasks": {},
        "window_sensitivity": {},
    }

    task_names = list(tasks.keys())
    all_layer_results = {}

    for task_name, data_dir in tasks.items():
        logging.info(f"Analyzing task: {task_name}")

        try:
            tokens, probs_by_layer, inds_by_layer, segments = load_task_data(
                data_dir, args.max_files, args.sample_start, args.sample_size
            )
        except Exception as e:
            logging.error(f"Failed to load {task_name}: {e}")
            summary["tasks"][task_name] = {"error": str(e)}
            continue

        layer_names = sorted(probs_by_layer.keys())
        n_tokens = len(tokens)

        # 计算 Top1 序列
        top1 = get_token_top1(layer_names, probs_by_layer, inds_by_layer, n_tokens)

        # 切换率 + run length
        switch_rate_val, n_pairs = compute_switch_rate(top1, segments)
        run_lengths_list = compute_run_lengths(top1, segments)

        # 窗口漂移（使用第一个窗口大小作为主指标）
        main_window = window_sizes[0] if window_sizes else 128
        stride = args.window_stride if args.window_stride > 0 else main_window

        jsd_vals, tv_vals, _ = compute_window_drift(
            layer_names, probs_by_layer, inds_by_layer, segments,
            main_window, stride, args.n_experts, use_other_bucket=True
        )

        # 也计算 Top1 频率分布方案（方案 B）
        jsd_vals_b, tv_vals_b, _ = compute_window_drift(
            layer_names, probs_by_layer, inds_by_layer, segments,
            main_window, stride, args.n_experts, use_other_bucket=False
        )

        # 置乱基线（可选，学习 v1 的开关设计）
        shuffle_metrics = {"switch_rate_shuffle": None, "window_jsd_mean_shuffle": None, "window_tv_mean_shuffle": None}
        if args.shuffle_baseline:
            shuffle_metrics = compute_shuffled_metrics(
                layer_names, probs_by_layer, inds_by_layer, segments,
                main_window, stride, args.n_experts, rng
            )

        # 边界分析（规则边界 + 数据驱动备选）
        boundaries = detect_boundaries(tokens, task_name)
        boundary_method = "rule_based"
        if not boundaries and jsd_vals:
            # 如果规则边界检测失败，使用数据驱动边界
            boundaries = detect_boundaries_data_driven(jsd_vals, main_window)
            boundary_method = "data_driven"
            logging.info(f"{task_name}: Using data-driven boundary detection (found {len(boundaries)} peaks)")

        boundary_results = compute_boundary_effect(
            layer_names, probs_by_layer, inds_by_layer, segments,
            boundaries, main_window, args.n_experts, rng
        )

        # Bootstrap CI（改进版：对 switch_rate 和 run_mean 做正确的 Bootstrap）
        jsd_mean, jsd_ci_low, jsd_ci_high = bootstrap_ci(jsd_vals, args.bootstrap_runs, 0.95, rng)
        switch_mean, switch_ci_low, switch_ci_high = bootstrap_switch_rate_ci(
            top1, segments, args.bootstrap_runs, 0.95, rng
        )
        run_mean_val, run_ci_low, run_ci_high = bootstrap_run_mean_ci(
            run_lengths_list, args.bootstrap_runs, 0.95, rng
        )

        # KS 检验：run length 是否符合几何分布
        ks_result = ks_test_geometric(run_lengths_list, switch_rate_val)

        # 短任务警告（HumanEval/Winogrande）
        short_task_warning = None
        if n_tokens < 1000:
            short_task_warning = f"Warning: Only {n_tokens} tokens. Results may be unreliable due to small sample size."
            logging.warning(f"{task_name}: {short_task_warning}")

        # 逐层分析
        if not args.skip_per_layer:
            layer_results = analyze_per_layer(layer_names, probs_by_layer, inds_by_layer, segments)
            all_layer_results[task_name] = layer_results

        # 检查 tokens_str 是否存在（参照 v1）
        tokens_str_present = any(str(t) for t in tokens)
        avg_segment_len = float(np.mean([e - s for s, e in segments])) if segments else 0.0

        # 每个窗口大小的统计（参照 v1 的 window_stats 结构）
        window_stats = {}
        for ws in window_sizes:
            ws_stride = args.window_stride if args.window_stride > 0 else ws
            ws_jsd, ws_tv, _ = compute_window_drift(
                layer_names, probs_by_layer, inds_by_layer, segments,
                ws, ws_stride, args.n_experts, use_other_bucket=True
            )
            # 每个窗口大小的边界分析
            ws_boundary = compute_boundary_effect(
                layer_names, probs_by_layer, inds_by_layer, segments,
                boundaries, ws, args.n_experts, rng
            )
            window_stats[str(ws)] = {
                "window_jsd_mean": float(np.mean(ws_jsd)) if ws_jsd else 0.0,
                "window_tv_mean": float(np.mean(ws_tv)) if ws_tv else 0.0,
                "boundary_count": ws_boundary["n_boundaries"],
                "boundary_jsd_mean": float(np.mean(ws_boundary["boundary_jsd"])) if ws_boundary["boundary_jsd"] else None,
                "boundary_tv_mean": float(np.mean(ws_boundary["boundary_tv"])) if ws_boundary["boundary_tv"] else None,
                "random_boundary_jsd_mean": float(np.mean(ws_boundary["random_jsd"])) if ws_boundary["random_jsd"] else None,
                "random_boundary_tv_mean": float(np.mean(ws_boundary["random_tv"])) if ws_boundary["random_tv"] else None,
            }
            # 每个窗口大小的漂移曲线图
            plot_window_drift_curve(
                task_name, ws_jsd, shuffle_metrics["window_jsd_mean_shuffle"],
                os.path.join(plot_dir, f"{task_name}_window_jsd_w{ws}.png")
            )

        # 置乱基线的每窗口统计
        window_stats_shuffle = {}
        for ws in window_sizes:
            ws_stride = args.window_stride if args.window_stride > 0 else ws
            n_tokens_for_shuffle = probs_by_layer[layer_names[0]].shape[0]
            shuffle_idx = shuffle_within_segments(n_tokens_for_shuffle, segments, rng)
            probs_shuffled = {ln: probs_by_layer[ln][shuffle_idx] for ln in layer_names}
            inds_shuffled = {ln: inds_by_layer[ln][shuffle_idx] for ln in layer_names}
            ws_jsd_shuf, ws_tv_shuf, _ = compute_window_drift(
                layer_names, probs_shuffled, inds_shuffled, segments,
                ws, ws_stride, args.n_experts, use_other_bucket=True
            )
            window_stats_shuffle[str(ws)] = {
                "window_jsd_mean_shuffle": float(np.mean(ws_jsd_shuf)) if ws_jsd_shuf else 0.0,
                "window_tv_mean_shuffle": float(np.mean(ws_tv_shuf)) if ws_tv_shuf else 0.0,
            }

        # 汇总（增强版，参照 v1 结构 + 新增 CI）
        # run length 直方图（用于真实分布图）
        run_hist = None
        run_hist_ratio = None
        if run_lengths_list:
            counts = Counter(run_lengths_list)
            max_bin = 10
            hist = {str(k): int(counts.get(k, 0)) for k in range(1, max_bin + 1)}
            hist[f">{max_bin}"] = int(sum(v for k, v in counts.items() if k > max_bin))
            total = float(sum(counts.values()))
            if total > 0:
                hist_ratio = {k: v / total for k, v in hist.items()}
            else:
                hist_ratio = {k: 0.0 for k in hist}
            run_hist = hist
            run_hist_ratio = hist_ratio

        task_out = {
            "n_tokens": n_tokens,
            "n_segments": len(segments),
            "avg_segment_len": avg_segment_len,
            "n_pairs": n_pairs,
            "tokens_str_present": tokens_str_present,
            "short_task_warning": short_task_warning,  # 短任务警告
            "switch_rate": switch_rate_val,
            "switch_rate_ci_95": [switch_ci_low, switch_ci_high],  # 新增：CI
            "switch_rate_shuffle": shuffle_metrics["switch_rate_shuffle"],
            "run_mean": run_mean_val,  # 使用 Bootstrap 均值
            "run_mean_ci_95": [run_ci_low, run_ci_high],  # 新增：CI
            "run_median": float(np.median(run_lengths_list)) if run_lengths_list else 0.0,
            "run_p90": float(np.percentile(run_lengths_list, 90)) if run_lengths_list else 0.0,
            "geom_run_mean": geometric_baseline_mean(switch_rate_val),  # 与 v1 命名一致
            "ks_test_geometric": ks_result,  # KS 检验结果
            "run_length_hist": run_hist,
            "run_length_hist_ratio": run_hist_ratio,
            "window_jsd_mean": jsd_mean,
            "window_jsd_ci_95": [jsd_ci_low, jsd_ci_high],
            "window_tv_mean": float(np.mean(tv_vals)) if tv_vals else 0.0,
            "window_jsd_mean_method_b": float(np.mean(jsd_vals_b)) if jsd_vals_b else 0.0,
            "window_jsd_mean_shuffle": shuffle_metrics["window_jsd_mean_shuffle"],
            "window_tv_mean_shuffle": shuffle_metrics["window_tv_mean_shuffle"],
            "boundary_method": boundary_method,  # 新增：边界检测方法
            "n_boundaries": boundary_results["n_boundaries"],
            "boundary_jsd_mean": float(np.mean(boundary_results["boundary_jsd"])) if boundary_results["boundary_jsd"] else None,
            "random_jsd_mean": float(np.mean(boundary_results["random_jsd"])) if boundary_results["random_jsd"] else None,
            "window_stats": window_stats,  # 每窗口大小的详细统计
            "window_stats_shuffle": window_stats_shuffle,  # 置乱基线的每窗口统计
        }
        summary["tasks"][task_name] = task_out

        # 绘图
        plot_run_length_distribution(
            task_name, run_lengths_list, switch_rate_val,
            os.path.join(plot_dir, f"{task_name}_run_length.png")
        )

    # 窗口敏感性分析
    logging.info("Running window sensitivity analysis...")
    for ws in window_sizes:
        summary["window_sensitivity"][ws] = {}
        for task_name, data_dir in tasks.items():
            if task_name not in summary["tasks"] or "error" in summary["tasks"][task_name]:
                continue
            try:
                tokens, probs_by_layer, inds_by_layer, segments = load_task_data(
                    data_dir, args.max_files, args.sample_start, args.sample_size
                )
                layer_names = sorted(probs_by_layer.keys())
                n_tokens = len(tokens)
                top1 = get_token_top1(layer_names, probs_by_layer, inds_by_layer, n_tokens)
                switch_val, _ = compute_switch_rate(top1, segments)

                stride = ws
                jsd_vals, tv_vals, _ = compute_window_drift(
                    layer_names, probs_by_layer, inds_by_layer, segments,
                    ws, stride, args.n_experts, use_other_bucket=True
                )
                summary["window_sensitivity"][ws][task_name] = {
                    "switch_rate": switch_val,
                    "window_jsd_mean": float(np.mean(jsd_vals)) if jsd_vals else 0.0,
                    "window_tv_mean": float(np.mean(tv_vals)) if tv_vals else 0.0,
                }
            except Exception as e:
                logging.warning(f"Window sensitivity failed for {task_name} ws={ws}: {e}")

    # 汇总图
    valid_tasks = [t for t in task_names if t in summary["tasks"] and "error" not in summary["tasks"][t]]

    if valid_tasks:
        # 切换率对比
        real_switch = [summary["tasks"][t]["switch_rate"] for t in valid_tasks]
        shuf_switch = [summary["tasks"][t]["switch_rate_shuffle"] for t in valid_tasks]
        plot_switch_rate_comparison(valid_tasks, real_switch, shuf_switch, os.path.join(plot_dir, "switch_rate_comparison.png"))

        # Run length 均值柱状图
        run_means = [summary["tasks"][t]["run_mean"] for t in valid_tasks]
        _bar_plot_simple(valid_tasks, run_means, "mean run length", "Mean Run Length",
                        os.path.join(plot_dir, "run_length_mean.png"))

        # 每窗口大小的汇总图（参照 v1）
        for ws in window_sizes:
            ws_str = str(ws)
            # JSD 汇总
            jsd_vals = [summary["tasks"][t]["window_stats"].get(ws_str, {}).get("window_jsd_mean", 0) for t in valid_tasks]
            jsd_base = [summary["tasks"][t]["window_stats_shuffle"].get(ws_str, {}).get("window_jsd_mean_shuffle", 0) for t in valid_tasks]
            _bar_plot_with_baseline(valid_tasks, jsd_vals, jsd_base, "window JSD mean",
                                   f"Window Drift (JSD, w={ws})", os.path.join(plot_dir, f"window_jsd_mean_w{ws}.png"))

            # TV 汇总
            tv_vals = [summary["tasks"][t]["window_stats"].get(ws_str, {}).get("window_tv_mean", 0) for t in valid_tasks]
            tv_base = [summary["tasks"][t]["window_stats_shuffle"].get(ws_str, {}).get("window_tv_mean_shuffle", 0) for t in valid_tasks]
            _bar_plot_with_baseline(valid_tasks, tv_vals, tv_base, "window TV mean",
                                   f"Window Drift (TV, w={ws})", os.path.join(plot_dir, f"window_tv_mean_w{ws}.png"))

            # 边界 JSD 汇总
            boundary_vals = [summary["tasks"][t]["window_stats"].get(ws_str, {}).get("boundary_jsd_mean") or 0 for t in valid_tasks]
            random_boundary_vals = [summary["tasks"][t]["window_stats"].get(ws_str, {}).get("random_boundary_jsd_mean") or 0 for t in valid_tasks]
            _bar_plot_with_baseline(valid_tasks, boundary_vals, random_boundary_vals, "boundary JSD mean",
                                   f"Boundary Drift (JSD, w={ws})", os.path.join(plot_dir, f"boundary_jsd_mean_w{ws}.png"))

        # 边界对比（主窗口）
        boundary_means = [summary["tasks"][t].get("boundary_jsd_mean") or 0 for t in valid_tasks]
        random_means = [summary["tasks"][t].get("random_jsd_mean") or 0 for t in valid_tasks]
        plot_boundary_comparison(valid_tasks, boundary_means, random_means, os.path.join(plot_dir, "boundary_comparison.png"))

        # 窗口敏感性
        plot_window_sensitivity(valid_tasks, summary["window_sensitivity"], os.path.join(plot_dir, "window_sensitivity.png"))

        # 逐层热力图
        if all_layer_results:
            plot_per_layer_heatmap(valid_tasks, all_layer_results, os.path.join(plot_dir, "per_layer_switch_heatmap.png"))

    # 保存结果（含逐层结果，便于离线复现绘图）
    if all_layer_results:
        summary["per_layer"] = all_layer_results

    # 保存结果
    with open(os.path.join(args.output_dir, "task_drift_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=_json_default)

    logging.info(f"Done. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
