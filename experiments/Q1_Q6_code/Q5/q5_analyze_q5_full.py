#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q5 完整分析脚本：路由置信度特征分析（5个证据）

证据1：任务间置信度对比
证据2：多层置信度一致性
证据3：置信度 vs 切换率关联
证据4：序列位置 vs 置信度
证据5：低置信度 token 特征分析

输出：
    - q5_full_results.json：完整结果
    - plots/：所有图表
"""

import argparse
import json
import logging
import math
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# ===========================================================================
# 数学工具
# ===========================================================================
def _safe_normalize(vec: np.ndarray) -> np.ndarray:
    total = float(np.sum(vec))
    if total <= 0:
        return np.ones_like(vec, dtype=np.float64) / float(len(vec))
    return vec.astype(np.float64) / total


def entropy(p: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, eps, 1.0)
    p = _safe_normalize(p)
    return float(-np.sum(p * np.log2(p)))


def normalized_entropy(p: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=np.float64)
    k = len(p)
    if k <= 1:
        return 0.0
    h = entropy(p, eps)
    max_h = math.log2(k)
    return h / max_h if max_h > 0 else 0.0


def ks_test(group1: List[float], group2: List[float]) -> Dict[str, float]:
    try:
        from scipy.stats import ks_2samp
        stat, pval = ks_2samp(group1, group2)
        return {"statistic": float(stat), "p_value": float(pval)}
    except:
        return {"statistic": 0.0, "p_value": 1.0}


def cohens_d(group1: List[float], group2: List[float]) -> float:
    if not group1 or not group2:
        return 0.0
    n1, n2 = len(group1), len(group2)
    m1, m2 = np.mean(group1), np.mean(group2)
    s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    pooled_std = math.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if pooled_std < 1e-10:
        return 0.0
    return float((m1 - m2) / pooled_std)


def pearson_corr(x: List[float], y: List[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    x, y = np.array(x), np.array(y)
    return float(np.corrcoef(x, y)[0, 1])


def compute_statistics(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0, "n": 0}
    arr = np.array(values)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
        "q25": float(np.percentile(arr, 25)),
        "q75": float(np.percentile(arr, 75)),
        "n": len(arr),
    }


# ===========================================================================
# 数据加载
# ===========================================================================
def unpack_object_array(arr: np.ndarray) -> np.ndarray:
    """处理 object 类型数组，展开为连续 2D 数组"""
    if arr.dtype != object:
        return arr

    # object 数组表示多样本打包，每个元素是一个样本的数据
    all_data = []
    for sample in arr:
        if sample is not None and hasattr(sample, '__len__') and len(sample) > 0:
            sample_arr = np.asarray(sample)
            if sample_arr.ndim == 1:
                sample_arr = sample_arr.reshape(-1, 1)
            all_data.append(sample_arr)

    if not all_data:
        return np.array([]).reshape(0, 1)

    # 连接所有样本
    return np.concatenate(all_data, axis=0)


def load_npz_data(npz_path: str, layer: int) -> Optional[Dict[str, Any]]:
    """加载单个 NPZ 文件"""
    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as e:
        return None

    prob_key = f"layers/MoE_Gate{layer}_out_probs"
    idx_key = f"layers/MoE_Gate{layer}_out_indices"

    if prob_key not in data or idx_key not in data:
        return None

    probs = data[prob_key]
    indices = data[idx_key]
    tokens_str = data.get("tokens_str", None)

    # 处理 object 类型数组（多样本打包格式）
    probs = unpack_object_array(probs)
    indices = unpack_object_array(indices)

    # 处理 tokens_str
    if tokens_str is not None and tokens_str.dtype == object:
        # tokens_str 可能也是打包的
        all_tokens = []
        for sample_tokens in tokens_str:
            if sample_tokens is not None:
                all_tokens.extend(list(sample_tokens))
        tokens_str = np.array(all_tokens) if all_tokens else None

    return {
        "probs": probs,
        "indices": indices,
        "tokens_str": tokens_str,
        "npz_path": npz_path,
    }


def load_npz_multilayer(npz_path: str, layers: List[int]) -> Optional[Dict[str, Any]]:
    """加载单个 NPZ 文件的多层数据"""
    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as e:
        return None

    tokens_str = data.get("tokens_str", None)
    # 处理 tokens_str
    if tokens_str is not None and tokens_str.dtype == object:
        all_tokens = []
        for sample_tokens in tokens_str:
            if sample_tokens is not None:
                all_tokens.extend(list(sample_tokens))
        tokens_str = np.array(all_tokens) if all_tokens else None

    result = {
        "tokens_str": tokens_str,
        "npz_path": npz_path,
        "layers": {}
    }

    for layer in layers:
        prob_key = f"layers/MoE_Gate{layer}_out_probs"
        idx_key = f"layers/MoE_Gate{layer}_out_indices"
        if prob_key in data and idx_key in data:
            probs = unpack_object_array(data[prob_key])
            indices = unpack_object_array(data[idx_key])
            result["layers"][layer] = {
                "probs": probs,
                "indices": indices,
            }

    if not result["layers"]:
        return None
    return result


def load_task_data(data_dir: str, layer: int, max_files: int = -1) -> List[Dict]:
    """加载单个任务的所有样本"""
    if not os.path.isdir(data_dir):
        logging.warning(f"目录不存在: {data_dir}")
        return []

    npz_files = sorted([
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith(".npz")
    ])

    if max_files > 0:
        npz_files = npz_files[:max_files]

    samples = []
    total_tokens = 0
    for npz_path in npz_files:
        data = load_npz_data(npz_path, layer)
        if data:
            samples.append(data)
            if data["probs"] is not None:
                total_tokens += data["probs"].shape[0]

    logging.info(f"  加载 {len(samples)} 个样本, 共 {total_tokens} tokens from {data_dir}")
    return samples


# ===========================================================================
# 证据1：任务间置信度对比
# ===========================================================================
def extract_token_confidence(probs: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
    """提取 token 级置信度特征"""
    # 检查 None
    if probs is None:
        return None

    # 处理 object 类型数组（不应该发生，因为已在加载时处理）
    if probs.dtype == object:
        logging.debug("extract_token_confidence: 遇到 object 类型数组（不应发生）")
        return None

    # 确保是 2D 数组
    if len(probs.shape) == 1:
        probs = probs.reshape(-1, 1)

    # 检查数据有效性
    if probs.shape[0] == 0:
        return None

    n_tokens = probs.shape[0]
    top_k = probs.shape[1]

    # 转换为 float，处理可能的类型问题
    try:
        probs = probs.astype(np.float64)
    except:
        return None

    top1_prob = np.max(probs, axis=1)

    if top_k >= 2:
        sorted_probs = np.sort(probs, axis=1)[:, ::-1]
        margin = sorted_probs[:, 0] - sorted_probs[:, 1]
    else:
        margin = np.ones(n_tokens)

    entropies = np.array([normalized_entropy(probs[t]) for t in range(n_tokens)])

    return {
        "top1_prob": top1_prob,
        "margin": margin,
        "entropy": entropies,
    }


def analyze_evidence1(tasks_data: Dict[str, List[Dict]]) -> Dict[str, Any]:
    """证据1：任务间置信度对比"""
    logging.info("=== 证据1：任务间置信度对比 ===")

    results = {"tasks": {}, "pairwise": {}, "ranking": []}
    task_features = {}

    for task_name, samples in tasks_data.items():
        logging.info(f"  处理: {task_name} ({len(samples)} 样本)")

        all_top1, all_margin, all_entropy = [], [], []
        sample_means = []

        for sample in samples:
            probs = sample["probs"]
            if probs is None or len(probs) == 0:
                continue

            feats = extract_token_confidence(probs)
            if feats is None:
                continue

            all_top1.extend(feats["top1_prob"].tolist())
            all_margin.extend(feats["margin"].tolist())
            all_entropy.extend(feats["entropy"].tolist())
            sample_means.append(float(np.mean(feats["top1_prob"])))

        task_features[task_name] = sample_means

        results["tasks"][task_name] = {
            "n_samples": len(samples),
            "n_tokens": len(all_top1),
            "top1_prob": compute_statistics(all_top1),
            "margin": compute_statistics(all_margin),
            "entropy": compute_statistics(all_entropy),
        }

    # 任务间两两对比
    task_names = list(task_features.keys())
    for i, t1 in enumerate(task_names):
        for t2 in task_names[i + 1:]:
            f1, f2 = task_features[t1], task_features[t2]
            if f1 and f2:
                results["pairwise"][f"{t1}_vs_{t2}"] = {
                    "ks_test": ks_test(f1, f2),
                    "cohens_d": cohens_d(f1, f2),
                }

    # 按置信度排序
    results["ranking"] = sorted(
        task_names,
        key=lambda t: results["tasks"][t]["top1_prob"]["mean"],
        reverse=True
    )

    return results


# ===========================================================================
# 证据2：多层置信度一致性
# ===========================================================================
def analyze_evidence2(data_dir: str, layers: List[int], max_files: int = 500) -> Dict[str, Any]:
    """证据2：多层置信度一致性"""
    logging.info(f"=== 证据2：多层置信度一致性 (layers={layers}) ===")

    npz_files = sorted([
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith(".npz")
    ])[:max_files]

    layer_top1_means = {l: [] for l in layers}

    for npz_path in npz_files:
        data = load_npz_multilayer(npz_path, layers)
        if not data:
            continue

        for layer in layers:
            if layer not in data["layers"]:
                continue
            probs = data["layers"][layer]["probs"]
            if probs is None or len(probs) == 0:
                continue
            feats = extract_token_confidence(probs)
            if feats is None:
                continue
            layer_top1_means[layer].append(float(np.mean(feats["top1_prob"])))

    # 计算层间相关
    results = {
        "layers": {},
        "layer_correlations": {},
        "conclusion": ""
    }

    for layer in layers:
        if layer_top1_means[layer]:
            results["layers"][f"layer_{layer}"] = compute_statistics(layer_top1_means[layer])

    # 两两相关
    for i, l1 in enumerate(layers):
        for l2 in layers[i + 1:]:
            v1, v2 = layer_top1_means[l1], layer_top1_means[l2]
            min_len = min(len(v1), len(v2))
            if min_len > 10:
                corr = pearson_corr(v1[:min_len], v2[:min_len])
                results["layer_correlations"][f"layer{l1}_vs_layer{l2}"] = corr

    # 结论
    if results["layer_correlations"]:
        avg_corr = np.mean(list(results["layer_correlations"].values()))
        if avg_corr > 0.8:
            results["conclusion"] = f"高度一致 (avg_corr={avg_corr:.3f})，单层可代表"
        elif avg_corr > 0.5:
            results["conclusion"] = f"中度一致 (avg_corr={avg_corr:.3f})，建议取平均"
        else:
            results["conclusion"] = f"一致性低 (avg_corr={avg_corr:.3f})，需分层建模"

    logging.info(f"  层间相关: {results['layer_correlations']}")
    logging.info(f"  结论: {results['conclusion']}")

    return results


# ===========================================================================
# 证据3：置信度 vs 切换率
# ===========================================================================
def analyze_evidence3(data_dir: str, layer: int, max_files: int = 500) -> Dict[str, Any]:
    """证据3：置信度 vs 切换率关联"""
    logging.info(f"=== 证据3：置信度 vs 切换率 (layer={layer}) ===")

    npz_files = sorted([
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith(".npz")
    ])[:max_files]

    # 收集每个 token 的置信度和是否切换
    confidence_bins = {
        "very_low": {"conf": [], "switch": []},   # top1 < 0.05
        "low": {"conf": [], "switch": []},        # 0.05-0.10
        "medium": {"conf": [], "switch": []},     # 0.10-0.20
        "high": {"conf": [], "switch": []},       # > 0.20
    }

    layers_to_check = [layer, layer + 1] if layer + 1 < 24 else [layer]

    for npz_path in npz_files:
        data = load_npz_multilayer(npz_path, layers_to_check)
        if not data or len(layers_to_check) < 2:
            continue
        if layer not in data["layers"] or layer + 1 not in data["layers"]:
            continue

        probs_l = data["layers"][layer]["probs"]
        indices_l = data["layers"][layer]["indices"]
        indices_l1 = data["layers"][layer + 1]["indices"]

        if len(probs_l) == 0:
            continue

        feats = extract_token_confidence(probs_l)
        if feats is None:
            continue
        top1_probs = feats["top1_prob"]

        # Top-1 专家 ID
        top1_l = indices_l[:, 0] if len(indices_l.shape) > 1 else indices_l
        top1_l1 = indices_l1[:, 0] if len(indices_l1.shape) > 1 else indices_l1

        min_len = min(len(top1_probs), len(top1_l), len(top1_l1))

        for t in range(min_len):
            conf = top1_probs[t]
            switched = int(top1_l[t] != top1_l1[t])

            if conf < 0.05:
                bin_name = "very_low"
            elif conf < 0.10:
                bin_name = "low"
            elif conf < 0.20:
                bin_name = "medium"
            else:
                bin_name = "high"

            confidence_bins[bin_name]["conf"].append(conf)
            confidence_bins[bin_name]["switch"].append(switched)

    results = {"bins": {}, "conclusion": ""}

    for bin_name, data in confidence_bins.items():
        if data["switch"]:
            switch_rate = float(np.mean(data["switch"]))
            results["bins"][bin_name] = {
                "n_tokens": len(data["switch"]),
                "avg_confidence": float(np.mean(data["conf"])),
                "switch_rate": switch_rate,
            }

    # 判断趋势
    if results["bins"]:
        rates = [(k, v["switch_rate"]) for k, v in results["bins"].items() if "switch_rate" in v]
        if len(rates) >= 2:
            order = ["very_low", "low", "medium", "high"]
            ordered_rates = [results["bins"].get(k, {}).get("switch_rate", 0) for k in order if k in results["bins"]]
            if len(ordered_rates) >= 2:
                if ordered_rates[0] > ordered_rates[-1] + 0.02:
                    results["conclusion"] = "低置信度切换率更高，假设成立"
                elif abs(ordered_rates[0] - ordered_rates[-1]) < 0.02:
                    results["conclusion"] = "切换率与置信度无明显关联"
                else:
                    results["conclusion"] = "高置信度切换率更高，与预期相反"

    logging.info(f"  各桶切换率: {results['bins']}")
    logging.info(f"  结论: {results['conclusion']}")

    return results


# ===========================================================================
# 证据4：序列位置 vs 置信度
# ===========================================================================
def analyze_evidence4(tasks_data: Dict[str, List[Dict]]) -> Dict[str, Any]:
    """证据4：序列位置 vs 置信度"""
    logging.info("=== 证据4：序列位置 vs 置信度 ===")

    results = {"tasks": {}}

    for task_name, samples in tasks_data.items():
        logging.info(f"  处理: {task_name}")

        # 分段统计
        segments = {
            "first_25pct": [],
            "middle_50pct": [],
            "last_25pct": [],
        }

        for sample in samples:
            probs = sample["probs"]
            if probs is None or len(probs) < 8:
                continue

            feats = extract_token_confidence(probs)
            if feats is None:
                continue
            n = len(feats["entropy"])

            q1, q3 = n // 4, 3 * n // 4
            segments["first_25pct"].extend(feats["entropy"][:q1].tolist())
            segments["middle_50pct"].extend(feats["entropy"][q1:q3].tolist())
            segments["last_25pct"].extend(feats["entropy"][q3:].tolist())

        task_result = {}
        for seg_name, values in segments.items():
            if values:
                task_result[seg_name] = compute_statistics(values)

        # 判断趋势
        if all(seg in task_result for seg in ["first_25pct", "last_25pct"]):
            first_mean = task_result["first_25pct"]["mean"]
            last_mean = task_result["last_25pct"]["mean"]
            diff = last_mean - first_mean

            if diff > 0.02:
                trend = "末段熵上升（长程不确定性增加）"
            elif diff < -0.02:
                trend = "末段熵下降（路由更确定）"
            else:
                trend = "无明显位置效应"
            task_result["trend"] = trend
            task_result["entropy_diff_last_minus_first"] = diff

        results["tasks"][task_name] = task_result

    return results


# ===========================================================================
# 证据5：低置信度 token 特征
# ===========================================================================
def analyze_evidence5(tasks_data: Dict[str, List[Dict]], entropy_threshold: float = 0.95) -> Dict[str, Any]:
    """证据5：低置信度 token 特征分析"""
    logging.info(f"=== 证据5：低置信度 token 分析 (entropy > {entropy_threshold}) ===")

    results = {"tasks": {}, "global": {}}

    # Token 类型分类器
    def classify_token(tok: str) -> str:
        if tok is None:
            return "unknown"
        tok = str(tok)
        # 去除特殊前缀
        tok_clean = tok.lstrip("Ġ▁")
        if not tok_clean:
            return "whitespace"
        if tok_clean in ".,;:!?()[]{}\"'`-–—…":
            return "punctuation"
        if tok_clean.isdigit():
            return "number"
        if len(tok_clean) == 1:
            return "single_char"
        if tok_clean.isalpha() and len(tok_clean) <= 3:
            return "short_word"
        if tok_clean.isalpha():
            return "word"
        if re.match(r'^[\u4e00-\u9fff]+$', tok_clean):
            return "chinese"
        return "other"

    global_type_counts = defaultdict(int)
    global_high_entropy_type_counts = defaultdict(int)

    for task_name, samples in tasks_data.items():
        logging.info(f"  处理: {task_name}")

        type_counts = defaultdict(int)
        high_entropy_type_counts = defaultdict(int)
        high_entropy_examples = []

        for sample in samples:
            probs = sample["probs"]
            tokens_str = sample.get("tokens_str")
            if probs is None or len(probs) == 0:
                continue

            feats = extract_token_confidence(probs)
            if feats is None:
                continue

            for t in range(len(feats["entropy"])):
                tok = tokens_str[t] if tokens_str is not None and t < len(tokens_str) else None
                tok_type = classify_token(tok)
                type_counts[tok_type] += 1
                global_type_counts[tok_type] += 1

                if feats["entropy"][t] > entropy_threshold:
                    high_entropy_type_counts[tok_type] += 1
                    global_high_entropy_type_counts[tok_type] += 1
                    if len(high_entropy_examples) < 20:
                        high_entropy_examples.append({
                            "token": str(tok),
                            "type": tok_type,
                            "entropy": float(feats["entropy"][t]),
                        })

        # 计算占比
        total_tokens = sum(type_counts.values())
        total_high_ent = sum(high_entropy_type_counts.values())

        task_result = {
            "total_tokens": total_tokens,
            "high_entropy_tokens": total_high_ent,
            "high_entropy_ratio": total_high_ent / total_tokens if total_tokens > 0 else 0,
            "type_distribution_all": {k: v / total_tokens for k, v in type_counts.items()} if total_tokens > 0 else {},
            "type_distribution_high_entropy": {k: v / total_high_ent for k, v in high_entropy_type_counts.items()} if total_high_ent > 0 else {},
            "examples": high_entropy_examples[:10],
        }

        # 判断
        if total_high_ent > 0:
            punc_ratio = high_entropy_type_counts.get("punctuation", 0) / total_high_ent
            word_ratio = (high_entropy_type_counts.get("word", 0) + high_entropy_type_counts.get("chinese", 0)) / total_high_ent
            if punc_ratio > 0.3:
                task_result["conclusion"] = f"低置信度集中在标点 ({punc_ratio:.1%})，可简化处理"
            elif word_ratio > 0.5:
                task_result["conclusion"] = f"低置信度多为内容词 ({word_ratio:.1%})，需完整建模"
            else:
                task_result["conclusion"] = "低置信度分布较均匀"

        results["tasks"][task_name] = task_result

    # 全局统计
    total_global = sum(global_type_counts.values())
    total_high_global = sum(global_high_entropy_type_counts.values())
    results["global"] = {
        "total_tokens": total_global,
        "high_entropy_tokens": total_high_global,
        "high_entropy_ratio": total_high_global / total_global if total_global > 0 else 0,
        "type_distribution_high_entropy": {k: v / total_high_global for k, v in global_high_entropy_type_counts.items()} if total_high_global > 0 else {},
    }

    return results


# ===========================================================================
# 主函数
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="Q5 完整分析：路由置信度特征（5个证据）")

    parser.add_argument("--task", action="append", dest="tasks", required=True,
                        help="任务配置，格式：name=path")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--layer", type=int, default=0, help="主分析层 (默认 0)")
    parser.add_argument("--layers_e2", type=str, default="0,4,8,12,16,20",
                        help="证据2多层分析的层列表")
    parser.add_argument("--max_files", type=int, default=-1, help="每任务最多文件数")
    parser.add_argument("--entropy_threshold", type=float, default=0.95,
                        help="证据5的高熵阈值")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    args = parser.parse_args()
    np.random.seed(args.seed)

    # 解析任务配置
    tasks_config = {}
    for task_spec in args.tasks:
        if "=" in task_spec:
            name, path = task_spec.split("=", 1)
            tasks_config[name.strip()] = path.strip()

    logging.info(f"任务: {list(tasks_config.keys())}")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载数据
    logging.info("=== 加载数据 ===")
    tasks_data = {}
    for task_name, data_dir in tasks_config.items():
        samples = load_task_data(data_dir, args.layer, args.max_files)
        if samples:
            tasks_data[task_name] = samples

    if not tasks_data:
        logging.error("无有效数据")
        return

    # ========== 执行5个证据分析 ==========
    results = {"config": vars(args), "evidence": {}}

    # 证据1：任务间置信度对比
    results["evidence"]["E1_task_comparison"] = analyze_evidence1(tasks_data)

    # 证据2：多层置信度一致性（用第一个任务的数据）
    layers_e2 = [int(x) for x in args.layers_e2.split(",")]
    first_task_dir = list(tasks_config.values())[0]
    results["evidence"]["E2_layer_consistency"] = analyze_evidence2(first_task_dir, layers_e2, max_files=500)

    # 证据3：置信度 vs 切换率（用第一个任务的数据）
    results["evidence"]["E3_confidence_vs_switch"] = analyze_evidence3(first_task_dir, args.layer, max_files=500)

    # 证据4：序列位置 vs 置信度
    results["evidence"]["E4_position_effect"] = analyze_evidence4(tasks_data)

    # 证据5：低置信度 token 特征
    results["evidence"]["E5_low_confidence_tokens"] = analyze_evidence5(tasks_data, args.entropy_threshold)

    # 保存结果
    output_file = os.path.join(args.output_dir, "q5_full_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logging.info(f"结果已保存到: {output_file}")

    # 打印摘要
    logging.info("\n" + "=" * 60)
    logging.info("Q5 分析摘要")
    logging.info("=" * 60)

    # E1 摘要
    e1 = results["evidence"]["E1_task_comparison"]
    logging.info("\n[证据1] 任务间置信度对比:")
    logging.info(f"  确定性排名: {e1['ranking']}")
    for task in e1["ranking"][:3]:
        stats = e1["tasks"][task]
        logging.info(f"    {task}: Top-1={stats['top1_prob']['mean']:.4f}, 熵={stats['entropy']['mean']:.4f}")

    # E2 摘要
    e2 = results["evidence"]["E2_layer_consistency"]
    logging.info(f"\n[证据2] 多层一致性: {e2['conclusion']}")

    # E3 摘要
    e3 = results["evidence"]["E3_confidence_vs_switch"]
    logging.info(f"\n[证据3] 置信度vs切换率: {e3['conclusion']}")

    # E4 摘要
    e4 = results["evidence"]["E4_position_effect"]
    logging.info("\n[证据4] 位置效应:")
    for task, data in e4["tasks"].items():
        if "trend" in data:
            logging.info(f"    {task}: {data['trend']}")

    # E5 摘要
    e5 = results["evidence"]["E5_low_confidence_tokens"]
    logging.info("\n[证据5] 低置信度token特征:")
    for task, data in e5["tasks"].items():
        if "conclusion" in data:
            logging.info(f"    {task}: {data['conclusion']}")


if __name__ == "__main__":
    main()
