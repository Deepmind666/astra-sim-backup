#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q5：路由置信度特征分析

核心问题：路由置信度能否解释任务难度，甚至预测样本答对/答错？

主指标：
    - Top-1 概率：最高路由概率
    - Margin：Top-1 与 Top-2 概率差
    - 路由熵：归一化不确定性

分析阶段：
    - Phase 1（无标签）：任务间置信度差异、与切换率关联
    - Phase 3（有标签）：正确 vs 错误对比、AUC、校准曲线

输出：
    - q5_confidence_results.json：完整结果
    - plots/：所有图表

注意：本脚本面向不会写代码的同事，因此注释极其详细。
"""

import argparse
import json
import logging
import math
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

# ---------------------------------------------------------------------------
# 1. 日志设置
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# ---------------------------------------------------------------------------
# 2. 数学工具：熵、归一化
# ---------------------------------------------------------------------------
def _safe_normalize(vec: np.ndarray) -> np.ndarray:
    """
    把向量归一化为概率分布（和为1）。
    如果向量总和为 0，就返回均匀分布（避免除零）。
    """
    total = float(np.sum(vec))
    if total <= 0:
        return np.ones_like(vec, dtype=np.float64) / float(len(vec))
    return vec.astype(np.float64) / total


def entropy(p: np.ndarray, eps: float = 1e-12) -> float:
    """
    分布熵（log2）。用于衡量路由不确定性。
    """
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, eps, 1.0)
    p = _safe_normalize(p)
    return float(-np.sum(p * np.log2(p)))


def normalized_entropy(p: np.ndarray, eps: float = 1e-12) -> float:
    """
    归一化熵：H / log2(K)，范围 [0, 1]。
    K 是概率分布的维度。
    """
    p = np.asarray(p, dtype=np.float64)
    k = len(p)
    if k <= 1:
        return 0.0
    h = entropy(p, eps)
    max_h = math.log2(k)
    return h / max_h if max_h > 0 else 0.0


# ---------------------------------------------------------------------------
# 3. 置信度特征提取：Token 级别
# ---------------------------------------------------------------------------
def extract_token_confidence(probs: np.ndarray) -> Dict[str, np.ndarray]:
    """
    从 Top-K 概率矩阵中提取 token 级置信度特征。

    参数：
        probs: shape (n_tokens, top_k)，每个 token 的 Top-K 概率

    返回：
        {
            "top1_prob": shape (n_tokens,) - Top-1 概率
            "margin": shape (n_tokens,) - Top-1 与 Top-2 差值
            "entropy": shape (n_tokens,) - 归一化路由熵
        }
    """
    n_tokens = probs.shape[0]
    top_k = probs.shape[1] if len(probs.shape) > 1 else 1

    # Top-1 概率：假设 probs[t, 0] 是概率最大的（如果已排序）
    # 为安全起见，取 max
    top1_prob = np.max(probs, axis=1) if top_k > 1 else probs.flatten()

    # Margin：Top-1 - Top-2
    if top_k >= 2:
        sorted_probs = np.sort(probs, axis=1)[:, ::-1]  # 降序排列
        margin = sorted_probs[:, 0] - sorted_probs[:, 1]
    else:
        margin = np.ones(n_tokens)

    # 归一化熵：对每个 token 的 Top-K 概率计算
    entropies = np.array([normalized_entropy(probs[t]) for t in range(n_tokens)])

    return {
        "top1_prob": top1_prob,
        "margin": margin,
        "entropy": entropies,
    }


def aggregate_sample_features(token_features: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    将 token 级特征聚合为 样本级 特征。

    返回：
        {
            "top1_prob_mean": float,
            "top1_prob_std": float,
            "margin_mean": float,
            "margin_std": float,
            "entropy_mean": float,
            "entropy_max": float,  # 最犹豫的 token
            "low_conf_ratio": float,  # Top-1 < 0.2 的 token 占比
        }
    """
    top1 = token_features["top1_prob"]
    margin = token_features["margin"]
    ent = token_features["entropy"]

    return {
        "top1_prob_mean": float(np.mean(top1)),
        "top1_prob_std": float(np.std(top1)),
        "margin_mean": float(np.mean(margin)),
        "margin_std": float(np.std(margin)),
        "entropy_mean": float(np.mean(ent)),
        "entropy_max": float(np.max(ent)),
        "entropy_std": float(np.std(ent)),
        "low_conf_ratio": float(np.mean(top1 < 0.2)),
        "n_tokens": len(top1),
    }


# ---------------------------------------------------------------------------
# 4. 数据读取
# ---------------------------------------------------------------------------
def _iter_npz_samples(npz_path: str, layer: int) -> List[Dict[str, np.ndarray]]:
    """
    从一个 NPZ 里提取所有样本。

    兼容两种格式：
    1) 常见格式：一个 NPZ = 一个样本（数组为 2D）。
    2) 旧格式：一个 NPZ = 多个样本（数组 dtype=object）。
    """
    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as e:
        logging.warning(f"[SKIP] 无法加载: {npz_path} - {e}")
        return []

    prob_key = f"layers/MoE_Gate{layer}_out_probs"
    idx_key = f"layers/MoE_Gate{layer}_out_indices"

    if prob_key not in data or idx_key not in data:
        logging.debug(f"[SKIP] 缺少关键键: {npz_path}")
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
                "tokens_str": tokens_str[i] if tokens_str is not None else None,
                "npz_path": npz_path,
                "sample_idx": i,
            }
            samples.append(sample)
    else:
        # 常见格式：单样本
        sample = {
            "probs": probs,
            "indices": indices,
            "tokens_str": tokens_str if tokens_str is not None else None,
            "npz_path": npz_path,
            "sample_idx": 0,
        }
        samples.append(sample)

    return samples


def load_task_data(
    data_dir: str,
    layer: int,
    max_files: int = -1,
    max_samples: int = 0,
) -> List[Dict[str, Any]]:
    """
    加载单个任务的所有样本。

    返回：
        List[{
            "probs": np.ndarray,
            "indices": np.ndarray,
            "tokens_str": Optional[np.ndarray],
            "npz_path": str,
            "sample_idx": int,
        }]
    """
    if not os.path.isdir(data_dir):
        logging.warning(f"[SKIP] 目录不存在: {data_dir}")
        return []

    npz_files = sorted([
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith(".npz")
    ])

    if max_files > 0:
        npz_files = npz_files[:max_files]

    all_samples = []
    for npz_path in npz_files:
        samples = _iter_npz_samples(npz_path, layer)
        all_samples.extend(samples)
        if max_samples > 0 and len(all_samples) >= max_samples:
            all_samples = all_samples[:max_samples]
            break

    logging.info(f"  加载 {len(all_samples)} 个样本 from {data_dir}")
    return all_samples


# ---------------------------------------------------------------------------
# 5. 标签加载（Phase 3）
# ---------------------------------------------------------------------------
def load_labels(labels_path: str) -> Dict[int, bool]:
    """
    加载 is_correct 标签文件。

    格式：JSON 数组，每个元素为 {"sample_id": int, "is_correct": bool}

    返回：
        {sample_id: is_correct}
    """
    if not os.path.isfile(labels_path):
        logging.warning(f"[SKIP] 标签文件不存在: {labels_path}")
        return {}

    with open(labels_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    labels = {}
    for item in data:
        sid = item.get("sample_id", item.get("id", None))
        correct = item.get("is_correct", item.get("correct", None))
        if sid is not None and correct is not None:
            labels[int(sid)] = bool(correct)

    logging.info(f"  加载 {len(labels)} 个标签 from {labels_path}")
    return labels


# ---------------------------------------------------------------------------
# 6. 统计分析工具
# ---------------------------------------------------------------------------
def compute_statistics(values: List[float]) -> Dict[str, float]:
    """
    计算基本统计量。
    """
    if not values:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "median": 0.0,
            "q25": 0.0,
            "q75": 0.0,
            "n": 0,
        }

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


def ks_test(group1: List[float], group2: List[float]) -> Dict[str, float]:
    """
    Kolmogorov-Smirnov 检验：两组分布是否相同。

    返回：
        {"statistic": float, "p_value": float}
    """
    try:
        from scipy.stats import ks_2samp
        stat, pval = ks_2samp(group1, group2)
        return {"statistic": float(stat), "p_value": float(pval)}
    except ImportError:
        logging.warning("scipy 未安装，跳过 KS 检验")
        return {"statistic": 0.0, "p_value": 1.0}
    except Exception as e:
        logging.warning(f"KS 检验失败: {e}")
        return {"statistic": 0.0, "p_value": 1.0}


def cohens_d(group1: List[float], group2: List[float]) -> float:
    """
    Cohen's d 效应量：衡量两组均值差异的标准化程度。
    """
    if not group1 or not group2:
        return 0.0

    n1, n2 = len(group1), len(group2)
    m1, m2 = np.mean(group1), np.mean(group2)
    s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)

    # 池化标准差
    pooled_std = math.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))

    if pooled_std < 1e-10:
        return 0.0

    return float((m1 - m2) / pooled_std)


def compute_auc(y_true: List[bool], y_scores: List[float]) -> float:
    """
    计算 AUC（ROC 曲线下面积）。

    y_true: 真实标签 (True=正确, False=错误)
    y_scores: 置信度分数
    """
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_true, y_scores))
    except ImportError:
        logging.warning("sklearn 未安装，使用简易 AUC 计算")
        # 简易实现：Mann-Whitney U / (n1 * n2)
        pos = [s for t, s in zip(y_true, y_scores) if t]
        neg = [s for t, s in zip(y_true, y_scores) if not t]
        if not pos or not neg:
            return 0.5
        count = sum(1 for p in pos for n in neg if p > n)
        count += 0.5 * sum(1 for p in pos for n in neg if p == n)
        return count / (len(pos) * len(neg))
    except Exception as e:
        logging.warning(f"AUC 计算失败: {e}")
        return 0.5


def compute_calibration(y_true: List[bool], y_scores: List[float], n_bins: int = 10) -> Dict[str, Any]:
    """
    计算校准曲线和 ECE（Expected Calibration Error）。

    返回：
        {
            "bins": List[float],  # 每个 bin 的平均置信度
            "accuracies": List[float],  # 每个 bin 的实际准确率
            "counts": List[int],  # 每个 bin 的样本数
            "ece": float,  # 期望校准误差
        }
    """
    if not y_true or not y_scores:
        return {"bins": [], "accuracies": [], "counts": [], "ece": 0.0}

    scores = np.array(y_scores)
    labels = np.array(y_true, dtype=bool)

    bins_conf = []
    bins_acc = []
    bins_count = []

    bin_edges = np.linspace(0, 1, n_bins + 1)

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        mask = (scores >= low) & (scores < high)
        if i == n_bins - 1:
            mask = (scores >= low) & (scores <= high)

        if np.sum(mask) > 0:
            avg_conf = float(np.mean(scores[mask]))
            avg_acc = float(np.mean(labels[mask]))
            count = int(np.sum(mask))
        else:
            avg_conf = (low + high) / 2
            avg_acc = 0.0
            count = 0

        bins_conf.append(avg_conf)
        bins_acc.append(avg_acc)
        bins_count.append(count)

    # ECE = sum(count * |acc - conf|) / total
    total = sum(bins_count)
    if total > 0:
        ece = sum(c * abs(a - conf) for c, a, conf in zip(bins_count, bins_acc, bins_conf)) / total
    else:
        ece = 0.0

    return {
        "bins": bins_conf,
        "accuracies": bins_acc,
        "counts": bins_count,
        "ece": float(ece),
    }


# ---------------------------------------------------------------------------
# 7. Phase 1：无标签分析
# ---------------------------------------------------------------------------
def analyze_phase1(
    tasks_data: Dict[str, List[Dict[str, Any]]],
    output_dir: str,
) -> Dict[str, Any]:
    """
    Phase 1：无标签分析

    内容：
        - E1: Top-1 概率分布
        - E2: 路由熵分布
        - E3-weak: 任务间置信度差异
    """
    logging.info("=== Phase 1: 无标签分析 ===")

    results = {
        "phase": 1,
        "tasks": {},
        "cross_task_comparison": {},
    }

    # 每个任务的样本级特征
    task_features = {}  # task_name -> List[sample_features]

    for task_name, samples in tasks_data.items():
        logging.info(f"  处理任务: {task_name} ({len(samples)} 样本)")

        sample_features_list = []
        all_top1 = []
        all_margin = []
        all_entropy = []

        for sample in samples:
            probs = sample["probs"]
            if probs is None or len(probs) == 0:
                continue

            # 提取 token 级特征
            token_feats = extract_token_confidence(probs)

            # 聚合为样本级特征
            sample_feats = aggregate_sample_features(token_feats)
            sample_features_list.append(sample_feats)

            # 收集用于分布分析
            all_top1.extend(token_feats["top1_prob"].tolist())
            all_margin.extend(token_feats["margin"].tolist())
            all_entropy.extend(token_feats["entropy"].tolist())

        task_features[task_name] = sample_features_list

        # 任务级统计
        results["tasks"][task_name] = {
            "n_samples": len(sample_features_list),
            "n_tokens": len(all_top1),
            "token_level": {
                "top1_prob": compute_statistics(all_top1),
                "margin": compute_statistics(all_margin),
                "entropy": compute_statistics(all_entropy),
            },
            "sample_level": {
                "top1_prob_mean": compute_statistics([s["top1_prob_mean"] for s in sample_features_list]),
                "margin_mean": compute_statistics([s["margin_mean"] for s in sample_features_list]),
                "entropy_mean": compute_statistics([s["entropy_mean"] for s in sample_features_list]),
            },
        }

    # 任务间对比（E3-weak）
    task_names = list(task_features.keys())
    if len(task_names) >= 2:
        logging.info("  计算任务间差异...")
        pairwise = {}
        for i, t1 in enumerate(task_names):
            for t2 in task_names[i + 1:]:
                feats1 = [s["top1_prob_mean"] for s in task_features[t1]]
                feats2 = [s["top1_prob_mean"] for s in task_features[t2]]

                if feats1 and feats2:
                    pairwise[f"{t1}_vs_{t2}"] = {
                        "top1_prob_ks": ks_test(feats1, feats2),
                        "top1_prob_cohens_d": cohens_d(feats1, feats2),
                    }

        results["cross_task_comparison"] = {
            "pairwise": pairwise,
            "task_ranking": sorted(
                task_names,
                key=lambda t: np.mean([s["top1_prob_mean"] for s in task_features[t]]) if task_features[t] else 0,
                reverse=True
            ),
        }

    return results


# ---------------------------------------------------------------------------
# 8. Phase 3：有标签分析
# ---------------------------------------------------------------------------
def analyze_phase3(
    tasks_data: Dict[str, List[Dict[str, Any]]],
    labels_dir: str,
    output_dir: str,
) -> Dict[str, Any]:
    """
    Phase 3：有标签分析

    内容：
        - E3-strong: 正确 vs 错误样本置信度对比
        - E5: 置信度预测正确性的 AUC
        - E6: 分桶准确率曲线（校准）
    """
    logging.info("=== Phase 3: 有标签分析 ===")

    results = {
        "phase": 3,
        "tasks": {},
        "overall": {},
    }

    all_labels = []
    all_scores = []

    for task_name, samples in tasks_data.items():
        logging.info(f"  处理任务: {task_name}")

        # 尝试加载标签
        label_file = os.path.join(labels_dir, f"{task_name}_is_correct.json")
        labels = load_labels(label_file)

        if not labels:
            logging.warning(f"    跳过：无标签文件")
            continue

        correct_feats = []
        incorrect_feats = []
        task_labels = []
        task_scores = []

        for i, sample in enumerate(samples):
            probs = sample["probs"]
            if probs is None or len(probs) == 0:
                continue

            # 尝试匹配标签
            sample_id = sample.get("sample_idx", i)
            if sample_id not in labels:
                continue

            is_correct = labels[sample_id]

            # 提取特征
            token_feats = extract_token_confidence(probs)
            sample_feats = aggregate_sample_features(token_feats)

            if is_correct:
                correct_feats.append(sample_feats)
            else:
                incorrect_feats.append(sample_feats)

            task_labels.append(is_correct)
            task_scores.append(sample_feats["top1_prob_mean"])

        if not correct_feats or not incorrect_feats:
            logging.warning(f"    跳过：正确/错误样本不足")
            continue

        # 正确 vs 错误对比
        correct_top1 = [s["top1_prob_mean"] for s in correct_feats]
        incorrect_top1 = [s["top1_prob_mean"] for s in incorrect_feats]

        correct_entropy = [s["entropy_mean"] for s in correct_feats]
        incorrect_entropy = [s["entropy_mean"] for s in incorrect_feats]

        correct_margin = [s["margin_mean"] for s in correct_feats]
        incorrect_margin = [s["margin_mean"] for s in incorrect_feats]

        # AUC
        auc_top1 = compute_auc(task_labels, task_scores)

        # 校准
        calibration = compute_calibration(task_labels, task_scores)

        results["tasks"][task_name] = {
            "n_correct": len(correct_feats),
            "n_incorrect": len(incorrect_feats),
            "comparison": {
                "top1_prob": {
                    "correct_stats": compute_statistics(correct_top1),
                    "incorrect_stats": compute_statistics(incorrect_top1),
                    "ks_test": ks_test(correct_top1, incorrect_top1),
                    "cohens_d": cohens_d(correct_top1, incorrect_top1),
                },
                "entropy": {
                    "correct_stats": compute_statistics(correct_entropy),
                    "incorrect_stats": compute_statistics(incorrect_entropy),
                    "ks_test": ks_test(correct_entropy, incorrect_entropy),
                    "cohens_d": cohens_d(correct_entropy, incorrect_entropy),
                },
                "margin": {
                    "correct_stats": compute_statistics(correct_margin),
                    "incorrect_stats": compute_statistics(incorrect_margin),
                    "ks_test": ks_test(correct_margin, incorrect_margin),
                    "cohens_d": cohens_d(correct_margin, incorrect_margin),
                },
            },
            "auc_top1": auc_top1,
            "calibration": calibration,
        }

        all_labels.extend(task_labels)
        all_scores.extend(task_scores)

    # 总体分析
    if all_labels and all_scores:
        results["overall"] = {
            "n_samples": len(all_labels),
            "n_correct": sum(all_labels),
            "n_incorrect": len(all_labels) - sum(all_labels),
            "auc_top1": compute_auc(all_labels, all_scores),
            "calibration": compute_calibration(all_labels, all_scores),
        }

    return results


# ---------------------------------------------------------------------------
# 9. 主入口
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Q5: 路由置信度特征分析")

    # 任务配置：--task name=path 格式
    parser.add_argument(
        "--task",
        action="append",
        dest="tasks",
        required=True,
        help="任务配置，格式：name=path，可多次使用"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="输出目录"
    )

    parser.add_argument(
        "--n_experts",
        type=int,
        default=60,
        help="专家数量"
    )

    parser.add_argument(
        "--layer",
        type=int,
        default=0,
        help="分析哪一层 (默认 0)"
    )

    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 3],
        default=1,
        help="分析阶段: 1=无标签, 3=有标签"
    )

    parser.add_argument(
        "--labels_dir",
        type=str,
        default="",
        help="标签目录 (Phase 3 需要)"
    )

    parser.add_argument(
        "--max_files",
        type=int,
        default=-1,
        help="每任务最多读取的 NPZ 文件数 (-1=全部)"
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="每任务最多样本数 (0=全部)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子"
    )

    args = parser.parse_args()

    # 解析任务配置
    tasks_config = {}
    for task_spec in args.tasks:
        if "=" in task_spec:
            name, path = task_spec.split("=", 1)
            tasks_config[name.strip()] = path.strip()
        else:
            logging.error(f"无效的任务配置: {task_spec}")
            return

    logging.info(f"配置: phase={args.phase}, layer={args.layer}, n_experts={args.n_experts}")
    logging.info(f"任务: {list(tasks_config.keys())}")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    plots_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # 加载数据
    logging.info("=== 加载数据 ===")
    tasks_data = {}
    for task_name, data_dir in tasks_config.items():
        samples = load_task_data(data_dir, args.layer, args.max_files, args.max_samples)
        if samples:
            tasks_data[task_name] = samples

    if not tasks_data:
        logging.error("无有效数据，退出")
        return

    # 执行分析
    if args.phase == 1:
        results = analyze_phase1(tasks_data, args.output_dir)
    else:  # phase == 3
        if not args.labels_dir:
            logging.error("Phase 3 需要 --labels_dir 参数")
            return
        results = analyze_phase3(tasks_data, args.labels_dir, args.output_dir)

    # 保存结果
    output_file = os.path.join(args.output_dir, "q5_confidence_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logging.info(f"结果已保存到: {output_file}")

    # 打印摘要
    logging.info("=== 分析摘要 ===")
    if args.phase == 1:
        for task_name, task_result in results.get("tasks", {}).items():
            top1_stats = task_result.get("sample_level", {}).get("top1_prob_mean", {})
            logging.info(f"  {task_name}: Top-1 均值={top1_stats.get('mean', 0):.4f}, "
                        f"熵均值={task_result.get('sample_level', {}).get('entropy_mean', {}).get('mean', 0):.4f}")
    else:
        for task_name, task_result in results.get("tasks", {}).items():
            auc = task_result.get("auc_top1", 0)
            n_correct = task_result.get("n_correct", 0)
            n_incorrect = task_result.get("n_incorrect", 0)
            logging.info(f"  {task_name}: AUC={auc:.4f}, 正确={n_correct}, 错误={n_incorrect}")

        overall = results.get("overall", {})
        if overall:
            logging.info(f"  总体: AUC={overall.get('auc_top1', 0):.4f}, "
                        f"ECE={overall.get('calibration', {}).get('ece', 0):.4f}")


if __name__ == "__main__":
    main()
