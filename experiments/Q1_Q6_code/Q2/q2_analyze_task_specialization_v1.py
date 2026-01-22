"""  # 模块级说明：三引号字符串不会影响执行
Q2 任务偏好（专家专门化）分析脚本 —— 超详细注释版

目标问题：
不同任务是否偏好不同专家？偏好强度有多大？

核心思路：
1) 为每个任务统计“专家使用分布”P(e|task)
   - Hard 口径：每层只看 Top1 专家，等价于“一层一票”
   - Soft 口径：把 Top-(K+2) 概率质量加到对应专家上
2) 用 JSD / TV 衡量任务分布差异
3) 用 TSS = JSD(P_task, P_global) 衡量“任务偏离全局”的程度
4) 用置乱基线验证“任务差异是否真实存在”
"""  # 文档字符串结束

import os  # 导入 os，用于路径拼接与目录创建
import glob  # 导入 glob，用于匹配 *.npz 文件
import json  # 导入 json，用于读写结果文件
import math  # 导入 math，用于 log 等数学函数
import argparse  # 导入 argparse，用于命令行参数解析
import logging  # 导入 logging，用于输出日志
from typing import Dict, List, Tuple  # 类型标注，便于阅读

import numpy as np  # 导入 numpy，并命名为 np
import matplotlib.pyplot as plt  # 导入 matplotlib，用于绘图

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")  # 设置日志格式


def _list_npz(data_dir: str, max_files: int) -> List[str]:  # 定义函数：列出 npz 文件
    """列出目录下的 npz 文件列表。"""  # 函数说明
    npz_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))  # 查找并排序
    if max_files > 0:  # 若设置最大数量
        npz_files = npz_files[:max_files]  # 只取前 max_files 个
    return npz_files  # 返回文件列表


def _layer_names_from_file(npz_path: str) -> List[str]:  # 定义函数：提取层名
    """从 npz 的 key 中提取层名。"""  # 函数说明
    with np.load(npz_path, allow_pickle=True) as data:  # 读取 npz
        layer_names = []  # 初始化层名列表
        for k in data.keys():  # 遍历 key
            if k.startswith("layers/") and k.endswith("_out_probs"):  # 只找 probs
                layer_name = k.split("/")[1].replace("_out_probs", "")  # 提取层名
                layer_names.append(layer_name)  # 追加到列表
        return sorted(layer_names)  # 排序并返回


def _ensure_2d_array(arr: np.ndarray, pad_value: float, dtype: np.dtype) -> np.ndarray:  # 定义函数：统一成 2D
    """将 ragged 数组统一成 2D，必要时做 padding。"""  # 函数说明
    if not isinstance(arr, np.ndarray):  # 如果不是 ndarray
        return np.asarray(arr, dtype=dtype)  # 直接转成 ndarray
    if arr.dtype == object or (arr.ndim == 1 and arr.size > 0 and isinstance(arr[0], (list, np.ndarray))):
        elems = [np.asarray(x) for x in arr]  # 把每个元素转成数组
        if elems and all(e.ndim == 2 for e in elems):  # 情况：每个元素是 2D
            max_cols = max(e.shape[1] for e in elems)  # 找最大列数
            padded = []  # 准备存放补齐结果
            for e in elems:  # 遍历每个 2D block
                e = e.astype(dtype)  # 转 dtype
                if e.shape[1] < max_cols:  # 列数不足则补齐
                    pad = np.full((e.shape[0], max_cols - e.shape[1]), pad_value, dtype=dtype)  # padding
                    e = np.concatenate([e, pad], axis=1)  # 拼接
                padded.append(e)  # 收集
            return np.concatenate(padded, axis=0)  # 合并成一个大矩阵
        max_len = max(e.shape[0] for e in elems) if elems else 0  # 计算最大长度
        out = np.full((len(elems), max_len), pad_value, dtype=dtype)  # 初始化输出
        for i, e in enumerate(elems):  # 遍历每行
            e = e.astype(dtype)  # 转 dtype
            out[i, : e.shape[0]] = e  # 拷贝内容
        return out  # 返回 2D
    if arr.ndim == 1:  # 若为 1D
        return arr.reshape(-1, 1).astype(dtype)  # 转成列向量
    return arr.astype(dtype)  # 其他情况直接转 dtype


def _token_count_from_probs(probs: np.ndarray) -> int:  # 定义函数：推断 token 数
    """从 probs 字段推断 token 数量。"""  # 函数说明
    if isinstance(probs, np.ndarray) and probs.dtype == object:  # 如果是 object
        elems = [np.asarray(x) for x in probs]  # 转成数组
        if elems and all(e.ndim == 2 for e in elems):  # 若每个元素是 2D
            return int(sum(e.shape[0] for e in elems))  # 总 token 数
        return int(len(elems))  # 否则按元素个数
    return int(probs.shape[0])  # 普通情况直接取第一维


def load_task_data(data_dir: str, max_files: int) -> Tuple[List[str], Dict[str, np.ndarray], Dict[str, np.ndarray]]:  # 定义函数：读取任务数据
    """  # 函数说明
    读取任务数据：
    - tokens_str: token 文本
    - probs_by_layer: 每层的 top-(K+2) 概率
    - inds_by_layer: 每层的 top-(K+2) 专家索引
    """  # 说明结束
    npz_files = _list_npz(data_dir, max_files)  # 列出 npz
    if not npz_files:  # 如果为空
        raise FileNotFoundError(f"No NPZ files found in {data_dir}")  # 抛错

    layer_names = _layer_names_from_file(npz_files[0])  # 从第一个文件取层名
    probs_by_layer: Dict[str, List[np.ndarray]] = {ln: [] for ln in layer_names}  # 初始化 probs 容器
    inds_by_layer: Dict[str, List[np.ndarray]] = {ln: [] for ln in layer_names}  # 初始化 inds 容器
    tokens_str: List[str] = []  # 初始化 token 文本列表

    logging.info("Found %d NPZ files.", len(npz_files))  # 输出日志
    for f in npz_files:  # 遍历文件
        with np.load(f, allow_pickle=True) as data:  # 读取 npz
            if "tokens_str" in data:  # 如果有 token 文本
                tokens_str.extend(list(data["tokens_str"]))  # 追加文本
            else:  # 如果没有 token 文本
                first_layer = layer_names[0]  # 取第一层
                token_count = _token_count_from_probs(data[f"layers/{first_layer}_out_probs"])  # 推断数量
                tokens_str.extend([""] * token_count)  # 用空串占位
            for ln in layer_names:  # 遍历每层
                probs = _ensure_2d_array(data[f"layers/{ln}_out_probs"], 0.0, np.float64)  # 读取并整理 probs
                inds = _ensure_2d_array(data[f"layers/{ln}_out_indices"], -1, np.int64)  # 读取并整理 inds
                probs_by_layer[ln].append(probs)  # 追加到列表
                inds_by_layer[ln].append(inds)  # 追加到列表

    probs_concat = {ln: np.concatenate(probs_by_layer[ln], axis=0) for ln in layer_names}  # 拼接 probs
    inds_concat = {ln: np.concatenate(inds_by_layer[ln], axis=0) for ln in layer_names}  # 拼接 inds
    return tokens_str, probs_concat, inds_concat  # 返回结果


def sample_indices(n_tokens: int, sample_size: int, seed: int) -> np.ndarray:  # 定义函数：抽样
    """随机抽样 token 索引。"""  # 函数说明
    if sample_size <= 0 or sample_size >= n_tokens:  # 若 sample_size 不合理
        return np.arange(n_tokens)  # 返回全量索引
    rng = np.random.default_rng(seed)  # 随机数生成器
    return rng.choice(n_tokens, size=sample_size, replace=False)  # 无放回抽样


def _normalize_rows(x: np.ndarray) -> np.ndarray:  # 定义函数：按行归一化
    """对每一行做归一化，使其和为 1。"""  # 函数说明
    row_sums = x.sum(axis=1, keepdims=True)  # 行和
    out = np.zeros_like(x, dtype=np.float64)  # 初始化输出
    np.divide(x, row_sums, out=out, where=row_sums > 0)  # 安全除法
    return out  # 返回


def _entropy(p: np.ndarray) -> np.ndarray:  # 定义函数：熵
    """计算每行熵 H(p) = -sum p log p。"""  # 函数说明
    eps = 1e-12  # 防止 log(0)
    p_safe = np.clip(p, eps, 1.0)  # 截断范围
    return -np.sum(p_safe * np.log(p_safe), axis=1)  # 返回熵


def build_token_vectors(  # 定义函数：构造 token 向量
    layer_names: List[str],  # 层名列表
    probs_by_layer: Dict[str, np.ndarray],  # 每层 probs
    inds_by_layer: Dict[str, np.ndarray],  # 每层 inds
    idxs: np.ndarray,  # 选中的 token 索引
    n_experts: int,  # 专家数
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:  # 返回 hard/soft 向量与统计
    """  # 函数说明
    构造每个 token 的 Hard/Soft 专家向量 + 置信度统计。
    - Hard: 每层 Top1 专家记 1 次
    - Soft: Top-(K+2) 概率质量加到对应专家
    """  # 说明结束
    n_tokens = idxs.shape[0]  # token 数
    n_layers = len(layer_names)  # 层数
    hard_vecs = np.zeros((n_tokens, n_experts), dtype=np.float64)  # Hard 初始化
    soft_vecs = np.zeros((n_tokens, n_experts), dtype=np.float64)  # Soft 初始化

    top1_prob = np.zeros((n_tokens, n_layers), dtype=np.float64)  # Top1 概率
    margin = np.zeros((n_tokens, n_layers), dtype=np.float64)  # Margin
    entropy = np.zeros((n_tokens, n_layers), dtype=np.float64)  # 熵

    for li, ln in enumerate(layer_names):  # 遍历层
        probs = probs_by_layer[ln][idxs]  # 取该层 probs
        inds = inds_by_layer[ln][idxs]  # 取该层 inds
        probs_norm = _normalize_rows(probs.astype(np.float64))  # 归一化

        top1_idx = np.argmax(probs, axis=1)  # top1 在 top-(K+2) 内的位置
        top1 = inds[np.arange(n_tokens), top1_idx]  # top1 专家编号
        top1_p = probs[np.arange(n_tokens), top1_idx]  # top1 概率
        top1_prob[:, li] = top1_p  # 记录 top1 概率

        if probs.shape[1] > 1:  # 如果有 top2
            sorted_probs = np.sort(probs, axis=1)  # 排序
            top2_p = sorted_probs[:, -2]  # 取 top2 概率
            margin[:, li] = top1_p - top2_p  # 计算 margin

        entropy[:, li] = _entropy(probs_norm)  # 计算熵

        valid_top1 = top1 >= 0  # top1 合法标记
        if np.any(valid_top1):  # 若存在合法
            np.add.at(hard_vecs, (np.arange(n_tokens)[valid_top1], top1[valid_top1]), 1.0)  # Hard 计票
        for col in range(inds.shape[1]):  # 遍历 top-(K+2)
            valid = inds[:, col] >= 0  # 合法标记
            if not np.any(valid):  # 若没有合法
                continue  # 跳过
            np.add.at(soft_vecs, (np.arange(n_tokens)[valid], inds[valid, col]), probs_norm[valid, col])  # Soft 累加

    hard_vecs /= float(n_layers)  # 对层取平均
    soft_vecs /= float(n_layers)  # 对层取平均

    stats = {  # 置信度统计
        "top1_prob": top1_prob.mean(axis=1),  # token 级均值
        "margin": margin.mean(axis=1),  # token 级均值
        "entropy": entropy.mean(axis=1),  # token 级均值
    }  # stats 结束
    return hard_vecs, soft_vecs, stats  # 返回


def normalize_counts(counts: np.ndarray, alpha: float) -> np.ndarray:  # 定义函数：平滑归一化
    """计数 + 平滑，转成概率分布。"""  # 函数说明
    smoothed = counts + alpha  # 加平滑
    total = smoothed.sum()  # 求和
    if total <= 0:  # 若总和为 0
        return np.full_like(smoothed, 1.0 / smoothed.size, dtype=np.float64)  # 返回均匀分布
    return smoothed / total  # 返回归一化


def jsd(p: np.ndarray, q: np.ndarray) -> float:  # 定义函数：JSD
    """Jensen-Shannon divergence，范围 [0,1]。"""  # 说明
    eps = 1e-12  # 防止 0
    p = np.clip(p, eps, 1.0)  # 截断
    q = np.clip(q, eps, 1.0)  # 截断
    p = p / p.sum()  # 归一化
    q = q / q.sum()  # 归一化
    m = 0.5 * (p + q)  # 中间分布
    kl_pm = np.sum(p * np.log2(p / m))  # KL(p||m)
    kl_qm = np.sum(q * np.log2(q / m))  # KL(q||m)
    return 0.5 * (kl_pm + kl_qm)  # JSD


def tv(p: np.ndarray, q: np.ndarray) -> float:  # 定义函数：TV 距离
    """Total variation distance，范围 [0,1]。"""  # 说明
    return 0.5 * float(np.abs(p - q).sum())  # 公式


def coverage_curve(p: np.ndarray, k_values: List[int]) -> Dict[int, float]:  # 定义函数：Top-k 覆盖率
    """Top-k 覆盖率曲线。"""  # 说明
    order = np.argsort(p)[::-1]  # 按概率降序排序
    cov = {}  # 初始化字典
    for k in k_values:  # 遍历 k
        k = min(k, p.size)  # 防止超界
        cov[k] = float(p[order[:k]].sum())  # 累加前 k 的概率质量
    return cov  # 返回


def conc_score(p: np.ndarray) -> float:  # 定义函数：集中度
    """集中度分数：1 - H(p)/log(E)。"""  # 说明
    eps = 1e-12  # 防止 0
    p = np.clip(p, eps, 1.0)  # 截断
    p = p / p.sum()  # 归一化
    ent = -np.sum(p * np.log(p))  # 熵
    return 1.0 - ent / math.log(len(p))  # 集中度


def token_type(tok: str) -> str:  # 定义函数：token 类型分桶
    """粗粒度 token 类型，用于匹配置乱基线。"""  # 说明
    s = str(tok)  # 转成字符串
    if s.strip() == "":  # 空白
        return "whitespace"  # 返回 whitespace
    if any("\u4e00" <= ch <= "\u9fff" for ch in s):  # 中文字符判断
        return "chinese_char"  # 返回中文
    if s.isalpha():  # 纯字母
        return "alpha"  # 返回 alpha
    if s.isdigit():  # 纯数字
        return "digit"  # 返回 digit
    if all(not ch.isalnum() for ch in s):  # 全是符号
        return "punct"  # 返回 punct
    return "other"  # 其他


def length_bin(s: str) -> str:  # 定义函数：长度分桶
    """长度分桶，用于匹配置乱。"""  # 说明
    n = len(str(s))  # 长度
    if n <= 1:  # 1 个字符
        return "len1"  # 返回 len1
    if n <= 3:  # 2-3 个字符
        return "len2_3"  # 返回 len2_3
    if n <= 6:  # 4-6 个字符
        return "len4_6"  # 返回 len4_6
    return "len7p"  # 返回 len7p


def quantile_bins(values: np.ndarray, n_bins: int) -> np.ndarray:  # 定义函数：分位数分桶
    """按分位数分桶，减少样本偏差。"""  # 说明
    qs = np.linspace(0, 1, n_bins + 1)  # 分位点
    edges = np.quantile(values, qs)  # 分位值
    bins = np.digitize(values, edges[1:-1], right=False)  # 分桶编号
    return bins  # 返回


def shuffle_labels(task_ids: np.ndarray, rng: np.random.Generator) -> np.ndarray:  # 定义函数：全局置乱
    """全局置乱任务标签，构造无任务信号基线。"""  # 说明
    perm = rng.permutation(task_ids.shape[0])  # 生成随机排列
    return task_ids[perm]  # 返回置乱后的标签


def matched_shuffle(task_ids: np.ndarray, bucket_ids: np.ndarray, rng: np.random.Generator) -> np.ndarray:  # 定义函数：匹配置乱
    """在同桶内置乱，控制 token 类型/长度/熵影响。"""  # 说明
    shuffled = task_ids.copy()  # 复制标签
    for b in np.unique(bucket_ids):  # 遍历桶
        idx = np.where(bucket_ids == b)[0]  # 取桶内索引
        if idx.size <= 1:  # 样本太少
            continue  # 跳过
        shuffled[idx] = task_ids[idx][rng.permutation(idx.size)]  # 桶内置乱
    return shuffled  # 返回


def compute_task_distributions(  # 定义函数：计算 P(e|task)
    hard_vecs: np.ndarray,  # Hard 向量
    soft_vecs: np.ndarray,  # Soft 向量
    task_ids: np.ndarray,  # 任务标签
    n_tasks: int,  # 任务数
    alpha: float,  # 平滑系数
) -> Tuple[List[np.ndarray], List[np.ndarray]]:  # 返回硬/软分布列表
    """计算每个任务的专家分布 P(e|task)。"""  # 说明
    p_hard = []  # Hard 分布列表
    p_soft = []  # Soft 分布列表
    for t in range(n_tasks):  # 遍历任务
        idx = np.where(task_ids == t)[0]  # 取该任务索引
        hard_counts = hard_vecs[idx].sum(axis=0)  # 求和得到计数
        soft_counts = soft_vecs[idx].sum(axis=0)  # 求和得到计数
        p_hard.append(normalize_counts(hard_counts, alpha))  # 归一化
        p_soft.append(normalize_counts(soft_counts, alpha))  # 归一化
    return p_hard, p_soft  # 返回


def main():  # 定义主函数
    """主流程：读取数据 -> 构建向量 -> 统计指标 -> 输出结果。"""  # 说明
    parser = argparse.ArgumentParser(description="Task specialization analysis (v1)")  # 参数解析器
    parser.add_argument("--task", action="append", default=[], help="task_name=path (repeatable)")  # 多任务参数
    parser.add_argument("--output_dir", required=True)  # 输出目录
    parser.add_argument("--n_experts", type=int, default=60)  # 专家数
    parser.add_argument("--max_files", type=int, default=-1)  # 最多文件数
    parser.add_argument("--sample_size", type=int, default=20000)  # 抽样 token 数
    parser.add_argument("--seed", type=int, default=42)  # 随机种子
    parser.add_argument("--alpha", type=float, default=1.0)  # 平滑系数
    parser.add_argument("--topn", type=int, default=10)  # Top-N 专家输出
    parser.add_argument("--min_support", type=float, default=0.0)  # 最小支持度
    parser.add_argument("--k_values", type=str, default="1,2,5,10,20")  # Top-k 覆盖率 k 值
    parser.add_argument("--shuffle_runs", type=int, default=20)  # 置乱次数
    parser.add_argument("--matched_shuffle", action="store_true")  # 是否匹配置乱
    parser.add_argument("--bootstrap_runs", type=int, default=0)  # bootstrap 次数
    args = parser.parse_args()  # 解析命令行

    if not args.task:  # 如果没有任务参数
        raise SystemExit("Please provide --task name=path (repeatable).")  # 退出并提示

    os.makedirs(args.output_dir, exist_ok=True)  # 创建输出目录
    plot_dir = os.path.join(args.output_dir, "plots")  # 绘图目录
    os.makedirs(plot_dir, exist_ok=True)  # 创建绘图目录

    tasks = []  # 保存任务列表
    for item in args.task:  # 遍历任务参数
        name, path = item.split("=", 1)  # 拆分 name=path
        tasks.append((name, path))  # 追加

    rng = np.random.default_rng(args.seed)  # 随机数生成器
    k_values = [int(x) for x in args.k_values.split(",") if x.strip()]  # 解析 k 值列表

    task_token_data = []  # 任务数据缓存
    for name, path in tasks:  # 遍历任务
        logging.info("Loading task %s from %s", name, path)  # 输出日志
        tokens_str, probs_by_layer, inds_by_layer = load_task_data(path, args.max_files)  # 读取数据
        layer_names = sorted(probs_by_layer.keys())  # 层名
        n_tokens = len(tokens_str)  # token 数
        idxs = sample_indices(n_tokens, args.sample_size, args.seed)  # 抽样索引

        hard_vecs, soft_vecs, stats = build_token_vectors(  # 构造向量
            layer_names, probs_by_layer, inds_by_layer, idxs, args.n_experts
        )
        toks = [tokens_str[i] for i in idxs]  # 抽样 token 文本
        task_token_data.append({  # 缓存
            "name": name,
            "hard_vecs": hard_vecs,
            "soft_vecs": soft_vecs,
            "tokens": toks,
            "stats": stats,
        })

    # Equalize task sizes to avoid data-volume bias.
    n_tasks = len(task_token_data)  # 任务数
    min_size = min(t["hard_vecs"].shape[0] for t in task_token_data)  # 最小样本数
    logging.info("Equal sampling size per task: %d", min_size)  # 输出日志

    all_hard = []  # 汇总 Hard
    all_soft = []  # 汇总 Soft
    all_tokens = []  # 汇总 tokens
    all_entropy = []  # 汇总 entropy
    all_task_ids = []  # 汇总 task_id
    for ti, t in enumerate(task_token_data):  # 遍历任务
        idx = rng.choice(t["hard_vecs"].shape[0], size=min_size, replace=False)  # 等量抽样
        all_hard.append(t["hard_vecs"][idx])  # 追加
        all_soft.append(t["soft_vecs"][idx])  # 追加
        all_tokens.extend([t["tokens"][i] for i in idx])  # 追加 token 文本
        all_entropy.append(t["stats"]["entropy"][idx])  # 追加熵
        all_task_ids.append(np.full(min_size, ti, dtype=np.int64))  # 追加任务标签

    hard_pool = np.concatenate(all_hard, axis=0)  # 合并 Hard
    soft_pool = np.concatenate(all_soft, axis=0)  # 合并 Soft
    task_ids = np.concatenate(all_task_ids, axis=0)  # 合并标签
    entropy_vals = np.concatenate(all_entropy, axis=0)  # 合并熵

    # Task-conditional distributions and pooled global baseline.
    p_hard, p_soft = compute_task_distributions(hard_pool, soft_pool, task_ids, n_tasks, args.alpha)  # 任务分布
    p_global_hard = normalize_counts(hard_pool.sum(axis=0), args.alpha)  # 全局 Hard
    p_global_soft = normalize_counts(soft_pool.sum(axis=0), args.alpha)  # 全局 Soft

    # Shuffle baseline (global or matched) to remove task signal.
    shuffle_p_hard = []  # 置乱 Hard
    shuffle_p_soft = []  # 置乱 Soft
    bucket_ids = None  # 分桶 id
    if args.matched_shuffle:  # 若启用匹配置乱
        tok_types = np.array([token_type(t) for t in all_tokens], dtype=object)  # 类型桶
        len_bins = np.array([length_bin(t) for t in all_tokens], dtype=object)  # 长度桶
        ent_bins = quantile_bins(entropy_vals, 4)  # 熵桶
        bucket_ids = np.array([f"{a}|{b}|{c}" for a, b, c in zip(tok_types, len_bins, ent_bins)])  # 合并桶

    for _ in range(args.shuffle_runs):  # 多次置乱
        if args.matched_shuffle:  # 匹配置乱
            perm_labels = matched_shuffle(task_ids, bucket_ids, rng)  # 桶内置乱
        else:  # 全局置乱
            perm_labels = shuffle_labels(task_ids, rng)  # 直接置乱
        ph, ps = compute_task_distributions(hard_pool, soft_pool, perm_labels, n_tasks, args.alpha)  # 计算分布
        shuffle_p_hard.append(ph)  # 追加
        shuffle_p_soft.append(ps)  # 追加

    shuffle_p_hard_mean = np.mean(np.stack(shuffle_p_hard, axis=0), axis=0)  # 置乱均值
    shuffle_p_soft_mean = np.mean(np.stack(shuffle_p_soft, axis=0), axis=0)  # 置乱均值

    # Task-to-task distance matrices (hard/soft).
    dist_hard = np.zeros((n_tasks, n_tasks))  # Hard JSD
    dist_soft = np.zeros((n_tasks, n_tasks))  # Soft JSD
    dist_tv_hard = np.zeros((n_tasks, n_tasks))  # Hard TV
    dist_tv_soft = np.zeros((n_tasks, n_tasks))  # Soft TV
    for i in range(n_tasks):  # 双层循环
        for j in range(n_tasks):
            dist_hard[i, j] = jsd(p_hard[i], p_hard[j])  # JSD hard
            dist_soft[i, j] = jsd(p_soft[i], p_soft[j])  # JSD soft
            dist_tv_hard[i, j] = tv(p_hard[i], p_hard[j])  # TV hard
            dist_tv_soft[i, j] = tv(p_soft[i], p_soft[j])  # TV soft

    # TSS (task vs global), coverage curves, and concentration scores.
    tss_hard = [jsd(p_hard[i], p_global_hard) for i in range(n_tasks)]  # TSS hard
    tss_soft = [jsd(p_soft[i], p_global_soft) for i in range(n_tasks)]  # TSS soft
    coverage_hard = [coverage_curve(p_hard[i], k_values) for i in range(n_tasks)]  # 覆盖率 hard
    coverage_soft = [coverage_curve(p_soft[i], k_values) for i in range(n_tasks)]  # 覆盖率 soft
    conc_hard = [conc_score(p_hard[i]) for i in range(n_tasks)]  # 集中度 hard
    conc_soft = [conc_score(p_soft[i]) for i in range(n_tasks)]  # 集中度 soft

    # Preference scores: score1 (expert -> task) and score2 (task -> expert).
    def score1_score2(p_tasks: List[np.ndarray], p_global: np.ndarray) -> np.ndarray:  # 定义内部函数
        p_tasks_arr = np.stack(p_tasks, axis=0)  # [task, expert]
        p_task_given_e = p_tasks_arr / np.maximum(p_tasks_arr.sum(axis=0, keepdims=True), 1e-12)  # P(task|expert)
        p_task = 1.0 / n_tasks  # P(task) 均匀先验
        score1 = np.log(p_task_given_e / p_task)  # score1 = log 比例
        score2 = p_tasks_arr / np.maximum(p_global, 1e-12)  # score2 = P(e|task)/P(e)
        return score1, score2  # 返回

    score1_hard, score2_hard = score1_score2(p_hard, p_global_hard)  # score1/2 hard
    score1_soft, score2_soft = score1_score2(p_soft, p_global_soft)  # score1/2 soft

    score1_hard_b, score2_hard_b = score1_score2(list(shuffle_p_hard_mean), np.mean(shuffle_p_hard_mean, axis=0))  # 基线 hard
    score1_soft_b, score2_soft_b = score1_score2(list(shuffle_p_soft_mean), np.mean(shuffle_p_soft_mean, axis=0))  # 基线 soft

    topn = args.topn  # topn 参数
    min_support = args.min_support  # 支持度阈值
    top_lists = {}  # 结果字典
    for ti, (name, _) in enumerate(tasks):  # 遍历任务
        delta = score1_soft[ti] - score1_soft_b[ti]  # delta = 真实 - 置乱
        order = np.argsort(delta)[::-1]  # 按 delta 降序
        top_entries = []  # 保存 top 专家
        for e in order:  # 遍历专家
            if p_soft[ti][e] < min_support:  # 支持度过滤
                continue  # 跳过
            top_entries.append({"expert": int(e), "delta_score": float(delta[e])})  # 记录
            if len(top_entries) >= topn:  # 数量够了
                break  # 停止
        top_lists[name] = top_entries  # 写入结果

    bootstrap = {}  # 初始化 bootstrap 结果
    if args.bootstrap_runs > 0:  # 如果需要 bootstrap
        logging.info("Running bootstrap: %d", args.bootstrap_runs)  # 输出日志
        tss_hard_boot = []  # 保存 tss hard
        tss_soft_boot = []  # 保存 tss soft
        conc_hard_boot = []  # 保存 conc hard
        conc_soft_boot = []  # 保存 conc soft
        coverage_hard_boot = []  # 保存 coverage hard
        coverage_soft_boot = []  # 保存 coverage soft
        for _ in range(args.bootstrap_runs):  # 多次重采样
            boot_hard = []  # boot hard
            boot_soft = []  # boot soft
            boot_labels = []  # boot labels
            for ti in range(n_tasks):  # 按任务采样
                idx = np.where(task_ids == ti)[0]  # 任务索引
                bidx = rng.choice(idx, size=idx.size, replace=True)  # 有放回采样
                boot_hard.append(hard_pool[bidx])  # 追加
                boot_soft.append(soft_pool[bidx])  # 追加
                boot_labels.append(np.full(idx.size, ti, dtype=np.int64))  # 追加标签
            boot_hard_pool = np.concatenate(boot_hard, axis=0)  # 合并
            boot_soft_pool = np.concatenate(boot_soft, axis=0)  # 合并
            boot_task_ids = np.concatenate(boot_labels, axis=0)  # 合并
            p_hard_b, p_soft_b = compute_task_distributions(  # 计算 boot 分布
                boot_hard_pool, boot_soft_pool, boot_task_ids, n_tasks, args.alpha
            )
            p_global_hard_b = normalize_counts(boot_hard_pool.sum(axis=0), args.alpha)  # boot 全局 hard
            p_global_soft_b = normalize_counts(boot_soft_pool.sum(axis=0), args.alpha)  # boot 全局 soft
            tss_hard_boot.append([jsd(p_hard_b[i], p_global_hard_b) for i in range(n_tasks)])  # 记录 tss hard
            tss_soft_boot.append([jsd(p_soft_b[i], p_global_soft_b) for i in range(n_tasks)])  # 记录 tss soft
            conc_hard_boot.append([conc_score(p_hard_b[i]) for i in range(n_tasks)])  # 记录 conc hard
            conc_soft_boot.append([conc_score(p_soft_b[i]) for i in range(n_tasks)])  # 记录 conc soft
            coverage_hard_boot.append([coverage_curve(p_hard_b[i], k_values) for i in range(n_tasks)])  # 记录覆盖率
            coverage_soft_boot.append([coverage_curve(p_soft_b[i], k_values) for i in range(n_tasks)])  # 记录覆盖率

        def _ci(arr: np.ndarray) -> Dict[str, List[float]]:  # 定义 CI 内部函数
            low = np.percentile(arr, 2.5, axis=0).tolist()  # 低分位
            high = np.percentile(arr, 97.5, axis=0).tolist()  # 高分位
            return {"low": low, "high": high}  # 返回

        bootstrap["tss_hard_ci"] = _ci(np.array(tss_hard_boot))  # 保存 tss hard CI
        bootstrap["tss_soft_ci"] = _ci(np.array(tss_soft_boot))  # 保存 tss soft CI
        bootstrap["conc_hard_ci"] = _ci(np.array(conc_hard_boot))  # 保存 conc hard CI
        bootstrap["conc_soft_ci"] = _ci(np.array(conc_soft_boot))  # 保存 conc soft CI

        cov_hard_ci = []  # 保存 coverage hard CI
        cov_soft_ci = []  # 保存 coverage soft CI
        for ti in range(n_tasks):  # 按任务计算
            per_k = {k: [] for k in k_values}  # 初始化
            for run in coverage_hard_boot:  # 遍历 boot
                for k in k_values:
                    per_k[k].append(run[ti][k])  # 收集
            cov_hard_ci.append({
                str(k): {
                    "low": float(np.percentile(per_k[k], 2.5)),
                    "high": float(np.percentile(per_k[k], 97.5)),
                } for k in k_values
            })  # 追加
            per_k = {k: [] for k in k_values}  # 重置
            for run in coverage_soft_boot:
                for k in k_values:
                    per_k[k].append(run[ti][k])
            cov_soft_ci.append({
                str(k): {
                    "low": float(np.percentile(per_k[k], 2.5)),
                    "high": float(np.percentile(per_k[k], 97.5)),
                } for k in k_values
            })  # 追加
        bootstrap["coverage_hard_ci"] = cov_hard_ci  # 保存 hard CI
        bootstrap["coverage_soft_ci"] = cov_soft_ci  # 保存 soft CI

    summary = {  # 结果汇总 JSON
        "tasks": [name for name, _ in tasks],  # 任务列表
        "p_hard": [p.tolist() for p in p_hard],  # hard 分布
        "p_soft": [p.tolist() for p in p_soft],  # soft 分布
        "p_global_hard": p_global_hard.tolist(),  # 全局 hard
        "p_global_soft": p_global_soft.tolist(),  # 全局 soft
        "p_hard_shuffle_mean": shuffle_p_hard_mean.tolist(),  # 置乱 hard
        "p_soft_shuffle_mean": shuffle_p_soft_mean.tolist(),  # 置乱 soft
        "tss_hard": tss_hard,  # tss hard
        "tss_soft": tss_soft,  # tss soft
        "conc_hard": conc_hard,  # conc hard
        "conc_soft": conc_soft,  # conc soft
        "coverage_hard": coverage_hard,  # 覆盖率 hard
        "coverage_soft": coverage_soft,  # 覆盖率 soft
        "top_experts_soft_score1_delta": top_lists,  # top 专家
        "jsd_hard": dist_hard.tolist(),  # jsd hard
        "jsd_soft": dist_soft.tolist(),  # jsd soft
        "tv_hard": dist_tv_hard.tolist(),  # tv hard
        "tv_soft": dist_tv_soft.tolist(),  # tv soft
        "shuffle_runs": args.shuffle_runs,  # 置乱次数
        "matched_shuffle": args.matched_shuffle,  # 是否匹配置乱
        "bootstrap_runs": args.bootstrap_runs,  # bootstrap 次数
        "bootstrap": bootstrap,  # bootstrap 结果
    }  # summary 结束
    with open(os.path.join(args.output_dir, "task_specialization_summary.json"), "w", encoding="utf-8") as f:  # 写文件
        json.dump(summary, f, ensure_ascii=False, indent=2)  # 写 JSON

    # plots: coverage
    plt.figure(figsize=(8, 5))  # 创建画布
    for ti, (name, _) in enumerate(tasks):  # 遍历任务
        ks = list(coverage_hard[ti].keys())  # k 值
        vals = [coverage_hard[ti][k] for k in ks]  # 覆盖率
        plt.plot(ks, vals, label=name)  # 绘制曲线
    plt.xlabel("Top-k")  # x 轴
    plt.ylabel("coverage (hard)")  # y 轴
    plt.title("Top-k coverage (hard)")  # 标题
    plt.legend()  # 图例
    plt.tight_layout()  # 紧凑布局
    plt.savefig(os.path.join(plot_dir, "coverage_hard.png"))  # 保存图片
    plt.close()  # 关闭图像

    plt.figure(figsize=(8, 5))  # 创建画布
    for ti, (name, _) in enumerate(tasks):  # 遍历任务
        ks = list(coverage_soft[ti].keys())  # k 值
        vals = [coverage_soft[ti][k] for k in ks]  # 覆盖率
        plt.plot(ks, vals, label=name)  # 绘制曲线
    plt.xlabel("Top-k")  # x 轴
    plt.ylabel("coverage (soft)")  # y 轴
    plt.title("Top-k coverage (soft)")  # 标题
    plt.legend()  # 图例
    plt.tight_layout()  # 紧凑布局
    plt.savefig(os.path.join(plot_dir, "coverage_soft.png"))  # 保存图片
    plt.close()  # 关闭图像

    # plots: TSS
    plt.figure(figsize=(8, 4))  # 创建画布
    x = np.arange(n_tasks)  # x 轴
    plt.bar(x - 0.2, tss_hard, width=0.4, label="hard")  # hard 条形图
    plt.bar(x + 0.2, tss_soft, width=0.4, label="soft")  # soft 条形图
    plt.xticks(x, [n for n, _ in tasks], rotation=30, ha="right")  # x 轴标签
    plt.ylabel("TSS (JSD)")  # y 轴
    plt.title("Task specialization score")  # 标题
    plt.legend()  # 图例
    plt.tight_layout()  # 紧凑布局
    plt.savefig(os.path.join(plot_dir, "tss.png"))  # 保存图片
    plt.close()  # 关闭图像

    logging.info("Done. Outputs saved to %s", args.output_dir)  # 输出日志


if __name__ == "__main__":  # 判断是否直接运行脚本
    main()  # 调用主函数
