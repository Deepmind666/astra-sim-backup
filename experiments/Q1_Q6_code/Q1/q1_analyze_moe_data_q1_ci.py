"""  # 使用三引号定义“模块级文档字符串”，用于说明脚本目的
Q1 相邻层相关性指标的置信区间估计（Bootstrap 版本）

这份脚本解决的问题：
Q1 在分析相邻层关系时，会给出几个核心指标（Top1 MI、Topk Jaccard、Topk Overlap、Sparse Cos）。
但这些指标是“点估计”，缺少统计不确定性。
本脚本通过 bootstrap 重采样给每个指标计算置信区间（CI），让结论更稳健。

核心思想（公式与代码对应关系）：
1) Top1 Mutual Information (归一化)：
   - 先统计相邻两层 Top1 专家的联合频次表 joint[e1, e2]
   - MI = sum p(e1,e2) * log( p(e1,e2) / (p(e1)*p(e2)) )
   - 归一化：MI / log(n_experts)
2) Topk Jaccard / Overlap：
   - 每个 token 只看 Top-(K+2) 专家集合
   - Jaccard = |A∩B| / |A∪B|
   - Overlap = |A∩B| / min(|A|,|B|)
3) Sparse Cosine：
   - 对每个 token 的 Top-(K+2) 概率向量做稀疏余弦相似度
   - 只在 Top-(K+2) 的专家上计算点积与范数
4) Bootstrap：
   - 在 token 维度做有放回采样
   - 每次重算所有相邻层指标
   - 取 2.5% / 97.5% 分位数作为 CI

输出：
router_structure_ci.json
"""  # 文档字符串结束

import argparse  # import 语法：导入 argparse，用于命令行参数解析
import glob  # import 语法：导入 glob，用于匹配文件名模式
import json  # import 语法：导入 json，用于读写 JSON 文件
import logging  # import 语法：导入 logging，用于输出日志
import math  # import 语法：导入 math，用于数学函数
import os  # import 语法：导入 os，用于路径与文件操作
from collections import defaultdict  # from...import 语法：导入 defaultdict（默认字典）
from typing import Dict, List, Tuple  # 类型标注：Dict/List/Tuple 帮助阅读与IDE提示

import numpy as np  # import 语法：导入 numpy 并命名为 np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")  # 配置日志格式与级别


def _to_2d_block(x: any, pad_value: float, dtype: np.dtype) -> np.ndarray:  # def 语法：定义函数
    """  # 三引号：函数文档字符串
    把输入转换成 2D 数组。
    这里处理的是 Stage1 采集的 ragged/嵌套数组：
    - 有时是 list of 1D
    - 有时是 list of 2D
    - 有时已经是标准 2D
    统一成 [n_tokens, k] 方便后续拼接。
    """  # 函数文档字符串结束
    x = np.asarray(x, dtype=object)  # 赋值语法：把 x 转成 numpy 对象数组
    if x.ndim == 2 and x.dtype != object:  # if 语法：条件判断（是否已是标准 2D）
        return x.astype(dtype)  # return 语法：直接返回并转换类型
    if x.dtype == object:  # if 语法：对象数组需要逐条处理
        rows = [np.asarray(r) for r in x]  # 列表推导：把每行转成数组
        # 情况1：每行是 1D，长度可能不同，按最大长度 padding
        if rows and all(r.ndim == 1 for r in rows):  # all：判断是否所有行都是 1D
            max_len = max(r.shape[0] for r in rows)  # max：取最长行长度
            out = np.full((len(rows), max_len), pad_value, dtype=dtype)  # np.full：用 pad_value 初始化
            for i, r in enumerate(rows):  # for 语法：遍历行
                r = np.asarray(r, dtype=dtype)  # 转换行类型
                out[i, : r.shape[0]] = r  # 切片赋值：把真实数据写入对应位置
            return out  # return：返回 2D 结果
        # 情况2：每行是 2D，列数可能不同，按最大列数 padding
        if rows and all(r.ndim == 2 for r in rows):  # 判断是否所有行都是 2D
            max_cols = max(r.shape[1] for r in rows)  # 取最大列数
            padded = []  # 空列表：用于收集 padded 结果
            for r in rows:  # 遍历每个 block
                r = r.astype(dtype)  # 转 dtype
                if r.shape[1] < max_cols:  # 若列数不足则补齐
                    pad = np.full((r.shape[0], max_cols - r.shape[1]), pad_value, dtype=dtype)  # 构造 padding
                    r = np.concatenate([r, pad], axis=1)  # 拼接到右侧
                padded.append(r)  # 追加到列表
            return np.concatenate(padded, axis=0)  # 合并成一个大矩阵
    if x.ndim == 1:  # 如果是一维数组
        return np.asarray(x, dtype=dtype).reshape(-1, 1)  # reshape 为列向量
    return np.asarray(x, dtype=dtype)  # 默认：返回转换后的数组


def _ensure_2d_array(arr: np.ndarray, pad_value: float, dtype: np.dtype) -> np.ndarray:  # 定义函数
    """  # 函数文档字符串
    把任意形态的输入转换成标准 2D 数组。
    这是 _to_2d_block 的批量版本，用于把“多个 npz 文件的列表”
    统一成一个大矩阵。
    """  # 文档字符串结束
    if isinstance(arr, np.ndarray) and arr.dtype != object:  # 判断是否已是非 object 的 ndarray
        if arr.ndim == 2:  # 如果是二维
            return arr.astype(dtype)  # 直接返回
        if arr.ndim == 1:  # 如果是一维
            return arr.reshape(-1, 1).astype(dtype)  # 变成二维列向量

    if isinstance(arr, np.ndarray) and arr.dtype == object:  # 如果是 object ndarray
        elems = list(arr)  # 转为 Python 列表
    elif isinstance(arr, list):  # 如果本身就是 list
        elems = arr  # 直接使用
    else:  # 其他情况
        elems = list(np.asarray(arr, dtype=object))  # 强制转 object 列表

    blocks = []  # 创建空列表用于收集 2D block
    for e in elems:  # 遍历每个元素
        try:  # try/except：避免单个坏数据导致整体失败
            blocks.append(_to_2d_block(e, pad_value, dtype))  # 转为 2D block
        except Exception:  # 捕获异常
            continue  # 出错就跳过该元素
    if not blocks:  # 若没有任何 block
        return np.zeros((0, 1), dtype=dtype)  # 返回空数组

    max_cols = max(b.shape[1] for b in blocks)  # 计算最大列数
    padded = []  # 用于存放补齐后的 block
    for b in blocks:  # 遍历每个 block
        if b.shape[1] < max_cols:  # 如果列数不足
            pad = np.full((b.shape[0], max_cols - b.shape[1]), pad_value, dtype=dtype)  # 创建 padding
            b = np.concatenate([b, pad], axis=1)  # 拼接 padding
        padded.append(b)  # 追加到列表
    return np.concatenate(padded, axis=0)  # 合并并返回


def load_data(data_dir: str, max_files: int = -1) -> Dict[str, any]:  # 定义数据读取函数
    """  # 文档字符串
    读取 Stage1 输出的多个 npz 文件，并拼成统一结构：
    - tokens_str：token 文本（仅用于可选的匹配置乱）
    - 每层的 probs/indices：形状统一成 [n_tokens, k_save]
    """  # 文档字符串结束
    npz_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))  # 查找并排序所有 npz 文件
    if max_files > 0:  # 若用户限制读取数量
        npz_files = npz_files[:max_files]  # 只保留前 max_files 个
    logging.info("Found %d NPZ files.", len(npz_files))  # 输出日志

    all_data = defaultdict(list)  # 用默认字典存储数据
    for f in npz_files:  # 遍历文件
        try:  # 捕获读取异常
            with np.load(f, allow_pickle=True) as data:  # 读取 npz 文件
                tokens_str = data["tokens_str"]  # 取 token 文本
                all_data["tokens_str"].extend(tokens_str)  # 追加到总列表
                for k in data.keys():  # 遍历键
                    if k.startswith("layers/") and k.endswith("_out_probs"):  # 只取 probs
                        layer_name = k.split("/")[1].replace("_out_probs", "")  # 提取层名
                        probs = data[k]  # 取 probs
                        indices = data[f"layers/{layer_name}_out_indices"]  # 取 indices
                        all_data[f"{layer_name}_probs"].append(probs)  # 追加 probs
                        all_data[f"{layer_name}_indices"].append(indices)  # 追加 indices
        except Exception as e:  # 捕获异常
            logging.warning("Error reading %s: %s", f, e)  # 输出警告

    # 统一 probs/indices 为 2D 形式
    for k in list(all_data.keys()):  # 遍历键
        if k == "tokens_str":  # 跳过文本
            continue  # 继续下一轮
        if k.endswith("_probs"):  # 如果是 probs
            all_data[k] = _ensure_2d_array(np.asarray(all_data[k], dtype=object), 0.0, np.float64)  # 转 2D
        else:  # 否则视为 indices
            all_data[k] = _ensure_2d_array(np.asarray(all_data[k], dtype=object), -1, np.int64)  # 转 2D
    return all_data  # 返回整合数据


def _sorted_layer_names(all_data: Dict[str, any]) -> List[str]:  # 定义层名提取函数
    """从 all_data 的键里提取层名，并排序。"""  # 简短说明
    layer_names = []  # 初始化列表
    for k in all_data.keys():  # 遍历键
        if k.endswith("_probs"):  # 只看 probs 键
            layer_names.append(k.replace("_probs", ""))  # 去掉后缀得到层名
    return sorted(layer_names)  # 排序后返回


def _mi_from_joint(joint: np.ndarray) -> float:  # 定义互信息函数
    """  # 文档字符串
    根据联合频次表计算互信息：
    MI = Σ p(x,y) * log( p(x,y) / (p(x)p(y)) )
    """  # 文档字符串结束
    joint = np.asarray(joint, dtype=np.float64)  # 转为 float 数组
    total = float(joint.sum())  # 计算总频次
    if total <= 0:  # 如果总频次为 0
        return 0.0  # 返回 0
    pxy = joint / total  # 联合概率
    px = pxy.sum(axis=1, keepdims=True)  # 边缘概率 p(x)
    py = pxy.sum(axis=0, keepdims=True)  # 边缘概率 p(y)
    with np.errstate(divide="ignore", invalid="ignore"):  # 忽略数值警告
        mi = np.nansum(pxy * (np.log(pxy + 1e-12) - np.log(px + 1e-12) - np.log(py + 1e-12)))  # 互信息
    return float(mi)  # 返回 Python float


def _prepare_pair_arrays(  # 定义相邻层预处理函数
    a_probs: np.ndarray,  # 上一层 probs
    a_inds: np.ndarray,  # 上一层 indices
    b_probs: np.ndarray,  # 下一层 probs
    b_inds: np.ndarray,  # 下一层 indices
    n_experts: int,  # 专家数
) -> Dict[str, np.ndarray]:  # 返回字典
    """  # 文档字符串
    为相邻层准备缓存数组，避免 bootstrap 中重复计算。
    这里先把 Top1、Topk 集合、稀疏余弦的原始材料算出来，
    后面只需要按索引取子集再聚合。
    """  # 文档字符串结束
    n = min(a_probs.shape[0], b_probs.shape[0])  # 取最小 token 数
    top1_a = a_inds[:n, 0].astype(np.int64, copy=False)  # 上一层 top1
    top1_b = b_inds[:n, 0].astype(np.int64, copy=False)  # 下一层 top1
    valid = (top1_a >= 0) & (top1_a < n_experts) & (top1_b >= 0) & (top1_b < n_experts)  # 有效标记

    jac = np.zeros(n, dtype=np.float64)  # 初始化 Jaccard 数组
    ov = np.zeros(n, dtype=np.float64)  # 初始化 Overlap 数组
    cos = np.zeros(n, dtype=np.float64)  # 初始化 Cosine 数组
    for t in range(n):  # 遍历每个 token
        # Top-(K+2) 专家集合
        set_a = set(int(x) for x in a_inds[t] if 0 <= int(x) < n_experts)  # 构造上一层集合
        set_b = set(int(x) for x in b_inds[t] if 0 <= int(x) < n_experts)  # 构造下一层集合
        if not set_a and not set_b:  # 若两者为空
            continue  # 跳过
        inter = set_a & set_b  # 交集
        union = set_a | set_b  # 并集
        if union:  # 若并集非空
            jac[t] = len(inter) / len(union)  # Jaccard
        denom = min(len(set_a), len(set_b))  # Overlap 分母
        if denom > 0:  # 分母非零
            ov[t] = len(inter) / denom  # Overlap

        if inter:  # 若交集非空
            # 稀疏余弦：只在 Top-(K+2) 专家上计算
            na = 0.0  # 上一层向量平方和
            nb = 0.0  # 下一层向量平方和
            for k in range(a_inds.shape[1]):  # 遍历上一层 Top-(K+2)
                eid = int(a_inds[t, k])  # 专家编号
                pv = float(a_probs[t, k])  # 概率值
                if 0 <= eid < n_experts:  # 合法专家
                    na += pv * pv  # 累加平方
            for k in range(b_inds.shape[1]):  # 遍历下一层 Top-(K+2)
                eid = int(b_inds[t, k])  # 专家编号
                pv = float(b_probs[t, k])  # 概率值
                if 0 <= eid < n_experts:  # 合法专家
                    nb += pv * pv  # 累加平方
            amap = {int(a_inds[t, k]): float(a_probs[t, k]) for k in range(a_inds.shape[1]) if 0 <= int(a_inds[t, k]) < n_experts}  # 上一层稀疏向量
            bmap = {int(b_inds[t, k]): float(b_probs[t, k]) for k in range(b_inds.shape[1]) if 0 <= int(b_inds[t, k]) < n_experts}  # 下一层稀疏向量
            dot = 0.0  # 点积
            for e in inter:  # 只遍历交集专家
                dot += amap.get(e, 0.0) * bmap.get(e, 0.0)  # 累加点积
            denom = (na ** 0.5) * (nb ** 0.5)  # 余弦分母
            if denom > 0:  # 分母非零
                cos[t] = dot / denom  # 余弦相似度

    return {  # 返回字典
        "top1_a": top1_a,  # 上一层 top1
        "top1_b": top1_b,  # 下一层 top1
        "valid": valid,  # 有效标记
        "jac": jac,  # Jaccard 数组
        "ov": ov,  # Overlap 数组
        "cos": cos,  # Cosine 数组
    }  # 字典结束


def _compute_pair_metrics(pair: Dict[str, np.ndarray], indices: np.ndarray, n_experts: int) -> Tuple[float, float, float, float]:  # 定义指标计算函数
    """  # 文档字符串
    对某一对相邻层计算 4 个指标：
    1) top1_mi_norm：Top1 互信息，按 log(n_experts) 归一化
    2) topk_jaccard：Topk 集合 Jaccard
    3) topk_overlap：Topk 集合 Overlap
    4) sparse_cos：Topk 概率向量稀疏余弦
    """  # 文档字符串结束
    top1_a = pair["top1_a"][indices]  # 取子集
    top1_b = pair["top1_b"][indices]  # 取子集
    valid = pair["valid"][indices]  # 取子集
    joint = np.zeros((n_experts, n_experts), dtype=np.int64)  # 初始化联合频次表
    if valid.any():  # 若存在有效样本
        np.add.at(joint, (top1_a[valid], top1_b[valid]), 1)  # 累加联合频次
    mi = _mi_from_joint(joint)  # 互信息
    mi_norm = mi / math.log(max(n_experts, 2))  # 归一化
    jac = float(np.mean(pair["jac"][indices])) if indices.size > 0 else 0.0  # 平均 Jaccard
    ov = float(np.mean(pair["ov"][indices])) if indices.size > 0 else 0.0  # 平均 Overlap
    cos = float(np.mean(pair["cos"][indices])) if indices.size > 0 else 0.0  # 平均 Cosine
    return mi_norm, jac, ov, cos  # 返回四个指标


def _ci(arr: np.ndarray, low: float = 2.5, high: float = 97.5) -> Tuple[float, float]:  # 定义 CI 函数
    """用分位数法给出置信区间。"""  # 简短说明
    if arr.size == 0:  # 空数组保护
        return 0.0, 0.0  # 返回 0
    return float(np.percentile(arr, low)), float(np.percentile(arr, high))  # 返回分位数


def _baseline_means(baseline_json: str) -> Dict[str, float]:  # 定义基线均值函数
    """  # 文档字符串
    读取 Q1 主脚本产生的 baseline 置乱均值，
    用于计算“超出基线的 CI”（excess_ci）。
    """  # 文档字符串结束
    if not baseline_json:  # 若未提供路径
        return {}  # 返回空字典
    with open(baseline_json, "r", encoding="utf-8") as f:  # 打开基线文件
        data = json.load(f)  # 读取 JSON
    adj = data.get("adjacent_layer_pairs", {})  # 取相邻层字典
    if not adj:  # 若为空
        return {}  # 返回空字典
    mi_vals = []  # 收集 MI
    jac_vals = []  # 收集 Jaccard
    ov_vals = []  # 收集 Overlap
    cos_vals = []  # 收集 Cosine
    for _, v in adj.items():  # 遍历相邻层
        base = v.get("baseline", {}).get("permutation", {})  # 取置乱基线
        if not base:  # 若为空
            continue  # 跳过
        mi_vals.append(base.get("top1_mi_norm_mean", 0.0))  # 追加 MI
        jac_vals.append(base.get("topk_jaccard_mean", 0.0))  # 追加 Jaccard
        ov_vals.append(base.get("topk_overlap_mean", 0.0))  # 追加 Overlap
        cos_vals.append(base.get("sparse_cos_mean", 0.0))  # 追加 Cosine
    if not mi_vals:  # 若没有数据
        return {}  # 返回空字典
    return {  # 返回均值
        "top1_mi_norm": float(np.mean(mi_vals)),  # 平均 MI
        "topk_jaccard": float(np.mean(jac_vals)),  # 平均 Jaccard
        "topk_overlap": float(np.mean(ov_vals)),  # 平均 Overlap
        "sparse_cos": float(np.mean(cos_vals)),  # 平均 Cosine
    }  # 字典结束


def main():  # 定义主函数
    # ========================================================================
    # 解析命令行参数
    # ========================================================================
    parser = argparse.ArgumentParser(description="Q1 bootstrap CI for adjacent-layer metrics")  # 构建解析器
    parser.add_argument("--data_dir", required=True)  # 输入数据目录
    parser.add_argument("--output_dir", required=True)  # 输出目录
    parser.add_argument("--n_experts", type=int, default=60)  # 专家数
    parser.add_argument("--max_files", type=int, default=-1)  # 最多读取文件数
    parser.add_argument("--sample_size", type=int, default=20000)  # 采样 token 数
    parser.add_argument("--bootstrap_runs", type=int, default=200)  # bootstrap 次数
    parser.add_argument("--seed", type=int, default=42)  # 随机种子
    parser.add_argument("--baseline_json", type=str, default="")  # 基线 JSON 路径
    args = parser.parse_args()  # 解析参数

    os.makedirs(args.output_dir, exist_ok=True)  # 创建输出目录
    rng = np.random.default_rng(args.seed)  # 创建随机数生成器

    # ========================================================================
    # 读取数据并确定层名
    # ========================================================================
    all_data = load_data(args.data_dir, max_files=args.max_files)  # 读取数据
    layer_names = _sorted_layer_names(all_data)  # 提取层名
    if len(layer_names) < 2:  # 若层数不足
        raise ValueError("Not enough layers for adjacent analysis")  # 抛出错误

    # ========================================================================
    # 统一 token 数量（不同层可能长度不同，取最小值）
    # ========================================================================
    lengths = []  # 保存每层 token 数
    for ln in layer_names:  # 遍历层
        probs = all_data[f"{ln}_probs"]  # 取 probs
        lengths.append(int(probs.shape[0]))  # 记录长度
    n_tokens = min(lengths) if lengths else 0  # 取最小长度
    if args.sample_size > 0:  # 若指定采样
        n_base = min(args.sample_size, n_tokens)  # 取较小值
        base_idx = rng.choice(n_tokens, size=n_base, replace=False)  # 随机采样
    else:  # 若不采样
        base_idx = np.arange(n_tokens, dtype=np.int64)  # 用全量索引

    # ========================================================================
    # 预计算相邻层的 token 级指标材料
    # ========================================================================
    pair_arrays = []  # 用于存储相邻层预计算结果
    for i in range(len(layer_names) - 1):  # 遍历相邻层
        la = layer_names[i]  # 上一层名
        lb = layer_names[i + 1]  # 下一层名
        a_probs = all_data[f"{la}_probs"][base_idx]  # 取上一层 probs
        a_inds = all_data[f"{la}_indices"][base_idx]  # 取上一层 indices
        b_probs = all_data[f"{lb}_probs"][base_idx]  # 取下一层 probs
        b_inds = all_data[f"{lb}_indices"][base_idx]  # 取下一层 indices
        pair_arrays.append(_prepare_pair_arrays(a_probs, a_inds, b_probs, b_inds, args.n_experts))  # 预处理

    base_n = base_idx.size  # 采样 token 数
    logging.info("Base tokens for CI: %d", base_n)  # 输出日志

    # ========================================================================
    # 观测值（不重采样）
    # ========================================================================
    obs_metrics = []  # 存放观测指标
    full_idx = np.arange(base_n, dtype=np.int64)  # 全量索引
    for pair in pair_arrays:  # 遍历相邻层
        obs_metrics.append(_compute_pair_metrics(pair, full_idx, args.n_experts))  # 计算指标
    obs_metrics = np.array(obs_metrics, dtype=np.float64)  # 转数组
    obs_mean = obs_metrics.mean(axis=0)  # 对层取平均

    # ========================================================================
    # Bootstrap 重采样：每次有放回抽取 token
    # ========================================================================
    boot = np.zeros((args.bootstrap_runs, 4), dtype=np.float64)  # 初始化 bootstrap 结果
    for b in range(args.bootstrap_runs):  # 遍历 bootstrap 次数
        boot_idx = rng.integers(0, base_n, size=base_n, dtype=np.int64)  # 有放回采样
        pair_vals = []  # 存放每层指标
        for pair in pair_arrays:  # 遍历相邻层
            pair_vals.append(_compute_pair_metrics(pair, boot_idx, args.n_experts))  # 计算指标
        pair_vals = np.array(pair_vals, dtype=np.float64)  # 转数组
        boot[b] = pair_vals.mean(axis=0)  # 对层取平均后写入

    # 4 个指标分别给 CI
    ci = np.array([_ci(boot[:, i]) for i in range(boot.shape[1])], dtype=np.float64)  # 逐列计算 CI

    # ========================================================================
    # 基线对齐：从主脚本 router_structure_stats.json 读取置乱均值
    # ========================================================================
    baseline = _baseline_means(args.baseline_json)  # 读取基线
    excess_ci = {}  # 初始化 excess_ci
    if baseline:  # 如果存在基线
        excess = boot - np.array([  # 计算“超出基线”的 bootstrap
            baseline.get("top1_mi_norm", 0.0),  # baseline MI
            baseline.get("topk_jaccard", 0.0),  # baseline Jaccard
            baseline.get("topk_overlap", 0.0),  # baseline Overlap
            baseline.get("sparse_cos", 0.0),  # baseline Cosine
        ], dtype=np.float64)  # 转为数组
        excess_ci = {  # 计算 excess 的 CI
            "top1_mi_norm": _ci(excess[:, 0]),  # MI excess
            "topk_jaccard": _ci(excess[:, 1]),  # Jaccard excess
            "topk_overlap": _ci(excess[:, 2]),  # Overlap excess
            "sparse_cos": _ci(excess[:, 3]),  # Cosine excess
        }  # 字典结束

    # ========================================================================
    # 写出结果 JSON
    # ========================================================================
    out = {  # 结果总字典
        "meta": {  # 元信息
            "data_dir": args.data_dir,  # 数据目录
            "n_experts": args.n_experts,  # 专家数
            "sample_size": int(base_n),  # 实际采样数
            "bootstrap_runs": args.bootstrap_runs,  # bootstrap 次数
            "seed": args.seed,  # 随机种子
            "baseline_json": args.baseline_json,  # 基线路径
        },  # meta 结束
        "adjacent_mean": {  # 相邻层平均指标
            "top1_mi_norm": {"mean": float(obs_mean[0]), "ci_low": float(ci[0][0]), "ci_high": float(ci[0][1])},  # MI
            "topk_jaccard": {"mean": float(obs_mean[1]), "ci_low": float(ci[1][0]), "ci_high": float(ci[1][1])},  # Jaccard
            "topk_overlap": {"mean": float(obs_mean[2]), "ci_low": float(ci[2][0]), "ci_high": float(ci[2][1])},  # Overlap
            "sparse_cos": {"mean": float(obs_mean[3]), "ci_low": float(ci[3][0]), "ci_high": float(ci[3][1])},  # Cosine
        },  # adjacent_mean 结束
        "baseline_mean": baseline,  # 基线均值
        "adjacent_mean_excess_ci": excess_ci,  # 超出基线的 CI
    }  # out 结束

    out_path = os.path.join(args.output_dir, "router_structure_ci.json")  # 输出路径
    with open(out_path, "w", encoding="utf-8") as f:  # 打开文件
        json.dump(out, f, ensure_ascii=False, indent=2)  # 写入 JSON
    logging.info("Wrote CI summary -> %s", out_path)  # 输出日志


if __name__ == "__main__":  # 判断是否直接运行脚本
    main()  # 调用主函数
