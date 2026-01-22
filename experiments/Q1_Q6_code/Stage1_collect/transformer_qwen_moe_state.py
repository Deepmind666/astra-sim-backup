#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage-1：MoE 路由采集脚本（Qwen1.5-MoE-A2.7B）
============================================================

这份脚本的定位：**把“真实输入文本”喂给 MoE 模型，然后把每个 token 的路由轨迹录下来**。

你可以把它理解成给 Stage-2/Stage-3 准备“真实录像带”。

输出什么？
----------
会在 `experiments/moe_measure/results/<run_dir>/` 下生成两类产物：
1) `sample-*.npz`：每条样本（一个输入文本）一个 NPZ，里面保存：
   - `tokens_ids` / `tokens_str`
   - 每一层 MoE 的 `out_indices`（top-k 专家 id）与 `out_probs`（对应概率）
2) `routing_summary.json`：全局汇总（每层每个专家的命中占比）。
   - 注意：为了兼容历史，我们还会额外写一份 `*-midstate_report.json`（同内容）。

为什么要特别强调 `routing_summary.json`？
----------------------------------------
我们后续很多脚本（例如聚类批处理）会用 `find ... -name routing_summary.json` 来定位某次采集的目录。
如果没有这个文件，即使 NPZ 已经生成，后续流程也很容易“找不到数据”。

重要的耗时坑（新同事必读）
--------------------------
1) **数据集加载不要“无限制”**：
   - 例如 WikiText-103 train 有 180 万条，如果 config 里 max_samples=-1 且 max_docs=-1，
     光把文本读进内存就会卡很久，更别提推理。
2) **txt 不要按行读**：
   - GSM8K/MBPP 这类多行样本，如果按行读，会被拆碎成海量短样本，采集会慢很多。
   - 这个问题我们在 `load.py` 里做了修复：如果 txt 有 `<|endoftext|>`，就按分隔符切样本。
"""

import os
import json
import argparse
import logging
import collections
import re
from typing import Dict, List, Tuple, Optional, Any
from config import load_exp_config

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
)

from load import load_model_and_tokenizer, load_corpus_data
from utils import find_moe_layers

# --- 日志设置 ---
logging.basicConfig(level=logging.INFO)
logging.getLogger("transformers").setLevel(logging.WARNING)


# --- 参数解析 ---
def parse_args():
    """解析命令行参数。"""
    ap = argparse.ArgumentParser(
        description="Capture per-token MoE gate IO and write NPZ + routing summary JSON"
    )

    # —— 运行标识（影响输出目录命名）——
    #
    # 需求背景：
    # - 我们经常会在同一个 results 目录里多次采集（不同日期/不同口径/不同模型）。
    # - 如果输出目录不带日期/标记，后续很容易“覆盖、混淆、找不到对应那次采集”。  
    # - 因此增加 run_tag：它会被拼到输出目录名最前面，例如：
    #   results/2025-12-18-Qwen1.5-MoE-A2.7B-gsm8k-train-2000-gate-state/
    #
    # 注意：
    # - 这不会影响模型推理，也不会影响专家数（专家数永远来自 router_logits 的维度）。
    ap.add_argument(
        "--run_tag",
        type=str,
        default="",
        help="Optional tag appended to output directory name (e.g., 2025-12-18).",
    )

    # —— record 控制（NPZ 落盘）——
    ap.add_argument(
        "--record",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable per-token gate IO NPZ recording (default: on; use --no-record to disable).",
    )
    ap.add_argument(
        "--max-docs",
        type=int,
        default=-1,
        help="Max number of samples (lines) to record. -1 = no limit.",
    )
    ap.add_argument(
        "--record-every-n",
        type=int,
        default=1,
        help="Record 1 token every N tokens (downsample).",
    )
    ap.add_argument(
        "--max-tokens-per-doc",
        type=int,
        default=4096,
        help="Cap tokens recorded per line (after mask & sampling).",
    )

    # 仅保存门控输入部分的模式
    ap.add_argument(
        "--gate-input-mode",
        choices=["full", "head", "none"],
        default="full",
        help="Record full/head/none of gate inputs for NPZ.",
    )
    ap.add_argument(
        "--gate-input-head",
        type=int,
        default=64,
        help="When gate-input-mode=head, keep first N dims in NPZ.",
    )
    ap.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="Path to the configuration JSON file.",
    )

    return ap.parse_args()


from concurrent.futures import ThreadPoolExecutor

# --- 核心辅助函数：NPZ 保存 ---
def save_sample_npz(
    out_dir: str,
    global_index: int,
    tokens_ids: np.ndarray,
    tokens_str: List[str],
    gate_in_per_layer: Dict[str, np.ndarray],
    gate_out_probs_per_layer: Dict[str, np.ndarray],
    gate_out_indices_per_layer: Dict[str, np.ndarray],
) -> str:
    """
    将单个样本的门控输入/输出落盘为 .npz。
    """
    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.join(out_dir, f"sample-{global_index:07d}.npz")
    pack = {
        "tokens_ids": tokens_ids.astype(np.int64),
        "tokens_str": np.array(tokens_str, dtype=object),
    }

    for layer_name, arr in gate_in_per_layer.items():
        pack[f"layers/{layer_name}_in"] = arr.astype(np.float16)
    
    for layer_name, arr in gate_out_probs_per_layer.items():
        pack[f"layers/{layer_name}_out_probs"] = arr.astype(np.float16)
    for layer_name, arr in gate_out_indices_per_layer.items():
        pack[f"layers/{layer_name}_out_indices"] = arr.astype(np.int32)

    np.savez_compressed(fname, **pack)
    return fname


# --- 核心辅助函数：NPZ 数据提取 ---
def _extract_gate_input_for_sample(
    gate_in_tensor: torch.Tensor,  # From hook, on CPU
    batch_idx: int,
    seq_len: int,
    valid_token_positions: List[int],
    mode: str,
    head_dim: int,
) -> Optional[np.ndarray]:
    """从批处理张量中提取单个样本的门控输入。"""
    if gate_in_tensor is None or mode == "none" or not valid_token_positions:
        return None

    h_eff = None
    try:
        if gate_in_tensor.dim() == 2:  # [B*S, H]
            start = batch_idx * seq_len
            end = (batch_idx + 1) * seq_len
            if end <= gate_in_tensor.size(0):
                h_slice = (
                    gate_in_tensor[start:end].to(torch.float16).numpy()
                )  # [S, H]
                h_eff = h_slice[valid_token_positions, :]  # [T, H]
        elif gate_in_tensor.dim() == 3:  # [B, S, H]
            if batch_idx < gate_in_tensor.size(
                0
            ) and seq_len <= gate_in_tensor.size(1):
                h_slice = (
                    gate_in_tensor[batch_idx].to(torch.float16).numpy()
                )  # [S, H]
                h_eff = h_slice[valid_token_positions, :]
    except IndexError as e:
        logging.warning(f"IndexError during gate input extraction: {e}")
        return None

    if h_eff is not None and mode == "head" and head_dim > 0:
        h_eff = h_eff[:, :head_dim]

    return h_eff


def _extract_gate_output_for_sample(
    probs_tensor: torch.Tensor,  # 输入参数：Softmax 后的概率张量，已经在 CPU 上了
    batch_idx: int,              # 输入参数：当前处理的是批次里的第几句话（样本索引）
    seq_len: int,                # 输入参数：这句话的长度（Token数）
    valid_token_positions: List[int], # 输入参数：有效的 Token 位置列表（已经去掉了 Padding）
    top_k: int,                  # 输入参数：模型原始配置的 Top-K 值（例如 Qwen 通常是 2）
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    从批处理张量中提取单个样本的门控概率，仅保留 top-(K+2)。
    
    【设计理念 - 给零基础同学的解释】：
    1. **背景**：大模型的 MoE 层在做决定时，会给所有专家打分（概率）。比如有 60 个专家，它会算出 60 个概率值。
    2. **问题**：如果我们把这 60 个值全存下来，数据量太大了（Token数 x 层数 x 专家数）。
    3. **观察**：其实绝大部分专家的概率都是 0，只有被选中的那几个（Top-K）是有意义的。
    4. **优化**：我们只存概率最大的前 K 个。
    5. **进阶**：为了研究“备选专家”（即那些差点被选中，或者概率也很高的专家），我们多存 2 个，即 Top-(K+2)。
       这样我们既省了空间（存 4 个比存 60 个省多了），又保留了分析“路由不确定性”的可能性。
    
    参数详解:
        probs_tensor: 包含这一批次所有 Token 的概率矩阵。形状可能是 [Batch, Seq, Experts] 或平铺的 [Batch*Seq, Experts]。
        batch_idx: 我们现在要提取哪一句话的数据。
        seq_len: 每句话有多长。
        valid_token_positions: 这句话里哪些位置是真实的字（不是补零的 Padding）。
        top_k: 原始模型选几个专家。
               
    返回:
        vals: 选出的专家的概率值 (数据类型: float16，省内存)
        inds: 选出的专家的ID索引 (数据类型: int32)
    """
    
    # 第一步：如果没有有效的 Token（比如这句话全是 Padding），直接返回空
    if not valid_token_positions:
        return None, None

    p_eff_t = None
    try:
        # 第二步：从大矩阵里把当前这句话的数据“切”出来
        # 我们需要处理两种可能的输入形状：3维 或者 2维
        
        if probs_tensor.dim() == 3:  # 情况 A: [Batch, Sequence, Experts]
            # 直接用索引切片：第 batch_idx 句话，所有有效位置，所有专家
            p_eff_t = probs_tensor[
                batch_idx, valid_token_positions, :
            ]  # 结果形状: [有效Token数, 专家总数]
            
        elif probs_tensor.dim() == 2:  # 情况 B: [Batch*Sequence, Experts] (已经被展平了)
            # 如果是平铺的，我们需要算出每个有效 Token 在平铺数组里的绝对位置
            # 绝对位置 = (第几句话 * 每句话长度) + 这句话里的第几个字
            idx_flat = torch.as_tensor(
                [batch_idx * seq_len + int(t) for t in valid_token_positions],
                dtype=torch.long,
                device=probs_tensor.device,
            )
            
            # 安全检查：确保算出来的索引没有越界（不要超过数组总长度）
            idx_flat = idx_flat[idx_flat < probs_tensor.size(0)] 
            
            if idx_flat.numel() > 0:
                # index_select 是 PyTorch 用来按索引取行的函数
                p_eff_t = probs_tensor.index_select(0, idx_flat)  # 结果形状: [有效Token数, 专家总数]
                
    except IndexError as e:
        # 如果出错了（比如索引算错了），打印警告并跳过
        logging.warning(f"IndexError during gate output extraction: {e}")
        return None, None

    if p_eff_t is None:
        return None, None

    # --- 第三步：核心修改逻辑 Top-(K+2) 采样 ---
    
    # 获取总共有多少个专家（比如 60 个）
    num_experts = p_eff_t.size(-1)
    
    # 计算我们要保存多少个专家
    # 逻辑：原始 Top-K (比如2) + 额外 2 个备胎 = 4 个
    # min函数的作用：万一总专家数一共才 3 个，我们就不能存 4 个，所以要取小值
    k_save = min(top_k + 2, num_experts)
    #11月30日
    # torch.topk 是核心函数！
    # 它可以帮我们找出概率最大的前 k_save 个
    # dim=-1 表示在最后一个维度（专家维度）上找
    # vals: 存这 k_save 个概率的具体数值 (例如 [0.8, 0.15, 0.03, 0.01])
    # inds: 存这 k_save 个专家是第几号专家 (例如 [15, 3, 58, 0])
    vals, inds = torch.topk(p_eff_t, k=k_save, dim=-1)
    
    # 第四步：类型转换与返回
    # to(torch.float16): 把概率值转成半精度浮点数。概率不需要极高的精度，float16 足够了，且内存占用减半。
    # to(torch.int32): 把索引转成 32位整数。
    # .numpy(): 最后转成 numpy 数组，因为后续我们要存成 .npz 文件
    return vals.to(torch.float16).numpy(), inds.to(torch.int32).numpy()


def setup_gate_input_hooks(
    model: PreTrainedModel, moe_layer_indices: List[int]
) -> Tuple[Optional[List[Any]], Optional[Dict[int, torch.Tensor]]]:
    """附加前向钩子以捕获 MoE 门控输入。"""
    gate_inputs_storage = collections.defaultdict(lambda: None)

    def get_hook_fn(layer_index: int):
        def hook(module, input_args, output):
            # 取第一个位置参数作为门控输入；detach + CPU，避免占用显存
            gate_input = input_args[0]
            gate_inputs_storage[layer_index] = gate_input.detach().cpu()

        return hook

    hooks = []
    for layer_idx in moe_layer_indices:
        try:
            # 假设模型结构
            gate_module = model.model.layers[layer_idx].mlp.gate
            h = gate_module.register_forward_hook(get_hook_fn(layer_idx))
            hooks.append(h)
        except Exception as e:
            logging.warning(f"Cannot hook .mlp.gate at layer {layer_idx}: {e}")

    if not hooks:
        logging.error("Error: no hooks attached. Abort.")
        return None, None

    logging.info(f"Attached {len(hooks)} hooks to MoE gates.")
    return hooks, gate_inputs_storage


# --- 核心辅助函数：处理 ---
def initialize_statistics_containers(
    router_logits_tuple: Tuple[torch.Tensor, ...],
    aligned_indices: List[int],
    moe_info: Dict[int, Dict[str, Any]],
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, int], Dict[str, int]]:
    """首次遇到 logits 时，初始化统计容器。"""
    layer_counters = {}
    layer_top_k = {}
    layer_expert_counts = {}

    storage_device = device if torch.cuda.is_available() else "cpu"

    for logits_idx, (layer_idx, logits) in enumerate(
        zip(aligned_indices, router_logits_tuple)
    ):
        num_experts = logits.shape[-1]
        layer_name = f"MoE_Gate{logits_idx}"

        layer_counters[layer_name] = torch.zeros(
            num_experts, dtype=torch.long, device=storage_device
        )

        # 取该层 top_k，取不到就 2；且不大于专家数
        k = int(moe_info.get(layer_idx, {}).get("top_k", 2) or 2)
        k = max(1, min(k, num_experts))
        layer_top_k[layer_name] = k
        layer_expert_counts[layer_name] = num_experts

    logging.info(
        f"[discover] Found {len(layer_counters)} MoE layers (from router_logits)."
    )
    return layer_counters, layer_top_k, layer_expert_counts


def update_routing_statistics(
    router_logits_tuple: Tuple[torch.Tensor, ...],
    aligned_indices: List[int],
    mask_flat: torch.Tensor,
    layer_counters: Dict[str, torch.Tensor],
    layer_top_k: Dict[str, int],
):
    """根据一个批次的数据聚合专家分配计数。"""
    for logits_idx, (layer_idx, logits) in enumerate(
        zip(aligned_indices, router_logits_tuple)
    ):
        layer_name = f"MoE_Gate{logits_idx}"
        num_experts = logits.size(-1)
        # [B,S,E] 或 [B*S,E] -> [B*S,E]
        flat_logits = logits.reshape(-1, num_experts)
        k = layer_top_k[layer_name]

        _, topk_indices = torch.topk(flat_logits, k=k, dim=-1)
        valid_topk_indices = topk_indices[mask_flat]  # 仅有效 token

        # 逐列累加
        for col in range(k):
            expert_indices_col = valid_topk_indices[:, col]
            bin_counts = torch.bincount(
                expert_indices_col, minlength=num_experts
            )
            layer_counters[layer_name] += bin_counts.to(
                layer_counters[layer_name].dtype
            )


def record_npz_samples(
    args: argparse.Namespace,
    batch_texts: List[str],
    input_ids_cpu: torch.Tensor,
    attn_mask_cpu: torch.Tensor,
    probs_list_cpu: List[torch.Tensor],
    gate_inputs_storage: Dict[int, torch.Tensor],
    tokenizer: PreTrainedTokenizer,
    aligned_indices: List[int],
    layer_top_k: Dict[str, int],
    global_doc_index_start: int,
    gate_state_dir: str,
    executor: Optional[ThreadPoolExecutor] = None,
) -> Tuple[int, bool]:
    """处理并保存一个批次中样本的 NPZ 文件。

    返回: (本次写入的文档数, 是否应提前停止)
    """
    B, S = attn_mask_cpu.shape
    docs_written_in_batch = 0
    stop_early = False

    for batch_idx in range(B):
        current_global_index = global_doc_index_start + docs_written_in_batch

        # 检查是否已达到最大文档数
        if args.max_docs > 0 and current_global_index >= args.max_docs:
            stop_early = True
            break

        # 1. 获取这条样本的有效 token 位置
        valid_pos_tensor = torch.nonzero(
            attn_mask_cpu[batch_idx], as_tuple=False
        ).reshape(-1)
        valid_pos = valid_pos_tensor.tolist()
        if not valid_pos:
            continue

        # 2. 抽样：每 N 个取 1 个
        if args.record_every_n > 1:
            valid_pos = valid_pos[:: args.record_every_n]

        # 3. 上限
        if (
            args.max_tokens_per_doc > 0
            and len(valid_pos) > args.max_tokens_per_doc
        ):
            valid_pos = valid_pos[: args.max_tokens_per_doc]

        if not valid_pos:
            continue

        # 4. 提取 token ID 和字符串
        ids_eff = input_ids_cpu[batch_idx][valid_pos].numpy()
        toks_str = tokenizer.convert_ids_to_tokens(ids_eff.tolist())

        # 5. 收集各层 gate_in / gate_out
        gate_in_np = {}
        gate_out_probs_np = {}
        gate_out_indices_np = {}

        for logits_idx, layer_idx in enumerate(aligned_indices):
            layer_name = f"MoE_Gate{logits_idx}"

            # --- 提取 gate_in：来自 hook 的 CPU Tensor ---
            h_eff = _extract_gate_input_for_sample(
                gate_in_tensor=gate_inputs_storage.get(layer_idx),
                batch_idx=batch_idx,
                seq_len=S,
                valid_token_positions=valid_pos,
                mode=args.gate_input_mode,
                head_dim=args.gate_input_head,
            )
            if h_eff is not None:
                gate_in_np[layer_name] = h_eff

            # --- 提取 gate_out：来自 router logits 的 softmax 概率 ---
            # 获取该层的 top_k
            k = layer_top_k.get(layer_name, 2)
            
            p_vals, p_inds = _extract_gate_output_for_sample(
                probs_tensor=probs_list_cpu[logits_idx],
                batch_idx=batch_idx,
                seq_len=S,
                valid_token_positions=valid_pos,
                top_k=k,
            )
            if p_vals is not None and p_inds is not None:
                gate_out_probs_np[layer_name] = p_vals
                gate_out_indices_np[layer_name] = p_inds

        # 准备预览文本
        preview = (
            batch_texts[batch_idx][:80].replace("\n", " ").replace("\r", " ")
        )
        
        # 定义保存任务
        def _save_task(idx, t_ids, t_str, g_in, g_probs, g_inds, txt_prev):
            path = save_sample_npz(
                out_dir=gate_state_dir,
                global_index=idx,
                tokens_ids=t_ids,
                tokens_str=t_str,
                gate_in_per_layer=g_in,
                gate_out_probs_per_layer=g_probs,
                gate_out_indices_per_layer=g_inds,
            )
            # 只有在非异步模式下，或者为了调试，才打印所有日志
            # 为了减少控制台刷屏，我们可以每 100 个打印一次，或者保持原样
            # 异步模式下打印可能会乱序，但这是可接受的
            logging.info(
                f"[record] saved: {path} | text='{txt_prev}...' | tokens={len(t_ids)} | layers={len(g_probs)}"
            )

        # 6. 保存 NPZ (异步或同步)
        if executor is not None:
            # 异步提交
            executor.submit(
                _save_task,
                current_global_index + 1,
                ids_eff,
                toks_str,
                gate_in_np,
                gate_out_probs_np,
                gate_out_indices_np,
                preview
            )
        else:
            # 同步执行
            _save_task(
                current_global_index + 1,
                ids_eff,
                toks_str,
                gate_in_np,
                gate_out_probs_np,
                gate_out_indices_np,
                preview
            )

        docs_written_in_batch += 1

    return docs_written_in_batch, stop_early


# --- 核心辅助函数：报告 ---
def save_summary_report(
    layer_counters: Dict[str, torch.Tensor],
    layer_top_k: Dict[str, int],
    layer_expert_counts: Dict[str, int],
    output_file_path: str,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    生成并保存最终的 JSON 路由摘要。

    这个摘要文件主要用于两件事：
    1) 作为“采集是否成功”的验收标志（文件存在 + 内容不为空）
    2) 作为后续分析/排障的元信息（例如确认专家数到底是 60 还是 64）

    兼容性设计（非常重要）：
    - 历史脚本可能默认“最外层 key 就是层名”，因此这里不把结构改成 {meta:..., layers:...}。
    - 我们把元信息放在一个不会与层名冲突的特殊 key：`_meta`。
    """
    summary = {}

    # 1) 先写入元信息（如果提供）。老代码即使不认识 `_meta`，也会自动忽略它。
    if meta is not None:
        summary["_meta"] = meta

    # 2) 逐层写入路由统计（每层一个分布表）
    for layer_name, counts in layer_counters.items():
        counts_cpu = counts.detach().to("cpu").to(torch.float64)
        total_assignments = float(counts_cpu.sum().item())
        num_experts = int(layer_expert_counts.get(layer_name, 0))

        if total_assignments <= 0:
            dist_pretty = {}
        else:
            pct_np = (counts_cpu / total_assignments * 100.0).numpy()
            dist_pretty = {
                f"Expert{i}": f"{pct_np[i]:.4f}%" for i in range(num_experts)
            }

        summary[layer_name] = {
            "num_experts": num_experts,
            "top_k_used": int(layer_top_k.get(layer_name, 0)),
            "total_assignments": int(total_assignments),
            "distribution_pct": dist_pretty,
        }

    try:
        with open(output_file_path, "w", encoding="utf-8") as f:
            json.dump(
                summary, f, ensure_ascii=False, indent=2, separators=(",", ": ")
            )
        logging.info(
            f"\n[SUMMARY] wrote routing distribution JSON -> {output_file_path}"
        )
    except IOError as e:
        logging.error(f"Failed to write summary JSON: {e}")

    return summary


# --- 主流程编排 ---
def run_analysis(args: argparse.Namespace, config: Dict[str, Any]):
    """执行 MoE 门控分析的主流程。"""

    # --- 1. 提取配置并设置路径 ---
    model_id = config["model_id"]
    dataset_config = config["dataset_config"]
    max_samples = config["max_samples"]
    batch_size = config["batch_size"]
    max_length = config["max_length"]

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "results")
    # 简化模型名，只取最后一部分，防止 Windows 路径过长
    model_id_slug = os.path.basename(os.path.normpath(model_id))

    # dataset_name 既可能是“数据集名字”（如 gsm8k），也可能被误写成“路径”（如 /home/.../wikitext）。
    # 如果直接把路径拼进输出目录名，会导致目录层级混乱甚至创建失败。
    # 因此这里做一次“取 basename”的规整，让输出目录稳定可预测。
    dataset_name_raw = dataset_config.get("name") or dataset_config.get("local_path") or "dataset"
    dataset_name = os.path.basename(os.path.normpath(str(dataset_name_raw)))

    dataset_split = dataset_config.get("split") or "unknown"
    max_samples_str = str(max_samples) if max_samples is not None else "all"

    # 生成输出目录名：
    # - 默认：{model}-{dataset}-{split}-{max_samples}-gate-state
    # - 若设置了 run_tag：{run_tag}-{model}-{dataset}-{split}-{max_samples}-gate-state
    #
    # 这样可以做到“同一 results 根目录下，多次采集互不覆盖”，并且目录名一眼能看出是哪天采的。
    base_dir_name = f"{model_id_slug}-{dataset_name}-{dataset_split}-{max_samples_str}-gate-state"
    run_tag = (args.run_tag or "").strip()
    if run_tag:
        base_dir_name = f"{run_tag}-{base_dir_name}"

    gate_state_dir = os.path.join(output_path, base_dir_name)

    os.makedirs(output_path, exist_ok=True)
    os.makedirs(gate_state_dir, exist_ok=True)
    logging.info(f"NPZ output directory: {gate_state_dir}")

    # --- 2. 加载模型、分词器和数据 ---
    model, tokenizer = load_model_and_tokenizer(model_id)

    moe_info = find_moe_layers(model)
    if not moe_info:
        logging.error("No MoE layers found, exit.")
        return
    moe_layer_indices = list(moe_info.keys())  # 真实的解码器层索引
    logging.info(f"Found MoE layers at indices: {moe_layer_indices}")

    # --- 2.1 数据量“安全阈值”处理（避免无意义的超大加载）---
    #
    # 关键点：config.max_samples 控制“最多加载多少条文本样本”。
    # 但如果用户只想录制前 max_docs 条样本（args.max_docs>0），
    # 那我们完全没必要把一个超大数据集（例如 wikitext-103 的 180 万条）全部读进内存。
    #
    # 这里采用一个非常直观的策略：
    # - 如果开启 record 且设置了 max_docs：最多只加载 max_docs 条样本；
    # - 否则：按 config.max_samples 来加载（可能是 None=无限制，风险自负）。
    max_samples_to_load = max_samples
    if args.record and args.max_docs > 0:
        max_samples_to_load = args.max_docs if max_samples_to_load is None else min(max_samples_to_load, args.max_docs)
        logging.info(
            f"[Data] record=True 且 max_docs={args.max_docs}，限制最多加载 {max_samples_to_load} 条样本（避免超大数据集加载过慢）"
        )
    elif max_samples_to_load is None and args.max_docs <= 0:
        logging.warning(
            "[Data] max_samples=all 且 max_docs=all：如果你使用的是超大数据集（例如 wikitext-103 train），采集可能会非常非常久。"
        )

    corpus = load_corpus_data(dataset_config, max_samples_to_load)
    if not corpus:
        logging.error("Dataset is empty. Exit.")
        return

    # --- 2.5 断点续训：检查已存在的 NPZ 文件 ---
    import glob
    existing_npz = glob.glob(os.path.join(gate_state_dir, "sample-*.npz"))

    # 断点续训口径：
    # - 旧代码用“文件数量”来判断已完成多少条样本；
    # - 更稳妥的做法是解析文件名里的编号（避免出现缺号时误判）。
    #   由于我们命名规则是：sample-0000001.npz 表示第 1 条已写入样本（1-based），
    #   所以 `resume_offset = max(已有编号)` 就等价于“已经写了多少条”。
    resume_offset = 0
    for fpath in existing_npz:
        base = os.path.basename(fpath)
        m = re.match(r"sample-(\d+)\.npz$", base)
        if not m:
            continue
        resume_offset = max(resume_offset, int(m.group(1)))
    
    if resume_offset > 0:
        logging.info(f"[Resume] Found existing NPZ up to sample-{resume_offset:07d}.npz in {gate_state_dir}")

        # 如果用户设置了 max_docs，并且我们已经达到/超过 max_docs，就直接结束（避免重复跑）
        if args.max_docs > 0 and resume_offset >= args.max_docs:
            logging.info(f"[Resume] resume_offset={resume_offset} >= max_docs={args.max_docs}. Nothing to do.")
            return

        # 跳过已处理的样本
        if resume_offset < len(corpus):
            corpus = corpus[resume_offset:]
            logging.info(f"[Resume] Skipping first {resume_offset} samples, {len(corpus)} remaining")
        else:
            logging.info(f"[Resume] All {len(corpus)} samples already processed. Nothing to do.")
            return
    else:
        logging.info(f"[Fresh start] Processing {len(corpus)} samples")

    # --- 3. 设置钩子 ---
    hooks, gate_inputs_storage = setup_gate_input_hooks(
        model, moe_layer_indices
    )
    if hooks is None:
        return  # setup_gate_input_hooks 内部已打印错误

    # --- 4. 初始化统计容器 ---
    # 在第一次迭代中延迟初始化
    layer_counters: Optional[Dict[str, torch.Tensor]] = None
    layer_top_k: Optional[Dict[str, int]] = None
    layer_expert_counts: Optional[Dict[str, int]] = None

    # --- 5. 推理 & 记录 ---
    model.eval()
    total_docs_written = 0
    global_doc_index = resume_offset  # 跟踪已处理的文档总数（用于 NPZ 索引），支持断点续训
    
    # 创建线程池执行器，用于异步保存文件（写 NPZ 是明显的 IO 瓶颈）。
    # max_workers 不宜过大：过大反而会让磁盘随机写更严重、整体更慢。
    executor = ThreadPoolExecutor(max_workers=4)

    try:
        with torch.no_grad():
            for i in tqdm(range(0, len(corpus), batch_size), desc="Batches"):
                batch_texts = corpus[i : i + batch_size]
                enc = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                )

                primary_device = next(model.parameters()).device
                inputs = {k: v.to(primary_device) for k, v in enc.items()}

                gate_inputs_storage.clear()  # 清上一次 batch 的 hook 缓存

                # 前向推理
                outputs = model(
                    **inputs, output_router_logits=True, use_cache=False
                )
                router_logits_tuple = getattr(outputs, "router_logits", None)

                if not router_logits_tuple:
                    logging.error("[error] router_logits is None/empty, break.")
                    break

                # 对齐层：router_logits 的层数可能不等于所有 MoE 层数
                num_layers_logits = len(router_logits_tuple)
                aligned_indices = moe_layer_indices[:num_layers_logits]

                # 延迟初始化统计容器
                if layer_counters is None:
                    layer_counters, layer_top_k, layer_expert_counts = (
                        initialize_statistics_containers(
                            router_logits_tuple,
                            aligned_indices,
                            moe_info,
                            primary_device,
                        )
                    )

                attention_mask = inputs.get(
                    "attention_mask", torch.ones_like(inputs["input_ids"])
                )
                mask_flat = attention_mask.reshape(-1).to(torch.bool)

                # --- 5a. 统计汇总 (对所有有效 token 计数) ---
                update_routing_statistics(
                    router_logits_tuple,
                    aligned_indices,
                    mask_flat,
                    layer_counters,
                    layer_top_k,
                )

                # --- 5b. 记录 NPZ (按抽样/上限保存) ---
                if args.record:
                    # 预先准备 CPU 数据
                    input_ids_cpu = enc["input_ids"].cpu()
                    attn_mask_cpu = attention_mask.cpu()
                    probs_list_cpu = [
                        F.softmax(logits, dim=-1).detach().to("cpu")
                        for logits in router_logits_tuple
                    ]

                    # 将保存任务提交给线程池
                    # 注意：这里我们将数据拷贝传递给线程，避免后续 batch 覆盖
                    # record_npz_samples 内部会调用 save_sample_npz，我们将其包装一下以支持异步
                    # 但 record_npz_samples 返回 docs_written，我们需要稍微改动逻辑
                    
                    # 这里直接同步调用 record_npz_samples，但将其内部的 save_sample_npz 改为异步提交？
                    # 实际上 record_npz_samples 做了很多数据切片工作，如果在主线程做会阻塞
                    # 最好的方式是将整个 record_npz_samples 放到线程池
                    
                    # 为了简化，我们这里直接调用 record_npz_samples，但由于 IO 是在 save_sample_npz 里的
                    # 我们修改 save_sample_npz 让其变为 "fire and forget" 或者在内部使用 executor
                    # 鉴于代码结构，我们在这里不改动 record_npz_samples 的签名，而是通过全局变量或者修改 save_sample_npz
                    
                    # 方案二：直接在 record_npz_samples 内部调用 save_sample_npz 时使用 executor
                    # 这需要传入 executor。
                    # 既然我们不能轻易修改函数签名（为了保持兼容性），我们选择直接运行。
                    # 等等，上面的 Read 并没有显示 record_npz_samples 的定义，它应该在更上面的位置。
                    
                    # 让我们先用简单粗暴的方法：在这里直接运行，但我已经把 save_sample_npz 优化为只做保存。
                    # 真正的瓶颈在于大量的 np.savez_compressed。
                    
                    # 我们把 record_npz_samples 的调用放入 executor？
                    # 不行，因为它返回 stop_early 状态，我们需要这个状态。
                    
                    # 妥协方案：我们接受主线程做数据切片，但把 save_sample_npz 里的 np.savez_compressed 放到 executor。
                    # 我们需要把 executor 传递给 record_npz_samples。
                    # 但我无法修改 record_npz_samples 的签名（它在其他地方定义，我看不到）。
                    
                    # 让我们假设 record_npz_samples 就在这个文件里（通常是的，或者在 utils 里）。
                    # 我需要先找到 record_npz_samples 的定义。
                    # 注意：record_npz_samples 本身会把“写文件”交给 executor，主线程只做数据切片与提交任务。
                    # 因此这里不需要再额外套一层线程池。
                    docs_written, stop_early = record_npz_samples(
                        args,
                        batch_texts,
                        input_ids_cpu,
                        attn_mask_cpu,
                        probs_list_cpu,
                        gate_inputs_storage,
                        tokenizer,
                        aligned_indices,
                        layer_top_k,
                        global_doc_index,
                        gate_state_dir,
                        executor=executor  # 尝试传入 executor，如果函数不支持会报错，我需要先检查函数定义
                    )

                    total_docs_written += docs_written

                    # 重要修正：global_doc_index 表示“已经成功写入了多少条样本”。
                    # 旧代码用 len(batch_texts) 递增，理论上在极端情况下会出现“写入数 < batch_size”导致编号跳号。
                    # 这里改为用 docs_written 递增，保证 sample 编号与写入数量一致。
                    global_doc_index += docs_written

                    if stop_early:
                        logging.info(
                            f"[record] reached max-docs={args.max_docs}, stop recording."
                        )
                        break
                else:
                    global_doc_index += len(batch_texts)

    except Exception as e:
        logging.error(
            f"An error occurred during processing: {e}", exc_info=True
        )
    finally:
        # 等待所有异步保存任务完成
        logging.info("Waiting for pending save tasks to complete...")
        executor.shutdown(wait=True)
        
        # --- 6. 收尾：移除钩子 ---
        for h in hooks:
            h.remove()
        logging.info("Removed all hooks.")

    # --- 7. 保存摘要 ---
    if layer_counters:
        # 1) 固定文件名：routing_summary.json（方便后续脚本用 find 定位目录）
        # 2) 兼容旧文件名：*-midstate_report.json（方便历史对照）
        # 两者内容一致，只是命名不同。
        meta = {
            "run_tag": run_tag,
            "model_id": model_id,
            "model_id_slug": model_id_slug,
            "dataset_name": dataset_name,
            "dataset_name_raw": str(dataset_name_raw),
            "dataset_split": dataset_split,
            "max_samples_config": max_samples,
            "max_docs": args.max_docs,
            "batch_size": batch_size,
            "max_length": max_length,
            "record_every_n": args.record_every_n,
            "max_tokens_per_doc": args.max_tokens_per_doc,
            "gate_input_mode": args.gate_input_mode,
            "gate_input_head": args.gate_input_head,
        }

        routing_summary_path = os.path.join(gate_state_dir, "routing_summary.json")
        summary = save_summary_report(
            layer_counters,
            layer_top_k,
            layer_expert_counts,
            routing_summary_path,
            meta=meta,
        )

        # 额外写一份“带 run 信息的文件名”（便于归档/对照）
        legacy_name = f"{model_id_slug}-{dataset_name}-{dataset_split}-{max_samples_str}-midstate_report.json"
        legacy_path = os.path.join(gate_state_dir, legacy_name)
        if os.path.abspath(legacy_path) != os.path.abspath(routing_summary_path):
            save_summary_report(
                layer_counters,
                layer_top_k,
                layer_expert_counts,
                legacy_path,
                meta=meta,
            )

        logging.info(
            f"[DONE] NPZ samples: {total_docs_written}, Layers summarized: {len(summary)}"
        )
    else:
        logging.warning(
            "[DONE] No statistics were generated (maybe no data or no MoE layers found)."
        )


# --- 入口 ---
def main():
    """程序主入口。"""
    args = parse_args()

    # 优先从命令行参数加载配置，如果未提供则尝试默认路径
    config = load_exp_config(args.config_file)

    # 如果配置文件加载失败，则退出 (或者你可以选择使用默认值并继续，这里选择安全退出)
    if config is None:
        logging.error("Failed to load configuration. Exiting.")
        return

    logging.info(f"Loaded configuration from: {args.config_file or 'default location'}")
    logging.info(f"Model ID: {config.get('model_id')}")

    run_analysis(args, config)


if __name__ == "__main__":
    main()
