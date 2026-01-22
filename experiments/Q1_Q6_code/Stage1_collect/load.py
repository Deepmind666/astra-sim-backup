# -*- coding: utf-8 -*-
"""
Stage-1 数据加载模块（给 MoE 采集脚本用）
============================================================

你可以把它理解成“把各种数据集，统一变成一个 List[str] 文本列表”的工具。

为什么要单独写一个 load.py？
---------------------------
我们的 Stage-1 采集脚本会做两件事：
1) 把文本喂给 Qwen-MoE 做前向推理；
2) 用 hook 把每个 token 的路由信息（top-k 专家索引/概率等）记录到 .npz。

因此：**文本的“切分方式”会直接影响采集质量和耗时**。

最常见的坑：把一个 `*.txt` 文件交给 HuggingFace 的 `load_dataset("text")` 去读。
它默认是“按行读取”，也就是：
  - 每一行都是一条样本；
  - 空行会被丢掉（或者过滤掉）；
  - 如果你的 `*.txt` 里用 `<|endoftext|>` 做分隔符，那这一行也会被当成一条样本。

后果：
  - GSM8K/MBPP 这类“多行结构”的数据会被拆碎成几十万条短样本；
  - 采集会变慢很多（样本数暴涨）；
  - 会出现大量只有 1 个 token 的“噪声样本”，污染路由统计与后续聚类。

为了解决这个问题，我们在读取本地 txt 时提供了**分块读取**：
  - 如果文件里出现 `<|endoftext|>` 分隔符，就按它来切样本（一个块 = 一个样本）；
  - 否则才退化为“按行读取”。

这样可以让 “GSM8K 一题（含推理与答案）” 作为一个完整样本输入模型，
更符合我们想采集的真实推理轨迹。
"""

import logging
import os
import sys
from typing import List, Tuple, Optional

from utils import print_device_summary, disable_qwen2moe_aux_loss

from datasets import load_dataset, load_from_disk

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)


def load_model_and_tokenizer(
    model_id: str,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """加载指定的模型和分词器。"""
    logging.info(f"Loading model: {model_id} with CPU Offload (to fix OOM without bitsandbytes)")
    
    # 策略调整：既然 bitsandbytes 与环境冲突，我们改用 "显存+内存" 混合加载。
    # 我们限制 GPU 只用 20GB (留余量给计算)，剩下的层自动放到 CPU 内存中。
    # 这样既不需要量化库，也不会爆显存。
    max_memory = {0: "20GiB", "cpu": "100GiB"}

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="auto",      # 自动分配
        max_memory=max_memory,  # 显存上限控制
        offload_folder="offload", # 必要的临时交换目录
        trust_remote_code=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
        logging.info(
            f"Set tokenizer.pad_token to tokenizer.eos_token ({tokenizer.eos_token})"
        )

    print_device_summary(model)
    disable_qwen2moe_aux_loss()  # 特定于 Qwen 的处理

    return model, tokenizer


def load_corpus_data(
    dataset_config: dict, max_samples: Optional[int]
) -> List[str]:
    """加载数据集。"""
    corpus = _load_data(dataset_config, max_samples)
    logging.info(f"{len(corpus)} samples loaded for analysis.")

    logging.info("Sample text:")
    for i in range(0, min(3, len(corpus))):
        preview = corpus[i][:100].replace("\n", " ")
        suffix = "..." if len(corpus[i]) > 100 else ""
        logging.info(f"  [{i}] {preview}{suffix}")

    if not corpus:
        logging.warning("Dataset is empty.")

    return corpus


def _load_local_text_file_as_samples(
    *,
    file_path: str,
    max_samples: Optional[int],
    delimiter_line: str = "<|endoftext|>",
) -> List[str]:
    """
    把一个本地 `*.txt` 文件读取成 “样本列表”。

    设计目标（非常重要）：
    - 如果文件包含 `<|endoftext|>` 分隔符：按“块”读取（一个块 = 一个样本）。
      这样能把一整题（多行）保留为一个整体输入，避免被拆成很多短句。
    - 如果文件不包含分隔符：退化为“按行读取”（兼容纯一行一条样本的文件）。

    参数说明：
    - file_path：txt 文件路径
    - max_samples：最多取多少条样本（None 表示不限制）
    - delimiter_line：分隔符行内容，默认 `<|endoftext|>`
    """
    samples: List[str] = []

    # 1) 先一次性读取所有行（大多数数据集 txt 都不算太大，这样实现最直观）
    #    如果以后出现超大文本（GB 级），再改成流式逐行处理也不难。
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 2) 判断是否存在分隔符（只要出现过一次，我们就启用“按块切分”）
    has_delimiter = any(line.strip() == delimiter_line for line in lines)

    if has_delimiter:
        # ------------------------------
        # 分块模式：遇到 <|endoftext|> 就结束当前样本
        # ------------------------------
        buf: List[str] = []
        for raw in lines:
            line = raw.rstrip("\n")
            if line.strip() == delimiter_line:
                block = "\n".join(buf).strip()
                if block:
                    samples.append(block)
                    if max_samples is not None and len(samples) >= max_samples:
                        break
                buf = []
                continue
            buf.append(line)

        # 文件末尾如果没有分隔符，也要把最后一个块收进去
        if (max_samples is None or len(samples) < max_samples) and buf:
            block = "\n".join(buf).strip()
            if block:
                samples.append(block)

        return samples

    # ------------------------------
    # 退化模式：按行读取（过滤空行）
    # ------------------------------
    for raw in lines:
        s = raw.strip()
        if not s:
            continue
        samples.append(s)
        if max_samples is not None and len(samples) >= max_samples:
            break

    return samples


def _load_data(dataset_config, max_samples=None, seed=42):
    """
    从 HuggingFace 或本地路径加载数据，并统一返回 List[str]。

    注意：这里的 `max_samples` 指“最多加载多少条样本（文本块）”，
    它直接决定 Stage-1 要跑多少个输入样本，进而决定采集时长。
    """
    source = dataset_config.get("source", "huggingface")

    if source == "huggingface":
        try:
            dataset = load_dataset(
                dataset_config["name"],
                dataset_config.get("config"),
                split=dataset_config.get("split", "test"),
                trust_remote_code=True,
            )
            text_field = dataset_config.get("text_field", "text")
            prompt_template = dataset_config.get("prompt_template", "{text}")

            # Deterministic shuffle for train splits
            if "train" in dataset_config.get("split", ""):
                dataset = dataset.shuffle(seed=seed)

            if dataset_config["name"] == "wikitext":

                def _keep(example):
                    s = (example.get("text") or "").strip()
                    return bool(s) and not (
                        s.startswith("=") and s.endswith("=")
                    )

                dataset = dataset.filter(_keep)

            n = (
                min(len(dataset), max_samples)
                if max_samples is not None
                else len(dataset)
            )
            if n <= 0:
                raise ValueError("No valid samples found.")
            dataset = dataset.select(range(n))

            formatted_texts = []
            for row in dataset:
                try:
                    format_args = dict(row)
                    format_args["text"] = row[text_field]
                    formatted = prompt_template.format(**format_args)
                    formatted_texts.append(formatted)
                except (KeyError, AttributeError):
                    formatted_texts.append(row.get(text_field, ""))

            return formatted_texts

        except Exception as e:
            print(f"Error loading HuggingFace dataset: {e}", file=sys.stderr)

    if source == "local":
        # 关键约定（避免踩坑）：
        # - local_path 才是“真正的本地路径”（文件或目录）
        # - name 只是数据集名字/标识（有时也会被写成路径，但我们不依赖它）
        local_path = dataset_config.get("local_path") or dataset_config.get("name")
        if not local_path:
            raise ValueError(
                "Local dataset path not specified in config (key: 'local_path')."
            )

        local_path = os.path.expanduser(str(local_path))
        logging.info(f"Loading local dataset from: {local_path}")

        try:
            # 1) 如果是单个 txt 文件：优先按“块”读取，避免按行拆碎
            if os.path.isfile(local_path):
                # 这里不使用 `load_dataset("text")`，原因见文件头注释。
                return _load_local_text_file_as_samples(
                    file_path=local_path,
                    max_samples=max_samples,
                )

            # 2) 如果是目录：优先尝试 load_from_disk（HF save_to_disk 格式）
            dataset = load_from_disk(local_path)

            # 2.1) DatasetDict（多 split）需要选择 split
            if hasattr(dataset, "keys") and callable(getattr(dataset, "keys", None)):
                split = dataset_config.get("split", "train")
                if split in dataset:
                    dataset = dataset[split]
                else:
                    # 如果配置的 split 不存在，就取第一个 split，避免直接报错卡死
                    first = list(dataset.keys())[0]
                    logging.warning(
                        f"Split '{split}' not found, fallback to '{first}'"
                    )
                    dataset = dataset[first]

            # 2.2) wikitext 的“标题行”过滤：形如 "====== xxx ======" 的行一般不是正文
            if "wikitext" in str(local_path).lower() or dataset_config.get(
                "config"
            ) in {"wikitext-103-raw-v1", "wikitext-2-raw-v1"}:

                def _keep(example):
                    s = (example.get("text") or "").strip()
                    return bool(s) and not (s.startswith("=") and s.endswith("="))

                dataset = dataset.filter(_keep)

            # 2.3) 截断到 max_samples（避免一次性把超大数据集全读进来）
            n = min(len(dataset), max_samples) if max_samples is not None else len(dataset)
            if n <= 0:
                raise ValueError("No valid samples found after filtering.")
            dataset = dataset.select(range(n))

            # 2.4) 从数据集中取出文本字段，并支持可选的 prompt_template
            text_field = dataset_config.get("text_field", "text")
            prompt_template = dataset_config.get("prompt_template", "{text}")

            formatted_texts: List[str] = []
            for row in dataset:
                try:
                    # 让模板里既能用 {text}，也能用 {question}/{answer} 等原始字段
                    format_args = dict(row)
                    format_args["text"] = row.get(text_field, "")
                    formatted_texts.append(prompt_template.format(**format_args))
                except Exception:
                    formatted_texts.append(row.get(text_field, ""))

            return formatted_texts

        except Exception as e:
            print(f"Error loading local dataset: {e}", file=sys.stderr)

    raise ValueError(
        "Failed to load dataset from both HuggingFace and local sources."
    )
