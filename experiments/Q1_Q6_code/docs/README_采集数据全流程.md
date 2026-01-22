# Stage1 数据采集全流程说明（新目录口径）

本说明仅对应当前新目录结构：
`Q1_Q6_code/Stage1_collect`

目标：把真实文本喂给 Qwen1.5-MoE-A2.7B，生成 `sample-*.npz` 与 `routing_summary.json`，供 Q1-Q6 后续使用。

## 一 采集代码所在位置

采集代码已放入：

`Q1_Q6_code/Stage1_collect`

包含必要文件：

- `transformer_qwen_moe_state.py` 主采集脚本
- `config/` 各数据集配置 JSON
- `config.py` 配置加载
- `load.py` 数据加载
- `utils.py` MoE 层定位
- `fix_bnb.py` 兼容修复

## 二 数据集位置

统一放在：

`Q1_Q6_code/datasets`

需要的 6 个数据集目录：

- `datasets/gsm8k`
- `datasets/humaneval`
- `datasets/cmrc2018`
- `datasets/wikitext2_raw`
- `datasets/piqa`
- `datasets/winogrande_xl`

## 三 配置文件已改为相对路径

配置文件在：

`Q1_Q6_code/Stage1_collect/config/`

**配置文件已经修改为相对路径**，无需手动修改。

当前配置示例（以 gsm8k 为例）：

```json
{
  "model_id": "Qwen/Qwen1.5-MoE-A2.7B",
  "dataset_config": {
    "source": "local",
    "name": "gsm8k",
    "local_path": "../datasets/gsm8k/gsm8k.txt",
    "split": "train",
    "text_field": "text"
  },
  "max_samples": -1,
  "batch_size": 4,
  "max_length": 2048
}
```

说明：
- `model_id`: 使用 HuggingFace 模型名，会自动下载；如需本地模型，改为本地路径
- `local_path`: 使用相对路径 `../datasets/...`，从 Stage1_collect 目录出发

## 四 运行命令（逐个数据集）

进入采集目录：

```bash
cd Stage1_collect
```

逐个运行：

```bash
python transformer_qwen_moe_state.py --config_file config/qwen_gsm8k.json --run_tag $(date +%Y-%m-%d)
python transformer_qwen_moe_state.py --config_file config/qwen_humaneval.json --run_tag $(date +%Y-%m-%d)
python transformer_qwen_moe_state.py --config_file config/qwen_cmrc2018.json --run_tag $(date +%Y-%m-%d)
python transformer_qwen_moe_state.py --config_file config/qwen_config.json --run_tag $(date +%Y-%m-%d)
python transformer_qwen_moe_state.py --config_file config/qwen_piqa.json --run_tag $(date +%Y-%m-%d)
python transformer_qwen_moe_state.py --config_file config/qwen_winogrande.json --run_tag $(date +%Y-%m-%d)
```

说明：
`run_tag` 会写进输出目录名，防止覆盖历史采集。

## 五 输出位置与检查方式

输出目录固定在：

`Stage1_collect/results/`

每个数据集会生成一个子目录，目录内应包含：

- `sample-0000000.npz` 等文件
- `routing_summary.json`

检查命令（Linux/macOS）：

```bash
ls Stage1_collect/results/
find Stage1_collect/results -name "routing_summary.json"
```

检查命令（Windows PowerShell）：

```powershell
dir Stage1_collect\results
Get-ChildItem -Path Stage1_collect\results -Recurse -Filter routing_summary.json
```

只要 `routing_summary.json` 存在，就说明采集成功。
