# Q1_Q6_code 新目录统一说明

这份文档是当前唯一有效的"路径 + 运行口径"说明。
旧目录下的说明一律不再使用。

## 一 统一根目录

本项目的唯一根目录是 `Q1_Q6_code`，所有路径均使用**相对路径**。

克隆仓库后，进入该目录即可运行所有脚本：

```bash
cd experiments/Q1_Q6_code
```

所有脚本、数据、输出都以这里为起点。

## 二 目录结构

- Q1 Q2 Q3 Q4 Q5 Q6  
  六个问题的主脚本与绘图脚本
- Stage1_collect  
  采集数据的主脚本与配置
- docs  
  本文档与补充说明
- datasets  
  本地数据集目录（用于另一台电脑的重跑）

## 三 本地数据集来源与放置

数据集需要放入 `Q1_Q6_code/datasets/` 目录下。

需要的 6 个核心数据集为：

- gsm8k
- humaneval
- cmrc2018
- wikitext2_raw
- piqa
- winogrande_xl  

注意：如果你的数据源目录中有 `192.168.1.13` 这样的子目录，它不是数据集，不要复制。

## 四 数据集复制命令

### Windows PowerShell

```powershell
$src = "<你的数据集源目录>"
$dst = "<Q1_Q6_code路径>/datasets"
New-Item -ItemType Directory -Force -Path $dst | Out-Null

robocopy "$src\gsm8k"         "$dst\gsm8k"         /E
robocopy "$src\humaneval"     "$dst\humaneval"     /E
robocopy "$src\cmrc2018"      "$dst\cmrc2018"      /E
robocopy "$src\wikitext2_raw" "$dst\wikitext2_raw" /E
robocopy "$src\piqa"          "$dst\piqa"          /E
robocopy "$src\winogrande_xl" "$dst\winogrande_xl" /E
```

### Linux/macOS

```bash
SRC="<你的数据集源目录>"
DST="<Q1_Q6_code路径>/datasets"
mkdir -p "$DST"

cp -r "$SRC/gsm8k"         "$DST/"
cp -r "$SRC/humaneval"     "$DST/"
cp -r "$SRC/cmrc2018"      "$DST/"
cp -r "$SRC/wikitext2_raw" "$DST/"
cp -r "$SRC/piqa"          "$DST/"
cp -r "$SRC/winogrande_xl" "$DST/"
```

复制完成后，`datasets/` 下应出现 6 个子目录。

## 五 运行口径（简要）

所有脚本都在 Q1_Q6_code 目录内运行。
详细运行命令写在根目录的 `README.md`。

本地运行时的统一输入路径口径：

`datasets/<dataset_name>`

例如：

`datasets/gsm8k`

`datasets/wiki` 对应本地目录名是 `datasets/wikitext2_raw`。

## 六 Stage1 数据采集指令

采集数据的完整流程单独写在：

`docs/README_采集数据全流程.md`

## 七 关于提交到 GitHub

`datasets/` 只用于本地重跑，不应提交到 GitHub。
如果需要上传代码，请确保只提交脚本与说明文档。
