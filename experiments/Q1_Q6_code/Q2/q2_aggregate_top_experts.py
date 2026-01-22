# 解释：导入标准库 json，用于读取/写入 JSON 文件。
import json
# 解释：从 pathlib 导入 Path，用来进行跨平台路径拼接和文件操作。
from pathlib import Path

# 解释：导入 matplotlib 的绘图接口，后面要画柱状图。
import matplotlib.pyplot as plt
# 解释：导入 numpy，用于数组计算和矩阵运算。
import numpy as np


# 解释：定义 score1 指标的计算函数。
# 解释：p_task_e 是每个任务在每个专家上的概率分布矩阵，形状 [T, E]。
# 解释：n_tasks 是任务数，用于计算均匀先验 1 / n_tasks。
def _score1(p_task_e: np.ndarray, n_tasks: int) -> np.ndarray:
    # 解释：这一行是原作者备注，说明 p_task_e 的形状是 [T, E]，并且每个任务内部已归一化。
    # p_task_e: [T, E] normalized per task
    # 解释：按列求和得到每个专家的总概率质量；用 maximum 加上 1e-12 防止除零。
    denom = np.maximum(p_task_e.sum(axis=0, keepdims=True), 1e-12)
    # 解释：计算 P(task | expert) = P(task, expert) / P(expert)，这里用任务内归一化矩阵近似。
    p_task_given_e = p_task_e / denom
    # 解释：均匀先验 P(task) = 1 / 任务数，表示“如果没有偏好，任务被选中的概率”。
    p_task = 1.0 / n_tasks
    # 解释：score1 定义为 log(P(task|expert) / P(task))，衡量“某专家对任务的偏好强度”。
    return np.log(p_task_given_e / p_task)


# 解释：读取单个任务的 summary JSON 文件。
def _load_summary(path: Path):
    # 解释：用 UTF-8 打开文件，确保中文字段不会乱码。
    with path.open("r", encoding="utf-8") as f:
        # 解释：json.load 直接把文件内容解析成 Python 字典。
        return json.load(f)


# 解释：从 delta（偏好增益）里挑出 topN 专家，同时满足最小支持度。
def _top_list(delta: np.ndarray, support: np.ndarray, topn: int, min_support: float):
    # 解释：argsort 会按升序排序索引；[::-1] 反转成降序，越大越靠前。
    order = np.argsort(delta)[::-1]
    # 解释：用列表收集最终通过筛选的专家编号。
    out = []
    # 解释：遍历排序后的专家编号，从高到低逐个检查。
    for e in order:
        # 解释：如果该专家的支持度太小（出现太少），就跳过。
        if support[e] < min_support:
            continue
        # 解释：把专家编号转成 int，避免 numpy 类型影响 JSON 序列化。
        out.append(int(e))
        # 解释：如果已经收集到 topn 个专家，就提前结束循环。
        if len(out) >= topn:
            break
    # 解释：返回最终筛选出的专家编号列表。
    return out


# 解释：主函数，串起读取、计算、输出和绘图。
def main():
    # 解释：base 指向 Q2 文件夹的上一级，用来拼接 seed 子目录。
    base = Path(__file__).resolve().parent.parent
    # 解释：这里固定 3 个随机种子，与实验记录保持一致。
    seeds = [11, 22, 33]
    # 解释：用于存放每个 seed 的 summary 内容。
    summaries = []
    # 解释：逐个 seed 读取对应的 task_specialization_summary.json。
    for s in seeds:
        # 解释：路径规则是 base/seedXX/task_specialization_summary.json。
        p = base / f"seed{s}" / "task_specialization_summary.json"
        # 解释：若文件不存在，说明实验没跑完或路径错误，直接报错。
        if not p.exists():
            raise FileNotFoundError(p)
        # 解释：读取 JSON 并存入 summaries。
        summaries.append(_load_summary(p))

    # 解释：任务列表（字符串数组），取第一个 seed 的结果即可。
    tasks = summaries[0]["tasks"]
    # 解释：任务数量 T。
    n_tasks = len(tasks)
    # 解释：专家数量 E，取 p_global_soft 的长度。
    n_experts = len(summaries[0]["p_global_soft"])

    # 解释：下面把不同 seed 的数组堆叠成 [S, T, E]，便于算均值/方差。
    # stack per-seed arrays
    # 解释：p_soft 是 soft 口径（概率分布）下的任务-专家矩阵。
    p_soft = np.stack([np.array(s["p_soft"]) for s in summaries], axis=0)  # [S, T, E]
    # 解释：p_hard 是 hard 口径（Top-1 频次）下的任务-专家矩阵。
    p_hard = np.stack([np.array(s["p_hard"]) for s in summaries], axis=0)
    # 解释：p_soft_shuffle 是 soft 口径的置乱基线，用于消除任务标签偏差。
    p_soft_shuffle = np.stack([np.array(s["p_soft_shuffle_mean"]) for s in summaries], axis=0)
    # 解释：p_hard_shuffle 是 hard 口径的置乱基线。
    p_hard_shuffle = np.stack([np.array(s["p_hard_shuffle_mean"]) for s in summaries], axis=0)

    # 解释：score1 的增益定义为“真实 - 置乱”，越大说明任务专用性越强。
    # score1 delta (real - shuffle)
    # 解释：存储 soft 口径的 delta 列表（每个 seed 一个）。
    delta_soft = []
    # 解释：存储 hard 口径的 delta 列表（每个 seed 一个）。
    delta_hard = []
    # 解释：遍历每个 seed，分别计算 delta。
    for si in range(len(seeds)):
        # 解释：真实 soft 口径的 score1。
        score1_soft = _score1(p_soft[si], n_tasks)
        # 解释：置乱 soft 口径的 score1（基线）。
        score1_soft_b = _score1(p_soft_shuffle[si], n_tasks)
        # 解释：delta = 真实 - 基线，表示超出随机的任务专用性。
        delta_soft.append(score1_soft - score1_soft_b)

        # 解释：真实 hard 口径的 score1。
        score1_hard = _score1(p_hard[si], n_tasks)
        # 解释：置乱 hard 口径的 score1（基线）。
        score1_hard_b = _score1(p_hard_shuffle[si], n_tasks)
        # 解释：delta = 真实 - 基线。
        delta_hard.append(score1_hard - score1_hard_b)

    # 解释：把每个 seed 的 delta 堆叠成 [S, T, E]。
    delta_soft = np.stack(delta_soft, axis=0)  # [S, T, E]
    # 解释：hard 口径同样堆叠。
    delta_hard = np.stack(delta_hard, axis=0)

    # 解释：跨 seed 求均值，得到稳定的任务专用性估计。
    delta_soft_mean = delta_soft.mean(axis=0)
    # 解释：跨 seed 求标准差，衡量稳定性。
    delta_soft_std = delta_soft.std(axis=0)
    # 解释：hard 口径的均值。
    delta_hard_mean = delta_hard.mean(axis=0)
    # 解释：hard 口径的标准差。
    delta_hard_std = delta_hard.std(axis=0)

    # 解释：support 表示专家在该任务中的平均出现概率/频次。
    # support (mean per task)
    # 解释：soft 口径 support。
    support_soft = p_soft.mean(axis=0)
    # 解释：hard 口径 support。
    support_hard = p_hard.mean(axis=0)

    # 解释：topn 表示每个任务最多保留多少个专家。
    topn = 10
    # 解释：min_support 是最小支持度阈值，太小的专家不计入候选。
    min_support = 0.002

    # 解释：把 delta/support 结果整理成表格字典，方便写 JSON 和 Markdown。
    def build_table(delta_mean, delta_std, support, label):
        # 解释：out 以任务名为 key，存该任务的专家列表。
        out = {}
        # 解释：遍历任务索引和任务名。
        for ti, task in enumerate(tasks):
            # 解释：先筛出该任务的 top 专家列表。
            experts = _top_list(delta_mean[ti], support[ti], topn, min_support)
            # 解释：rows 保存当前任务的专家信息字典。
            rows = []
            # 解释：遍历每个通过筛选的专家编号。
            for e in experts:
                # 解释：将该专家的 delta/support 数值打包成字典。
                rows.append({
                    # 解释：专家编号。
                    "expert": int(e),
                    # 解释：delta 的平均值。
                    "delta_mean": float(delta_mean[ti, e]),
                    # 解释：delta 的标准差。
                    "delta_std": float(delta_std[ti, e]),
                    # 解释：该专家在该任务的平均支持度。
                    "support_mean": float(support[ti, e]),
                })
            # 解释：把该任务的 rows 放入输出字典。
            out[task] = rows
        # 解释：返回整理好的表格数据。
        return out

    # 解释：构建 soft 口径的 top 专家表。
    top_soft = build_table(delta_soft_mean, delta_soft_std, support_soft, "soft")
    # 解释：构建 hard 口径的 top 专家表。
    top_hard = build_table(delta_hard_mean, delta_hard_std, support_hard, "hard")

    # 解释：输出目录统一放在 base/analysis。
    out_dir = base / "analysis"
    # 解释：如果目录不存在就创建，parents=True 会递归创建。
    out_dir.mkdir(parents=True, exist_ok=True)
    # 解释：写出 soft 口径的 JSON 文件。
    with (out_dir / "top_experts_soft_score1_delta_mean.json").open("w", encoding="utf-8") as f:
        # 解释：ensure_ascii=False 保留中文，indent=2 方便阅读。
        json.dump(top_soft, f, ensure_ascii=False, indent=2)
    # 解释：写出 hard 口径的 JSON 文件。
    with (out_dir / "top_experts_hard_score1_delta_mean.json").open("w", encoding="utf-8") as f:
        # 解释：同样用 UTF-8 输出，保证 GitHub 不乱码。
        json.dump(top_hard, f, ensure_ascii=False, indent=2)

    # 解释：下面生成一个 Markdown 摘要，方便直接贴到 issue。
    # markdown summary for issue use
    # 解释：md_lines 用来存储每一行文字。
    md_lines = []
    # 解释：先写 soft 再写 hard，两种口径分开。
    for label, data in [("soft", top_soft), ("hard", top_hard)]:
        # 解释：每种口径先写一个标题行。
        md_lines.append(f"Top experts ({label}) by score1 delta mean (min_support={min_support})")
        # 解释：逐个任务写明该任务的 top 专家。
        for task in tasks:
            # 解释：任务名作为小标题行。
            md_lines.append(f"{task}:")
            # 解释：取出该任务对应的专家列表。
            rows = data[task]
            # 解释：若为空，说明没有任何专家超过 min_support。
            if not rows:
                md_lines.append("  - (no experts meet min_support)")
                # 解释：继续下一个任务。
                continue
            # 解释：逐个专家输出 delta 均值/方差和支持度。
            for r in rows:
                # 解释：这里把专家编号、delta_mean、delta_std 和 support 格式化成一行字符串。
                md_lines.append(
                    f"  - expert {r['expert']}: delta_mean={r['delta_mean']:.4f} +/- {r['delta_std']:.4f}, support={r['support_mean']:.4f}"
                )
        # 解释：不同口径之间插入空行分隔。
        md_lines.append("")
    # 解释：把所有行拼成一个字符串并写入文件。
    (out_dir / "top_experts_summary.md").write_text("\n".join(md_lines), encoding="utf-8")

    # 解释：下面开始绘图，把 top 专家画成柱状图。
    # plots
    # 解释：定义绘图函数，输入是任务->专家列表和输出文件名。
    def plot_top_experts(data, fname):
        # 解释：创建 2x3 子图布局，figsize 控制图尺寸。
        fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
        # 解释：把二维 axes 展平为一维，便于索引。
        axes = axes.ravel()
        # 解释：遍历任务列表，逐个画柱状图。
        for idx, task in enumerate(tasks):
            # 解释：取当前子图。
            ax = axes[idx]
            # 解释：取出当前任务的专家列表。
            rows = data[task]
            # 解释：如果没有专家，就只显示标题并关闭坐标轴。
            if not rows:
                ax.set_title(task)
                ax.axis("off")
                continue
            # 解释：把专家编号转换成字符串，作为 x 轴标签。
            labels = [str(r["expert"]) for r in rows]
            # 解释：取 delta_mean 作为柱子的高度。
            vals = [r["delta_mean"] for r in rows]
            # 解释：绘制柱状图，x 轴用 range(len(vals))。
            ax.bar(range(len(vals)), vals)
            # 解释：设置 x 轴刻度位置。
            ax.set_xticks(range(len(vals)))
            # 解释：设置 x 轴标签并旋转，避免重叠。
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
            # 解释：设置子图标题为任务名。
            ax.set_title(task)
            # 解释：y 轴标签说明柱子含义。
            ax.set_ylabel("delta_mean")
        # 解释：整张图的主标题用文件名去掉 .png 得到。
        fig.suptitle(fname.replace(".png", ""))
        # 解释：保存图像到输出目录。
        fig.savefig(out_dir / fname)
        # 解释：关闭图像，释放内存。
        plt.close(fig)

    # 解释：绘制 soft 口径的柱状图。
    plot_top_experts(top_soft, "top_experts_soft_bar.png")
    # 解释：绘制 hard 口径的柱状图。
    plot_top_experts(top_hard, "top_experts_hard_bar.png")


# 解释：Python 入口判断，确保脚本直接运行时才会执行 main。
if __name__ == "__main__":
    # 解释：调用主函数。
    main()
