"""Write the dataset documentation and the Stage 2 task specification from built artifacts."""

from __future__ import annotations

import json

from wwbench.paths import DOCS, METADATA, RESULTS, ensure_dirs
from wwbench.spec import INDICATORS


def write_docs() -> None:
    ensure_dirs()
    catalog = json.loads((METADATA / "catalog.json").read_text(encoding="utf-8"))
    summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    synthetic = json.loads((RESULTS / "synthetic_eval.json").read_text(encoding="utf-8"))
    scenario = json.loads((RESULTS / "bsm1_scenario_classification.json").read_text(encoding="utf-8"))
    bundled = [item for item in catalog if item["bundled"]]
    samples = sum(item["n_samples"] or 0 for item in bundled)
    lines = [
        "# 污水处理监测 Benchmark 数据集说明",
        "",
        "本仓库把公开、可访问、许可清楚的污水处理相关数据整理成一个 Benchmark。原始文件保留在 `data/raw`，标准化结果在 `data/processed`，字段和任务关系在 `data/metadata` 与 `data/benchmark.sqlite`。",
        "",
        f"当前登记 {len(catalog)} 个数据集，其中 {len(bundled)} 个已入库，可处理样本 {samples:,} 条。样本数不把未下载的图像和禁止再分发的 SWaT 算进去。",
        "",
        "## 为什么不把所有 CSV 拼在一起",
        "",
        "各数据集的变量、单位和采样频率不同。UCI 是日尺度水质，BSM1 是 15 分钟仿真进水，丹麦和科隆数据是流量与液位，AquaFetch 是实验室回收实验。拼接会把缺失测项伪装成数值，也会把 m³/d、m³/h 和 L/s 当成同一个流量。因此每个数据集保留自己的原生表，另外提供列名对齐的标准化表；对不存在的测项留空。",
        "",
        "## 标签政策",
        "",
        "UCI 的数据文件没有故障标签。`.names` 中的 13 个类别是原作者概念聚类的产物，解析后写入 `author_cluster_*`，不写入 `operating_status`，也不作为异常检测金标准。官方缺失个数与当前文件逐列一致，见统计页。",
        "",
        "BSM1 三个文件是旱天、雨天和暴雨进水情景。雨天和暴雨文件的第一周仍接近旱天，所以文件名不是逐时刻天气标签，也不是传感器故障。",
        "",
        "SWaT A1 有正常/攻击标签，但 iTrust 条款禁止再分发，本仓库不含其数据。MD-AS-2025 标注集约 2.0 GB，本仓库只保留类别索引和来源。",
        "",
        "## 数据集",
        "",
    ]
    for item in catalog:
        lines += [
            f"### {item['name']}",
            "",
            f"- 编号：`{item['dataset_id']}`",
            f"- 类别：{item['category']}；模态：{item['modality']}；领域：{item['domain']}",
            f"- 入库：{'是' if item['bundled'] else '否'}；样本：{item['n_samples'] if item['n_samples'] is not None else '未入库'}",
            f"- 采样：{item['sampling_rate']}",
            f"- 许可：{item['license']}",
            f"- 来源：{item['url']}",
            f"- 引用：{item['citation']}",
            f"- 标签：{item['label_note']}",
            "",
            item["description"],
            "",
            "| 任务 | 支持 | 理由 |",
            "| --- | --- | --- |",
        ]
        for task in item["tasks"]:
            lines.append(f"| {task['name']} | {task['support']} | {task['rationale']} |")
        lines.append("")
    lines += [
        "## 统一列契约",
        "",
        "时序标准化列：" + "、".join(json.loads((METADATA / "schema.json").read_text(encoding="utf-8"))["column_contracts"]["timeseries"]["columns"]) + "。",
        "",
        "处理效果列：sample_id、dataset_id、treatment_method、input_parameters、influent_quality、effluent_quality、removal_efficiency。",
        "",
        "显微图像列保留在元数据中。MD-AS 索引没有本地 image_path。",
        "",
        "## 指标覆盖",
        "",
        "水厂实测、实验室、仿真和缺失是分开记录的。覆盖的指标包括：" + "、".join(label for _, label in INDICATORS) + "。",
        "",
        "溶解氧、MLSS 和总磷在已入库的水厂时序里没有实测。氨氮只存在于 BSM1 仿真进水。AquaFetch 的温度是实验温度，不是污水水温，因此不记入水温覆盖。",
        "",
        "## 复现",
        "",
        "```bash",
        "pip install -r requirements.txt",
        "python scripts/build_benchmark.py",
        "streamlit run app.py",
        "```",
        "",
        "构建脚本会重新读取原始文件、生成标准化表、SQLite、筛查结果和本说明中的模型数字。",
        "",
    ]
    (DOCS / "dataset_documentation.md").write_text("\n".join(lines), encoding="utf-8")

    uci = synthetic["uci_water_treatment"]
    metric_lines = []
    for name, metrics in uci["metrics"].items():
        latency = uci["latency"][name]
        delay = "—" if latency["mean_latency_hours"] is None else f"{latency['mean_latency_hours']:.2f} 小时"
        auc = "—" if metrics["roc_auc"] is None else f"{metrics['roc_auc']:.3f}"
        metric_lines.append(
            f"| {name} | {metrics['precision']:.3f} | {metrics['recall']:.3f} | {metrics['f1']:.3f} | {auc} | {delay} |"
        )
    task = f"""# Stage 2 任务说明：污水处理异常筛查与水质监测

## 任务

输入是 Stage 1 入库的时序监测数据，输出是相对参照期的离群点、变量偏离和一份监测报告。默认数据是 UCI 城市污水厂日监测，并用 BSM1 旱天进水做参照工况扩展。

## 输入与输出

- 输入：Benchmark 中的时序表，或用户上传的 CSV。
- 处理：参照期拟合基线，缺失坐标不贡献异常分数；子日尺度序列先减去同一时刻的参照中位数。
- 输出：离群时间、异常分数、贡献最大的变量、统计区间和 Markdown 报告。
- 模型：Z-score、Isolation Forest；在合成标签存在时增加 Random Forest。

## 评价

合成测试只注入在参照期之后的测试段。点异常幅度为参照标准差的 3.5–6 倍，短时漂移为连续 4 个点、2.5 倍标准差。Z-score 阈值固定为 3，孤立森林 contamination 固定为 0.05。随机森林的阈值只在另一套训练集合成标签上按 F1 选取。

UCI 合成测试结果：

| 模型 | Precision | Recall | F1 | ROC-AUC | 平均检出延迟 |
| --- | --- | --- | --- | --- | --- |
{chr(10).join(metric_lines)}

隔离森林的合成 F1 为 {summary['primary_f1_synthetic']:.3f}。这不是真实故障 F1。

BSM1 进水文件情景分类的宏平均 F1 为 {scenario['macro_f1']:.3f}，准确率为 {scenario['accuracy']:.3f}。标签是文件身份。由于雨天和暴雨文件前半段接近旱天，这个分数低是标签定义的结果，不是把模型调失败了。

## 不声称的事情

- 不把 UCI 概念聚类当成故障金标准。
- 不把合成标签的分数写成现场检测性能。
- 不把 SWaT 的攻击标签算作本仓库已经完成的实验。
- 不把 BSM1 进水偏离说成污水厂设备故障。
"""
    (DOCS / "task_specification.md").write_text(task, encoding="utf-8")
