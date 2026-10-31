# 污水处理监测 Benchmark

从公开数据到可交互的污水监测应用。Stage 1 整理可处理的数据资源和任务目录，Stage 2 在这些数据上做轻量级离群筛查。

## 运行

```bash
pip install -r requirements.txt
python scripts/build_benchmark.py
streamlit run app.py
```

构建脚本会读取 `data/raw`，生成标准化表、SQLite、筛查结果和 `docs/` 中的说明。

## 已入库的数据

- UCI Water Treatment Plant：城市污水厂日监测，CC BY 4.0
- IWA BSM1 进水：旱天、雨天、暴雨仿真文件
- DHI 丹麦污水厂进水：CC BY-NC 4.0
- 科隆 Weiden 进水与管网液位：CC BY 4.0
- AquaFetch 氮、磷回收实验表：CC BY 4.0

MD-AS-2025 和 SWaT 只登记来源。前者标注集约 2 GB，后者的使用条款禁止再分发。

## 结果怎么读

异常检测页的分数是相对参照期的统计离群。模型评价页的 Precision、Recall 和 F1 来自合成异常，不是真实故障标签。UCI `.names` 中的类别是原作者的概念聚类，没有被当作金标准。
