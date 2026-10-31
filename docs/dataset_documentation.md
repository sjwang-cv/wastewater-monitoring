# 污水处理监测 Benchmark 数据集说明

本仓库把公开、可访问、许可清楚的污水处理相关数据整理成一个 Benchmark。原始文件保留在 `data/raw`，标准化结果在 `data/processed`，字段和任务关系在 `data/metadata` 与 `data/benchmark.sqlite`。

当前登记 7 个数据集，其中 5 个已入库，可处理样本 121,653 条。样本数不把未下载的图像和禁止再分发的 SWaT 算进去。

## 为什么不把所有 CSV 拼在一起

各数据集的变量、单位和采样频率不同。UCI 是日尺度水质，BSM1 是 15 分钟仿真进水，丹麦和科隆数据是流量与液位，AquaFetch 是实验室回收实验。拼接会把缺失测项伪装成数值，也会把 m³/d、m³/h 和 L/s 当成同一个流量。因此每个数据集保留自己的原生表，另外提供列名对齐的标准化表；对不存在的测项留空。

## 标签政策

UCI 的数据文件没有故障标签。`.names` 中的 13 个类别是原作者概念聚类的产物，解析后写入 `author_cluster_*`，不写入 `operating_status`，也不作为异常检测金标准。官方缺失个数与当前文件逐列一致，见统计页。

BSM1 三个文件是旱天、雨天和暴雨进水情景。雨天和暴雨文件的第一周仍接近旱天，所以文件名不是逐时刻天气标签，也不是传感器故障。

SWaT A1 有正常/攻击标签，但 iTrust 条款禁止再分发，本仓库不含其数据。MD-AS-2025 标注集约 2.0 GB，本仓库只保留类别索引和来源。

## 数据集

### UCI Water Treatment Plant

- 编号：`uci_water_treatment`
- 类别：A；模态：Time Series；领域：Municipal
- 入库：是；样本：527
- 采样：1 天（日历上有缺日，不是严格连续日序列）
- 许可：CC BY 4.0
- 来源：https://archive.ics.uci.edu/dataset/106/water+treatment+plant
- 引用：Poch, M. (1993). Water Treatment Plant [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5FS4C
- 标签：数据文件没有故障标签。.names 中的 13 个类别是原作者概念聚类的结果，不是独立确认的设备故障，因此不作为异常检测金标准。

西班牙某城市污水厂约 1990–1991 年的日尺度记录，覆盖进水、初沉、二沉和出水，以及原文件给出的去除表现。

| 任务 | 支持 | 理由 |
| --- | --- | --- |
| 水质预测 | partial | 有日尺度进出水 BOD、COD、悬浮固体和 pH，可以构造预测任务，但没有官方的预测划分，采样也有缺日。 |
| 运行状态分类 | partial | .names 给出 13 个概念聚类，类别极不平衡，而且聚类不是独立的工况巡检记录。 |
| 异常检测 | partial | 适合做无监督筛查和单独报告的合成异常测试。没有可当作金标准的故障标签。 |
| 处理效率预测 | partial | 全流程 BOD、COD、悬浮固体和沉降物有进水、出水和原文件记录的去除表现。 |
| 微生物识别 | none | 没有显微图像。 |
| 处理过程性能分析 | full | 同一天记录了工艺段水质和去除表现，可以直接做过程性能分析。 |

### IWA BSM1 进水（旱天 / 雨天 / 暴雨）

- 编号：`bsm1_influent`
- 类别：A；模态：Time Series；领域：Simulation
- 入库：是；样本：4035
- 采样：15 分钟
- 许可：IWA 任务组公开提供的基准进水文件；使用时引用技术报告。不是开源软件许可证。
- 来源：https://iwa-mia.org/benchmarking/
- 引用：IWA Task Group on Benchmarking of Control Strategies for WWTPs. BSM1 influent files Inf_dry_2006, Inf_rain_2006 and Inf_strm_2006. https://iwa-mia.org/benchmarking/
- 标签：文件名给出进水情景（旱天、雨天、暴雨）。雨天和暴雨文件的第一周仍接近旱天，因此文件级标签不是逐时刻天气标签。这些文件不含传感器故障标签，也不含出水。

BSM1 基准的 14 天动态进水，包含 ASM1 组分和流量。溶解氧、硝酸盐、自养菌和颗粒产物在进水中按基准定义为 0，不当作实测溶解氧。

| 任务 | 支持 | 理由 |
| --- | --- | --- |
| 水质预测 | partial | 15 分钟仿真进水可以做负荷预测。它不是实测出水水质。 |
| 运行状态分类 | partial | 三个进水文件是明确的旱天、雨天、暴雨情景，但文件级标签不等于逐时刻天气，也不是厂内控制状态。 |
| 异常检测 | partial | 可以用旱天作为参照工况做偏离筛查。进水文件本身没有故障标签；带故障的是另一个基准 BSM1_LT，未纳入本仓库。 |
| 处理效率预测 | none | 只有进水，没有出水，不能计算去除率。 |
| 微生物识别 | none | 没有图像。 |
| 处理过程性能分析 | partial | 这是控制策略基准的标准进水，但本仓库没有模拟出水和控制输出。 |

### DHI 丹麦污水厂进水流量

- 编号：`dhi_wwtp_inflow`
- 类别：A；模态：Time Series；领域：Municipal
- 入库：是；样本：11257
- 采样：1 小时
- 许可：CC BY-NC 4.0
- 来源：https://doi.org/10.5281/zenodo.15301164
- 引用：DHI (2025). Inflow to a Waste Water Treatment Plant: observations and processed data [Data set]. Zenodo. https://doi.org/10.5281/zenodo.15301164
- 标签：没有异常或故障标签。气温、气压和降水来自气象站，不是污水水温。

约 15 个月的污水厂进水流量，以及丹麦气象局的降水、气压、辐射、湿度和气温。厂站位置已匿名。

| 任务 | 支持 | 理由 |
| --- | --- | --- |
| 水质预测 | partial | 可以预测进水流量，文件中没有污染物浓度。 |
| 运行状态分类 | none | 没有工况或状态标签。 |
| 异常检测 | partial | 只有无监督筛查可用，没有故障标签。 |
| 处理效率预测 | none | 没有进出水水质。 |
| 微生物识别 | none | 没有图像。 |
| 处理过程性能分析 | partial | 能分析进水流量和气象驱动，不能分析处理效率。 |

### 科隆 Weiden 污水厂进水与管网液位

- 编号：`koeln_weiden`
- 类别：A；模态：Time Series；领域：Municipal
- 入库：是；样本：105120
- 采样：15 分钟；交互式检测使用小时均值
- 许可：CC BY 4.0
- 来源：https://doi.org/10.5281/zenodo.6992694
- 引用：Sonnenschein, B., & Ziel, F. (2022). Wastewater Treatment Plant Inflow Rate, Network Water Levels and Rain Rate for Koeln Weiden, Germany [Data set]. Zenodo. https://doi.org/10.5281/zenodo.6992694
- 标签：没有异常标签。雨量是气象驱动，不默认送入异常检测，避免把降雨事件直接当成设备故障。

科隆 Weiden 污水系统的进水流量、三处管网液位、调蓄液位和雨强。原始记录为 15 分钟。

| 任务 | 支持 | 理由 |
| --- | --- | --- |
| 水质预测 | partial | 可以预测进水流量和管网液位，没有污染物浓度。 |
| 运行状态分类 | none | 没有工况标签。 |
| 异常检测 | partial | 多传感器时序可做无监督筛查，没有故障标签。 |
| 处理效率预测 | none | 没有去除率。 |
| 微生物识别 | none | 没有图像。 |
| 处理过程性能分析 | partial | 能分析降雨、管网液位和进水的水力关系，不能分析生化处理效果。 |

### AquaFetch 氮磷回收实验

- 编号：`aquafetch_nutrient_recovery`
- 类别：D；模态：Tabular；领域：Laboratory
- 入库：是；样本：714
- 采样：每行一次实验，不是水厂时序
- 许可：CC BY 4.0
- 来源：https://doi.org/10.5281/zenodo.14586314
- 引用：Abbas, A. (2024). Experimental data for P and N recovery from wastewater [Data set]. Zenodo. https://doi.org/10.5281/zenodo.14586314. 实验来源：Leng 等, Journal of Water Process Engineering, 2024, https://doi.org/10.1016/j.jwpe.2024.104896
- 标签：目标是氮回收率或磷回收率，不是水厂故障标签。

实验室吸附回收氮、磷的实验条件与回收率。用于处理效率任务，不能代表市政污水厂在线监测。

| 任务 | 支持 | 理由 |
| --- | --- | --- |
| 水质预测 | none | 不是水厂水质时间序列。 |
| 运行状态分类 | none | 没有运行状态标签。 |
| 异常检测 | none | 实验表不适合作为水厂异常检测数据。 |
| 处理效率预测 | full | 实验条件与氮、磷回收率成对出现，可以直接做处理效率预测。范围限于实验室回收实验。 |
| 微生物识别 | none | 没有图像。 |
| 处理过程性能分析 | partial | 能分析实验参数和回收率的关系，不能分析全流程污水厂。 |

### MD-AS-2025 活性污泥微型动物图像

- 编号：`mdas_2025`
- 类别：B；模态：Image；领域：Laboratory / Municipal
- 入库：否；样本：未入库
- 采样：显微静态图像；10× 物镜，1824×1216 像素
- 许可：CC BY 4.0
- 来源：https://doi.org/10.5281/zenodo.15210103
- 引用：Pang, Z. 等 (2025). Image Dataset of Microfauna in Activated Sludge. Scientific Data. https://doi.org/10.1038/s41597-025-06228-6
- 标签：标注集含边界框，类别为有壳虫、蛭态轮虫、单巢轮虫、线虫和腹毛类。没有分割掩膜。

4,000 张带框全图和 14,257 张单目标裁剪图，服务微生物识别。不能用于 COD 预测。

| 任务 | 支持 | 理由 |
| --- | --- | --- |
| 水质预测 | none | 图像不能直接预测 COD。 |
| 运行状态分类 | none | 没有工艺运行状态标签。 |
| 异常检测 | none | 没有过程异常标签。 |
| 处理效率预测 | none | 没有去除率。 |
| 微生物识别 | full | 有五类微型动物和边界框，支持分类与目标检测。图像未打包进本仓库。 |
| 处理过程性能分析 | none | 没有同步的工艺水质表。 |

### SWaT 水处理测试平台（A1，2015-12）

- 编号：`swat_a1`
- 类别：C；模态：Time Series；领域：Industrial
- 入库：否；样本：未入库
- 采样：A1 为 11 天连续运行记录；采样间隔以获批数据包为准
- 许可：iTrust 使用条款：须署名、不得向他人再分发
- 来源：https://www.sutd.edu.sg/itrust/request-for-datasets/
- 引用：iTrust, Centre for Research in Cyber Security, Singapore University of Technology and Design. SWaT dataset. https://www.sutd.edu.sg/itrust/itrust-labs/datasets/dataset-characteristics/swat/
- 标签：A1 含正常与攻击标签：7 天正常运行、4 天攻击情景、51 个传感器和执行器通道、41 次攻击。这是水处理测试平台，不是市政污水处理厂。

工业控制系统安全研究常用的水处理测试平台数据。目录中保留它，是为了说明带真实攻击标签的公开资源存在，但不能放进可再分发的仓库。

| 任务 | 支持 | 理由 |
| --- | --- | --- |
| 水质预测 | none | 测试平台不是市政污水水质监测，本仓库也没有获准使用其传感器数值。 |
| 运行状态分类 | partial | 正常与攻击可以作为状态分类，但是网络攻击情景，不是污水工艺工况。 |
| 异常检测 | full | A1 提供正常/攻击标签。数据不能再分发，因此这里只登记任务适用性。 |
| 处理效率预测 | none | 没有污染物去除率。 |
| 微生物识别 | none | 没有显微图像。 |
| 处理过程性能分析 | none | 不能代表市政污水处理性能。 |

## 统一列契约

时序标准化列：dataset_id、sample_id、timestamp、pH、DO、COD、BOD、NH4_N、TN、TP、MLSS、SS、conductivity、flow_rate、temperature、operating_status。

处理效果列：sample_id、dataset_id、treatment_method、input_parameters、influent_quality、effluent_quality、removal_efficiency。

显微图像列保留在元数据中。MD-AS 索引没有本地 image_path。

## 指标覆盖

水厂实测、实验室、仿真和缺失是分开记录的。覆盖的指标包括：pH、溶解氧、COD、BOD、氨氮、总氮、总磷、MLSS、悬浮固体、流量、水温、电导率。

溶解氧、MLSS 和总磷在已入库的水厂时序里没有实测。氨氮只存在于 BSM1 仿真进水。AquaFetch 的温度是实验温度，不是污水水温，因此不记入水温覆盖。

## 复现

```bash
pip install -r requirements.txt
python scripts/build_benchmark.py
streamlit run app.py
```

构建脚本会重新读取原始文件、生成标准化表、SQLite、筛查结果和本说明中的模型数字。
