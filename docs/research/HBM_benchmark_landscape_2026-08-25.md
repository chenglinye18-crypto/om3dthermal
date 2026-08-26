# HBM容量、带宽与DAC baseline证据地图

日期：2026-08-25

面向：Orthogonal M3D Memory DAC Research Lead
目的：用公开的一手标准、厂商规格与芯片论文补充DreamRAM，建立可复现、不过度宣称的HBM/Orthogonal-Si比较口径。

## 1. Executive answer

当前项目不能再把“DreamRAM输出”直接等同于“真实HBM或真实Si DRAM benchmark”。最稳健的做法是保留DreamRAM作为组织、能量分解和refresh的分析引擎，同时增加三层外部证据：

1. **商品GPU现实锚点**：A100、H100、H200、MI300X、MI325X、MI355X、Blackwell Ultra，回答市场上整颗加速器实际公开了多少HBM容量和峰值带宽。
2. **HBM硅论文锚点**：HBM2E 16 GB/640 GB/s、HBM3 16 GB/1024 GB/s、HBM3 24 GB/896 GB/s，回答单个真实HBM器件做到过什么。
3. **架构研究锚点**：MOSAIC和DreamRAM，回答如何在iso-bandwidth、iso-capacity、iso-power、thermal constraints下组织比较。

最重要的结论有五个：

- 公开HBM规格必须分为**单stack**和**整GPU聚合**；二者不能混用。
- 当前 `39.2 Tb/s = 4.9 TB/s` 与H200整GPU的4.8 TB/s接近，但不是两个当前HBM3E stack能够提供的带宽。
- 当前Conventional 2x1软件配置把两个11x22 mm可见热区域解释为两个物理stack，因而反推出每stack约61.6 GB、2.45 TB/s；但其legacy来源明确把每个11x22 mm区域定义为两个沿y合并的11x11 mm stack-equivalent。当前per-stack语义存在冲突，系统总容量可作为matched analytical value，不能据此声称两颗commodity stack capability。
- DreamRAM对容量、带宽和面积做过外部验证，但其论文明确显示公开HBM3/HBM2E目标没有可比的访问能量数据；当前1.3677 pJ/bit是**analytical nominal**，不是测量验证值。
- 最新整GPU HBM3E已经达到256-288 GB和6-8 TB/s，因此只对比144 GB HBM会被认为偏旧。论文至少应增加一个“current high-capacity HBM3E”敏感度锚点，但不能把其1000-1400 W整卡热设计直接塞入当前300 W双stack热模型。

## 2. 先统一四种常被混淆的数字

### 2.1 单stack容量和带宽

这是HBM厂商或芯片论文给出的一个封装堆栈的数据。HBM2-HBM3E通常采用1024-bit接口，理论峰值带宽为：

\[
BW_{stack}=R_{pin}\times 1024/8
\]

例如2.4 Gb/s/pin对应307.2 GB/s。

### 2.2 整GPU容量和带宽

这是GPU上所有HBM stack聚合后的产品规格。例如H200报告141 GB HBM3E与4.8 TB/s。它不说明应用一定能持续获得4.8 TB/s，也不能在不知道stack数量和组织时直接除回单stack能力。

### 2.3 论文架构带宽

MOSAIC的4.9 TB/s来自通道数、每通道速率与die数量的架构乘法，不是98-die完整系统的实测payload。其原型验证的是单条contactless通信路径在4 Gb/s下恢复信号。

### 2.4 Workload实际payload

这是应用运行时真正从memory读取/写入的数据率，通常低于datasheet peak，并受访问模式、bank冲突、协议、调度和缓存影响。当前项目的39.2 Tb/s明确属于matched-reference bandwidth，不是测得payload，也不是BWcoil capability。

## 3. 单stack与HBM硅benchmark

| 代际/来源 | 容量 | 堆叠 | Pin rate | 峰值带宽/stack | 证据状态 |
|---|---:|---:|---:|---:|---|
| Samsung Aquabolt HBM2 | 8 GB | 8H | 2.4 Gb/s | 307.2 GB/s | 2018量产产品公告 |
| Samsung Flashbolt HBM2E | 16 GB | 8H | 3.2 Gb/s | 410 GB/s | 2020产品公告；另有4.2 Gb/s/538 GB/s maximum-tested点 |
| SK hynix HBM2E | 16 GB | 8H | 3.6 Gb/s | 460 GB/s | 2019产品开发公告 |
| Chun et al. HBM2E | 16 GB | 未在摘要中固定 | 5.0 Gb/s | 640 GB/s | JSSC硅论文，105 C稳定cell operation |
| Park et al. HBM3 | 24 GB | 12H | 未单列 | 896 GB/s | ISSCC硅论文 |
| Ryu et al. HBM3 | 16 GB | 未在摘要中固定 | 8.0 Gb/s | 1024 GB/s | JSSC硅论文，105 C稳定运行 |
| Micron HBM3E | 24/36 GB | 8H/12H | >9.2 Gb/s | >1.2 TB/s | 24 GB shipping、36 GB production-capable |
| SK hynix HBM3E | 36 GB | 12H | 9.6 Gb/s | >1.23 TB/s | 2024量产公告 |
| Samsung HBM3E | 36 GB | 12H | 约10 Gb/s（由带宽反推） | 1.28 TB/s | 2024 developed product |
| Micron HBM4 | 36/48 GB | 12H/16H | >11 Gb/s | >2.8 TB/s | 2026 product/sample信息；2048-bit接口 |

来源：[Samsung HBM2](https://news.samsung.com/global/samsung-starts-producing-8-gigabyte-high-bandwidth-memory-2-with-highest-data-transmission-speed)、[Samsung HBM2E](https://news.samsung.com/global/samsung-to-advance-high-performance-computing-systems-with-launch-of-industrys-first-3rd-generation-16gb-hbm2e)、[SK hynix HBM2E](https://news.skhynix.com/en/sk-hynix-develops-worlds-fastest-high-bandwidth-memory-hbm2e/)、[SK hynix HBM3](https://product.skhynix.com/products/dram/hbm/hbm3.go)、[Micron HBM3E](https://www.micron.com/products/memory/hbm/hbm3e)、[SK hynix HBM3E](https://news.skhynix.com/en/sk-hynix-begins-volume-production-of-the-world-first-12-layer-hbm3e/)、[Samsung HBM3E](https://news.samsung.com/global/samsung-develops-industry-first-36gb-hbm3e-12h-dram)、[Micron HBM4](https://www.micron.com/products/memory/hbm/hbm4)。芯片论文见[HBM2E JSSC](https://ieeexplore.ieee.org/document/9240974/)、[HBM3 ISSCC](https://ieeexplore.ieee.org/document/9731562/)和[HBM3 JSSC](https://ieeexplore.ieee.org/document/10005600/)。

### 3.1 从这张表应该学到什么

容量、pin速率和stack高度并不是一条固定代际曲线。同一代产品可以在容量、stack高度、数据率、热设计和良率之间选择不同点。不能用“HBM3就是24 GB/819 GB/s”这样的单一句子代表整个HBM3。

HBM芯片论文常见写法是直接在题目和摘要中交代：

- 工艺节点；
- 容量；
- stack高度；
- pin rate；
- aggregate stack bandwidth；
- 关键电路贡献；
- 高温工作点或可靠性结果。

它们通常不会给一个可直接拿来做系统read/write pJ/bit的完整数值。因此，容量和带宽可作为强锚点，能量不能默认同等级使用。

## 4. 整GPU/加速器benchmark

下表是整颗GPU或加速器公开的聚合峰值，不是单stack规格，也不是应用实测payload。

| 加速器 | Memory | 容量 | 峰值带宽 | TDP/TBP | 理论全容量扫描率* |
|---|---|---:|---:|---:|---:|
| AMD MI100 | HBM2 | 32 GB | 1.2 TB/s | 300 W | 37.5 /s |
| NVIDIA A100 PCIe | HBM2E | 80 GB | 1.935 TB/s | 300 W | 24.2 /s |
| NVIDIA A100 SXM | HBM2E | 80 GB | 2.039 TB/s | 400 W | 25.5 /s |
| AMD MI250X | HBM2E | 128 GB | 3.2 TB/s | 560 W | 25.0 /s |
| NVIDIA H100 SXM | HBM3 | 80 GB | 3.35 TB/s | up to 700 W | 41.9 /s |
| NVIDIA H100 NVL | HBM3 | 94 GB | 3.938 TB/s | 350-400 W | 41.9 /s |
| NVIDIA H200 | HBM3E | 141 GB | 4.8 TB/s | up to 700/600 W | 34.0 /s |
| AMD MI300X | HBM3 | 192 GB | 5.3 TB/s | 750 W | 27.6 /s |
| AMD MI325X | HBM3E | 256 GB | 6.0 TB/s | 1000 W | 23.4 /s |
| NVIDIA HGX B300 | HBM3E | 270 GB | 7.7 TB/s | up to 1100 W | 28.5 /s |
| NVIDIA GB300 | HBM3E | 279 GB | 8.0 TB/s | up to 1400 W | 28.7 /s |
| AMD MI355X | HBM3E | 288 GB | 8.0 TB/s | 1400 W | 27.8 /s |

\* `bandwidth/capacity`，只表示按datasheet峰值顺序扫完整个memory的理论次数，不是测得tokens/s。

来源：[NVIDIA A100 datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-nvidia-us-2188504-web.pdf)、[NVIDIA H100](https://www.nvidia.com/en-us/data-center/h100/)、[NVIDIA H100 NVL brief](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/h100/PB-11773-001_v01.pdf)、[NVIDIA H200 datasheet](https://dam-cdn.nvd.orangelogic.com/AssetLink/5o2qgy5d2835ve2pm11i62kv8mphqta8.pdf)、[AMD MI100](https://www.amd.com/en/products/accelerators/instinct/mi100.html)、[AMD MI250X](https://www.amd.com/en/products/accelerators/instinct/mi200/mi250x.html)、[AMD MI300X](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html)、[AMD MI325X](https://www.amd.com/en/products/accelerators/instinct/mi300/mi325x.html)、[AMD MI355X](https://www.amd.com/en/products/accelerators/instinct/mi350/mi355x.html)、[NVIDIA Blackwell Ultra datasheet](https://dam-cdn.nvd.orangelogic.com/AssetLink/1k0p832eq8r5ca0u5383ie5o4tp3bst1.pdf)。

### 4.1 对本项目最关键的现实变化

早期80-144 GB是合理的HBM-on-GPU参考区间，但高端加速器已经公开到256-288 GB。正交M3D当前460.37 GB仍然更高，但相对最新整GPU HBM3E的总容量优势约为：

- 对H200 141 GB：约3.26x；
- 对MI325X 256 GB：约1.80x；
- 对MI355X 288 GB：约1.60x。

这些不是iso-area比值，因为产品封装、stack数量、GPU面积和TDP不同。论文应将其用于“现实产品上下文”，而不是直接替换thermal mesh。

## 5. MOSAIC、DreamRAM与当前项目的对应关系

### 5.1 MOSAIC如何写baseline

本地Ref 8.1的比较表同时列出：memory type、placement、cube size、die thickness、stacked dies、cube count、GPU power density、total capacity、Tmax、aggregated bandwidth。它不是只列容量和温度。

它的核心对照是：

| 架构 | 容量 | 带宽 | Tmax |
|---|---:|---:|---:|
| HBM baseline | 144 GB | 4.8 TB/s | 80.0 C |
| MOSAIC Si | 294 GB | 4.9 TB/s | 81.3 C |
| MOSAIC thin-Si projection | 882 GB | 4.9 TB/s | 81.3 C |

论文明确把thermal simulation和interface prototype分开：98-die/8 Gb/s/channel是架构假设；原型是8-die测试cube、4 Gb/s和6.9 pJ/bit。其4 pJ/bit memory access又是引用的模型假设，其中0.5 pJ/bit归于I/O。参见本地[Ref 8.1](../../ref/T8.1_Wednesday_Mitarai_0021.pdf)和其引用的[Fine-Grained DRAM](https://doi.org/10.1145/3123939.3124545)。

### 5.2 DreamRAM真正验证了什么

[DreamRAM论文](https://arxiv.org/abs/2512.12106)的Table I采用`Real / Model / Error`结构：

| Target | Bandwidth real/model | Capacity real/model | Energy real/model | Area error |
|---|---|---|---|---:|
| HBM3 | 1024/1024 GB/s | 16/16 GB | NA / 0.98-3.01 pJ/b | -8.3% |
| HBM2E | 640/741 GB/s | 16/16 GB | NA / 1.46-3.61 pJ/b | -0.6% |

这说明：

- DreamRAM容量与HBM3/HBM2E目标一致；
- HBM3带宽吻合，HBM2E带宽高估15.7%；
- 面积误差有量化；
- 公开目标没有可比较的访问能量与miss latency，因此能量/延迟是模型输出，不是外部测量验证。

这并不削弱DreamRAM作为分析工具的价值，但决定了项目里的措辞必须是：

```text
DreamRAM analytical read-energy nominal
```

而不是：

```text
validated silicon read energy
```

### 5.3 当前Conventional 2x1实际上是什么

当前软件配置解析为：

```text
2 thermal-visible merged groups
4 physical stack equivalents
total capacity = 108 GiB = 115.9641 GB
matched bandwidth = 4.9 TB/s
GPU power = 300 W
```

每个physical stack的geometry-driven analytical capacity为：

```text
27 GiB = 28.991 GB
aggregate matched bandwidth is not allocated as a validated per-stack capability
```

Gate A审计确认legacy语义有来源支持：两个11x22 mm可见区域分别合并两个11x11 mm stack-equivalent，即2 visible groups、4 physical stack equivalents。canonical现在将10.8x21.8 mm thermal-visible group与10.8x10.8 mm capacity instance解耦；114.75 GiB/57.375 GiB-per-group旧口径已退役。

沿用当前12H integer packing的正式结果为4x27 GiB=108 GiB/system；若使用DreamRAM原生16 GiB HBM3组织，则4 stack为64 GiB。当前108 GiB仍是geometry-driven analytical matched reference，不是commodity SKU。

公开HBM3E 12H通常为36 GB和约1.2-1.28 TB/s；Micron HBM4 12H为36 GB、>2.8 TB/s，16H sample为48 GB。因此当前2x1不是一个现成commodity HBM SKU，而是继承MOSAIC/IEDM几何和聚合带宽的matched-reference baseline。

推荐论文名称：

```text
Geometry-derived HBM-on-GPU matched reference
```

而不是无修饰的：

```text
two commodity HBM3/HBM3E stacks
```

### 5.4 当前Orthogonal Si的证据差距

| 指标 | 当前模型 | MOSAIC reference | 判断 |
|---|---:|---:|---|
| Capacity | 251.56 GB | 294 GB | DreamRAM integer packing更保守，需要解释14%差异 |
| Aggregated BW | 4.9 TB/s | 4.9 TB/s | matched architecture reference，不是完整实测capability |
| Read E/bit | 1.3677 pJ/b | 4 pJ/b total access assumption | 相差约2.9x；必须做敏感度 |
| Interface E/bit | 0.5 pJ/b | 0.5 pJ/b advanced reference | 一致但不是98-die系统实测 |
| Refresh | 1.6685 W | 未报告 | DreamRAM analytical |
| Logic/background | 0 W | 未报告 | 显式lower-bound modeling choice，不应写resolved physical zero |

## 6. 对LLM workload研究最有用的新指标

仅比较GB和TB/s还不够。对decode更直观的指标是：

\[
R_{scan}=\frac{BW_{peak}}{Capacity}
\]

它近似回答“如果顺序读取全部已驻留数据，一秒理论上最多扫几遍”。当前matched场景：

| Architecture | Capacity | Matched BW | Peak scans/s |
|---|---:|---:|---:|
| Current Conventional 2x1 | 123.21 GB | 4.9 TB/s | 39.8 |
| Current Orthogonal Si | 251.56 GB | 4.9 TB/s | 19.5 |
| Current M3D-IGZO | 460.37 GB | 4.9 TB/s | 10.6 |

这揭示了本项目真正的权衡：M3D扩大了可驻留容量，但在不增加带宽时，单位驻留容量对应的带宽下降。它适合“避免offload/分片”和“提高可容纳batch”，但不会让超大模型单请求自动变快。

因此论文应同时报告：

- local-resident capacity envelope；
- external overflow requirement；
- peak/full-capacity scan balance；
- matched-bandwidth tokens/s；
- capacity-enabled maximum batch与aggregate throughput。

## 7. E/bit和power为什么不能直接从产品表得到

厂商公开资料主要提供：

- 容量；
- pin rate；
- peak bandwidth；
- 相对能效改善；
- 整GPU TDP/TBP。

但通常不提供具有完整边界的：

- DRAM-core read pJ/bit；
- write pJ/bit；
- ACT/PRE与row-hit比例；
- PHY/interface pJ/bit；
- refresh/background分解；
- memory-only static power。

整GPU 700-1400 W不能当作HBM功率。厂商“低30%功耗”也不能转换成绝对pJ/bit。对当前项目最可靠的做法是：

1. 保留DreamRAM component decomposition；
2. 将1.3677 pJ/bit标为analytical nominal；
3. 将MOSAIC使用的4 pJ/bit设为literature-reference point；
4. 用1.37-4 pJ/bit做Si read-energy sensitivity；
5. write energy继续单独扫描，不从read静默推导；
6. refresh/background单独保留provenance和lower-bound状态。

在39.2 Tb/s下，1.3677 pJ/bit对应约53.6 W dynamic memory power，而4 pJ/bit对应约156.8 W。这个差距足以改变thermal结论，因此是load-bearing uncertainty。

## 8. 推荐给论文使用的baseline层级

### Layer A - Reality anchors

正文相关工作或一张context table中至少列：

- H200：141 GB、4.8 TB/s；
- MI300X：192 GB、5.3 TB/s；
- MI325X：256 GB、6 TB/s；
- Blackwell Ultra或MI355X：270-288 GB、7.7-8 TB/s。

作用：证明没有只挑旧HBM。

### Layer B - Canonical thermal reference

保留当前Conventional 2x1和MOSAIC-derived 300 W geometry，用于严格的空间thermal comparison。必须明确它是future/matched reference，不是H200复现。

### Layer C - Silicon anchors

用HBM2E 16 GB/640 GB/s、HBM3 16 GB/1024 GB/s、HBM3 24 GB/896 GB/s约束stack-level可实现范围。

### Layer D - Analytical decomposition

DreamRAM负责：

- packing；
- area；
- row-policy-dependent read energy；
- component energy breakdown；
- refresh；
- spatial source mapping所需分解。

### Layer E - Forward-looking sensitivity

HBM4只放在sensitivity/outlook：36 GB 12H或48 GB 16H、>2.8 TB/s/stack。不要把sample或future信息写成当前普遍部署。

## 9. 论文写法可以直接学习的四种范式

### 9.1 芯片论文：数字写进题目，边界写进摘要

HBM2E/HBM3 JSSC/ISSCC论文在题目中给容量与带宽，在摘要中给工艺、pin rate、高温工作和电路贡献。适合学习“一个数字必须绑定到具体硅、工艺和测试条件”。

### 9.2 DreamRAM：Real / Model / Error，缺失就写NA

DreamRAM没有因为论文没报告energy就假装完成能量验证，而是在表里保留NA。这是本项目最应该模仿的科学写法。

### 9.3 MOSAIC：先定义iso约束，再比较结果

MOSAIC先固定GPU power和iso-bandwidth，再比较capacity与Tmax；prototype measurement与system projection分开。你的论文也应先写：

```text
iso package footprint / matched bandwidth / matched GPU power
```

再展示M3D结果。

### 9.4 厂商datasheet：产品规格和workload benchmark分栏

H200 datasheet一边列141 GB/4.8 TB/s，一边用脚注明确模型、精度、batch和软件栈后报告LLM speedup。论文不能只抄相对倍数而忽略脚注条件。

## 10. 推荐的项目决策

### 立即采用

- 在论文与结果中区分`STACK_PEAK`、`GPU_AGGREGATE_PEAK`、`ARCHITECTURE_DERIVED`、`MATCHED_REFERENCE`和`MEASURED_PAYLOAD`。
- 保留来源原单位GB/TB/s；进入代码时显式记录GB到bytes或GiB的转换。
- 把当前Conventional 2x1称为geometry-derived matched HBM-on-GPU reference；在per-stack语义审计完成前，只使用系统总容量，不声称2-stack商品能力。
- 把H200作为最接近当前144 GB/4.9 TB/s的商品现实锚点。
- 增加MI325X或Blackwell Ultra高容量HBM3E sensitivity，避免过时baseline质疑。
- 把Orthogonal Si的1.3677 pJ/bit改为`DREAMRAM_ANALYTICAL_NOMINAL_NOT_MEASUREMENT_VALIDATED`的论文状态。
- 使用1.37和4 pJ/bit作为Si read-energy的关键两点，必要时在二者之间扫值。

### 暂时不要做

- 不要用整GPU TDP反推HBM E/bit；
- 不要用raw peak bandwidth直接声称LLM payload；
- 不要把两个HBM3E stack写成4.9 TB/s capability；
- 不要把HBM4 sample当当前主baseline；
- 不要因为文献4 pJ/bit看起来更有利于M3D就覆盖DreamRAM nominal；
- 不要同时重做thermal geometry和power model。

## 11. 推荐阅读顺序

1. [MOSAIC T8.1本地论文](../../ref/T8.1_Wednesday_Mitarai_0021.pdf)：先看Table I与Fig. 3，学习iso-bandwidth/thermal comparison。
2. [DreamRAM论文](https://arxiv.org/abs/2512.12106)：重点看Table I、Fig. 5/6和validation措辞。
3. [16 GB 1024 GB/s HBM3 JSSC](https://ieeexplore.ieee.org/document/10005600/)：看真实HBM芯片如何报告工艺、pin rate和高温稳定性。
4. [16 GB 640 GB/s HBM2E JSSC](https://ieeexplore.ieee.org/document/9240974/)：看上一代芯片的同类写法。
5. [24 GB 12-high 896 GB/s HBM3 ISSCC](https://ieeexplore.ieee.org/document/9731562/)：看容量/层数/带宽的另一设计点。
6. [NVIDIA H200 datasheet](https://dam-cdn.nvd.orangelogic.com/AssetLink/5o2qgy5d2835ve2pm11i62kv8mphqta8.pdf)：看产品规格与LLM benchmark脚注如何分开。
7. [Micron HBM4产品页](https://www.micron.com/products/memory/hbm/hbm4)：理解capacity和bandwidth是两个独立维度，也用于了解HBM4的2048-bit变化。

## 12. 仍然开放的证据缺口

- HBM/HBM3E完整read/write access energy accounting boundary；
- HBM refresh/background/PHY静态功率的公开、可迁移数据；
- peak raw bandwidth到LLM decode delivered payload的效率；
- 当前2x12Hi HBM reference的确切per-stack channel/PHY组织；
- MOSAIC 4 pJ/bit相对当前DreamRAM row-utilization与component边界的逐项对照；
- 商品H200/MI300X等的真实HBM物理布局、功率图和memory-only thermal贡献。

这些缺口不会阻止容量与峰值带宽benchmark，但会阻止把产品资料直接变成精确J/token和Tmax。

## 13. Stop criterion

本轮检索在以下条件满足后停止：HBM2-HBM4单stack代表点已有一手来源；A100到当前高容量HBM3E整GPU代表点已有一手datasheet；DreamRAM能量验证边界已由原论文确认；当前项目与MOSAIC、H200和最新HBM3E之间的口径冲突已能够解释。继续收集更多厂商重复规格不会改变上述决策；下一轮应转向绝对能量/功率边界，而不是继续扩充容量/带宽列表。
