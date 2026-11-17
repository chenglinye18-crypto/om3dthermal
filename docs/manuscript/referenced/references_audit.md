# IOM3D-HBM 文献审计（仅添加引用版）

核验日期：2026-11-17。根据用户最新要求，本版本撤销上一版所有正文措辞、图题文字及图片修改。问题只列于本报告，不自动修正文稿。本报告供作者使用，不作为匿名投稿材料。

## 修改范围与验证

- 源文件：终稿目录 `DAC2027V1(1).docx`，未改动源文件。
- 输出仍为 `DAC2027V1_referenced.docx`、`DAC2027V1_referenced.pdf`，覆盖前一次交付版本。
- 正文只插入引用编号、更新原有编号和有充分依据的引用占位符；所有其他文字及标点恢复原稿，包括原稿中的错误表述和排版字符串。没有新增相关工作句子、解释句或脚注文字。
- 逐段验证前 110 个原始段落：剔除引用编号及插入编号带来的空格后，正文和图题与原稿完全一致。全部 21 个图片媒体资源与原稿逐字节一致；所有公式、绘图和嵌入对象节点一致。
- 保留必要的字号/段距/分页调整以满足先前 6+1 页要求：正文与参考文献 9 pt，图片尺寸不变；图片与图题同页。最终 PDF 7 页，结论在第 6 页，第 7 页仅参考文献，已 Word 原生打开、导出和逐页渲染检查。
- 最终 **28 篇引用，其中 IEEE/ACM（含 DATE）21 篇**。编号 [1]–[28] 连续，按首次出现排序，无未使用或重复条目。为严格遵守“不添加正文句子”，没有强行插入需要额外解释才能避免误导的 Shiba/DRAMsim3 引文，二者保留为审计候选；数量服从论据匹配。
- **保留三处不能直接修复的 [X]**：正文“100 µm 已制造并组装”和“8 Gb/s/ch、0.5 pJ/bit 实测”两处，及 Fig. 3 图片内部的 GPU 能量来源占位符。现有证据不能完整支持这些原句；用编号替换会造成错误背书，修改句子/图片又违反最新要求，因此只在下文指出。
- 本次不运行仿真，不改实现、参数、图表、实验数据或结论。PDF 作者元数据为空；未增加作者或单位信息。

联网核验使用 IEEE/ACM 正式记录、Crossref、会议节目、作者官网、模型报告和 NVIDIA 官方资料。Mitarai VLSI 2026 与 Tang IEDM 2023 另核查本地原文；部分付费文章采用官方可索引摘要/结论及作者发表记录交叉验证，不能宣称全部全文均已获取。ECTC MOSAIC/TCPMT 付费全文未取得。正式 DOI、作者和页码与 Crossref 核对，引用不代表已复现其全部实验。

## 引用清单与论据范围

| 编号 | 文献标题 | 发表平台/年份 | DOI/URL | 引用位置 | 支持范围 | 核实状态 |
|---|---|---|---|---|---|---|
| [1] | The Llama 3 Herd of Models | arXiv / 2024 | [官方来源](https://arxiv.org/abs/2407.21783) | I Introduction；II-A；V-A | Llama 3.1 模型家族、长上下文；V-A 模型来源。 | [证据页](https://arxiv.org/abs/2407.21783)；书目/公开证据核验，细节范围见下文 |
| [2] | Efficient Memory Management for Large Language Model Serving with PagedAttention | in Proc. ACM SOSP / 2023 | [10.1145/3600006.3613165](https://doi.org/10.1145/3600006.3613165) | I Introduction；II-A | KV cache 容量、批处理和内存管理。 | [证据页](https://web.stanford.edu/class/cs240/readings/vllm.pdf)；书目/公开证据核验，细节范围见下文 |
| [3] | Orca: A Distributed Serving System for Transformer-Based Generative Models | USENIX OSDI / 2022 | [官方来源](https://www.usenix.org/conference/osdi22/presentation/yu) | I Introduction | 并发 serving、iteration-level scheduling、selective batching。 | [证据页](https://www.usenix.org/conference/osdi22/presentation/yu)；书目/公开证据核验，细节范围见下文 |
| [4] | NVIDIA H200 Tensor Core GPU | NVIDIA 官方规格 / 2026 | [官方来源](https://www.nvidia.com/en-us/data-center/h200/) | I Introduction | H200 141 GB HBM3e、4.8 TB/s；不支持论文 GPU 动态能耗系数。 | [证据页](https://www.nvidia.com/en-us/data-center/h200/)；书目/公开证据核验，细节范围见下文 |
| [5] | Splitwise: Efficient Generative LLM Inference Using Phase Splitting | in Proc. ACM/IEEE ISCA / 2024 | [10.1109/isca59077.2024.00019](https://doi.org/10.1109/isca59077.2024.00019) | I Introduction | Prefill/decode 资源需求不同。 | [证据页](https://www.microsoft.com/en-us/research/wp-content/uploads/2023/12/Splitwise_ISCA24.pdf)；书目/公开证据核验，细节范围见下文 |
| [6] | FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness | NeurIPS / 2022 | [官方来源](https://proceedings.neurips.cc/paper_files/paper/2022/hash/67d57c32e20fd0a7a302cb81d36e40d5-Abstract-Conference.html) | I Introduction | IO-aware attention 减少 HBM/SRAM 数据移动；不作为本文加速比证据。 | [证据页](https://proceedings.neurips.cc/paper_files/paper/2022/hash/67d57c32e20fd0a7a302cb81d36e40d5-Abstract-Conference.html)；书目/公开证据核验，细节范围见下文 |
| [7] | FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU | ICML / 2023 | [官方来源](https://proceedings.mlr.press/v202/sheng23a.html) | I Introduction | GPU/CPU/磁盘 offloading 的内存与数据传输权衡。 | [证据页](https://proceedings.mlr.press/v202/sheng23a.html)；书目/公开证据核验，细节范围见下文 |
| [8] | An LPDDR-based CXL-PNM Platform for TCO-efficient Inference of Transformer-based Large Language Models | in Proc. IEEE HPCA / 2024 | [10.1109/hpca57654.2024.00078](https://doi.org/10.1109/hpca57654.2024.00078) | I Introduction | 近存 LLM 加速的背景及 memory-centric 设计。 | [证据页](https://snu.elsevierpure.com/en/publications/an-lpddr-based-cxl-pnm-platform-for-tco-efficient-inference-of-tr)；书目/公开证据核验，细节范围见下文 |
| [9] | Stratum: System-Hardware Co-Design with Tiered Monolithic 3D-Stackable DRAM for Efficient MoE Serving | in Proc. IEEE/ACM MICRO / 2025 | [10.1145/3725843.3756043](https://doi.org/10.1145/3725843.3756043) | I Introduction | M3D NMP/GPU 协同、MoE expert tiering 和迁移。 | [证据页](https://arxiv.org/abs/2510.05245)；书目/公开证据核验，细节范围见下文 |
| [10] | Near-Memory LLM Inference Processor based on 3D DRAM-to-logic Hybrid Bonding | in Proc. ACM/IEEE DAC / 2025 | [10.1109/dac63849.2025.11132870](https://doi.org/10.1109/dac63849.2025.11132870) | I Introduction | hybrid-bonded DRAM/logic、集中控制器及 dual-IO GEMV/GEMM。 | [证据页](https://ieeexplore.ieee.org/document/11132870/)；书目/公开证据核验，细节范围见下文 |
| [11] | 3D-PATH: A Hierarchy LUT Processing-in-memory Accelerator with Thermal-aware Hybrid Bonding Integration | in Proc. IEEE/ACM MICRO / 2025 | [10.1145/3725843.3756087](https://doi.org/10.1145/3725843.3756087) | I Introduction | LUT-PIM 和 thermal-aware hybrid bonding；非本文 MAC FEOL。 | [证据页](https://doi.org/10.1145/3725843.3756087)；书目/公开证据核验，细节范围见下文 |
| [12] | Breaking Thermal Bottleneck in 3D HBM-on-GPU Integration via System-Technology Co-Optimization | in Proc. IEEE IEDM / 2025 | [10.1109/iedm50572.2025.11353711](https://doi.org/10.1109/iedm50572.2025.11353711) | I Introduction | HBM-on-GPU 热瓶颈和系统技术协同优化。 | [证据页](https://static1.squarespace.com/static/67a3eee4385dfb3390804f02/t/692722bf0dad5d330cf60d98/1764172479207/IEDM%202025%20Final%20Program-v5.pdf)；书目/公开证据核验，细节范围见下文 |
| [13] | Massive Orthogonal Stacking Assembly of IC (Mosaic) Cube with Inductive Coupling for Exascale Memory Applications | in Proc. IEEE ECTC / 2025 | [10.1109/ectc51687.2025.00227](https://doi.org/10.1109/ectc51687.2025.00227) | I Introduction | MOSAIC 正交组装与感应耦合；非 8 Gb/s 实测依据。 | [证据页](https://fukushima.mech.tohoku.ac.jp/news/?p=4)；书目/公开证据核验，细节范围见下文 |
| [14] | A Thermal and System Analysis of SoC-Memory 3-D Stacking Using Massive Orthogonal Stacking Assembly | IEEE Trans. Compon., Packag., Manuf. Technol. / 2026 | [10.1109/tcpmt.2025.3648414](https://doi.org/10.1109/tcpmt.2025.3648414) | I Introduction | 正交堆叠热/性能权衡；2026 正式卷期而非 DOI 年份。 | [证据页](https://klab.t.u-tokyo.ac.jp/en/international-journal-papers/y-2026/)；书目/公开证据核验，细节范围见下文 |
| [15] | AttAcc! Unleashing the Power of PIM for Batched Transformer-based Generative Model Inference | in Proc. ACM ASPLOS / 2024 | [10.1145/3620665.3640422](https://doi.org/10.1145/3620665.3640422) | II-A | batched Transformer 的 attention PIM；DNS/CPA 对比。 | [证据页](https://scale.snu.ac.kr/papers/2024-04-Conference-ASPLOS-AttAcc.pdf)；书目/公开证据核验，细节范围见下文 |
| [16] | NeuPIMs: NPU-PIM Heterogeneous Acceleration for Batched LLM Inferencing | in Proc. ACM ASPLOS / 2024 | [10.1145/3620666.3651380](https://doi.org/10.1145/3620666.3651380) | II-A | GEMM/GEMV 异构协同与重叠调度；不能把异构调度本身当作独创。 | [证据页](https://www.asplos-conference.org/asplos2024/main-program/abstracts/)；书目/公开证据核验，细节范围见下文 |
| [17] | Capacitor-less, Long-Retention (>400s) DRAM Cell Paving the Way towards Low-Power and High-Density Monolithic 3D DRAM | in Proc. IEEE IEDM / 2020 | [10.1109/iedm13553.2020.9371900](https://doi.org/10.1109/iedm13553.2020.9371900) | II-B | IGZO 低漏电、长保持时间、stacking-compatible 2T0C。 | [证据页](https://ieee-iedm.org/wp-content/uploads/2026/05/2020-IEDM-Archive.pdf)；书目/公开证据核验，细节范围见下文 |
| [18] | Tailoring IGZO-TFT architecture for capacitorless DRAM, demonstrating > 10³s retention, >10¹¹ cycles endurance and Lg scalability down to 14nm | in Proc. IEEE IEDM / 2021 | [10.1109/iedm19574.2021.9720596](https://doi.org/10.1109/iedm19574.2021.9720596) | II-B | IGZO 2T0C retention/endurance 与尺寸缩放。 | [证据页](https://imec-publications.be/entities/publication/b1855549-195d-4537-b0c4-3eb4546b036e/full)；书目/公开证据核验，细节范围见下文 |
| [19] | Novel Vertical Channel-All-Around (CAA) In-Ga-Zn-O FET for 2T0C-DRAM With High Density Beyond 4F² by Monolithic Stacking | IEEE Trans. Electron Devices / 2022 | [10.1109/ted.2022.3154693](https://doi.org/10.1109/ted.2022.3154693) | II-B | CAA IGZO 垂直器件、BEOL 工艺和 4F² 概念。 | [证据页](https://xplorestaging.ieee.org/document/9732906/figures)；书目/公开证据核验，细节范围见下文 |
| [20] | 30 Mb/mm²/layer 3D eDRAM Computing-in-Memory with Embedded BEOL Peripherals and Local Layer-wise Calibration based on First-Demonstrated Vertically-stacked CAA-IGZO 4F² 2T0C Cell | in Proc. IEEE IEDM / 2023 | [10.1109/iedm45741.2023.10413756](https://doi.org/10.1109/iedm45741.2023.10413756) | II-B；III-C | 0.023 µm²、20 s measured cell；30 Mb/mm²/layer 是论文 CiM 架构评估密度。 | [证据页](https://static1.squarespace.com/static/67a3eee4385dfb3390804f02/t/684859e2d4527577597b1474/1749572070397/2023-IEDM-Archive.pdf)；书目/公开证据核验，细节范围见下文 |
| [21] | Monolithic 3D Integration: A Path from Concept to Reality | in Proc. DATE / 2015 | [10.7873/date.2015.1111](https://doi.org/10.7873/date.2015.1111) | II-B | 细粒度层间连接和 M3D 集成背景，不验证本文每项 RC。 | [证据页](https://past.date-conference.com/proceedings-archive/2015/pdf/1111.pdf)；书目/公开证据核验，细节范围见下文 |
| [22] | 3D Orthogonal Die Stacking Technology for DRAM-on-GPU Integration Using Contactless Die-to-Die Interface | in Proc. IEEE/JSAP Symp. VLSI Technol. Circuits / 2026 | [10.1109/vlsitechnologyandcir65830.2026.11577466](https://doi.org/10.1109/vlsitechnologyandcir65830.2026.11577466) | II-C | 侧壁供电；4 Gb/s/ch、6.9 pJ/bit 原型；100 µm、8 Gb/s/0.5 pJ 是投影/假设。 | [证据页](https://doi.org/10.1109/VLSITechnologyandCir65830.2026.11577466)；书目/公开证据核验，细节范围见下文 |
| [23] | A case for exploiting subarray-level parallelism (SALP) in DRAM | ACM SIGARCH Comput. Archit. News / 2012 | [10.1145/2366231.2337202](https://doi.org/10.1145/2366231.2337202) | III-C | ISCA 2012 工作的 SIGARCH 版本；共享外围导致串行化、独立 subarray 带来并行性。 | [证据页](https://users.ece.cmu.edu/~yoonguk/papers/kim-isca12.pdf)；书目/公开证据核验，细节范围见下文 |
| [24] | Fine-Grained DRAM: Energy-Efficient DRAM for Extreme Bandwidth Systems | in Proc. IEEE/ACM MICRO / 2017 | [10.1145/3123939.3124545](https://doi.org/10.1145/3123939.3124545) | III-C | 阵列与 I/O 数据移动能量、细粒度 DRAM 组织。 | [证据页](https://research.nvidia.com/sites/default/files/pubs/2017-10_Fine-Grained-DRAM%3A-Energy-Efficient/oconnor_and_chatterjee.micro2017.pdf)；书目/公开证据核验，细节范围见下文 |
| [25] | DreamRAM: A Fine-Grained Configurable Design Space Modeling Tool for Custom 3D Die-Stacked DRAM | in Proc. DATE / 2026 | [10.23919/date69613.2026.11539402](https://doi.org/10.23919/date69613.2026.11539402) | III-C | HBM 的 MAT/subarray/bank/stack 物理、延迟和能量建模；不支持本文新建执行模型。 | [证据页](https://github.com/harvard-acc/DreamRAM)；书目/公开证据核验，细节范围见下文 |
| [26] | HotSpot: a compact thermal modeling methodology for early-stage VLSI design | IEEE Trans. VLSI Syst. / 2006 | [10.1109/tvlsi.2006.876103](https://doi.org/10.1109/tvlsi.2006.876103) | IV-A | 紧凑热阻网络方法背景；不表示本文使用其瞬态方法。 | [证据页](https://www.cs.virginia.edu/~skadron/Papers/hotspot_tvlsi06.pdf)；书目/公开证据核验，细节范围见下文 |
| [27] | Qwen2.5 Technical Report | arXiv / 2024 | [官方来源](https://arxiv.org/abs/2412.15115) | V-A | Qwen2.5 模型家族；具体 32B 长上下文配置需另核 model card。 | [证据页](https://arxiv.org/abs/2412.15115)；书目/公开证据核验，细节范围见下文 |
| [28] | NVIDIA Grace Hopper Superchip Architecture In-Depth | NVIDIA 技术文档 / 2022 | [官方来源](https://developer.nvidia.com/blog/nvidia-grace-hopper-superchip-architecture-in-depth/) | V-A | Grace/Hopper coherency 与 900 GB/s 双向 C2C；不是 H200 SXM 自带 C2C 的证明。 | [证据页](https://developer.nvidia.com/blog/nvidia-grace-hopper-superchip-architecture-in-depth/)；书目/公开证据核验，细节范围见下文 |

## 发现的问题：仅指出，未修改正文或图片

| 位置 | 问题与证据 | 建议（未执行） |
|---|---|---|
| II-C 100 µm [X] | VLSI [22] 将 300 µm 与“with thinned Si”100 µm 分开，100 µm 是减薄扩展分析，不是文稿所说的实测组装证明 | 应区分已组装原型与 100 µm 设计投影；当前保留原句和 [X] |
| II-C 8 Gb/s/ch、0.5 pJ/bit [X] | VLSI 原型为 0.18 µm、4 Gb/s/ch、6.9 pJ/bit。8/0.5 是先进工艺预测/建模条件。Shiba 的 12.8 Gb/s、0.5 pJ/bit 来自另一种 7-nm SRAM 感应接口 | 不能写成正交 MOSAIC 已实测 8/0.5。未替换该 [X]，未改变原数字或措辞；侧壁供电处有明确依据，已引用 [22] |
| II-B 30 Mb/mm²/layer | Tang [20] 同时包含器件测量和 CiM 架构评估；30 是该 CiM 架构密度，不等于整阵列制造实测 | “memories demonstrate”可能过强，应明确评估层级，当前原句不改 |
| Fig. 2 器件与能耗 | 0.023 µm²、约 20 s 可追溯 Tang 原文；20 s 有特定状态与保持阈值。0.288 µm、512×512 和 0.60/368/0.9/370 fJ 不应一概归为同篇实测 | 区分实测器件、设计输入和 SPICE 提取。项目配置将能耗归于本地 IEDM2026_HaotongZhu_V5 的 MAT operation table；未找到可公开验证正式版本，未收入 References |
| Fig. 3 密度38.82与容量1.49 TB | 512×512×64×280/(121×10⁶)=38.82330975 Mb/mm²/layer；容量乘8层×318dies再除8/10¹²=1.49384331264 TB | 这是本文结构计算，外围和口径不同于 Tang 的30；建议补解释，但本次不增正文 |
| Fig. 3 GPU [X] | 15.29 pJ/bit、0.633 pJ/FLOP 在项目CSV为模型推导，涉及带宽利用率和功率比例，不是 NVIDIA H200 直接实测规格 | 图片完全恢复，保留 [X]，不能以 H200 产品页强配引用；需要完整推导或实测证据 |
| Fig. 3 H200算力 | 原图985.5 TFLOP/s与项目平台表989.5不一致 | 作者核查笔误/模型取值，未改图 |
| NMP 0.604 pJ/MAC等 | 未核实公开论文能直接支持该FEOL组合 | 保持设计参数，不能借其他NMP论文证明数值 |
| III-C RC | 项目部分互连参数标为 MODELING_CHOICE；原稿声称“parameters from the literature”过于笼统 | 应给具体工艺/提取表；MIV电阻一定低于TSV的泛化也需谨慎。此次不改原句 |
| III-C DreamRAM [25] | 正式DATE2026版本可验证；主分支与DATE2026分支有不同更新 | 明确哪几个阵列/外围参数由哪个commit/config导出，本文调度、争用、placement不等于DreamRAM原功能。保留原有DreamRAM-derived表述，未新增说明 |
| V-A Grace/NVLink-C2C [28] | 官方900 GB/s是双向总带宽，适用于Grace/Hopper链路；不能据此认为H200 SXM本身具备该连接 | 说明SXM-class计算核加Grace-like offload的组合假设，当前不改变正文 |
| V-A Qwen126K | 原模型报告与官方model card区分模型配置和YaRN/长上下文设置 | 物理仿真不等于任务精度验证，需checkpoint、base/instruct、RoPE/YaRN设置；未改工作负载 |
| IV-A 热模型 | HotSpot [26]仅支持热网络方法背景，不证明本文边界条件、85°C阈值或PCG验证 | 原模型和阈值不变，外部引用不作为3.10×、3.34×等自有结果的证明 |
| 其他原稿问题 | Fig.5 “Thermal s4calability”、III-B字面 LaTeX `\(8\times8\)`、HoG/HBM-GPU命名、热讨论与die-grouped BEOL-uniform映射口径 | 全部保留，仅记录 |

补充核查来源：[Shiba 2023](https://doi.org/10.1109/LSSC.2023.3252607)、[NVIDIA H200](https://www.nvidia.com/en-us/data-center/h200/)、[Qwen model card](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct)、[xPU-athalon 原始稿](https://arxiv.org/abs/2604.10852)。这些来源不自动验证本文各项设计参数。

## 相关工作区别（仅在审计中分析）

- Stratum [9]：MoE专家概率、Mono3D tiering和专家迁移；本文DNS/CPA面向resident weights/KV和orthogonal FEOL group/region/port争用。不能把M3D+GPU/NMP的总体组合当作从未出现；未更改贡献主张，也未加入原稿之外的对比句。
- Han DAC [10]：集中控制器、dual-IO与hybrid-bonded DRAM/logic的GEMV/GEMM；本文分布式FEOL MAC与正交封装不同，没有统一配置实验便不能声称更优。
- 3D-PATH [11]：层级LUT、fast-LUT及热点相关电路；本文MAC FEOL和以时延为目标的CPA不同。该引用位于Introduction的广义prior memory organizations句，不暗示采用其LUT方法。
- AttAcc [15]/NeuPIMs [16]：已有attention-PIM、GEMM/GEMV异构分工和调度；引用在II-A的内存/数据移动背景，不把它们写作本文DNS的外部验证。
- SALP [23]/FGDRAM [24]：共享资源和阵列/I/O背景；并未证明本文执行事件模型已正确校准。DreamRAM [25]的工具范围也不能覆盖本稿全部模型。

## 未入选候选与可扩充文献

候选库按检索过程扩充；表中未入选不表示无关，也不表示每篇全文均已审阅。正式 References 优先直接论据，保留 28 篇，不以泛背景凑 35 篇。新增候选若只得到初步题录，明确标为候选而非已完成论据核验。

| 候选 | 来源 | 不入选原因/价值 |
|---|---|---|
| Ramulator 2.0: A Modern, Modular, and Extensible DRAM Simulator | https://doi.org/10.1109/LCA.2023.3333759 | 正式书目核对；与已选 DRAMsim3 的时序仿真背景重叠，本文未使用它 |
| GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints | https://aclanthology.org/2023.emnlp-main.298/ | 官方全文/书目；适合补模型 KV-head 配置，正文未改注意力机制 |
| High Thermal Conductivity AlN Films for Advanced 3D Chiplets | https://doi.org/10.1109/VLSITechnologyandCir46783.2024.10631317 | 正式书目和作者 PDF；本文不采用其 AlN 工艺 |
| Thermal Analysis and Design Guideline of Massive Orthogonal Stacking Assembly of IC (MOSAIC) Cube with Bump Connection Enabling SoC-DRAM Direct Stacking | https://doi.org/10.1109/ECTC51687.2025.00272 | Crossref+VLSI 原文引用；与已选 TCPMT/VLSI 热分析重叠 |
| High-density three-dimensional integration of dynamic random-access memory using vertical dual-gate IGZO TFTs | https://www.nature.com/articles/s41467-025-65925-3 | 出版社原文；不同双栅器件，不能替代 Tang CAA 参数 |
| Disrupting the DRAM roadmap with capacitor-less IGZO-DRAM technology | https://doi.org/10.1038/s44287-025-00162-w | 作者综述；优先保留其原始 IEDM 工作 |
| A Modern Primer on Processing in Memory | https://doi.org/10.1007/978-981-16-7487-7_7 | Crossref 书目；通用背景，直接 LLM PIM 工作更贴近 |
| DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving | https://www.usenix.org/system/files/osdi24_slides-zhong-yinmin.pdf | 官方会议材料；与 Splitwise 阶段分离背景重叠 |
| MOONCAKE: Trading More Storage for Less Computation—A KVCache-centric Architecture for Serving LLM Chatbot | https://www.usenix.org/system/files/fast25-qin.pdf | 正式全文；适合 KV 重用/分布式 serving 扩展，本文未讨论该机制 |
| Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve | https://www.usenix.org/conference/osdi24/presentation/agrawal | 官方书目/摘要；可扩充 online serving，当前模型没有完整 SLO/到达流建模 |
| PIM GPT a hybrid process in memory accelerator for autoregressive transformers | https://www.nature.com/articles/s44335-024-00004-2 | 正式期刊候选；当前保留更贴近 3D 与异构调度的工作 |
| PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU | https://doi.org/10.1145/3694715.3695964 | 正式版确认；利用激活稀疏性，与本文稠密 FP16 物理模型不同 |
| vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention | https://doi.org/10.1145/3669940.3707256 | 正式版/作者仓库；KV 软件管理背景重复 |
| FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning | https://proceedings.iclr.cc/paper_files/paper/2024/hash/98ed250b203d1ac6b24bbcf263e3d4a7-Abstract-Conference.html | ICLR 正式版；已保留原始 IO-aware 论据 |
| FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision | https://papers.nips.cc/paper_files/paper/2024/file/7ede97c3e082c6df10a8d6103a2eebd2-Paper-Conference.pdf | 正式会议原文；本文没有相应 kernel/低精度方案 |
| PNMAX: Mapping-Architecture Co-Exploration Framework for Processing Near Memory | https://dynamo.ethz.ch/wp-content/uploads/2026/11/Yang_PNMAX_MICRO26.pdf | 新近作者正式稿候选；与 CPA 更相关，建议后续逐算法比较；本次未做全文机制审计，不据此扩展创新主张 |
| The xPU-athalon: Quantifying the Competition of AI Acceleration | https://arxiv.org/abs/2604.10852 | 作者稿标注 ISPASS 2026 accepted；项目能量假设的相关来源，但没有直接验证 Fig. 3 的 H200 系数，故不硬插 |

可能缺少的关键证据不是更多背景引文，而是：MAT SPICE 提取材料的可公开版本、H200 动态能耗系数的可复现实测或完整推导、100-µm 318-die 正交组装及电源完整性证据、同配置相关 NMP 系统比较、Qwen 126K 配置说明。没有用题目相近的论文填补这些证据缺口。


另外两篇从原入选清单移到候选：

- Shiba 等，A 12.8-Gb/s 0.5-pJ/b Encoding-Less Inductive Coupling Interface Achieving 111-GB/s/W 3D-Stacked SRAM in 7-nm FinFET，IEEE SSC-L 2023，doi:10.1109/LSSC.2023.3252607。不同接口，需解释区别才能避免为原稿错误实测归属背书。
- Li 等，DRAMsim3: A Cycle-Accurate, Thermal-Capable DRAM Simulator，IEEE CAL 2020，doi:10.1109/LCA.2020.2973991。原稿没有使用该工具的声明，不能为保留30篇而添加新正文介绍或暗示采用。

## 完整性记录

此次候选总数仍为47，28篇入选、19篇候选/补充。数目低于原目标30–35，优先遵守最新“只加引用、不改正文和图片”要求及引用真实性。三个[X]明确记录，不宣称已全部处理。

- `DAC2027V1(1).docx` SHA-256：`3a24cf51e08225841c389679014a5a3fb79b107e5dc38a585a05132fe445f71e`
- `DAC2027V1_referenced.docx` SHA-256：`c29777f72049b337e02289804f39e9608f0224f48f997205dc22beca903c38c1`
- `DAC2027V1_referenced.pdf` SHA-256：`4f4444485015e4be50b8f34d970fa3a640181b4208a096c286e3ad925d8e819d`
