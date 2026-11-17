# IOM3D-HBM DAC 2027 参考文献审计

核验日期：2026-11-17（Asia/Shanghai）。本文件供作者审计使用，不属于投稿正文或匿名补充材料。

## 交付与检查结果

- 实际源文件为 `DAC 2026/终稿/DAC2027V1(1).docx`；要求中的 `DAC2027V1(1)(1).docx` 未找到。按照用户指定的“终稿”目录处理，未改动源文件。
- 输出 `DAC2027V1_referenced.docx`、`DAC2027V1_referenced.pdf` 和本审计文件。保留原 IEEE 双栏、图片尺寸、公式、模型参数、实验数字与结论。
- 筛选记录含 47 个候选条目（30 个入选、17 个未入选/补充）。最终 **30 篇，其中 23 篇为 IEEE/ACM 会议或期刊文献（含 DATE）**；其余为 NeurIPS、ICML、USENIX、模型技术报告和 NVIDIA 原始资料。
- 正文手动编号按首次出现顺序为 [1]–[30]；全部在正文使用；无重复条目、跳号、悬空引用和未使用条目。未新增 Zotero/BibTeX/Word 引用域。
- 三处正文 `[X]` 已替换。Fig. 3 的 EMF 图片内另有一处 `[X]`，仅将其同长度替换成 `[*]`；图题说明这是模型推导的 GPU 动态能耗系数，不给它强配外部实测引文。
- 最终 PDF **恰好 7 页**，Conclusion 完整位于第 6 页，第 7 页仅 References。正文及参考文献 9 pt；章节标题沿用原模板。未缩放图片或修改页边距；调整正文段距、图题整段保留及图片与图题同页，References 前强制分页。
- 已用 Word 2021 原生打开 DOCX 并导出 PDF，逐页渲染并检查图像、公式、图题和参考文献。文档技能的 `render_docx.py` 在指定 Conda 环境因缺少 `pdf2image` 不能运行；使用 Word 原生排版 + PyMuPDF 渲染替代，未另换 Python 环境。
- 原文件全部 `m:oMath`、`w:drawing`、`w:object` XML 节点与输出逐项相同；21 个 media 资源中仅上述 EMF 占位符文字发生变化，其余逐字节相同。最终版本前六页与已逐页目视检查版本的渲染逐像素一致。
- PDF author 元数据为空；DOCX creator/lastModifiedBy 清空。没有加入作者单位、致谢、实验路径或私有稿件作者信息。
- 本次不运行仿真、参数扫描或 pytest，不更改项目实现、配置、实验数据及 third_party。仅文献与文档结构/排版检查。

DAC 2027 的官方要求已联网核查：[Research Manuscript Submissions](https://dac.com/2027/program/research-manuscript-submissions)。该页面明确 IEEE 双栏、9/10 pt、6 页正文加 1 页仅参考文献，以及匿名和 PDF 元数据要求。本次仅确认这些条件，不能将本审计等同于会议全面形式审查。

## 核验方法与可访问性边界

逐篇检索 IEEE/ACM DOI、出版社页面、会议网站、作者机构网页或原始技术报告；用 Crossref API 返回的完整作者、标题、日期、页码、卷期交叉核对正式文献。部分 IEEE/ACM 直接页面遇到访问限制，采用可索引的出版社摘要、作者官网、正式会议节目及本地原文交叉验证，未把“DOI 存在”当作参数已验证。

重点全文核查了本地 `T8.1_Wednesday_Mitarai_0021.pdf` 与 Tang IEDM 2023 PDF；它们与联网确认的题名、作者及正式 DOI 对应。ECTC 2025 MOSAIC 与 TCPMT 的完整付费正文未获取，分别核查 Crossref、作者发表记录、公开摘要及 VLSI 2026 原文对它们的引用；不声称审阅这两篇的全部实验细节。3D-PATH 用 ACM 可索引摘要及结论核查架构；Han 用 IEEE 摘要；Stratum 用作者原始稿摘要及 ACM 正文可索引段落核查 tiering/迁移机制。未验证的字段不补造。

以下表格给出正式 DOI/原始 URL 及每篇的论据范围；“证据页面”可能与正式 DOI 页不同。完整作者字段可从对应正式记录取得，排版按 IEEE 多作者规则使用姓名缩写或 et al.。Qwen 引用原始技术报告的团队署名；Llama 使用当前原始报告的作者署名，而非混用旧版首作者。

## 入选文献审计表

| 编号 | 文献标题 | 发表平台/年份 | DOI/URL | 引用位置 | 所支持的论断 | 核实状态 |
|---|---|---|---|---|---|---|
| [1] | The Llama 3 Herd of Models | arXiv / 2024 | [官方来源](https://arxiv.org/abs/2407.21783) | I Introduction；II-A；V-A | Llama 3.1 模型家族、长上下文；V-A 模型来源。 | 官方页面或作者原始技术报告核对；[证据页面](https://arxiv.org/abs/2407.21783) |
| [2] | Efficient Memory Management for Large Language Model Serving with PagedAttention | in Proc. ACM SOSP / 2023 | [10.1145/3600006.3613165](https://doi.org/10.1145/3600006.3613165) | I Introduction；II-A | KV cache 容量、批处理和内存管理。 | Crossref 书目信息交叉核对；出版社/作者/会议摘要或正文核对；[证据页面](https://web.stanford.edu/class/cs240/readings/vllm.pdf) |
| [3] | Orca: A Distributed Serving System for Transformer-Based Generative Models | USENIX OSDI / 2022 | [官方来源](https://www.usenix.org/conference/osdi22/presentation/yu) | I Introduction | 并发 serving、iteration-level scheduling、selective batching。 | 官方页面或作者原始技术报告核对；[证据页面](https://www.usenix.org/conference/osdi22/presentation/yu) |
| [4] | NVIDIA H200 Tensor Core GPU | NVIDIA 官方规格 / 2026 | [官方来源](https://www.nvidia.com/en-us/data-center/h200/) | I Introduction；III-A / Fig. 3 | H200 141 GB HBM3e、4.8 TB/s；不支持论文 GPU 动态能耗系数。 | 官方页面或作者原始技术报告核对；[证据页面](https://www.nvidia.com/en-us/data-center/h200/) |
| [5] | Splitwise: Efficient Generative LLM Inference Using Phase Splitting | in Proc. ACM/IEEE ISCA / 2024 | [10.1109/isca59077.2024.00019](https://doi.org/10.1109/isca59077.2024.00019) | I Introduction；II-A | Prefill/decode 资源需求不同。 | Crossref 书目信息交叉核对；出版社/作者/会议摘要或正文核对；[证据页面](https://www.microsoft.com/en-us/research/wp-content/uploads/2023/12/Splitwise_ISCA24.pdf) |
| [6] | FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness | NeurIPS / 2022 | [官方来源](https://proceedings.neurips.cc/paper_files/paper/2022/hash/67d57c32e20fd0a7a302cb81d36e40d5-Abstract-Conference.html) | I Introduction | IO-aware attention 减少 HBM/SRAM 数据移动；不作为本文加速比证据。 | 官方页面或作者原始技术报告核对；[证据页面](https://proceedings.neurips.cc/paper_files/paper/2022/hash/67d57c32e20fd0a7a302cb81d36e40d5-Abstract-Conference.html) |
| [7] | FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU | ICML / 2023 | [官方来源](https://proceedings.mlr.press/v202/sheng23a.html) | I Introduction | GPU/CPU/磁盘 offloading 的内存与数据传输权衡。 | 官方页面或作者原始技术报告核对；[证据页面](https://proceedings.mlr.press/v202/sheng23a.html) |
| [8] | An LPDDR-based CXL-PNM Platform for TCO-efficient Inference of Transformer-based Large Language Models | in Proc. IEEE HPCA / 2024 | [10.1109/hpca57654.2024.00078](https://doi.org/10.1109/hpca57654.2024.00078) | I Introduction | 近存 LLM 加速的背景及 memory-centric 设计。 | Crossref 书目信息交叉核对；出版社/作者/会议摘要或正文核对；[证据页面](https://snu.elsevierpure.com/en/publications/an-lpddr-based-cxl-pnm-platform-for-tco-efficient-inference-of-tr) |
| [9] | Breaking Thermal Bottleneck in 3D HBM-on-GPU Integration via System-Technology Co-Optimization | in Proc. IEEE IEDM / 2025 | [10.1109/iedm50572.2025.11353711](https://doi.org/10.1109/iedm50572.2025.11353711) | I Introduction | HBM-on-GPU 热瓶颈和系统技术协同优化。 | Crossref 书目信息交叉核对；出版社/作者/会议摘要或正文核对；[证据页面](https://static1.squarespace.com/static/67a3eee4385dfb3390804f02/t/692722bf0dad5d330cf60d98/1764172479207/IEDM%202025%20Final%20Program-v5.pdf) |
| [10] | Massive Orthogonal Stacking Assembly of IC (Mosaic) Cube with Inductive Coupling for Exascale Memory Applications | in Proc. IEEE ECTC / 2025 | [10.1109/ectc51687.2025.00227](https://doi.org/10.1109/ectc51687.2025.00227) | I Introduction | MOSAIC 正交组装与感应耦合；非 8 Gb/s 实测依据。 | 书目与作者记录核实；摘要/相关原文交叉验证，未获取完整付费正文；[证据页面](https://fukushima.mech.tohoku.ac.jp/news/?p=4) |
| [11] | A Thermal and System Analysis of SoC-Memory 3-D Stacking Using Massive Orthogonal Stacking Assembly | IEEE Trans. Compon., Packag., Manuf. Technol. / 2026 | [10.1109/tcpmt.2025.3648414](https://doi.org/10.1109/tcpmt.2025.3648414) | I Introduction | 正交堆叠热/性能权衡；2026 正式卷期而非 DOI 年份。 | 书目与作者记录核实；摘要/相关原文交叉验证，未获取完整付费正文；[证据页面](https://klab.t.u-tokyo.ac.jp/en/international-journal-papers/y-2026/) |
| [12] | Capacitor-less, Long-Retention (>400s) DRAM Cell Paving the Way towards Low-Power and High-Density Monolithic 3D DRAM | in Proc. IEEE IEDM / 2020 | [10.1109/iedm13553.2020.9371900](https://doi.org/10.1109/iedm13553.2020.9371900) | II-B / Fig. 2 | IGZO 低漏电、长保持时间、stacking-compatible 2T0C。 | Crossref 书目信息交叉核对；出版社/作者/会议摘要或正文核对；[证据页面](https://ieee-iedm.org/wp-content/uploads/2026/05/2020-IEDM-Archive.pdf) |
| [13] | Tailoring IGZO-TFT architecture for capacitorless DRAM, demonstrating > 10³s retention, >10¹¹ cycles endurance and Lg scalability down to 14nm | in Proc. IEEE IEDM / 2021 | [10.1109/iedm19574.2021.9720596](https://doi.org/10.1109/iedm19574.2021.9720596) | II-B / Fig. 2 | IGZO 2T0C retention/endurance 与尺寸缩放。 | Crossref 书目信息交叉核对；出版社/作者/会议摘要或正文核对；[证据页面](https://imec-publications.be/entities/publication/b1855549-195d-4537-b0c4-3eb4546b036e/full) |
| [14] | Novel Vertical Channel-All-Around (CAA) In-Ga-Zn-O FET for 2T0C-DRAM With High Density Beyond 4F² by Monolithic Stacking | IEEE Trans. Electron Devices / 2022 | [10.1109/ted.2022.3154693](https://doi.org/10.1109/ted.2022.3154693) | II-B / Fig. 2 | CAA IGZO 垂直器件、BEOL 工艺和 4F² 概念。 | Crossref 书目信息交叉核对；出版社/作者/会议摘要或正文核对；[证据页面](https://xplorestaging.ieee.org/document/9732906/figures) |
| [15] | 30 Mb/mm²/layer 3D eDRAM Computing-in-Memory with Embedded BEOL Peripherals and Local Layer-wise Calibration based on First-Demonstrated Vertically-stacked CAA-IGZO 4F² 2T0C Cell | in Proc. IEEE IEDM / 2023 | [10.1109/iedm45741.2023.10413756](https://doi.org/10.1109/iedm45741.2023.10413756) | II-B / Fig. 2；III-B；III-C | 0.023 µm²、20 s measured cell；30 Mb/mm²/layer 是论文 CiM 架构评估密度。 | 联网正式会议/Crossref + 本地原文全文；器件实测/架构评估已区分；[证据页面](https://static1.squarespace.com/static/67a3eee4385dfb3390804f02/t/684859e2d4527577597b1474/1749572070397/2023-IEDM-Archive.pdf) |
| [16] | Monolithic 3D Integration: A Path from Concept to Reality | in Proc. DATE / 2015 | [10.7873/date.2015.1111](https://doi.org/10.7873/date.2015.1111) | II-B / Fig. 2 | 细粒度层间连接和 M3D 集成背景，不验证本文每项 RC。 | Crossref 书目信息交叉核对；出版社/作者/会议摘要或正文核对；[证据页面](https://past.date-conference.com/proceedings-archive/2015/pdf/1111.pdf) |
| [17] | 3D Orthogonal Die Stacking Technology for DRAM-on-GPU Integration Using Contactless Die-to-Die Interface | in Proc. IEEE/JSAP Symp. VLSI Technol. Circuits / 2026 | [10.1109/vlsitechnologyandcir65830.2026.11577466](https://doi.org/10.1109/vlsitechnologyandcir65830.2026.11577466) | II-C；III-A / Fig. 3 | 侧壁供电；4 Gb/s/ch、6.9 pJ/bit 原型；100 µm、8 Gb/s/0.5 pJ 是投影/假设。 | 联网 Crossref + 本地原文全文；实测/设计投影已区分；[证据页面](https://doi.org/10.1109/VLSITechnologyandCir65830.2026.11577466) |
| [18] | A 12.8-Gb/s 0.5-pJ/b Encoding-Less Inductive Coupling Interface Achieving 111-GB/s/W 3D-Stacked SRAM in 7-nm FinFET | IEEE Solid-State Circuits Lett. / 2023 | [10.1109/lssc.2023.3252607](https://doi.org/10.1109/lssc.2023.3252607) | II-C；III-A / Fig. 3 | 7-nm SRAM 感应接口的 0.5 pJ/bit，非正交 MOSAIC 原型。 | Crossref 书目信息交叉核对；出版社/作者/会议摘要或正文核对；[证据页面](https://klab.t.u-tokyo.ac.jp/en/international-journal-papers/y-2023/) |
| [19] | DRAMsim3: A Cycle-Accurate, Thermal-Capable DRAM Simulator | IEEE Comput. Archit. Lett. / 2020 | [10.1109/lca.2020.2973991](https://doi.org/10.1109/lca.2020.2973991) | III-C | DRAM timing/controller 建模相关工作；本文未声称使用 DRAMsim3。 | Crossref 书目信息交叉核对；出版社/作者/会议摘要或正文核对；[证据页面](https://github.com/umd-memsys/DRAMsim3) |
| [20] | A case for exploiting subarray-level parallelism (SALP) in DRAM | ACM SIGARCH Comput. Archit. News / 2012 | [10.1145/2366231.2337202](https://doi.org/10.1145/2366231.2337202) | III-C | ISCA 2012 工作的 SIGARCH 版本；共享外围导致串行化、独立 subarray 带来并行性。 | Crossref 书目信息交叉核对；出版社/作者/会议摘要或正文核对；[证据页面](https://users.ece.cmu.edu/~yoonguk/papers/kim-isca12.pdf) |
| [21] | DreamRAM: A Fine-Grained Configurable Design Space Modeling Tool for Custom 3D Die-Stacked DRAM | in Proc. DATE / 2026 | [10.23919/date69613.2026.11539402](https://doi.org/10.23919/date69613.2026.11539402) | III-C | HBM 的 MAT/subarray/bank/stack 物理、延迟和能量建模；不支持本文新建执行模型。 | Crossref 书目信息交叉核对；出版社/作者/会议摘要或正文核对；[证据页面](https://github.com/harvard-acc/DreamRAM) |
| [22] | Fine-Grained DRAM: Energy-Efficient DRAM for Extreme Bandwidth Systems | in Proc. IEEE/ACM MICRO / 2017 | [10.1145/3123939.3124545](https://doi.org/10.1145/3123939.3124545) | III-C | 阵列与 I/O 数据移动能量、细粒度 DRAM 组织。 | Crossref 书目信息交叉核对；出版社/作者/会议摘要或正文核对；[证据页面](https://research.nvidia.com/sites/default/files/pubs/2017-10_Fine-Grained-DRAM%3A-Energy-Efficient/oconnor_and_chatterjee.micro2017.pdf) |
| [23] | HotSpot: a compact thermal modeling methodology for early-stage VLSI design | IEEE Trans. VLSI Syst. / 2006 | [10.1109/tvlsi.2006.876103](https://doi.org/10.1109/tvlsi.2006.876103) | IV-A | 紧凑热阻网络方法背景；不表示本文使用其瞬态方法。 | Crossref 书目信息交叉核对；出版社/作者/会议摘要或正文核对；[证据页面](https://www.cs.virginia.edu/~skadron/Papers/hotspot_tvlsi06.pdf) |
| [24] | AttAcc! Unleashing the Power of PIM for Batched Transformer-based Generative Model Inference | in Proc. ACM ASPLOS / 2024 | [10.1145/3620665.3640422](https://doi.org/10.1145/3620665.3640422) | IV-B DNS | batched Transformer 的 attention PIM；DNS/CPA 对比。 | Crossref 书目信息交叉核对；出版社/作者/会议摘要或正文核对；[证据页面](https://scale.snu.ac.kr/papers/2024-04-Conference-ASPLOS-AttAcc.pdf) |
| [25] | NeuPIMs: NPU-PIM Heterogeneous Acceleration for Batched LLM Inferencing | in Proc. ACM ASPLOS / 2024 | [10.1145/3620666.3651380](https://doi.org/10.1145/3620666.3651380) | IV-B DNS | GEMM/GEMV 异构协同与重叠调度；不能把异构调度本身当作独创。 | Crossref 书目信息交叉核对；出版社/作者/会议摘要或正文核对；[证据页面](https://www.asplos-conference.org/asplos2024/main-program/abstracts/) |
| [26] | Stratum: System-Hardware Co-Design with Tiered Monolithic 3D-Stackable DRAM for Efficient MoE Serving | in Proc. IEEE/ACM MICRO / 2025 | [10.1145/3725843.3756043](https://doi.org/10.1145/3725843.3756043) | IV-B DNS；IV-C CPA | M3D NMP/GPU 协同、MoE expert tiering 和迁移。 | Crossref 书目信息交叉核对；出版社/作者/会议摘要或正文核对；[证据页面](https://arxiv.org/abs/2510.05245) |
| [27] | Near-Memory LLM Inference Processor based on 3D DRAM-to-logic Hybrid Bonding | in Proc. ACM/IEEE DAC / 2025 | [10.1109/dac63849.2025.11132870](https://doi.org/10.1109/dac63849.2025.11132870) | IV-B DNS | hybrid-bonded DRAM/logic、集中控制器及 dual-IO GEMV/GEMM。 | Crossref 书目信息交叉核对；出版社/作者/会议摘要或正文核对；[证据页面](https://ieeexplore.ieee.org/document/11132870/) |
| [28] | 3D-PATH: A Hierarchy LUT Processing-in-memory Accelerator with Thermal-aware Hybrid Bonding Integration | in Proc. IEEE/ACM MICRO / 2025 | [10.1145/3725843.3756087](https://doi.org/10.1145/3725843.3756087) | IV-B DNS | LUT-PIM 和 thermal-aware hybrid bonding；非本文 MAC FEOL。 | Crossref 书目信息交叉核对；出版社/作者/会议摘要或正文核对；[证据页面](https://doi.org/10.1145/3725843.3756087) |
| [29] | Qwen2.5 Technical Report | arXiv / 2024 | [官方来源](https://arxiv.org/abs/2412.15115) | V-A | Qwen2.5 模型家族；具体 32B 长上下文配置需另核 model card。 | 官方页面或作者原始技术报告核对；[证据页面](https://arxiv.org/abs/2412.15115) |
| [30] | NVIDIA Grace Hopper Superchip Architecture In-Depth | NVIDIA 技术文档 / 2022 | [官方来源](https://developer.nvidia.com/blog/nvidia-grace-hopper-superchip-architecture-in-depth/) | V-A | Grace/Hopper coherency 与 900 GB/s 双向 C2C；不是 H200 SXM 自带 C2C 的证明。 | 官方页面或作者原始技术报告核对；[证据页面](https://developer.nvidia.com/blog/nvidia-grace-hopper-superchip-architecture-in-depth/) |

## 参数与论断逐项审计

| 位置/参数 | 分类与核验结论 | 本次处理/剩余问题 |
|---|---|---|
| II-B 30 Mb/mm²/layer | Tang IEDM 2023 的器件测量 + CiM 架构评估组合；30 是其所提带 BEOL 外围的 CiM 密度，不能说整阵列已实测达该密度 | 将“memories demonstrate”改为“CiM design reports … in architectural evaluation”；保留数值 |
| Fig. 2 0.023 µm² 与 20 s | Tang 原文明确 scaled cell area 与数据“1”存储节点下降 0.1 V 时的保持时间约 20 s | 引用 [15]；不能推广为任意温度、任意工艺与所有状态都保证 20 s |
| Fig. 2 0.288 µm pitch、512×512 | 项目设计/提取输入；未找到公开论文逐项给出这组完整参数 | 图题明确属于 model inputs/SPICE extraction，未包装为全部外部实测 |
| Fig. 2 0.60/368 fJ read 与 0.9/370 fJ refresh | `orthogonal_m3d_igzo.yaml` 记录来自本地 `IEDM2026_HaotongZhu_V5`，为 SPICE_EXTRACTED_MAT_LOCAL_OPERATION_ENERGY，排除 FEOL SA/selectors/NMP/routing/interface | 联网未找到该稿可核验正式出版信息，不收入 References，不暴露其作者身份；公开文献不能独立验证这组值。保留输入并标为模型/SPICE，需作者保管完整提取记录 |
| Fig. 3 38.82 Mb/mm²/layer | 本文计算：512×512×64×280/(5.5×22×10⁶) = 38.82330975 Mb/mm²/layer，采用十进制 Mb | 与 Tang 30 的外围、布局及口径不同。正文 III-B 明确为本架构计算密度，不称为文献实测 |
| 1.49 TB、318 dies、8 layers | 本文计算：512×512×64×280×8×318/8/10¹² = 1.49384331264 TB | 无需外部文献“验证”本文结果；保留所有输入 |
| II-C 100 µm | VLSI 原文 Table I 将 300 µm 与“with thinned Si”100 µm 分列，最后一段是进一步减薄的扩展分析 | 原稿“已制备并组装 100 µm”过强，改为分析投影和本文设计取值；不修改实际模拟厚度 |
| II-C 8 Gb/s/ch、0.5 pJ/bit | VLSI 原型 0.18 µm，实测 4 Gb/s/ch、平均 6.9 pJ/bit；8/0.5 为先进工艺预测/热模型采用值。Shiba 2023 是另一种 7-nm 3D SRAM 感应链路，12.8 Gb/s、0.5 pJ/bit | 改为“modeled here”，补上原型值和 [17]/[18] 归属。不能把 Shiba 的平行堆叠 SRAM 接口等同正交 MOSAIC 实测 |
| 侧壁供电 | VLSI 原文说明 sidewall metal contacts、solder 与 FPC | 用 [17] 替换占位符；不额外声称本文 318-die 制造良率、电源完整性已验证 |
| H200 | NVIDIA 官方：141 GB HBM3e、4.8 TB/s；官方 tensor throughput 区分稀疏与稠密 | Fig. 3 原稿 985.5 TFLOP/s 与项目表 989.5 不一致，未改数字；需作者确认是笔误还是自定义模型参数 |
| GPU 15.29 pJ/bit、0.633 pJ/FLOP | 项目 CSV 标为 DERIVED，使用 H100 相关功率比例和 H200 平台假设、带宽利用率归一化；不是 NVIDIA 实测规格 | Fig. 3 [X] 改脚注 [*] 并说明 model-derived。不能由 H200 产品页面直接证明。候选 xPU-athalon 包含 A100/H100/MI300X 等测试，也不直接测得本稿 H200 两项系数 |
| NMP 0.604 pJ/MAC、1 GHz/512 MACs | 本文 FEOL 设计/能量建模输入；本轮没有查到支持该具体组合的独立测量来源 | 保留原设计，不将 Stratum/Han/3D-PATH 当作数值来源 |
| DreamRAM | DATE 2026 正式 DOI 已确认，正式作者为 V. Cai、J. Zhou、D. Brooks、G.-Y. Wei。GitHub 当前主分支比 DATE2026 分支多若干 timing/布局更新 | [21] 使用正式版本；III-C 限定为 HBM array/peripheral 物理模型。项目确切 commit/config 和所有参数导出表仍需实验复现材料说明；不能声称整体执行模型来自 DreamRAM |
| RC 参数 | 全局互连在配置中标为 MODELING_CHOICE；M3D 文献支持细粒度连接思路，不支持每个具体 R/C | 将“parameters from the literature”调整为 design-specific assumptions；具体 RC 的工艺提取依据仍未公开。MIV 比 TSV 电阻一定更低的泛化措辞也应在后续技术审稿中谨慎检查 |
| Grace/NVLink-C2C | 官方 900 GB/s 为双向总带宽，Grace/Hopper coherent link；H200 SXM 不因此自动具有 Grace C2C | 正文写为“modeled … path”，保留现有模型。若投稿强调商用现货平台，应进一步澄清 SXM-class 计算核与 Grace-like offload 的组合假设；未改仿真 |
| Qwen2.5-32B 126K | 原始报告支持模型家族；[官方 Instruct model card](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct) 对长输入说明 YaRN/长上下文设置 | 物理执行分析不等于已验证 126K 任务精度。作者应明确具体 checkpoint、base/instruct 和 RoPE/YaRN 设置；本轮不更改工作负载 |
| 85°C、热带宽、3.10×、3.34×等 | 85°C 是本文设计约束，后者是本文模型/仿真结果 | 不引用其他论文当成数值证明；未改变稳态方程、边界条件、PCG 或结果 |

## 直接相关工作的区别与创新边界

- **Stratum [26]**：Mono3D DRAM 的层级时延、MoE 专家访问概率与迁移/分层放置，配合 GPU/NMP 和 hybrid-bonded logic。本文 DNS 面向 resident weights/KV 的 decode 映射，CPA 处理 orthogonal FEOL 内 group/region/port 争用。不能把“使用 M3D + GPU/NMP 协同”整体视为从未出现；本次增加客观 related-work 衔接，不改变原贡献列表，也不宣称已完成优于 Stratum 的比较实验。
- **Han DAC [27]**：centralized controller、dual-IO、可重构 GEMV/GEMM 的 hybrid-bonded DRAM-to-logic。本文的正交封装与分布式 FEOL MAC/共享端口层级不同；仅能说明机制区别，不能在缺少统一配置下声称优于它。
- **3D-PATH [28]**：层级 DRAM-LUT/logic fast-LUT，支持 multiplier-free/alignment-free 浮点；self-throttling sense amplifier 和 low-toggle adder 控制热点。本文使用 MAC FEOL，CPA 的直接目标是时延，而非该工作中的 LUT 数据通路与热点电路。
- **AttAcc [24] / NeuPIMs [25]**：已探索 attention/PIM 与 NPU/PIM 异构执行、GEMM/GEMV 分工及调度。DNS 的创新边界应放在本文物理组织下的映射和数据路径，不是异构拆分这一普遍概念。CPA 的 contention-aware placement 应与其批处理/重叠调度区分。
- **PagedAttention [2] / Orca [3]**：软件 KV 分配和请求/迭代级 serving；不直接实现本文物理 FEOL 数据/计算映射。**Splitwise [5]** 为 prefill/decode 阶段资源分离提供背景，不是本稿固定 DNS 策略的实测验证。
- **DreamRAM [21] / DRAMsim3 [19] / SALP [20]**：分别提供物理阵列建模、时序/控制器仿真、共享 subarray 资源分析的相关背景；本文新增的 port serialization、事件计费、DNS/CPA 与 steady-state 热闭环不归属为这些工具自带功能。

## 未入选候选与可扩充文献

候选库按检索过程扩充；表中未入选不表示无关，也不表示每篇全文均已审阅。正式 References 优先直接论据，保留 30 篇，不以泛背景凑 35 篇。新增候选若只得到初步题录，明确标为候选而非已完成论据核验。

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

## 其他原稿问题（按要求保留）

Fig. 5 caption 的 “Thermal s4calability”、III-B 的字面 LaTeX `\(8\times8\)`、Fig. 8/9 的 HoG 与正文 HBM-GPU 命名、图中 H200-class 指标与原始 GPU 规格差异，均列为原稿问题，未进行无关的大规模改写。热讨论提到分散 FEOL activity，但方法使用 die-grouped BEOL-uniform memory/NMP mapping；不应据此宣称已验证 tile-level hotspots，本次不改变热模型或结论。

## 可复核完整性记录

源 DOCX SHA-256：`3a24cf51e08225841c389679014a5a3fb79b107e5dc38a585a05132fe445f71e`

- `DAC2027V1_referenced.docx`：`56653c18a07f30004c6e6ffa7a35fc22347f2aea874a41abb91d3a66c72e2d67`
- `DAC2027V1_referenced.pdf`：`7839f861e33f13088472775e7ddfa2e203308fb3dfce4caef022b288747b4b9d`

正文引文与引用段落按 OOXML 文本验证；图/公式按原始 XML 与媒体字节验证；页数、Conclusion/References 分界及匿名元数据按最终 PDF 验证。引用文本为手工编号。IEEE 格式统一题名、作者缩写、会议简称、年份、页码、DOI；无法确认的页码未补造。SALP 使用已核验的 SIGARCH 40(3) 正式条目，未混用 ISCA proceedings 与期刊卷期。
