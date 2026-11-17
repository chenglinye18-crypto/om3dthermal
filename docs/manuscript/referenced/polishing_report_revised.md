# 保留原稿逻辑的语言润色修正版

本版取代上一版过度压缩的润色稿。以 DAC2027V1_EndNote_10pt_single.docx 为基础，恢复原稿段落顺序、论证层次、解释内容和问题—方案衔接，仅对 40 个段落作局部语言调整。未沿用上一版新增的强制分页和分栏。

## 篇幅与逻辑

同一空白分词口径，含标题、正文及图注，不含参考文献：原稿 3748 词，本版 3762 词，增加 14 词（约 0.4%）；上一版为 2968 词。本版没有删除整段或整句解释。保留摘要的“两项挑战—为解决这两项挑战提出架构—DNS/CPA—实验结果”逻辑，保留原稿贡献段的组织和完整的背景、数据流、瓶颈变化及实验条件分析。

## 代表性局部修改

| Before | After |
|---|---|
| To address both, we present | To address these two challenges, we present |
| Their inference consists of two successive phases | Inference proceeds in two successive phases |
| A total of 318 100-µm-thick dies | A total of 318 dies, each 100 µm thick, |
| Memory and communication resources are explicitly contention-aware. | The model explicitly accounts for contention in memory and communication resources. |
| Consequently, instead of assumed peak bandwidth, execution latency is derived from workload-dependent physical resource usage. | Execution latency is therefore derived from workload-dependent physical resource usage rather than assumed peak bandwidth. |
| More importantly, this execution policy naturally matches | This execution policy also matches |
| yet improve by only 9.6% on geometric average | with a geometric-mean improvement of 9.6% |
| Thermal-Constrained Performance Headroom | Thermally Constrained Performance Headroom |

## 技术与完整性

没有修改架构、模型假设、工作负载、实验数值或公式。将概述中的 average 明确为原评估使用的 geometric mean，未重新计算结果。只调整个别强调或防御性措辞，没有删除其技术原因或适用条件。

逐项核对通过：21 个媒体文件字节一致；11 个图表容器 XML 一致；EndNote 域指令、公式、嵌入对象、参考文献和分节设置一致；数值集合一致；所有图题的 PDF 页码及垂直位置与原稿一致。原始包内除正文 XML 外的文件均字节一致。

通过 Word 原生导出 PDF，并检查全部 7 页：6 页正文、1 页参考文献。保留 10 pt 正文、单倍行距、双栏排版。未运行仿真或改动项目代码。

## 原稿待确认问题（本版未擅自修复）

- 未解决的 `[X]` 引用占位符。
- III-B 中直接显示的 LaTeX `\\(8\\times8\\)`。
- Fig. 5 的 “Thermal s4calability” 拼写，以及 Fig. 2/3 图题标点。
- 热模型将 NMP 功率分配于 BEOL，与架构中 FEOL NMP 的表述关系需作者确认。
- 图中 HoG 与正文 HBM-GPU 的基线命名，以及 Qwen 126K 配置的具体依据需作者确认。

交付：DAC2027V1_polished_revised.docx、DAC2027V1_polished_revised.pdf。本版另存，避免覆盖正在使用的旧润色稿。
