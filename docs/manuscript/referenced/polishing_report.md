# IOM3D-HBM 全文润色报告

修改基础仅为终稿目录中的 `DAC2027V1_EndNote_10pt_single.docx`。交付：`DAC2027V1_polished.docx`、`DAC2027V1_polished.pdf`。原稿保留。

## 主要修改

- **Abstract / Introduction**：按容量与流量挑战、硬件组织、DNS/CPA、实验收益展开；三条贡献分别对应 Orthogonal IGZO M3D Memory、DNS、CPA，保留其协同关系。
- **Background**：精简 prefill/decode 和 KV 增长的重复解释，保留器件参数、技术条件及引用。含未核实 `[X]` 的 II-C 两段未改写。
- **Architecture / Co-Optimization**：按存储组织、访问路径、执行位置、资源竞争和热约束说明机制；减少反复出现的 “physically grounded”等抽象修饰。
- **Evaluation / Conclusion**：先给收益，再说明局部数据处理、资源竞争和热路径的作用；保留基线策略、工作负载、频率范围及统计条件。结论收束三条贡献，不逐项复述摘要。

共润色 52 个段落。按可提取文本的空白分词统计，标题、正文及图注合计由 3748 词降至 2968 词，约减少 20.8%；不含参考文献。这是明确的语言精简，并非仅调整版式。

## 代表性 Before / After

以下为对应原句或连续节选。

1. **Abstract**
   - Before: “Thermally constrained FEOL frequency scaling further improves additional throughput by 15.5% on average, reaching 31.9% at maximum.”
   - After: “Thermally constrained FEOL frequency scaling provides a further 15.5% geometric-mean throughput improvement, with a maximum of 31.9%.”

2. **Background**
   - Before: “Consequently, long-context decode faces two coupled memory bottlenecks: intensive weight/KV data movement and insufficient local capacity for the expanding working set.”
   - After: “An effective memory architecture must address local capacity and weight/KV movement together.”

3. **Execution modeling**
   - Before: “Consequently, instead of assumed peak bandwidth, execution latency is derived from workload-dependent physical resource usage.”
   - After: “Workload-dependent resource usage determines latency and achieved bandwidth.”

4. **DNS**
   - Before: “More importantly, this execution policy naturally matches the bandwidth hierarchy of IOM3D-HBM.”
   - After: “The mapping follows the memory bandwidth hierarchy.”

5. **Physical execution analysis**
   - Before: “DNS keeps bulk weight/KV processing near the resident data, reducing the layer latency to 439.2 µs, but consequently shifts the critical resources inward to the FEOL fabric and inter-region NoC, while Softmax and reduction remain GPU-side.”
   - After: “DNS reduces layer latency from 938.1 µs with M3D-GPU to 439.2 µs by processing bulk weight/KV operands locally.”
   - 后续句仍保留 FEOL fabric / NoC 成为关键资源，以及 Softmax / reduction 留在 GPU 上的解释。

6. **HBM-resident workloads**
   - Before: “showing that the dominant NMP benefit is not solely caused by avoiding capacity overflow.”
   - After: “These resident-workload gains demonstrate the benefit of locality-aware NMP execution alongside the capacity benefit.”

7. **Thermal behavior**
   - Before: “The lower NMP temperatures do not arise simply from lower total power; distributed FEOL execution spreads activity across the orthogonal memory stack and exploits its short heat-flow paths, reducing hotspot concentration.”
   - After: “Distributed FEOL execution spreads activity across the orthogonal stack and uses its short heat-flow paths, reducing hotspot concentration.”

8. **Frequency scaling**
   - Before: “At B=1, all workloads remain feasible at 3 GHz, yet improve by only 9.6% on geometric average because memory-array service dominates and does not scale with FEOL frequency.”
   - After: “At B=1, all workloads remain feasible at 3 GHz, with a 9.6% geometric-mean gain; memory-array service remains critical.”

## 精简与技术边界

删除了对既有工作的冗长否定、重复解释每个 decode token 都读取权重/KV 的句子、DNS 多次重复的“避免把大数据搬到 GPU”总结，以及 “not solely”“do not arise simply”“yet improve by only”等自我辩护。短上下文/大 batch 的资源成本和低收益案例保留为简短的物理解释，未隐藏其数值或适用条件。

未改架构、模型假设、实验数值、几何与带宽参数、工作负载、公式含义或引用归属。1.49 TB、各架构吞吐/能效比、三段温度范围、938.1/439.2/385.8 µs、频率与 batch 条件均保留；低收益案例中的 1.05×、9.6% 等也仍保留。摘要和结论的 “average” 明确为原 Evaluation 已定义的 geometric mean，没有重算或改变统计口径。没有将模型结果改写为芯片实测。

## 单独记录的原稿问题（未修正）

- **三处 `[X]`**：II-C 的 100 µm 制造/正交组装归属、8 Gb/s/ch 与 0.5 pJ/bit 的 “demonstrated” 表述，以及 Fig. 3 图内 GPU 能耗参数来源。已有文献审计指出实测、减薄扩展分析和先进工艺建模之间的区别；本轮保留原文及标记，不用语言润色替代证据核实。
- **公式显示**：III-B 仍有字面 LaTeX `\(8\times8\)`，并非原生排版的乘法式。该串和原公式对象均未改。其余公式按原对象保留，未重新键入。
- **图题**：Fig. 5 的 “Thermal s4calability” 拼写，以及 Fig 2 / Fig 3 与其他图号的句点格式不一致，均保留。
- **模型叙述歧义**：IV-A 把包括 NMP 在内的功率描述为按 die 均匀分布在 BEOL，而架构/温度讨论强调 FEOL 执行。需要作者确认功率映射与解释的一致性，本轮没有擅自改模型描述。
- **基线名称与工作负载说明**：图中 HoG 与正文 HBM-GPU 的命名对应、Qwen 126K 的具体模型/长上下文配置仍沿用原稿，未补入未经核验的新事实。

## 页数与完整性

- 最终 **6 页正文 + 1 页参考文献**，Word 原生导出 PDF 后逐页检查。
- 保持 IEEE 双栏、10 pt 正文和单倍行距；通过章节分页及末页分栏组织精简后的文字，没有放大字体或修改页边距填页。
- 11 张浮动图表及图注的 OOXML 完全一致；PDF 中全部图注页码及垂直坐标与修改基础一致。
- 21 个媒体文件逐字节一致，所有原生公式、嵌入对象及绘图 XML 一致。
- EndNote 域指令逐项一致，参考文献部分 XML 完全一致，28 条文献及编号保留。未执行转纯文本或重新生成引用。
- 核对了数值及其对应条件；正文和图注中的不同数值集合完全一致，允许删除重复出现的数字叙述。
- 使用保留原压缩包及 XML 结构的局部修改，没有用普通 python-docx 读取后重建全文；未运行仿真或修改实验数据。
