# 存储器能耗拆分口径与 contactless 接口能耗锚点调研

日期：2026-09-05
面向：IOM3D-HBM 论文能耗建模辩护（interface 0.5 pJ/bit 的 provenance 升级）
方法：本地参考论文全文提取（`ref/T8.1_Wednesday_Mitarai_0021.pdf`）+ scholar 检索 + 网络一手来源核查。所有数值均标注来源；未核实项显式列出。

---

## 1. 结论先行

1. **当前模型中的 0.5 pJ/bit 接口项不是无锚点 placeholder。** MOSAIC 原论文（Mitarai et al., VLSI Symp. Circuits 2026，本地 `ref/T8.1`）明确写出其热/功耗估算口径：数据访问 4 pJ/b [O'Connor MICRO 2017]，**其中 I/O 能量 0.5 pJ/b [Shiba SSC-L 2023]**。项目取值与 MOSAIC 官方假设完全一致，provenance 可从 `MODELING_CHOICE (constant)` 升级为 `PAPER_REPORTED (Shiba SSC-L 2023, adopted by Mitarai VLSI 2026)`。
2. **0.5 pJ/b 在 contactless 接口文献中处于已被硅验证的区间**（7 nm 实测 0.5–0.7 pJ/b，12.8/8.5 Gb/s per link）；MOSAIC 原型在 0.18 µm 下实测 6.9 pJ/b，论文自己预期先进工艺降到 0.5 pJ/b @ 8 Gb/s。
3. **"memory 与 interface 分开比较"确有先例**，且恰好是本架构的直接前辈：MOSAIC 就是 access（引用 DRAM 文献值）+ I/O（引用接口电路文献值）分开相加。O'Connor MICRO 2017 则把 HBM2 拆成 activation / data movement / IO wire 三项。DRAM 厂商功率模型（Micron power model、DramDSE）同样 core power 与 I/O power 分开计算。
4. **需要注意的口径风险**：项目 HBM baseline 为 DreamRAM analytical 1.397 pJ/bit，而 O'Connor 的 HBM2 测量级模型为 ~3.9 pJ/bit。两者 accounting boundary 不同（row locality、data-movement 距离、ECC、节点）。论文比较时必须显式说明边界，否则 0.855 vs 1.397 的优势叙事会被 "HBM 文献值明明是 ~4 pJ/bit" 这一类质疑牵连。

---

## 2. Contactless / 电感耦合接口能耗锚点（硅实测）

| 来源 | 工艺 | 速率 | 能耗 | 备注 |
|---|---|---:|---:|---|
| Shiba et al., SSC-L 2023（= MOSAIC ref [8]） | 7 nm FinFET | 12.8 Gb/s | **0.5 pJ/b** | encoding-less 电感耦合，111 GB/s/W 3D-stacked SRAM |
| Shiba et al., JSSC 2023 | 7 nm FinFET | 8.5 Gb/s/link | 0.7 pJ/b | over-SRAM coil，Manchester 编码同步收发 |
| Shiba et al., TCAS-I 2021 | 40 nm CMOS | 3.6 Gb/s/pin | 1.5 pJ/bit/pin | 96 MB 3D-stacked SRAM，0.4 V TX + 12:1 SerDes |
| Mitarai et al., VLSI 2026（MOSAIC 原型，本地 T8.1） | 0.18 µm | 4 Gb/s/ch | 6.9 pJ/b 实测 | 论文预期先进工艺 8 Gb/s @ 0.5 pJ/b |
| Miura et al., ISSCC 2011（TCI） | 0.18 µm | — | 0.9 pJ/b/chip | NAND flash 堆叠，1 coil/channel |

来源：Kuroda 研究室论文列表（kuroda.t.u-tokyo.ac.jp，含 SSC-L 2023 条目 #512 与 MOSAIC VLSI 2026 条目 #536）、Kosuge Lab research theme 页、kotashiba.com publication 页、ISCAS 2020 讲稿对比表、本地 T8.1 全文。

**待核实项**：Shiba SSC-L 2023 的 0.5 pJ/b 的覆盖边界——标题强调 "encoding-less" 且采用 clocked hysteresis comparator，推断为 link 级 TX+RX 能耗；是否包含时钟分配与 SerDes 需读全文确认（IEEE 正文本次未能获取）。论文中引用时建议措辞为 "link energy reported for a 7-nm inductive-coupling transceiver"，并配敏感性。

## 3. HBM 访问能耗总量与拆分（对照组口径）

| 来源 | 对象 | 总量 | 拆分 |
|---|---|---:|---|
| O'Connor et al., MICRO 2017（= MOSAIC ref [7]） | HBM2 | ~3.92–3.97 pJ/bit | activation 1.21（~31%）+ die 内 data movement 2.24（~57%）+ interposer IO wire 0.30（<8%）+ ECC |
| 同上 | QB-HBM / FGDRAM | 3.83 / 1.95 pJ/bit | FGDRAM 靠 grain 本地接口削减 data movement |
| D.U. Lee et al., JSSC 2015（经 ISCAS'20 对比表转引） | HBM (29 nm) | — | I/O energy 3.8 pJ/bit/pin（厂商硅论文口径，含数据通路） |
| K. Sohn et al., JSSC 2017（同上转引） | HBM2 (20 nm) | — | I/O energy 2 pJ/bit/pin |
| Li et al., MEMSYS 2018 | HBM/HMC/GDDR5 | — | core power 与 I/O power 分开建模（Micron power model + 文献 I/O）；结论：高速接口 I/O 功耗占主导 |
| Ha et al., Stanford 学位论文 2018（DramDSE） | HBM Gen1 | — | row/column/refresh/background 分项；IO 为 column 功耗主要成分之一 |

来源：O'Connor MICRO 2017 正文（cs.utexas.edu 公开 PDF）、MEMSYS 2018 正文、DramDSE 学位论文、ISCAS 2020 讲稿对比表。

**关键解读**：O'Connor 口径下 HBM 的 "interface"（interposer 走线）只占 <8%，能耗大头是 die 内 data movement（57%）。这意味着：
- 与 HBM 比能耗时，真正可比的是 "把 bit 从 cell 送到封装外" 的完整路径。本项目的 M3D 拆分（internal 0.186 + MIV 0.0024 + FEOL 0.167 + interface 0.5 = 0.855）恰好对应 O'Connor 的 activation + data movement + IO 全路径，结构上是可辩护的；
- MOSAIC 的 4 pJ/b access + 0.5 pJ/b I/O 与本项目的 "array 侧解析值 + 接口文献值" 是同构口径。

## 4. PIM/存算论文的能耗口径（审稿先例）

| 论文 | 能耗报告方式 |
|---|---|
| Samsung HBM-PIM（ISSCC 2021） | 报告总 pJ/bit 含 PIM 逻辑；后续论文（DEAR-PIM, DATE 2025）引用其 2.75 pJ/bit |
| SK hynix GDDR6-AiM（ISSCC 2022） | 不报告绝对 pJ/bit；主打 "数据搬运减少 → 系统功耗降 80%"、1.25 V、1 TFLOPS MAC |
| J. Song, ISSCC 2025（存储技术综述报告） | HBM-PIM 代际比较：1.9 pJ/bit vs 3.5 pJ/bit |
| Wang et al., VLSI 2023（Xtacking 类堆叠 DRAM） | 0.66 pJ/bit 总访问能耗（hybrid bonding + mini-TSV，85 GB/s/Gbit） |
| O'Connor FGDRAM（MICRO 2017） | 分项拆分（activation / data movement / IO），逐 workload 报告 |

**规律**：电路级论文（ISSCC/JSSC）报总量或 I/O 单项，架构级论文（MICRO/HPCA/DATE）报分项拆分；PIM 论文几乎从不把 "array 能耗" 与 "接口能耗" 混合成单一不可分解数字。本项目目前的四段分解（internal/MIV/FEOL/interface）比多数先例更细，符合架构论文惯例。

## 5. 对本项目的行动建议

1. **升级 provenance**：power config 中 interface 项从 `MODELING_CHOICE (constant)` 改为 `PAPER_REPORTED: Shiba SSC-L 2023 (0.5 pJ/b @ 12.8 Gb/s, 7 nm FinFET inductive coupling), consistent with Mitarai VLSI 2026 assumption`。同步更新 `PROJECT_STATUS.md` 的可信度表（⭐☆☆ → ⭐⭐⭐）与 `docs/architecture/E2E_ARCHITECTURE.md` 的 claim boundary 措辞。
2. **保留敏感性**：建议 0.25 / 0.5 / 1.0 pJ/bit 三档（覆盖 "先进工艺进一步优化" 到 "保守含时钟/SerDes"），证明 E2E 结论不依赖点值。当前 M3D 能耗优势在该区间内方向不变即可写稳。
3. **核对 Shiba 原文覆盖边界**（时钟/SerDes 是否计入），决定敏感性上界取 1.0 还是更高。
4. **统一 HBM 对照口径**：论文能耗比较表中加一列 "accounting boundary"，显式说明 DreamRAM 1.397 与 O'Connor ~3.9 的差异来源；避免审稿人用文献 HBM 值反推本项目 baseline 造假。
5. **引用清单**（论文 related work / 建模依据可直接使用）：
   - K. Shiba et al., "A 12.8-Gb/s 0.5-pJ/b Encoding-Less Inductive Coupling Interface Achieving 111-GB/s/W 3D-Stacked SRAM in 7-nm FinFET," IEEE SSC-L, vol. 6, pp. 65–68, 2023. DOI: 10.1109/LSSC.2023.3252607
   - K. Shiba et al., "A 7-nm FinFET 1.2-TB/s/mm² 3D-Stacked SRAM Module With 0.7-pJ/b Inductive Coupling Interface...," IEEE JSSC, vol. 58, no. 7, pp. 2075–2086, 2023. DOI: 10.1109/JSSC.2022.3224421
   - M. O'Connor et al., "Fine-Grained DRAM: Energy-Efficient DRAM for Extreme Bandwidth Systems," MICRO 2017, pp. 41–54. DOI: 10.1145/3123939.3124545
   - Y. Mitarai et al., "3D Orthogonal Die Stacking Technology for DRAM-on-GPU Integration Using Contactless Die-to-Die Interface," IEEE Symp. VLSI Circuits, 2026.
   - S. Li et al., "A Performance & Power Comparison of Modern High-Speed DRAM Architectures," MEMSYS 2018.
   - Lee et al., "A 1ynm 1.25V 8Gb 16Gb/s/pin GDDR6-based Accelerator-in-Memory...," ISSCC 2022.

## 6. 本次调研未覆盖

- Shiba SSC-L 2023 全文（IEEE 付费墙，未获取；覆盖边界待核实，见 §2）
- HBM3/HBM3E 厂商最新 I/O 能耗（硅论文通常不给可直接用的 pJ/bit，见 `HBM_benchmark_landscape_2026-08-25.md` §3.1）
- IGZO 工艺下实现 coil transceiver 的可行性文献（ TFT 级 RF/高速接口，若审稿人质疑 "7 nm FinFET 的 0.5 pJ/b 能否迁移到 IGZO BEOL"，这是下一个需要锚的点）
