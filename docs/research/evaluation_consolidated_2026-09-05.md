# IOM3D-HBM 评估现状总览（截至 2026-09-05）

本文档汇总当前仓库已完成的全部评估工作、关键数字、provenance 状态与待办项，
作为论文实验章节的工作底稿。单一事实来源仍分别为各专项文档与代码，
本文件只做索引与一致性检查。

2026-09-05 复核更新：Conventional 2x1 是唯一论文基线，退役对照已从本文移除。
MAC/GPU 算子分工及其性能/能耗估算仍为待验证草案，不能作为实现依据或论文结果；
先完成逐算子记账、硬件能力、依赖时序和资源争用验证，再确定归属。

GPU 能耗/热源统一后的真实求解发现：当前满利用率 2x1 输入为 355.535414 W，
Tmax 81.925634 °C；差异来自先前容量几何变更所影响的刷新功率，修改前 HEAD
产生相同热源。下文 355.58349 W / 81.93349 °C 保留为历史锚点，不能声称已
精确复现。见 `gpu_power_thermal_unification_2026-09-05.md` 的诊断记录。

---

## 1. 流水线总览（E1–E8）

```text
实验 YAML → 架构/平台/workload 描述符 → 容量/流量/FLOPs
→ matched-reference 性能 → 条件存储能耗 → E8 GPU decode 仿射功率/能耗
→ workload 功率（共用 E8 GPU W）→ 热映射
→ FP64 matrix-free GPU-PCG（冻结）→ Tmax 观测
```

| 阶段 | 内容 | 状态 |
|---|---|---|
| E1–E6 | 容量 / 流量 / 性能 / 能耗 / 功率 / 热 | 冻结，canonical gate CONDITIONAL_PASS |
| E7 | canonical E2E 表（`configs/cases/conventional_hbm_2x1.yaml`） | 冻结 |
| E8 | GPU decode 能耗（仿射利用率模型） | **baseline 能耗/封装功率/热源已统一**；offload 路径（8.4）待实施 |
| — | 接触式接口能耗 0.5 pJ/bit | provenance 已升级为 PAPER_REPORTED |

## 2. 冻结不变量（不得漂移）

| 量 | 冻结值 | 来源 |
|---|---:|---|
| Conventional 2x1 网格 | 859,596 cells / 2,531,340 edges | GPU-PCG 基线 |
| 解析封装输入功率 | ≈ 355.58349 W | E7 |
| Conventional 2x1 Tmax | ≈ 81.93349 °C（canonical 容差） | E7 |
| 求解器 | FP64 matrix-free GPU-PCG + Jacobi，真 KCL 残差 | 冻结 |

## 3. 已完成评估与关键数字

### 3.1 Canonical E2E（LLaMA-3.1-8B 级，B=1, S=128K, fp16）

最新一次完整运行：`runs/e2e_canonical/summary.json`（2026-09-01，
**早于 E8 与算子分工修订，见 §6 注意事项**）。

| 指标 | 数值 |
|---|---:|
| 容量门 | PASS：working set 33.18 GB / 总容量 460.4 GB，headroom 427 GB（利用率 11.4%） |
| 无 offload 吞吐 | 147.7 tok/s（6.77 ms/token，接口 39.2 Tb/s 满载） |
| locality-only gain | 1.99×（294.5 tok/s） |
| balanced gain | 2.57×（378.8 tok/s） |
| ideal gain | 4.07×（600.5 tok/s） |
| 外部流量削减 | 99.97%（旧"全 offload"口径，**将随算子分工修订变为 ~48%**） |
| M3D+NMP 能耗 | 0.0857 J/token（仅存储侧 + NMP，非系统级） |
| 热 | global Tmax 83.85 °C（GPU 区）；M3D Tmax 73.60 °C，ΔT=6.38 °C |
| M3D 聚合 NMP 功率 | 32.45 W（mac 9.46 W + read 22.94 W） |

### 3.2 接口能耗 0.5 pJ/bit（commit 1b60079）

- provenance：`MODELING_CHOICE` → **`PAPER_REPORTED_INDUCTIVE_LINK_ENERGY`**，
  锚定 Shiba et al. SSC-L 2023（7 nm FinFET 电感耦合，12.8 Gb/s，0.5 pJ/b），
  与 MOSAIC（Mitarai et al., VLSI Symp. Circuits 2026）官方假设完全一致。
- 硅实测区间支持：0.5–0.7 pJ/b（7 nm）；MOSAIC 原型 0.18 µm 实测 6.9 pJ/b、
  预期先进工艺回到 0.5 pJ/b。
- "memory 与 interface 分开比较"先例确认：MOSAIC（4 pJ/b access + 0.5 pJ/b I/O）、
  O'Connor MICRO 2017（HBM2 三分项）、Micron power model 均分项。
- **口径风险已记录**：项目 HBM baseline 1.397 pJ/bit（DreamRAM 解析）vs
  O'Connor HBM2 ~3.9 pJ/bit，accounting boundary 不同，论文比较表须显式标注。
- 专项文档：`interface_energy_landscape_2026-09-05.md`（含完整引用清单）。
- 建议敏感性档位：0.25 / 0.5 / 1.0 pJ/bit（待实施）。

### 3.3 E8：GPU decode 仿射能耗（commit 197bcb1）

模型：`P_gpu = P_static + (P_peak − P_static)·u`，u = GPU 侧字节率 / 峰值带宽。

- 参数（`configs/platform/gpu_package_300w_reference.yaml`）：
  P_static = 100 W，P_peak = 300 W，BW_peak = 4.9 TB/s；
  状态 `PARAMETRIC_NOMINAL_WITHIN_MEASURED_REFERENCE_RANGE`。
- 证据锚点（MEASURED_REFERENCE）：ML.ENERGY longitudinal（Llama 3.1 8B on H100）、
  TokenPowerBench（AAAI 2026，decode 相功率平稳且低于 prefill ~90 W）、
  From Words to Watts（HPEC 2023，decode 对功耗帽不敏感）。
- baseline（u≈1）结果：E_gpu ≈ 300 W × 6.77 ms ≈ **2.03 J/token**；
  系统 J/token ≈ **2.26 J**（GPU 2.03 + 存储 0.227），
  claim 升级为 `ANALYTICAL_CALIBRATED_BY_MEASURED_REFERENCE`。
- 实现：`platform/gpu_power.py` + `evaluator/llm_decode_gpu_energy.py`，
  runner 已接入；u=1 时精确恢复旧 300 W 固定假设（回归安全）。

- 2026-09-05 修订：E8 前移至热求解之前，GPU W 同时用于能耗、封装总功率
  和 GPU 热源；`fixed_gpu_power_W` 仅保留配置参考，实际值为 `gpu_power_W`。
  该改动不改变 GPU-PCG 算子或数值实现，也不代表 offload 分工已经验证。
- 专项文档：`gpu_power_model_spec_2026-09-05.md`（§8.1–8.3 已完成）。

### 3.4 MAC/GPU 算子分工与带宽放大（2026-09-05，SPEC 已实现前分析）

候选方案（待验证）：**FEOL-MAC 接权重流 GEMV，GPU 接 attention
与通用算子。** 当前假设下 attention 算术强度约 4 FLOP/byte，仍受带宽限制；
归属必须依据硬件能力、依赖与通信成本验证，不能按“计算密集”标签确定。

| 划分 | FLOPs/token | 数据/token |
|---|---:|---|
| MAC：Q/K/V/O + FFN + Vocab 投影 | 15.01 G（18%） | 权重 ~16 GB，本地不过 coil |
| GPU：QK^T + softmax + PV | 68.72 G（82%） | KV 17.18 GB，过 coil |

量化结论（canonical matched 场景）：

- 接口流量 33.18 → 17.18 GB/token → **吞吐 147.7 → ~285 tok/s（1.93×）**；
- MAC 算力门槛 24 → **4.3 TFLOPS**（降低 5.6×，FEOL 预算可辩护）；
- 系统 J/token ~2.26 → **~1.2 J（~1.9×）**（hand-check 级，以实现后为准）；
- 放大倍数 regime：A(S) = 1 + 16 GB / KV(S)：
  8K → 15.9×，32K → 4.7×，128K → 1.93×，1M → 1.12×；
  **MAC 管权重主导区间，容量墙管 KV 主导区间**——三墙叙事分工自洽。
- 专项文档：`mac_gpu_operator_division_spec_2026-09-05.md`；
  图：`figures/mac_gpu_operator_division_v0.png`。

## 4. 测试状态

- 提交 1b60079 / 197bcb1 时全量套件通过；
- 2026-09-05 复验：gpu_power / gpu_energy / interface 相关 **37 项全部通过**；
- 冻结热求解器与 E7 表未被今日改动触碰。

## 5. 三墙叙事 ↔ 评估映射（论文故事线自查）

| 创新点 | 对应评估 | 现状 |
|---|---|---|
| 正交 M3D 架构 → 热墙 | 唯一 Conventional 基线为 2x1（canonical 约 81.93349 °C）；旧 M3D NMP 点 83.85 °C 尚未与其匹配工作点 | 热优势未建立；需同条件比较及温限内容量/性能评估 |
| 超大容量 → 容量墙 | E2E 容量门：460 GB 总容量，headroom 427 GB | 有数字；**多层堆叠容量叙事未跑** |
| FEOL-MAC → 接口带宽墙 | 算子分工草案；1.93× 暂仅作理想流量上界 | **建模待验证，代码未实施** |

## 6. 待办与注意事项（按优先级）

1. **先验证算子分工建模，再实施代码**（spec 顶部验证门及 §4）；
   实施后 `runs/nmp_*` 与 canonical E2E 的 traffic reduction（~100%→~48%）、
   balanced/ideal gain、NMP per-die 热基线都会变化，属口径修正而非模型错误，
   重跑后需更新审计文档。
2. **E8 接入 offload 路径**（spec §8.4：u_off、E_mac/token）——依赖第 1 项。
3. **完整形式实验重跑**：现有 `runs/e2e_canonical/summary.json` 为 09-01 旧版，
   不含 gpu_decode_energy；完整运行超前台时限，需用户自行整跑。
4. **敏感性扫描**：接口 0.25/0.5/1.0 pJ/bit；P_static/P_peak ±30%；E_mac_per_op 区间。
5. **文献缺口**：Shiba SSC-L 2023 全文覆盖边界（时钟/SerDes 是否计入，IEEE 付费墙）；
   IGZO BEOL 实现 coil transceiver 的可行性锚点（审稿人可能质疑 7 nm FinFET 值的可迁移性）。
6. **可选**：4070 SUPER 本地测量（llama.cpp GGUF + pynvml，仅验证仿射模型形式，
   不用于 nominal 取值）。
7. **未决定**：论文题目（候选："IOM3D-HBM: Breaking the Thermal, Capacity, and
   Bandwidth Walls of LLM Inference with IGZO Orthogonal Monolithic 3D Memory"）；
   热建模粗细程度（J/bit 简略口径 vs 适度细化）。

## 7. 文档索引

| 文件 | 内容 |
|---|---|
| `PROJECT_STATUS_AND_RESEARCH_DIRECTION_2026-08-25.md` | 项目状态与研究方向（08-25 版） |
| `HBM_benchmark_landscape_2026-08-25.md` | HBM 基准数据调研 |
| `interface_energy_landscape_2026-09-05.md` | 接口能耗锚点与拆分口径 |
| `gpu_power_model_spec_2026-09-05.md` | GPU 仿射能耗模型 spec（§8.1–8.3 已实现） |
| `mac_gpu_operator_division_spec_2026-09-05.md` | 算子分工 spec（未实现） |
| `figures/mac_gpu_operator_division_v0.png` | 带宽放大三联示意图 |
