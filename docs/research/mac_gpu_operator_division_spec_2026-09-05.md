# MAC/GPU 算子分工 Spec v0 — 按算术强度划分 decode 计算

日期：2026-09-05
状态：待验证草案，未实现；2026-09-05 复核后暂停作为实现依据。
候选方案：FEOL-MAC 执行权重 GEMV，GPU 执行 attention 与通用算子。
该归属尚未验证，不能仅按 FLOPs 占比或“计算密集/访存密集”标签确定。

**实施前验证门：**

1. 逐算子闭合 FLOPs、活跃/驻留权重与 KV/激活流量，补齐 vocab、归约和非线性的计数或明确开销范围。
2. 在每个 B/S/精度点计算算术强度并对照有效计算与带宽能力；当前 B=1 attention 为约 4 FLOP/byte，在所用平台假设下仍然受带宽限制。
3. 明确 MAC 的指令、累加精度、缓冲及归约能力；比较候选归属，而不是预先认定 GPU 或 MAC 必然更好。
4. 按 QKV → attention → O → FFN 的依赖及 bank/die/接口共享资源构造时序，解释允许重叠的条件。
5. 用算子实际活跃 die 数闭合面积、计算/本地带宽与功率预算；报告相同工作点的性能、能耗及热源。
6. 通过跨模块守恒、串行/重叠边界、B/S regime 与消融检查后，才冻结分工并修改生产路径。

**下文保留原始草案供逐项验证，不能引用为正式结果。** 原表权重字节数、attention 的计算密集标签、全局 max 时序、4.3 TFLOPS 门槛及 1.93×/1.2 J 预测均待修订；“热结论方向不变”也尚未成立。

---

## 1. 划分规则

一个 decode 算子划归 FEOL-MAC，当且仅当：

1. 其数据源是**驻留权重**（每 token 完整流一遍、无跨 token 复用）；
2. 算术强度 ≈ 1 FLOP/byte（GEMV 形态，纯带宽浪费型）；
3. 无数据依赖的全局归约、无非线性（softmax 类）。

反之留在 GPU：需要 tensor core 密度的矩阵结构、长序列归约、非线性、
逐请求动态形状的算子。

## 2. 算子归属表

**Dense decode（LLaMA-3.1-8B 类，B=1, S=131072, fp16）：**

| 算子 | FLOPs/token | 数据/token | 归属 | 理由 |
|---|---:|---:|---|---|
| Q/K/V 投影 | 1.61 G | 权重 0.81 GB | **MAC** | 权重流 GEMV |
| O 投影 | 1.07 G | 权重 0.54 GB | **MAC** | 权重流 GEMV |
| FFN gate/up/down | 11.28 G | 权重 5.64 GB | **MAC** | 权重流 GEMV |
| Vocab 投影 | 1.05 G | 权重 1.05 GB | **MAC** | 权重流 GEMV |
| Attention QK^T + softmax + PV | 68.72 G | KV 17.18 GB | **GPU** | 长序列归约 + softmax + tensor core |
| **MAC 小计** | **15.01 G（18%）** | 权重 ~16 GB 本地 | | |
| **GPU 小计** | **68.72 G（82%）** | KV 17.18 GB 过 coil | | |

**MoE decode（Mixtral-8x7B 类）**：现有口径已一致——expert MLP 是
权重流 GEMV → MAC；attention、shared 权重、routing → GPU。本次划分
把 dense 路径对齐到同一规则，两条路径统一语义：
**"权重流在 memory，其余在 GPU"。**

KV 数据的驻留位置不变（仍在 M3D，享受容量墙收益）；GPU 过 coil 流式
读 KV，与 baseline 从 HBM 读 KV 完全同口径，对比公平。

## 3. 量化影响（一阶估算，canonical matched 场景：coil 39.2 Tb/s，GPU 100 TFLOP/s）

### 3.1 时序与吞吐（B=1）

| 量 | 现状（全 offload） | 新划分 |
|---|---:|---:|
| GPU 过 coil 流量 | ~0（全本地） | 17.18 GB/token（KV） |
| GPU memory time | — | 3.506 ms |
| GPU compute time | — | 0.687 ms（attention，tensor core） |
| MAC compute 需求 | 83.73 GFLOP/token | **15.01 GFLOP/token** |
| 不成为瓶颈所需 MAC 算力 | ≥ ~24 TFLOPS | **≥ ~4.3 TFLOPS** |
| token 时间（roofline） | max(MAC 时序) | max(3.506, 0.687, MAC) = 3.506 ms |
| tokens/s | 受 MAC 算力限制 | **~285（vs 无 offload 147.7，1.93×）** |

关键结论：新划分把"不拖后腿所需的最小 MAC 算力"降低 **5.6×**
（24 → 4.3 TFLOPS），FEOL MAC 预算从"难以辩护"变为"温和"；同时
GPU 承担 82% FLOPs，是系统的注意力引擎，不存在"GPU 多余"问题。

### 3.2 能耗（一阶，含 E8 GPU 能耗）

| 项 | 无 offload（现 E7+E8） | 新划分（估算） |
|---|---:|---:|
| GPU E/token | 由 367.568 W nominal bandwidth-saturated decode 工作点 × token time 派生 | 同一 bandwidth-boundary 模型按实际带宽派生 |
| Memory E/token | 0.227 J（全部流量 × 0.855 pJ/bit） | ~0.14 J（KV 走全路径 0.855；权重走 MAT-local ~0.2） |
| MAC E/token | — | ~0.008 J（15 GFLOP × ~0.5 pJ/FLOP 级锚点） |
| **系统 J/token** | **~2.26 J** | **~1.2 J（~1.9×）** |

以上为 hand-check 级估算，正式值以实现后的 evaluator 输出为准；
E_mac_per_op 锚点与敏感性按 GPU spec §5 处理。

### 3.3 热

attention 计算功率回到 GPU die——GPU 热源叙事与 E8/热模型自洽
（GPU 热输入使用 E8 解析的同一工作点）；MAC 侧功率仅权重 GEMV 部分，
per-die NMP 功率图相应下降。热结论方向不变，需重跑确认。

## 4. 代码改动点（确认后实施）

| # | 位置 | 改动 |
|---|---|---|
| 4.1 | `placement/nmp_locality_e2e.py::build_dense_decode_placement_units` | ATTENTION_KV 不再是 NMP 计算 unit：KV 字节保留 placement（驻留 M3D），其 FLOPs 划入 GPU 侧 |
| 4.2 | NMP 时序模型 | 增加 GPU attention compute time 与 KV coil 流量项；MAC 只计权重 GEMV |
| 4.3 | E8 GPU 能耗（offload 路径） | u_off = (KV + 激活字节)/(BW_peak × T_token)，接入 spec §5 |
| 4.4 | `evaluator/canonical_e2e.py` | gain 口径更新：offload 收益 = 权重流量移除（1.93× @ B=1）而非全流量 |
| 4.5 | 测试 | 更新 canonical NMP 测试期望值（traffic reduction、gain、MAC 算力门限）；新增"算子归属"语义测试 |
| 4.6 | 文档 | NMP 相关 claim 统一为"权重流 GEMV offload"；`nmp_feasibility` 的 MINIMUM_USEFUL_NMP_TFLOPS 口径同步 |

## 5. 会变化的已发布数字（需重跑并更新审计）

- `runs/nmp_*` 与 canonical E2E 的 external traffic reduction（~100% → ~48%）、
  balanced/ideal gain；
- NMP per-die 热基线（MAC 功率下降，GPU 侧不变）；
- MoE 路径数字不变（口径本来就一致）。

这些数字变化是**口径修正**，不是模型错误；重跑后需要更新对应审计文档
并在文档中注明修正原因（算子归属语义收紧）。

## 6. 不变的部分

- MoE expert-only offload 语义；
- KV/权重驻留 M3D 的容量墙叙事与 serving/placement 框架；
- E8 baseline 路径（无 offload 场景）已交付内容；
- 冻结热求解器与 E7 表。
