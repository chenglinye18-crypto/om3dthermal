# GPU 功耗/能效三级仿射模型 — Analytical Spec v0

日期：2026-09-05
状态：8.1–8.3 已实现（platform/gpu_power.py、platform YAML `gpu_decode_power`
块、evaluator/llm_decode_gpu_energy.py 与 runner 接入；测试 31 项通过）。
E8 阶段输出挂在 summary.json 的 `gpu_decode_energy` 键下，冻结 E7 行与
热路径不变。8.4–8.7 待评审后实施。
目标：将 `GPU_ENERGY_MODEL = NOT_AVAILABLE` 升级为
`ANALYTICAL_CALIBRATED_BY_MEASURED_REFERENCE`，使 E2E 表可以输出系统级
J/token（含 GPU），覆盖 baseline 与 FEOL-MAC offload 两种场景。

---

## 1. 设计原则

- 一阶透明公式，可 hand-check；不引入 cycle simulator，不引入新物理。
- 所有参数标注 provenance：`MEASURED_REFERENCE`（文献实测锚点）/
  `MODELING_CHOICE`（模型形式选择）/ `MEASURED_LOCAL`（本机 4070 SUPER
  验证，仅用于模型形式 sanity check，不用于 nominal 取值）。
- nominal 取值只来自数据中心级 GPU（H100 类）的文献实测；本机测量只验证
  曲线形状。
- 每个标量结论都配敏感性区间，结论稳健性不依赖点值。

## 2. 模型形式

decode 相（memory-bound）GPU 功耗对其动态活动近似仿射：

```text
P_gpu = P_static + (P_peak − P_static) × u
u     = GPU 侧实际搬运字节率 / GPU 峰值显存带宽        (decode 相)
u     = GPU 侧实际 FLOP 率 / 峰值 FLOP 率              (compute-bound 相，v0 不用)
E_gpu/token = P_gpu(u) × T_token
```

证据基础（`MEASURED_REFERENCE`）：

- TokenPowerBench（AAAI 2026）：prefill/decode 分相实测，decode 相峰值功率
  比 prefill 低 ~90 W，decode 相功率平稳——支持"decode 相用单一 u 参数化"；
- ML.ENERGY longitudinal analysis（Chung et al.）：GPU power draw 是
  utilization 的直接指示器；小 batch 下静态功耗占比升高、J/token 恶化——
  支持仿射形式与 P_static 项的存在；
- From Words to Watts（HPEC 2023）：250→175 W power cap 下推理时间仅增
  6.7%，能耗降 23%——decode 对功耗帽不敏感（memory-bound），间接支持
  u 由带宽而非功耗帽决定。

## 3. 参数与锚点

| 参数 | 含义 | Nominal 来源 | 敏感性 |
|---|---|---|---|
| `P_static` | GPU 静态/idle 功耗 | ML.ENERGY / TokenPowerBench 实测区间（H100 类） | ±30% |
| `P_peak` | decode 相满载功耗 | 同上；TokenPowerBench decode 相实测 | ±30% |
| `BW_peak` | GPU 峰值显存带宽 | 平台描述符既有值（39.2 Tb/s matched 场景下为平台事实） | 不扫 |
| `T_token` | 每 token 时间 | 现有 E2E evaluator 输出 | 不扫 |

数值锚点（`MEASURED_REFERENCE`，引用用）：

- Llama 3.1 8B on H100：batch 64 约 0.12–0.20 J/token（ML.ENERGY
  longitudinal，vLLM V2/V3 区间）——与本项目冻结 workload 同族模型，
  是系统 J/token 的直接 sanity anchor；
- Llama 2 70B on 8×H200（MLPerf Inference v4.1 closed）：offline 34,864
  tok/s @ 700 W/GPU ≈ GPU 侧 0.16 J/token——高度优化大批量下界参考；
- LLaMA 65B on A100 类（From Words to Watts）：3–4 J/token 量级——
  旧代际/小批量上界参考。

## 4. Baseline（无 offload）场景

```text
u_base = read_bytes_per_token / (BW_peak × T_token)
```

现有 E2E 已输出 `read_bytes_per_token` 与 `T_token`（memory time），
u_base 是派生量，不需要新测量。在 39.2 Tb/s matched 场景下
u_base ≈ 1（带宽即瓶颈），E_gpu/token ≈ P_gpu(u≈1) × T_token——
这等价于现表中的固定 300 W 假设的精细化版本，可用于解释现有结果的
一致性。

## 5. FEOL-MAC offload 场景

卸载后 GPU 不再搬运被 offload 的权重流，剩下的流量为 KV、激活与
非卸载算子权重：

```text
gpu_remaining_bytes/token = KV_read + KV_write + activations
                          + non_offloaded_weight_bytes
u_off  = gpu_remaining_bytes/token / (BW_peak × T_token_off)
E_gpu/token = [P_static + (P_peak − P_static) × u_off] × T_token_off
```

- `gpu_remaining_bytes_per_decode_step` 已在
  `placement/nmp_feasibility.py` 的 workload closure 中存在，直接复用；
- `T_token_off` 取现有 NMP/placement evaluator 的时序输出；
- memory 侧新增 MAC 能耗：`E_mac/token = MACs/token × E_mac_per_op`，
  `E_mac_per_op` 用文献 pJ/MAC 锚点（候选：ISSCC 类 MAC 实测值，IGZO TFT
  MAC 若无可引值则用相近节点 Si 值 + 大敏感性），标注
  `MODELING_CHOICE / NOT_HARDWARE_VALIDATED`；
- 系统 J/token = E_gpu/token + E_mem/token（现有 E4 输出）+ E_mac/token。

## 6. 本机测量（MEASURED_LOCAL，可选，仅验证模型形式）

- 硬件：RTX 4070 SUPER（12 GB GDDR6X，220 W 级）。12 GB 装不下 fp16
  LLaMA-8B（16 GB），用 Q8/Q4 GGUF（llama.cpp）或 LLaMA-3.2-3B fp16；
- 方法：照抄 From Words to Watts——pynvml/nvidia-smi 100 ms 采样，
  prefill/decode 分相，扫 batch ∈ {1, 2, 4, 8, 16} 与 context；
- 产出：P(batch) 曲线形状（验证仿射 + P_static 占比）、decode 相功率
  平稳性（验证单 u 参数化）；
- 纪律：4070 是 GDDR6X 消费卡，绝对值不外推；论文中仅作为
  "model form validated against local measurement" 一句话证据。

## 7. 输出与 claim boundary 升级

实现后 E2E 表新增字段：

```text
gpu_power_model_status = ANALYTICAL_CALIBRATED_BY_MEASURED_REFERENCE
gpu_static_power_provenance = MEASURED_REFERENCE (ML.ENERGY / TokenPowerBench)
gpu_model_form = AFFINE_UTILIZATION_MODEL (MODELING_CHOICE)
system_j_per_token = E_gpu + E_mem (+ E_mac)
system_j_per_token_status = ANALYTICAL_WITH_MEASURED_ANCHORS
```

E2E_ARCHITECTURE.md 的 claim boundary 中
"GPU energy and complete system J/token are unavailable" 一条相应改写；
4070 测量若完成，追加 `model_form_validation = MEASURED_LOCAL_CONSUMER_GPU`。

## 8. 实现拆分（评审通过后）

| # | 任务 | 依赖 |
|---|---|---|
| 8.1 | `src/om3dthermal/platform/gpu_power.py`：仿射模型 + 参数 schema | 无 |
| 8.2 | `configs/platform/gpu_package_300w_reference.yaml` 增加 P_static/P_peak/BW_peak 与 provenance 字段 | 8.1 |
| 8.3 | E2E evaluator 增加 u 计算与 E_gpu/token 输出（baseline 路径） | 8.1 |
| 8.4 | NMP/placement 路径接入 u_off 与 E_mac/token | 8.3 + 现有 nmp closure |
| 8.5 | 敏感性：P_static/P_peak ±30%，E_mac_per_op 区间扫描 | 8.4 |
| 8.6 | （可选）4070 SUPER 本地测量脚本与报告 | 无，独立 |
| 8.7 | 文档与 claim boundary 升级；测试：单调性（u 升 → E_gpu 升）、边界（u=0 → P_static；u≥1 截断）、closure | 全部 |

## 9. 引用清单

- S. Samsi et al., "From Words to Watts: Benchmarking the Energy Costs of
  Large Language Model Inference," IEEE HPEC 2023. arXiv:2310.03003
- C. Niu et al., "Benchmarking the Power Consumption of LLM Inference"
  (TokenPowerBench), AAAI 2026. arXiv:2512.03024
- J. W. Chung et al., "The ML.ENERGY Benchmark," + ML.ENERGY longitudinal
  analysis blog (ml.energy), Llama 3.1 8B/70B on H100
- MLPerf Inference v4.1 closed results, Llama 2 70B on 8×H200
  (mlcommons.org; NVIDIA developer blog 2024-08-28)
- A. Tschand et al., "MLPerf Power: Benchmarking the Energy Efficiency of
  Machine Learning Systems," 2025. arXiv:2410.12032

## 10. 明确不做

- 不做 GPU cycle simulator / 功耗 trace 级建模；
- 不做 prefill 相建模（v0 只覆盖 decode；prefill 能量在论文中单独说明）；
- 不把 4070 测量值用于 nominal；
- 不声称 IGZO MAC 能耗已验证（文献锚点 + 大敏感性）。
