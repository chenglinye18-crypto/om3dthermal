# Orthogonal M3D Memory DAC — PROJECT STATUS

> 总控 Agent 状态报告
> 生成时间: 2026-08-18
> 核查范围: repo 源代码 + configs + tests + ref/

---

## 1. CURRENT STATE（已核实）

### 1.1 Geometry / Architecture — ✅ IMPLEMENTED

| 架构 | Config 文件 | 状态 |
|------|------------|------|
| Conventional HBM | `configs/cases/conventional_hbm_2x1.yaml` | ✅ 完整，可作为 baseline |
| Orthogonal M3D-IGZO | `configs/cases/orthogonal_m3d_igzo.yaml` | ✅ 完整，主 proposed design |
| Orthogonal Si | `configs/cases/orthogonal_si.yaml` | ✅ 完整，ablation 保留 |

**已实现接口 (Architecture Contract 雏形):**
- `capacity`: 由 geometry → packing 自动计算 (bits_per_die / bits_per_slab / layers)
- `read_energy_pj_per_bit` / `write_energy_pj_per_bit`: power model 输出 `E_access_total_pJ_per_bit`
- `bandwidth_capability`: **只有配置值 `read_bandwidth_gbps=39200`，没有 capability closure**
- `physical/power mapping metadata`: `map_system_power_to_thermal()` 已实现 coarse mapping

**关键代码:**
- `src/om3dthermal/power/config.py` — Pydantic schema，含 provenance 字段
- `src/om3dthermal/power/system.py` — `ResolvedSystemPower`, `ResolvedThermalPowerMapping`
- `src/om3dthermal/architecture_comparison.py` — `run_architecture_comparison()` 端到端 pipeline

### 1.2 Power Accounting — ✅ IMPLEMENTED (v0 backend)

**能量分解已闭合 (M3D IGZO):**
| Component | Energy (pJ/bit) | 来源 |
|-----------|----------------|------|
| internal (MAT local) | ~0.186 | Zhu 2026 IEDM operation_table |
| vertical (MIV) | ~0.0024 | DreamRAM length-scaled reference |
| FEOL route | ~0.167 | `calculate_feol_route()` analytical |
| interface | ~0.5 | `constant` (当前最大 uncertainty) |
| **total** | **~0.855** | ✅ energy closure 通过 |

**HBM baseline:** ~1.397 pJ/bit (DreamRAM analytical)

**关键代码:**
- `src/om3dthermal/power/model.py` — `calculate_memory_power()`
- `src/om3dthermal/power/m3d_subarray.py` — Tang embedded subarray topology
- `src/om3dthermal/power/feol_route.py` — Nearest-edge IO model
- `src/om3dthermal/power/miv.py` — MIV electrical model

### 1.3 Thermal — ✅ IMPLEMENTED & MATURE

- Production solver: GPU FP64 matrix-free PCG + Jacobi (也保留 CPU relaxation)
- Sweep 框架已跑通: `configs/sweeps/memory_internal_v0.yaml` (OFAT sweep)
- 34-point sweep 已验证通过
- Solver speed: ~1-2 s/point

**当前 limitation:**
- `interface Tx/Rx power` 的空间映射仍使用 coarse uniform volumetric mapping，未按实际 Tx/Rx placement 拆分
- 不影响 v0，但属于 thermal 的最后一个 physical-model issue

**关键代码:**
- `src/om3dthermal/thermal/gpu_pcg.py` — GPU PCG solver
- `src/om3dthermal/case_runner.py` — `run_steady_pipeline()`
- `src/om3dthermal/architecture_comparison.py` — `compile_case_thermal()`

### 1.4 MAT Sweep — ✅ IMPLEMENTED (已降级)

- Sweep aliases: `mat_rows`, `mat_cols`, `m3d_subarray_rows`, `m3d_subarray_cols`, `activated_row_data_utilization`
- `configs/sweeps/memory_internal_v0.yaml` 定义了 OFAT sweep
- 当前 sweep 主要用于 mechanism study / appendix sensitivity

**注意:**
- Zhu 512×512 anchor 是 validated primitive
- 256/1024 scaling 是 exploratory，不作为主结论

### 1.5 Bandwidth — ❌ NOT IMPLEMENTED (当前最大硬件缺口)

**现状:**
- 仅有 `workload.read_bandwidth_gbps = 39200` (matched/reference value)
- **没有** `BWsystem = min(BWarray, BWMIV, BWFEOL, BWinterface/coil)` 的 closure 计算
- 没有验证 interface/coil 是否是真正的 bottleneck

**代码中完全缺失:**
- `BWarray` (subarray-level aggregate bandwidth)
- `BWMIV` (MIV serialization / pitch limited)
- `BWFEOL` (FEOL routing channel limited)
- `BWcoil` (contactless interface coil limited)

### 1.6 Workload — ❌ NOT IMPLEMENTED (当前最高优先级缺口)

**现状:**
- `WorkloadInput` schema 中只有:
  - `read_bandwidth_gbps` / `write_bandwidth_gbps`
  - `read_data` (p0/p1 probability)
  - `refresh_data`
  - `row_policy` (HBM only)
  - `control_address_reuse` (M3D only)
- **完全没有 LLM decode workload model**

**缺失的全部内容:**
- Model weight footprint 计算
- KV-cache footprint 计算
- `read_bits/token` / `write_bits/token`
- `compute/token`
- Capacity feasibility check
- `tokens/s` = 1 / max(Tcompute, Tmemory)
- `J/token` = Etotal / token
- Workload-dependent power → thermal mapping

### 1.7 End-to-End Evaluator — ⚠️ PARTIAL

当前 `run_architecture_comparison()` 输出:
- ✅ Capacity
- ✅ E/bit (access_energy_pJ_per_bit)
- ✅ Bandwidth (配置值, 非 capability)
- ✅ Tmax
- ✅ Power closure
- ❌ tokens/s
- ❌ J/token
- ❌ capacity feasibility (只有 capacity GiB 数值，没有与 workload footprint 比较)

---

## 2. ASSUMPTIONS & PROVENANCE（当前锚点）

| 参数 | 值 | Provenance | 可信度 |
|------|-----|-----------|--------|
| IGZO read_0 / read_1 | 0.00060 / 0.368 pJ/bit | PAPER_REPORTED (Zhu IEDM 2026) | ⭐⭐⭐ 高 |
| IGZO MAT size anchor | 512×512 | PAPER_REPORTED | ⭐⭐⭐ 高 |
| M3D total E/bit | ~0.855 pJ/bit | DERIVED_FROM_PAPER + MODELING_CHOICE | ⭐⭐☆ 中 |
| Interface E/bit | ~0.5 pJ/bit | MODELING_CHOICE (constant) | ⭐☆☆ 低 (待验证) |
| BW reference | 39.2 Tb/s | MATCHED_REFERENCE_NOT_CAPABILITY_VALIDATED | ⭐☆☆ 低 |
| MIV energy | ~0.0024 pJ/bit | DREAMRAM_LENGTH_SCALED_REFERENCE | ⭐⭐☆ 中 |
| FEOL energy | ~0.167 pJ/bit | MODELING_CHOICE (analytical) | ⭐⭐☆ 中 |
| GPU power | 300 W | MODELING_CHOICE (fixed) | ⭐⭐☆ 中 |
| Package / GPU footprint | 65×65 mm / 30×22 mm | MODELING_CHOICE | ⭐⭐☆ 中 |

---

## 3. OPEN QUESTIONS（维护中）

1. **Zhu 64-layer SPICE anchor 与当前 8-layer M3D 的 operation-energy boundary** — 已处理: 使用 512×512 MAT-local primitive + size scaling，不重复加入 Si-style PRE/ACT
2. **Interface energy sensitivity** — OPEN: 0.5 pJ/bit 是 placeholder，需要 sensitivity analysis
3. **Interface Tx/Rx thermal placement** — OPEN: 当前是 homogenized mapping
4. **Coil/interface bandwidth capability** — OPEN: 没有 BW closure 模型
5. **Array/MIV/FEOL bandwidth ceiling** — OPEN: 没有计算
6. **Workload read/write traffic** — OPEN: 没有 LLM workload model
7. **Data-pattern sensitivity** — OPEN: 当前 read_data p0/p1=0.5 是简化假设
8. **Competitor architecture comparison** — BACKLOG: iso-thermal / iso-throughput 放在第二阶段

---

## 4. DECISIONS（已做）

| 决策 | 时间 | 理由 |
|------|------|------|
| Thermal solver 不再优化 | 当前 | 已足够快 (~1-2s/point)，重点转向 physical model |
| MAT sweep 降级 | 当前 | 不是 DAC 主线，只做 sensitivity / appendix |
| 第一版比较策略: iso-package | 当前 | 最简单、最容易解释的 nominal comparison |
| 第一版 workload: LLM autoregressive decode | 当前 | 最高优先级场景 (long-context, small/medium batch) |
| 不做完整 GPU cycle simulator | 当前 | 第一版用透明 analytical model |
| Orthogonal Si 保留但不阻塞主线 | 当前 | 作为 ablation / mechanism study |
| Tang SPICE / TCAD 进 backlog | 当前 | 等端到端闭环后再决定 |

---

## 5. BACKLOG（明确不现在做）

- [ ] Tang-based simplified SPICE / MAT scaling
- [ ] TCAD 重新搭建
- [ ] Complex GPU cycle simulator
- [ ] Large parameter sweeps (MAT-size optimizer)
- [ ] Detailed MIV optimization
- [ ] MoE decode workload (v1 以后)
- [ ] Iso-thermal / iso-throughput 比较 (第二阶段)

---

## 6. PAPER CLAIMS NOT YET PROVEN

| Claim | 当前状态 |  blocker |
|-------|---------|----------|
| M3D IGZO 比 Conventional HBM 在端到端 LLM decode 中有更好的 J/token | ❌ 未证明 | 缺少 workload model |
| M3D IGZO 比 Conventional HBM 有更高的 tokens/s | ❌ 未证明 | 缺少 BW closure + workload model |
| M3D IGZO 在 iso-package 下容量更大 | ✅ 可证明 | architecture_comparison 已输出 capacity |
| M3D IGZO 的热特性优于 HBM | ⚠️ 部分 | 有 Tmax，但缺少 workload-dependent power mapping |
| 39.2 Tb/s 是 proposed architecture 的 capability | ❌ 未证明 | 缺少 BW closure |

---

## 7. MILESTONES & 最小任务拆分

### Milestone 1 — Architecture Contract (巩固现有接口)

**目标:** 让每个 architecture 稳定暴露统一接口，不做大重构。

| # | 任务 | 工作量 | 状态 |
|---|------|--------|------|
| 1.1 | 确认 `conventional_hbm_2x1.yaml` 输出完整的 ArchitectureMetrics | 小 | ✅ 已实现 |
| 1.2 | 确认 `orthogonal_m3d_igzo.yaml` 输出完整的 ArchitectureMetrics | 小 | ✅ 已实现 |
| 1.3 | **新增:** `bandwidth_capability` 从静态配置升级为 computed property (即使初版是 placeholder) | 小 | 🆕 建议做 |
| 1.4 | 统一 `ArchitectureMetrics` schema，增加 `bandwidth_capability_gbps` 字段 | 小 | 🆕 建议做 |

**验收标准:**
- `python -m om3dthermal power configs/cases/orthogonal_m3d_igzo.yaml` 输出包含所有 energy decomposition
- `run_architecture_comparison()` 对两个主 case 都 PASS

---

### Milestone 2 — LLM Decode Workload v0

**目标:** 建立透明 analytical workload model，输入 model config，输出 traffic / compute per token。

| # | 任务 | 工作量 | 依赖 |
|---|------|--------|------|
| 2.1 | **新建模块** `src/om3dthermal/workload/llm_decode.py` | 中 | 无 |
| 2.2 | 实现 `model_footprint_bytes()` : weight + KV-cache | 小 | 无 |
| 2.3 | 实现 `read_bits_per_token()` : 基于 model size, batch, context, precision | 中 | 无 |
| 2.4 | 实现 `write_bits_per_token()` : KV-cache writeback | 小 | 2.3 |
| 2.5 | 实现 `compute_per_token()` : 透明 analytical (FLOPs → 秒，基于 GPU TFLOPS) | 中 | 无 |
| 2.6 | 实现 `capacity_feasible()` : total_memory_footprint < system_capacity_GiB | 小 | 2.2 |
| 2.7 | **新增 config 块** `workload: {type: llm_decode, model_size_GB, precision, batch_size, context_length, generated_tokens}` | 小 | 无 |
| 2.8 | **验证:** 至少跑一个 concrete example (e.g., Llama-3-8B, batch=1, context=128K) | 小 | 全部 |

**关键设计约束:**
- 第一版只做 analytical equations，不要 cycle simulator
- 明确标注所有 assumption: `MODELING_CHOICE` / `DERIVED_FROM_REFERENCE`
- 至少支持 fp16 / bf16
- 公式透明，可以 hand-check

**验收标准:**
- 输入: model=8B, precision=fp16, batch=1, context=128K
- 输出: weight_footprint_GB, kv_cache_GB, total_GB, read_bits/token, write_bits/token, compute_FLOPs/token, capacity_feasible=True/False

---

### Milestone 3 — End-to-End Evaluator v0

**目标:** 统一计算完整 DAC observable，产出第一张 comparison table。

| # | 任务 | 工作量 | 依赖 |
|---|------|--------|------|
| 3.1 | **新建模块** `src/om3dthermal/evaluator/` 或扩展 `architecture_comparison.py` | 中 | M1, M2 |
| 3.2 | 实现 `effective_memory_time_per_token = read_bits/token / effective_BW` | 小 | 2.3, BW |
| 3.3 | 实现 `compute_time_per_token = compute_FLOPs/token / GPU_TFLOPS` | 小 | 2.5 |
| 3.4 | 实现 `tokens_per_second = 1 / max(T_compute, T_memory)` | 小 | 3.2, 3.3 |
| 3.5 | 实现 `memory_J_per_token = (read_bits*Eread + write_bits*Ewrite) / token` | 小 | power model |
| 3.6 | 实现 `total_J_per_token = E_GPU_per_token + E_memory_per_token` | 小 | 3.5 |
| 3.7 | 实现 `workload_dependent_memory_power = memory_J_per_token * tokens/s` | 小 | 3.4, 3.5 |
| 3.8 | 将 workload-dependent power 传入 thermal mapper + PCG → Tmax | 小 | 现有 thermal |
| 3.9 | **输出:** 包含所有 main system outputs 的 comparison table | 小 | 全部 |

**Main system outputs (必须包含):**
- tokens/s
- J/token
- Tmax
- capacity feasibility

**Supporting outputs:**
- Capacity (GiB)
- BW (Gb/s) — 当前仍是配置值
- pJ/bit

**验收标准:**
- 对 Conventional HBM vs Orthogonal M3D-IGZO 跑同一 workload
- 产出 summary.csv / summary.json
- 所有 power / thermal closure 检查通过
- **STOP，不要继续启动 TCAD / SPICE / 大 sweep**

---

## 8. 近期推荐执行顺序

```
Step 0 (本报告) → 用户确认
    ↓
Step 1: Milestone 1 巩固 (1-2 天)
    - 确认 architecture contract
    - 统一接口 schema
    ↓
Step 2: Milestone 2 核心 (3-5 天)
    - LLM decode workload model
    - 这是当前最大缺口，优先投入
    ↓
Step 3: Milestone 3 整合 (2-3 天)
    - End-to-end evaluator
    - 产出第一张 comparison table
    ↓
Step 4: STOP & Review
    - 检查科学正确性
    - 决定下一阶段 (BW closure? iso-thermal? sensitivity?)
```

---

## 9. RISK ASSESSMENT

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| BW closure 发现 interface/coil 是 bottleneck，且远低于 39.2 Tb/s | 中 | 高 | 提前管理预期；v0 先用配置值，明确标注 `MATCHED_REFERENCE` |
| LLM workload model 假设过多，评审质疑 | 中 | 中 | 所有假设必须标注 `MODELING_CHOICE`；公式透明可 hand-check |
| M3D IGZO 端到端优势不明显甚至劣势 | 低 | 高 | **真实报告，不要调模型** |
| Thermal solver 在 workload-dependent power 下收敛变慢 | 低 | 中 | 现有 solver 已 robust；如出问题可降 rtol 到 1e-3 |

---

## 10. 文件清单 (本次核查)

**已核查的源文件:**
- `src/om3dthermal/power/config.py` — Pydantic schema & validation
- `src/om3dthermal/power/model.py` — `calculate_memory_power()`
- `src/om3dthermal/power/system.py` — System-level power resolution
- `src/om3dthermal/architecture_comparison.py` — End-to-end comparison pipeline
- `src/om3dthermal/case_runner.py` — Steady-state thermal pipeline
- `src/om3dthermal/cli.py` — CLI entry points
- `src/om3dthermal/sweep.py` — OFAT sweep framework

**已核查的配置文件:**
- `configs/cases/orthogonal_m3d_igzo.yaml`
- `configs/cases/conventional_hbm_2x1.yaml`
- `configs/cases/orthogonal_si.yaml`
- `configs/sweeps/memory_internal_v0.yaml`

**已核查的参考文献:**
- `ref/IEDM2026_HaotongZhu_V5.pdf` — Zhu 2026 IEDM (IGZO 2T0C MAT energy anchor)
- `ref/--30-Mb-mm-sup-2--sup--layer-3D-eDRAM-Computin.pdf` — Tang 2023 IEDM

---

## STOP

等待用户确认本状态报告，并授权进入下一步 (Milestone 1/2/3 选择)。
