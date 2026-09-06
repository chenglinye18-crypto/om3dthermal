# rev v2 优化总结与未来工作报告（2026-09-07）

用途：交接/写作速查。记录平台修订 v2（Platform Revision v2）已完成的优化、
当前端到端收益数字（非热），以及未来还需要做的优化项。
详细规格见 `platform_revision_v2_spec_2026-09-06.md`，参数分级见
`parameter_status_2026-09-06.md`。

## 一、本次优化了什么

### 1. GPU 功耗模型：300 W 名义值 → H200 实测锚定仿射模型

- 旧：fixed 300 W 无依据名义值。
- 新：P = P_static + e_decode·BW_eff·8，P_static = 74 W（H200 SXM 实测
  idle floor），e_decode = 5.10 pJ/bit（die-only 模型主值，落在板级扣存后
  区间 4.88–7.61 内），P_peak 派生 = **269.84 W**（= 74 + 5.10e-12×4.8e12×8）。
- 三架构（baseline / proposed / H200 参考行）同值同硅片论证；平台峰值带宽
  4.9 → 4.8 TB/s（H200 厂商值），场景 cap 4.9 TB/s → u=1.02 钳位于 1。
- 平台文件：`configs/platform/gpu_package_h200_reference.yaml`
  （旧 300w 文件已删除）；300 W 硬编码校验器改写为
  `CANONICAL_GPU_POWER_W = 269.84` 常量 + 豁免 legacy unresolved 案例。

### 2. 封装几何：对齐 H200 GH100-class

- GPU die 30×22 → **32×24 mm**；HBM stack 11×22 → **11.8×12.2 mm**；
  组 footprint 12.4×24.0、中心 ±9.8、thermal Si 条 8.0 → 7.2 mm；
  0.2 mm 间隙保留。
- 容量：baseline 116.0 → **145.0 GB**（135.0 GiB）；IOM3D 460.4 →
  **497.9 GB**（463.75 GiB，3.4× baseline）。
- M3D：slab_count 98 → **106**，cube 30×22 → 31.8×22 mm，slab 设计本身
  （22×5.5 mm、300 µm pitch、8 bitcell 层）冻结，容量随 slab 数线性扩展。
- 热几何硬编码（30/22/8 mm）全部派生化，改几何不再需要动代码。

### 3. A/B 双臂热消融机制（代码就绪，求解待跑）

- M3D y 方向两侧各 1 mm 边条：`edge_strip_material` 机制 +
  `Thermal_Silicon`（140 W/mK）材料。
- A 臂（主结果）= Mold 填充，与 conventional 背景材料一致；
  B 臂（对照）= Si bar，用于回应"热优势来自正交结构还是 Si 导热条"。
- 新 case：`configs/cases/orthogonal_m3d_igzo_edge_si_bar.yaml`。

### 4. 带宽派生 + 封顶全链路统一

- 场景 matched 带宽 = slab_count×50 ch×8 Gbps 派生、cap = 39.2 Tb/s；
  106-slab 能力 42.4 Tb/s 作设计冗余（写作：理想 vs 实际 + GPU 侧接口限制）。
- 新增公共解析 `resolve_scenario_matched_bandwidth_bits_per_second()`，
  formal runner 与 6 个评估脚本统一走它（修复了 4 个脚本读字面值 None 的
  回归）。

### 5. 验证与清理

- 全量测试 **996 passed**（约 20+ 文件的冻结断言按新物理值重冻结，
  全部先记录漂移原因再冻结，非凑通过）；Gate 6–9 通过。
- 唯一未重跑：legacy MOSAIC 98-die CPU relaxation 稳态测试（单次 >290 s，
  用未改动的 legacy 配置，不受影响）。
- `test_llm_decode_e2e.py` 的 THERMAL/SIZES 热冻结表**故意保持旧值**，
  待真实 GPU-PCG 重跑后替换。
- 旧 v0 审计报告（7 份，全部含旧 300 W/旧几何冻结值）已删除；
  `runs/` 旧输出已清理，仅保留 2026-09-07 重跑的 6 项最新结果。

## 二、当前端到端收益（rev v2，非热，2026-09-07 重跑）

Llama-3.1-8B BF16 @128K，balanced 放置：

| N | balanced 增益 | step 时间 | NMP 功率 | 能耗/token | 理想上界 |
|---|---|---|---|---|---|
| 1 | 2.565× | 2.640 ms | 32.45 W | 85.7 mJ | 4.398× |
| 8 | 3.950× | 7.928 ms | 60.60 W | 60.1 mJ | 4.394× |
| 16 | **4.277×**（旧 3.744×） | 13.879 ms | 67.12 W | 58.2 mJ | 4.394× |

- N=16 增益上移是 cap 修复后平衡点正确化的结果，已达理想上界 97.3%。
- die 局部放置（106 die）：1.78/1.73/1.61×（FUSED），Dopt=106 满铺。
- MoE（Mixtral 8×7B）可行性：一阶 NMP 目标 >128 TFLOP/s；N=16 全程
  memory-bound 815.7 tok/s。
- 分层物理服务：18.0 → 10.1–11.1 ns，服务速率 1.62–1.79×。
- 结果位置：`runs/nmp_locality_placement/`、`runs/die_local_placement/`、
  `runs/nmp_feasibility/`、`runs/stratum_tier_service/`、
  `runs/hierarchical_memory_service.json`、`runs/fiddler_moe_placement.json`。

## 三、未来还需要优化 / 待办

按优先级排序：

| # | 事项 | 状态 / 说明 |
|---|---|---|
| 1 | **算子分工（MAC/GPU split）** | 未实施。当前 NMP 包揽 decode 全部 82.68 GFLOP/token，其中 ATTENTION_KV 68.7 GFLOP（83.1%）不应由 FEOL MAC 承担。目标：MAC 只接权重 GEMV（~14 GFLOP/token，memory-bound），GPU 接 attention。规格草稿：`mac_gpu_operator_division_spec_2026-09-05.md`。改完 NMP 增益预计从 ~4.4× 回落但更可信，GPU 利用率叙事闭合 |
| 2 | **全量 formal 热重跑** | 三架构 × rho(0/1/100/1000) 真实 GPU-PCG，产出新 Tmax/cells 冻结值，替换 `test_llm_decode_e2e.py` 旧锚点与 `thermal_results_overview.md` 历史表 |
| 3 | **A/B 双臂热消融求解** | 机制与 case 就绪，跑两臂 Tmax 对比，分离 Si 条贡献 |
| 4 | **e_decode 区间敏感性** | 主值 5.10 pJ/bit；板级扣存区间端点（4.88 / 7.61）做敏感性行 |
| 5 | **compute-bound GPU power** | 已建立 dynamic-only 0.4557857504–0.6326427489 pJ/FLOP sensitivity range；不把含 static 的 0.531–0.707 pJ/FLOP 直接用于 `P_static + dynamic`，且未选择 nominal |
| 6 | **host 链路敏感性** | 主结果 PCIe Gen5 64 GB/s；C2C 450 GB/s robustness 行待跑 |
| 7 | **M3D slab IO 物理论证** | 50 ch/slab、8 Gbps/ch 为设计值，论文补通道数×pin rate 可行性一段 |
| 8 | **写能耗（rho）** | 维持 NOT_VALIDATED + rho 扫描叙事；若找到 IGZO 写能耗文献锚点可收窄 |
| 9 | **legacy 稳态测试提速或退役** | CPU relaxation >290 s 无法前台回归；考虑 GPU 化或移出常测 |

## 四、写作口径提醒

- 三创新点主线：正交结构热路径 → 深层堆叠（热瓶颈）；M3D 大容量本地
  驻留（容量瓶颈）；FEOL MAC 卸载 memory-bound decode（接口带宽瓶颈）。
  NMP 是第三个创新点的手段，不是文章主线。
- 带宽：场景固定 4.9 TB/s，42.4 Tb/s 能力作冗余，文字说明理想/实际差距。
- 容量叙事：同一根 DREAM 标定过的 slab，容量随 slab 数线性扩展
  （98→106），最干净的扩展声明。
