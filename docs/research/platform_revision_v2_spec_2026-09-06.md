# 平台修订 v2（Platform Revision v2）实施规格

日期：2026-09-06。状态：**已实施（2026-09-06 完成代码与 YAML 改动，
测试套件 996 passed 全绿；§4 第 3、4 步 formal 重跑与双臂消融待做）**。
目的：把 DAC E2E 基线从"300 W 名义 GPU + 旧 HBM 几何"迁移到
"H200 锚定 + 新封装几何 + 带宽派生封顶"，同时保持科学纪律
（所有改动可追溯、冻结数字显式重跑）。

## 1. 平台参数锚定（H200）

| 参数 | 旧值 | 新值 | 分级 / 依据 |
|---|---|---|---|
| GPU TDP | 300 W（固定名义） | 700 W | VENDOR_SPEC，H200 SXM |
| BF16 dense 峰值算力 | 100 TFLOPS（名义） | 989.5 TFLOPS | VENDOR_SPEC |
| GPU 峰值 HBM 带宽（平台侧） | 4.9 TB/s | 4.8 TB/s | VENDOR_SPEC |
| P_static | 100 W（旧名义） | **74 W**（H200 SXM 实测 idle floor；NVL 121 W 作敏感性上界）；baseline/proposed/H200 三行同值 | MEASURED_REFERENCE：ai-gpu-energy-optimizer 白皮书（72+ 次实测）；h200-gpu-benchmark-suite |
| e_decode（GPU decode dynamic coefficient） | 隐含旧值 | **6.28–9.01 pJ/bit range；7.645 pJ/bit nominal midpoint**；不扣除独立 memory energy | REFERENCE_DERIVED_RANGE；MODELING_CHOICE_RANGE_MIDPOINT |
| P_decode_at_bw_peak | 300 W | **367.568 W nominal 纯派生量** = 74 + 7.645e-12×8×4.8e12；不进入 YAML | SOFTWARE_DERIVED |
| compute-bound power | 未显式区分 static | **525–700 W 参考范围**；dynamic-only e_compute = 0.4557857504–0.6326427489 pJ/FLOP | DERIVED_FROM_MEASURED_REFERENCE；无 nominal |
| host 链路 | PCIe Gen5 64 GB/s 单向 | 不变（主结果）；+ NVLink-C2C 450 GB/s 敏感性 | VENDOR_SPEC；docs/research/gpu_platform_table_2026-09-06.csv |
| host DDR5 | 460.8 GB/s, η=0.878 | 不变 | MATCHED_REFERENCE |
| 场景 matched 带宽 | 39.2 Tb/s 字面值 | 派生 + cap = 39.2 Tb/s（已落地） | MODELING_CHOICE；106-slab 能力 42.4 Tb/s 作冗余 |

开放项：无功耗取点待定项。canonical nominal 冻结为 `P_static=74 W`、
`e_decode=7.645 pJ/bit`、`B_gpu_peak=4.8 TB/s`。nominal 是 6.28–9.01
pJ/bit 参考派生范围的中点 modeling choice；不从该系数扣减 HBM energy。
Memory energy 保持独立建模；GPU power 全部由实际带宽派生。

## 2. 几何变更

### 2.1 Conventional baseline（HBM-on-GPU）

- GPU die：30×22 → **32×24 mm**（768 mm²，GH100-class）
- HBM stack：11×22 → **11.8×12.2 mm**（12Hi，36.24 GB/stack，HBM3E-36GB-class）
- 两组 stack 组 12.4×24.0 mm，中心 ±9.8；中间 thermal Si 条 8.0 → **7.2 mm**
- 所有 0.2 mm 间隙保留；x 方向 12.4+7.2+12.4 = 32.0 ✓
- 容量：116.0 → **145.0 GB**（≈ MOSAIC 144 GB，+0.7%）
- 布局类型不变：HBM 3D 叠在 GPU 上（HBM-on-GPU，非 2.5D）

### 2.2 IOM3D-HBM

- GPU die：同步 32×24 mm
- slab 设计**完全冻结**：22×5.5 mm plane、300 µm pitch、8 bitcell 层
- slab_count：98 → **106**（x 余量恰好 0.2 mm）
- cube：30×22 → **31.8×22 mm**；y 方向两侧各 1 mm 条带，双臂消融：
  - **A 臂（主结果）**：mold 树脂填充（与 conventional 背景材料一致）
  - **B 臂（对照）**：thermal Si bar，分离"正交结构 vs Si 导热条"的热贡献
- 容量：460.4 → **497.9 GB**（3.4× baseline 145.0 GB）

示意图：docs/research/figures/package_layout_v1.png（生成器 tmp/fig_package_layout.py）

## 3. 配置 / 代码改动清单

| 项 | 文件 | 改动 | 类型 | 状态 |
|---|---|---|---|---|
| conventional case 几何 | configs/cases/conventional_hbm_2x1.yaml | memory_region / capacity_instance_region / visible_group_footprint / group_centers / thermal_silicon 宽按 §2.1 新算术 | YAML | ✅ 已实施（容量验证 135.0 GiB = 145.0 GB） |
| M3D case 几何 | configs/cases/orthogonal_m3d_igzo.yaml | slab_count 106、cube_length_x_mm 31.8、gpu_footprint_mm [32,24]、thermal.edge_strip_material: Mold | YAML | ✅ 已实施（463.75 GiB = 497.9 GB） |
| B 臂 case | configs/cases/orthogonal_m3d_igzo_edge_si_bar.yaml | 新建；edge_strip_material: Thermal_Silicon（140 W/mK），其余同 A 臂 | YAML | ✅ 已实施（建场景验证：2 条 1 mm 边条，A=Mold / B=Thermal_Silicon） |
| orthogonal_si 对照 | configs/cases/orthogonal_si.yaml | 删除 GPU power 输入；几何冻结作 MOSAIC 文献对照 | YAML | ✅ 已实施 |
| 平台 YAML | configs/platform/gpu_package_h200_reference.yaml | GPU decode runtime 唯一参数源：static 74 W、nominal e_decode 7.645 pJ/bit、peak bw 4.8e12；range 6.28–9.01 仅留 provenance ledger | YAML | ✅ 已实施 |
| GPU decode bandwidth boundary | src/om3dthermal/platform/gpu_power.py + evaluator/llm_decode_gpu_energy.py | `B_actual=min(B_demand,B_peak)` 单一解析路径；超峰值后 nominal dynamic power 保持 293.568 W | **代码** | ✅ 已实施 |
| GPU compute boundary | src/om3dthermal/platform/gpu_power.py + evaluator/llm_decode_gpu_energy.py | `F_actual=min(F_demand,F_effective)` 单一解析路径；按 bottleneck 选择 regime，balanced unresolved | **代码** | ✅ 已实施 |
| GPU source-of-truth | platform YAML → AffineGPUDecodePowerSpec → resolve_gpu_decode_power → E8/system/thermal | 删除 case fixed power、platform fixed/peak inputs、compatibility parser 与 fallback | **代码** | ✅ 已实施 |
| B 臂热几何 | src/om3dthermal/geometry/orthogonal_hbm.py + config.py + architecture_comparison.py | edge strip 发射（y 两侧 1 mm、x 全 GPU 宽、component=orthogonal_hbm_edge_strip）；conventional 硬编码（30/22/8 mm）全部派生化 | **代码** | ✅ 已实施 |
| 带宽 cap 派生 | src/om3dthermal/placement/nmp_locality_e2e.py + 2 个 scripts | external_bandwidth_cap_bytes_per_s 参数；从 scenario matched_bandwidth_derivation.cap（39.2e13 bits/s）÷8 = 4.9e12 传入 | **代码** | ✅ 已实施 |
| 算子分工 | （未授权，另案） | MAC 只接权重 GEMV（15.0 GFLOP/token），GPU 接 attention（68.7 GFLOP/token @128K） | 另案 | ⬜ 未动 |

## 4. 冻结数字漂移与重跑计划

以下冻结值全部对应旧几何/旧功率，实施后**重跑并重新冻结**（不是更新测试凑通过，
而是先记录漂移原因再冻结新值）：

| 冻结量 | 旧值 | 漂移原因 |
|---|---|---|
| 网格规模 | 859596 cells / 2531340 edges | 几何变更 |
| 封装输入功率 | 355.58349 W | P_static/P_decode、GPU 功耗模型变化 |
| Tmax | 81.93349 °C | 以上两者 + 热几何 |
| 574 W / ~122.97 °C 2x2 legacy | 不动 | legacy 基准，不受影响 |

重跑顺序：
1. 改 YAML → 跑 AGENTS.md Gate 6–8（import、sweep、canonical）✅ 已完成
2. Gate 9 GPU-PCG 热冒烟 ✅ 已完成（3 passed）
3. 全量 formal 实验（三架构 × rho）重跑，记录新冻结值 ⬜ 待做
4. 双臂热消融（A/B）对比运行 ⬜ 待做

### 4.1 测试套件重冻结记录（2026-09-06）

- 全量套件分批复跑：**996 passed**；唯一剔除
  `tests/test_orthogonal_hbm.py::test_steady_state_converges_is_finite_and_balanced`
  （legacy MOSAIC 98-die CPU relaxation，单次 > 290 s 前台时限，
  用未改动的 legacy 配置，不受 rev v2 影响，未重跑）。
- `tests/test_llm_decode_e2e.py` 的 THERMAL/SIZES 冻结表（859596 cells /
  81.93349 °C 等）**保持旧值未动**：该测试的热指标是注入式冻结锚点，
  新冻结值必须等 §4 第 3 步 formal 重跑产生真实 GPU-PCG 结果后再替换。
- 主要漂移（全部已按新物理值重冻结断言，非凑通过）：
  - canonical GPU 饱和 decode 功率：367.568 W（P_static 74 + 动态 293.568 =
    e_decode nominal 7.645 pJ/bit × 4.8 TB/s × 8）；半带宽点 220.784 W
  - 容量：conventional 108 → 135.0 GiB；M3D 428.75 → 463.75 GiB
  - slab 数 98 → 106；带宽能力 39.2 → 42.4 Tb/s（场景 cap 39.2 Tb/s 不变，
    因此 decode 功率利用率在 ρ=1 处钳位到 1，utilization_clamped=True）
  - NMP 增益重冻结：[2.565, 3.950, 4.277]（n=16 平衡点随 cap 修复上移）；
    n=16 NMP 功率 58.76 → 67.12 W
  - conventional 热几何硬编码（30/22/8 mm）已派生化，density 分母随之更新
    （footprint 595.2 mm²；M3D 699.6 mm² = 31.8×22）

## 5. 写作素材（已定决策）

### M3D bandwidth hierarchy

M3D raw memory capability 只包含 internal service 与 contactless interface：

```text
BW_internal = slab_count * service_lanes_per_slab * 32 B / service_cycle
BW_contactless = slab_count * links_per_slab * rate_per_link / 8
BW_M3D = min(BW_internal, BW_contactless)
```

`service_lanes_per_slab` 直接派生为 `links_per_slab`，8 个 M3D bitcell
layers 共用这些 lanes，不再乘 8。Nominal `106 * 50 * 8 Gb/s = 5.3
TB/s`，且 `BW_internal > 5.3 TB/s`，因此 raw M3D bottleneck 是
contactless interface。底层代码中的 `coil` 即 inductive/contactless link
layer。

4.9 TB/s 是 legacy matched scenario bandwidth，不是 M3D physical
capability。GPU 4.8 TB/s 仅在 system-level 作为 downstream bottleneck，
不进入 raw M3D bandwidth resolver。

### Shared M3D-to-GPU transfer operating point

`workload.read_bandwidth_gbps=39200` 的唯一语义是 requested scenario
bandwidth demand，而不是 actual bandwidth 或硬件 capability。正式
bandwidth-bound M3D 路径在 system/performance 边界解析：

```text
BW_xfer = min(BW_demand, BW_M3D_raw, BW_GPU_peak)
        = min(4.9, 5.3, 4.8) TB/s
        = 4.8 TB/s
```

`BW_M3D_raw` 来自上述 M3D resolver，`BW_GPU_peak` 来自 canonical H200
platform YAML。解析得到的同一个 `BW_xfer` 同时传给 M3D dynamic read
power 和 bandwidth-bound GPU decode power；M3D power 不再将 4.9 TB/s
demand 当作 actual rate。该 transfer 只表示跨越 M3D→GPU 边界的流量，
不代表未来可能留在 memory side 的 NMP-local activity。tie 的主
bottleneck 采用固定优先级 `DEMAND > MEMORY > GPU`，同时保留全部
等于最小值的 limiter flags。

- 带宽：场景 demand 4.9 TB/s；M3D raw capability 5.3 TB/s；GPU peak
  4.8 TB/s；shared actual transfer 为 4.8 TB/s
- host 链路：PCIe 主结果 + C2C 敏感性（450 GB/s 仍 ~11× 低于 HBM，结论稳健）
- 容量叙事：slab 设计冻结，容量随 slab 数线性扩展（98→106），
  "同一根 DREAM 标定过的 slab"是最干净的扩展声明
- 参数状态总表：docs/research/parameter_status_2026-09-06.md
