# IOM3D-HBM 参数状态清单（2026-09-06）

用途：写作与实验规划时快速查"哪个参数已定、哪个还要校准"。
分级沿用仓库纪律：`PAPER_REPORTED` / `DERIVED_FROM_PAPER_FIGURE` /
`MATCHED_REFERENCE` / `MODELING_CHOICE` / `NUMERICAL_CHOICE` / `PLANNED_REV_V2`。

## 已处理好的参数

| 参数 | 值 | 分级 | 出处 / 说明 |
|---|---|---|---|
| 接口能耗 | 0.5 pJ/bit | PAPER_REPORTED | Shiba SSC-L 2023，能找到的最小接口能耗文献值（commit 1b60079） |
| GPU 仿射功耗模型形式 | P = P_static + (P_peak − P_static)·u | MODELING_CHOICE（形式有三个实测锚点） | docs/research/gpu_power_model_spec_2026-09-05.md；E8 已接入 runner（commit 197bcb1） |
| matched 带宽派生机制 | slab_count × 50 ch × 8 Gbps，带 cap = 39.2 Tb/s | MODELING_CHOICE（派生 + 封顶） | 98 slabs → 39.2 Tb/s；106 slabs 能力 42.4 Tb/s 但场景带宽钉在 39.2（98-slab 等效），多出部分作设计冗余；代码已落地（`cap_bits_per_second`），改 slab 数不需要再手动同步 |
| 带宽冗余表述 | 场景带宽固定 4.9 TB/s，M3D 实际能力 5.3 TB/s 的余量在文字中说明（理想 vs 实际、GPU 侧接口受限） | 写作决策 | 2026-09-06 讨论结论（推翻早前"算出来多少就是多少"） |
| M3D 读能耗（模型值） | 0.855 pJ/bit | 模型分解：内部 0.186 + 垂直 0.002 + FEOL 0.167 + 接口 0.5 | canonical case `orthogonal_m3d_igzo` 解析输出 |
| conventional HBM 读能耗 | 1.397 pJ/bit | DreamRAM 解析标称 | canonical case `conventional_hbm_2x1` |
| 几何 rev v2（**已实施**） | GPU 32×24 mm；stack 11.8×12.2；thermal Si 7.2；106 slabs；y 两侧各 1 mm | IMPLEMENTED_REV_V2 | 图：docs/research/figures/package_layout_v1.png；代码/YAML 已落地 |
| 容量 rev v2（**已实施**） | baseline 145.0 GB（135.0 GiB）；IOM3D 497.9 GB（463.75 GiB，3.4×） | IMPLEMENTED_REV_V2 | 测试断言已按新值重冻结 |
| 双臂热消融设计（**代码已实施**） | A 臂 1 mm mold 填充（主结果，edge_strip_material: Mold）；B 臂 1 mm thermal Si bar（edge_strip_material: Thermal_Silicon，独立 case 文件） | IMPLEMENTED_REV_V2 | 用于分离"正交结构 vs Si 导热条"的热贡献；建场景验证通过，热求解待跑 |
| host DDR5 能力 | 460.8 GB/s | MATCHED_REFERENCE | AMD EPYC 9654 厂商值 |
| H2D 链路效率 η | 0.878125 | MATCHED_REFERENCE | Tyan H100 实测 56.2 / min(460.8, 64.0) |
| GPU↔DDR 链路 | **PCIe Gen5 x16 = 64 GB/s 单向**（128 双向）；备选 NVLink-C2C 450 GB/s 单向（GH200 风格） | VENDOR_SPEC + MEASURED | 表：docs/research/gpu_platform_table_2026-09-06.csv（host_link 段）。主结果用 PCIe 64 GB/s（x86 host 主流、DGX H200 即此），C2C 450 GB/s 放 robustness 敏感性证明容量瓶颈结论不变 |

## 继续校准 / 待定的参数

| 参数 | 现状 | 需要做什么 |
|---|---|---|
| P_static（GPU 静息功耗） | **74 W**（H200 SXM 实测 idle floor，白皮书 72+ 次实测；H200 NVL 121 W 留作敏感性上界）；baseline/proposed 与 H200 同值 | ✅ 已定并实施：platform YAML（gpu_package_h200_reference.yaml）static=74 W |
| e_decode 动态（仿射模型参数） | 模型主值 5.10 pJ/bit（die-only，三架构同值同硅片论证），落在扣后区间 4.88–7.61 内 | ✅ 已定并实施：**固定不变量（板级动态）** H200 锚定区间 **6.28–9.01 pJ/bit**；板级值含 HBM 动态能耗，E2E 在 E4 单独记存储能耗；CSV rev v2 只记 static_power_W + e_decode_dynamic 两个模型参数 |
| P_peak_decode | **269.84 W**（=74 + 5.10e-12×4.8e12×8） | ✅ 已实施：派生量，platform YAML 三字段自洽（fixed=peak=269.84） |
| GPU 峰值带宽（平台侧） | **4.8 TB/s**（H200 厂商值） | ✅ 已实施；场景带宽 cap 4.9 TB/s → u = 4.9/4.8 ≈ 1.02 截断于 1，测试断言 utilization_clamped=True 已重冻结 |
| e_compute（compute-bound 能耗不变量） | 0.531 pJ/FLOP（H200，prefill 75% TDP 档） | 算子分工实施后才真正使用（GPU 只接 attention）；届时确认是否取 75% 还是 100% TDP 档 |
| M3D slab IO 的物理论证 | 50 ch/slab、8 Gbps/ch 是设计值 | 论文中补一句通道数 × pin rate 的可行性问题（preempt 审稿人） |
| B 臂热几何 | ✅ 代码已实施（edge_strip_material 机制 + Thermal_Silicon 材料 140 W/mK） | 剩余：A/B 双臂热求解运行 |
| 冻结基线数字 | 859596 cells / 355.58 W / Tmax 81.93 °C 对应旧几何旧功率；`test_llm_decode_e2e.py` THERMAL/SIZES 表保持旧值未动 | 剩余：全量 formal 实验（三架构×rho）重跑产生真实 GPU-PCG 新值后重新冻结（spec §4 第 3 步） |

## 备注

- GPU 平台 + host 链路 + idle 锚点统一总表：
  `docs/research/gpu_platform_table_2026-09-06.csv`
  （生成器 `tmp/gen_gpu_platform_table.py`；category 列区分
  gpu_energy / host_link / idle_power_anchor；含 rev v2 计划行，标注 PLANNED）。
- 算子分工原则（MAC 只接权重 GEMV 约 15.0 GFLOP/token，GPU 接 attention 约
  68.7 GFLOP/token @128K）已确认，代码未实施、未授权。
