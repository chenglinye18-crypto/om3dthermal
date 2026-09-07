# Current DAC thermal baseline

Conventional 2x1 is the only Conventional reference. Its canonical case is
`configs/cases/conventional_hbm_2x1.yaml`.

> **rev v2 状态（2026-09-07）**：平台修订 v2 已实施（H200 锚定 GPU 功率
> 367.568 W nominal、新封装几何 32×24 mm、106 slabs）。下表为 rev v2 之前的
> 历史回归锚点，对应旧几何/旧功率；新冻结值待全量 formal 实验
> （三架构 × rho）真实 GPU-PCG 重跑后替换。实施细节与漂移记录见
> [platform_revision_v2_spec](../research/platform_revision_v2_spec_2026-09-06.md) §4。

## 历史锚点（rev v2 之前，非当前值）

| Quantity | Canonical reference |
|---|---:|
| Cells | 859,596 |
| Internal edges | 2,531,340 |
| GPU input | 300 W |
| Analytical memory input, rho = 1 reference | 55.5834875816 W |
| Package input, rho = 1 reference | 355.5834875816 W |
| Global Tmax at canonical tolerances | 81.933485 degC |
| Solver | FP64 matrix-free GPU-PCG with Jacobi preconditioning |

These were regression anchors under the pre-rev-v2 platform (300 W nominal
GPU, 30×22 mm GPU die, 98 slabs). 旧 v0 审计报告（含旧冻结值明细）已随
rev v2 退役删除；验证与诊断记录见
[GPU power/thermal unification](../research/gpu_power_thermal_unification_2026-09-05.md)。
The fixed physical operator, tolerances and numerical implementation remain
unchanged by rev v2.

## Architecture comparison status

The earlier M3D NMP result of approximately 83.85 degC uses an old operator
assignment and its own workload power. It is not a matched-work comparison
against the Conventional reference above. No thermal improvement is claimed
from subtracting those values.

Before making a thermal claim, report the same cooling and boundary assumptions,
geometry/package constraints, power accounting and workload definition. Separate
an equal-power geometry comparison from workload-dependent power results.
Report GPU and memory Tmax separately, together with the applicable temperature
limits, capacity and throughput. MAC/GPU partition results require validation
and a new evaluation before inclusion.

rev v2 新增 A/B 双臂热消融设计（y 方向两侧 1 mm 边条：A 臂 mold 填充为
主结果，B 臂 thermal Si bar 为对照），用于分离"正交结构 vs Si 导热条"
的热贡献；双臂热求解尚未运行。

## Historical results

Superseded fixed-power result tables and comparison plots have been removed
from the current paper entry point. Text records remain under `docs/archive/`
for provenance, and historical configurations remain test fixtures only.
The older fixed-power Conventional 2x1 results also must not be substituted
for the analytical baseline above.
