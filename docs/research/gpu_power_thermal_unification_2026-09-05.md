# GPU 能耗与稳态热源统一（2026-09-05）

## 修改与范围

按用户要求修正正式端到端路径中 GPU 能耗采用仿射功耗、热源却固定 300 W 的不一致。

runner 现在先计算 GPU decode 功率/能耗，再将同一 `GPUDecodeEnergyMetrics` 传入 workload power。封装总功率、GPU 热源和 E7 行都消费 `gpu_power_W`。`fixed_gpu_power_W` 仅保留原 case 参考值，不再驱动配置了 E8 的热路径，也不会额外加进总功率。附带 M3D logic-background 敏感性同步接入平台 GPU 模型。

运行时检查 architecture/rho/容量可行性一致、token 时间一致、memory dynamic energy 一致，并验证：

```text
GPU J/token × aggregate tok/s = GPU W = GPU thermal source W
package W = GPU W + memory workload W
```

E8 的 scoped system J/token 仍是 GPU + memory dynamic，不包含 memory refresh/background/logic。因此完整的封装功率检查还需要把这些静态项加一次。该修改没有升级 GPU 仿射模型的实测可信度，也没有实现待验证的 MAC/GPU offload 分工。

这里使用 workload 平均稳态功率，不涉及瞬态热模型。物理算子、GPU-PCG、网格、容差、选择器/空间映射及 benchmark 配置值均未修改。没有配置 E8 的兼容调用保留显式固定功率。

## 测试

- GPU energy、workload power、thermal mapping、E7 aggregation、formal runner：93 passed。
- M3D parameter sensitivity 兼容测试：1 passed。
- 环境门：原生 Windows `om3dthermal` Python，CPU imports、CuPy 13.6.0 和 GPU kernel probe 通过。
- `tests/test_sweep.py`：25 passed。
- `tests/test_thermal_relaxation.py -k gpu_pcg`：3 passed。

新增覆盖低利用率、满利用率、B=1/2 的 aggregate token 时间语义、无 E8 的固定参考兼容、敏感性路径、工作点混用拒绝、GPU 能耗与功率不守恒拒绝、即使封装总功率相同也拒绝错误 GPU 热源。

## 两个历史 GPU-PCG 接线验证点（已被 rev v2 参数取代）

以下数值只记录 2026-09-05 的旧 100/300 W、4.9 TB/s 参数接线验证，
不是当前 canonical 结果。当前模型使用 74 W、6.28–9.01 pJ/bit reference-derived
range、7.645 pJ/bit nominal midpoint 和 4.8 TB/s；bandwidth-saturated nominal
decode 工作点为 367.568 W。独立 memory energy 不从 GPU coefficient 中扣除。

| 量 | 历史满带宽点 | 历史半带宽验证点 |
|---|---:|---:|
| 接口带宽 Tb/s | 39.2 | 19.6 |
| GPU 利用率 | 1 | 0.5 |
| GPU 模型功率 W | 300 | 200 |
| 离散后 GPU 热源 W | 299.999999999999 | 199.9999999999998 |
| GPU J/token | 2.0314285871 | 2.7085714495 |
| aggregate tok/s | 147.6793237551 | 73.8396618775 |
| 封装功率 W | 355.5354142536 | 228.1522937512 |
| Tmax °C | 81.9256343842 | 59.7943576004 |
| 真相对残差 | 4.5906e-5 | 8.0349e-5 |
| 离散功率闭合误差 W | 4.09e-12 | 2.33e-12 |

两点均收敛，均为 859596 cells / 2531340 edges、FP64、Jacobi GPU-PCG，迭代期间无 full-vector D2H。第二点的 memory dynamic power 也随吞吐降低，因此温差不能全归因于 GPU 功率。GPU J/token 增加是因为该测试点变慢，不能把降低瓦数解释为能效改善。

结果 bundle 与诊断位于（保持 untracked）：

`runs/gpu_energy_thermal_closure_20260905_213529/`

其中 `verification.json` 记录两点检查；两个子目录包含完整输入、power/thermal/energy、provenance 和 checksum。`baseline_drift_diagnosis.json` 记录下面的历史锚点差异。

## 独立发现：历史基线绝对值存在先前漂移

原历史锚点保持为约 355.58349 W、81.93349 °C，未改成此次结果以使检查通过。首次实际重跑对历史功率/温度的严格检查未通过；GPU 接线验证通过不等于该历史锚点复现通过。

诊断证据：

1. 当前刷新功率为 0.769173248754 W；历史表为 0.817246576801 W，差约 0.048073328047 W。dynamic access 与 GPU 功率没有对应变化。
2. 提取本次修改前 HEAD 的 workload-power 与 thermal-mapping 实现，在同一当前配置上执行，封装功率同样为 355.535414253556 W；新旧每个热源的名称、功率、选择器完全相同。
3. 2026-08-26 的 `3fcb308003e49d7936b4649b8ec46be0cdda5828` 将容量实例从两个连续大区域改为四个 10.8×10.8 mm 物理 stack equivalents。刷新事件数跟随几何派生。
4. 仅在内存中的诊断副本恢复该提交前的容量几何/数量，刷新功率恢复到 0.817246576801 W。没有写回配置或更改物理方程。

结论：这是先前容量几何修订与旧冻结表之间的差异，不由此次 GPU 接线引入。保留原锚点并单独记录，后续需要审计该容量/刷新口径及历史证据版本；本次不调整刷新假设、不更新原数值期望。
