# GPU decode bandwidth/compute regime power model

日期：2026-09-05；2026-09-07 bandwidth/compute boundary 收口。

状态：已实现。canonical 解析位于
`src/om3dthermal/platform/gpu_power.py` 的 decode/compute resolvers。E8 按
performance bottleneck 选择一个 regime；bandwidth 与 compute dynamic power
不相加。这是稳态 decode 平均功耗模型，不是瞬态模型。

## 1. 统一性能模型

```text
t_memory = traffic_bits_per_token / BW_effective + optional_physical_latency
t_compute = FLOPs_per_token / F_effective
t_token_equivalent = max(t_memory, t_compute)
aggregate_tokens_per_second = 1 / t_token_equivalent
per_sequence_tokens_per_second = aggregate_tokens_per_second / batch_size
```

当 `t_compute > t_memory` 时，`tokens/s = F_effective/FLOPs_per_token`；当
`t_memory > t_compute` 时，`tokens/s = BW_effective/traffic_bits_per_token`（无
optional latency 时）。现有 batch semantics 和 `BALANCED` 判定不变。

`F_peak_vendor=989.5 TFLOP/s` 是 H200 BF16 dense vendor capability。
`F_effective` 是 scenario/modeling input，可写成
`eta_compute*F_peak_vendor`，其中 `0 < eta_compute <= 1`；当前兼容场景继续使用
100 TFLOP/s numerical choice，不自动替换为 vendor peak，也不为 eta 选经验值。

## 2. Memory-bound decode

GPU decode 功耗只由 GPU 侧实际服务的带宽决定：

```text
B_actual = min(B_demand, B_gpu_peak)
bit_rate_actual = 8 * B_actual
P_gpu_dynamic = e_decode * bit_rate_actual
P_gpu = P_static + P_gpu_dynamic
```

等价的分段式为：

```text
B_demand <  B_gpu_peak: P_gpu = P_static + e_decode * 8 * B_demand
B_demand >= B_gpu_peak: P_gpu = P_static + e_decode * 8 * B_gpu_peak
```

边界连续。超过 `B_gpu_peak` 后，bandwidth-dependent dynamic power 保持常数；
性能和服务速率继续由既有 bandwidth bottleneck 模型限制，功耗模型不对需求带宽
作超峰值线性外推。

`bandwidth_saturated` 冻结为严格超限语义：仅当
`B_demand > B_gpu_peak` 时为 `true`。在精确边界处 utilization 为 1，
但 `bandwidth_saturated=false`。

## 3. Compute-bound decode

```text
F_actual = min(F_demand, F_effective)
P_gpu_dynamic_compute = e_compute_dynamic * F_actual
P_gpu = P_static + P_gpu_dynamic_compute
```

`F_demand` 是 workload 为匹配另一资源服务时间所要求的 GPU compute rate；
`F_effective` 是明确的 scenario/allowed ceiling，不等同于 vendor peak。边界连续，
超过 ceiling 后功耗保持不变。`compute_saturated` 与 bandwidth 侧相同，采用严格
超限语义：只在 `F_demand > F_effective` 时为 true，精确边界为 false。

H200 参考的 compute-bound total power range 是 525–700 W。CSV 中旧
0.531–0.707 pJ/FLOP 是包含 static 的 total-equivalent coefficient，不能直接
放入 `P_static + dynamic`。dynamic-only 系数为：

```text
e_compute_dynamic_min = (525 - 74) / 989.5e12
                      = 0.4557857503789793 pJ/FLOP
e_compute_dynamic_max = (700 - 74) / 989.5e12
                      = 0.6326427488630622 pJ/FLOP
```

schema 保存 min/max sensitivity range，不选 nominal。compute-bound E8 调用必须
显式选择该范围内的 coefficient；缺少选择时拒绝计算。`BALANCED` power 暂标
unresolved，不发明混合规则，也不把两种 dynamic power 相加。

## 4. 参数、单位和 accounting boundary

| 量 | canonical nominal | 状态 |
|---|---:|---|
| `P_static` | 74 W | H200 SXM measured-reference idle floor |
| `e_decode` range | 6.28–9.01 pJ/bit | REFERENCE_DERIVED_RANGE |
| `e_decode` nominal | 7.645 pJ/bit | MODELING_CHOICE_RANGE_MIDPOINT |
| `B_gpu_peak` | 4.8 TB/s | H200 vendor peak HBM3e bandwidth |
| `P_decode_at_bw_peak` | 367.568 W | nominal 派生/校验量 |
| `F_peak_vendor` | 989.5 TFLOP/s | H200 BF16 dense VENDOR_SPEC |
| `P_compute_bound` | 525–700 W | measured-reference range |
| `e_compute_dynamic` | 0.4557857504–0.6326427489 pJ/FLOP | static 扣除后的 sensitivity range |

```text
P_decode_at_bw_peak
= 74 + 7.645e-12 * 8 * 4.8e12
= 367.568 W
```

`e_decode` 是项目 bandwidth-bounded GPU power model 使用的 GPU decode
dynamic energy-per-bit coefficient。保留的参考派生范围是 6.28–9.01 pJ/bit，
nominal 取区间中点 7.645 pJ/bit；中点是 modeling choice，不是 paper-reported
measurement。这里不应用 HBM-energy subtraction。HBM/M3D memory energy 仍在
独立 memory model 中建模，但不会改变 GPU decode coefficient。GPU power 不按
token 定义；W、B/s、bit/s 与 FLOP/s 是基本量。token time 仅用于把 workload
量换算为 rate，以及将已解析的 W 换算成 J/token。两个 regime 共用同一个
`P_static=74 W`，且 static 只加一次。

YAML 只保存 `P_static`、`e_decode` 和 `B_gpu_peak` 三个独立 decode 参数。
峰值 decode 功率由只读 property 按
`P_static + e_decode * 8 * B_gpu_peak` 即时派生，不是配置输入。

## 5. Canonical data flow and closure

```text
performance bottleneck
  -> MEMORY: resolve_gpu_decode_power (唯一 bandwidth clamp/power 路径)
  -> COMPUTE: resolve_gpu_compute_power (唯一 compute clamp/power 路径)
  -> BALANCED: unresolved
  -> E8 gpu_decode_power_W
  -> workload/package gpu_power_W
  -> thermal source "gpu"
```

不存在 legacy fixed-power 入口或无 E8 fallback。formal runner、GPU energy、
package power、E2E row 和 thermal source 必须使用同一个 resolver operating
point 的 `gpu_decode_power_W`。

闭环关系：

```text
gpu_energy_j_per_token * aggregate_tokens_per_second = gpu_power_W
package_power_W = gpu_power_W + memory_total_power_W
thermal.source_power_breakdown_W["gpu"] = gpu_power_W
```

E8 的 `system_energy_j_per_token` 仅覆盖 GPU + memory dynamic energy；memory
refresh/background/logic 仍由 E5 各加一次，因此不能把这些静态项重复加入 E8。

## 6. Hand checks

| Demand | Actual | Dynamic | Total | Saturated |
|---|---:|---:|---:|---|
| `0.5 * B_peak = 2.4 TB/s` | 2.4 TB/s | 146.784 W | 220.784 W | false |
| `B_peak = 4.8 TB/s` | 4.8 TB/s | 293.568 W | 367.568 W | false |
| `1.2 * B_peak = 5.76 TB/s` | 4.8 TB/s | 293.568 W | 367.568 W | true |

满带宽范围端点闭合：`315.152 / 367.568 / 419.984 W` 分别对应
`6.28 / 7.645 / 9.01 pJ/bit`。

Compute lower endpoint (`e_dynamic_min`)：

| Demand | Actual | Dynamic | Total | Saturated |
|---|---:|---:|---:|---|
| `0.5 * F_ceiling` | 494.75 TFLOP/s | 225.5 W | 299.5 W | false |
| `F_ceiling = 989.5 TFLOP/s` | 989.5 TFLOP/s | 451 W | 525 W | false |
| `1.2 * F_ceiling` | 989.5 TFLOP/s | 451 W | 525 W | true |

上端点闭合：`74 + 0.6326427488630622e-12 * 989.5e12 = 700 W`。

## 7. Claim boundary

模型形式是一阶、透明的 `MODELING_CHOICE`，参数由 measured references 锚定，
不构成某一具体 workload/GPU 的逐点实测复现。冻结范围是 GPU decode power
accounting、bandwidth boundary 与 compute boundary。不改变 workload FLOP、
memory physical energy、NMP placement、MAC/GPU operator division 或 thermal
solver；本次未运行 formal thermal experiment。
