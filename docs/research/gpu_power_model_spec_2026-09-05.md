# GPU decode bandwidth-boundary power model

日期：2026-09-05；2026-09-07 bandwidth-boundary 收口。

状态：已实现。canonical 解析位于
`src/om3dthermal/platform/gpu_power.py::resolve_gpu_decode_power`；E8、formal
runner、workload/package power 和 thermal GPU source 共用同一个解析工作点。
这是稳态 decode 平均功耗模型，不是瞬态模型。

## 1. 冻结的物理定义

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

## 2. 参数、单位和 accounting boundary

| 量 | canonical nominal | 状态 |
|---|---:|---|
| `P_static` | 74 W | H200 SXM measured-reference idle floor |
| `e_decode` | 5.10 pJ/bit | GPU-side effective decode coefficient |
| `B_gpu_peak` | 4.8 TB/s | H200 vendor peak HBM3e bandwidth |
| `P_decode_at_bw_peak` | 269.84 W | 派生/校验量 |

```text
P_decode_at_bw_peak
= 74 + 5.10e-12 * 8 * 4.8e12
= 269.84 W
```

`e_decode` 从实测 decode 动态功耗反推，并扣除了 E4 memory energy 模块已经
单独计账的部分。它是 GPU-side effective decode coefficient，不能重新解释为
单纯 memory-I/O energy。模型不含 `e_compute`，也不按 token 定义功率；W、B/s
和 bit/s 是基本量。token time 仅用于将每 token GPU 流量换算成 `B_demand`，以及
将已解析的 W 换算成 J/token。

YAML 中为兼容性保留 `peak_decode_power_W`，但 schema 强制它等于
`P_static + e_decode * 8 * B_gpu_peak`。它不是可独立标定或扫描的物理参数。

## 3. Canonical data flow and closure

```text
performance bytes/token + token time
  -> B_demand
  -> resolve_gpu_decode_power (唯一 clamp / power 解析路径)
  -> E8 gpu_decode_power_W
  -> workload/package gpu_power_W
  -> thermal source "gpu"
```

动态路径不得重新读取 legacy fixed power。`fixed_gpu_power_W` 只为没有 E8
模型的兼容调用和 case/platform 一致性检查保留；formal runner 配置 E8 时，GPU
energy、package power、E2E row 和 thermal source 必须使用完全相同的
`gpu_decode_power_W`。

闭环关系：

```text
gpu_energy_j_per_token * aggregate_tokens_per_second = gpu_power_W
package_power_W = gpu_power_W + memory_total_power_W
thermal.source_power_breakdown_W["gpu"] = gpu_power_W
```

E8 的 `system_energy_j_per_token` 仅覆盖 GPU + memory dynamic energy；memory
refresh/background/logic 仍由 E5 各加一次，因此不能把这些静态项重复加入 E8。

## 4. Hand checks

| Demand | Actual | Dynamic | Total | Saturated |
|---|---:|---:|---:|---|
| `0.5 * B_peak = 2.4 TB/s` | 2.4 TB/s | 97.92 W | 171.92 W | false |
| `B_peak = 4.8 TB/s` | 4.8 TB/s | 195.84 W | 269.84 W | false |
| `1.2 * B_peak = 5.76 TB/s` | 4.8 TB/s | 195.84 W | 269.84 W | true |

## 5. Claim boundary

模型形式是一阶、透明的 `MODELING_CHOICE`，参数由 measured references 锚定，
不构成某一具体 workload/GPU 的逐点实测复现。冻结范围是 GPU decode power
accounting 与 bandwidth boundary；不引入 compute-power 模型，不改变 workload
FLOP、memory physical energy、NMP placement、MAC/GPU operator division 或 thermal
solver。
