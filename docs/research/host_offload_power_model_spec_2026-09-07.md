# Host DDR + PCIe offload dynamic power model（2026-09-07）

## Scope

本模型只描述 host DDR + PCIe offload 的增量动态功耗。它不包含 host
static/idle power，不存在 compute-bound branch，也不改变 serving overlap、GPU
power、local memory energy 或最终 energy-efficiency reporting。

## Frozen performance boundary

```text
BW_host_eff = eta_offload * min(BW_DDR, BW_PCIe)
BW_host_actual = min(BW_host_demand, BW_host_eff)
host_transfer_time = host_transfer_bytes / BW_host_eff
```

主配置为 `460.8 GB/s` DDR、`64.0 GB/s` 单向 PCIe、`eta=0.878125`，所以
`BW_host_eff=56.2 GB/s`。`host_bandwidth_saturated` 使用严格超限语义：仅当
`BW_host_demand > BW_host_eff` 时为 true，exact boundary 为 false。

在 serving evaluator 中，功耗 operating point 表示 host transfer 活动期间的功耗。
冻结的 `bytes / BW_host_eff` 关系意味着：零 offload 流量解析为 0 B/s；正 offload
传输的活动速率解析在 exact boundary。该报告路径不参与性能计算。

## Incremental dynamic power

```text
bit_rate_host_actual = 8 * BW_host_actual
P_pcie_dynamic = e_pcie_dynamic * bit_rate_host_actual
P_ddr_dynamic = e_ddr_dynamic * bit_rate_host_actual
P_host_offload_dynamic = P_pcie_dynamic + P_ddr_dynamic
```

PCIe 使用 Zhao et al.《Quantifying Interconnect Energy Efficiency on Perlmutter》
Table III 的 `167.9 +/- 10.5 pJ/bit`。这是 A100/Perlmutter host-GPU path 的
cross-platform measured reference，不是 H200 PCIe Gen5 direct measurement。

DDR 系数仅由同一论文 Table II 的单次代表性 H2D 行派生：

```text
e_ddr_dynamic = 4.93 W / (24.94e9 byte/s * 8 bit/byte)
              = 24.709302325581394 pJ/bit
```

其 provenance 是 `SOFTWARE_DERIVED_FROM_PAPER_REPRESENTATIVE_RUN`，限定为
A100 + DDR4 Perlmutter reference；它不是 Table III 多次运行统计结果，也不是
H200 DDR5 direct measurement。

两项之和为 `192.6093023255814 pJ/bit`，但配置仍独立保存 PCIe 与 DDR 分量。
在 `56.2 GB/s` 满活动带宽下：

```text
P_pcie_dynamic = 75.48784 W
P_ddr_dynamic = 11.109302325581394 W
P_host_offload_dynamic = 86.5971423255814 W
```

原论文无法可靠测量 PCIe/NVLink static energy，因此 `P_host_static=UNRESOLVED`。
上述功耗只能标记为 `INCREMENTAL_DYNAMIC_OFFLOAD_POWER`。
