# M3D-GPU port-balanced placement diagnostic

Status: **DIAGNOSTIC_ONLY**. Verdict: **PORT_BALANCED_PLACEMENT_EFFECTIVE**.

## Method

The old policy is unchanged UNIFORM_STRIPING: lane = group × 318 + slab, die-fastest cyclic ownership, GROUP_DIRECT nearest-Manhattan external port routing. The new GPU_PORT_BALANCED policy starts with that exact capacity-legal allocation and applies one bijective permutation to physical group/slab lanes. Consecutive logical atoms first visit distinct reachable ports, then slabs, then the second group sharing each port. Equal-load group choices prefer shorter existing FEOL startup. Each operator retains its logical cyclic atom sequence and allocation offset.

The optimization is a deterministic port-first round robin, not a claim of globally optimal placement. Any short contiguous operator allocation now spreads over distinct (slab,port) resources instead of repeatedly using a small set of groups. The global bijection preserves every slot occupancy exactly, so no capacity heuristic or atom splitting is needed. QK/AV pairing, row/KV atom granularity, total bytes and logical prefix/append identity remain intact.

No service equation is changed: external = max(total bytes / thermal cap, max_port(port bytes / port_Bps + route startup)); GPU operator time = max(array, external, GPU). The exact frozen thermal cap is 3.3957835581187483 TB/s. Physical port rate, GPU compute and bandwidth, geometry, MAT/MIV, energy coefficients, NMP and CPA are unchanged. GPU_PORT_BALANCED rejects ATTENTION_NMP and MAC_NMP before execution. Tile IDs are inert legal bookkeeping, never an optimization target.

All 18 existing data-chain Prefill totals and Prefill energy are reused exactly. Only Decode is replayed under the new placement. Old Uniform Decode is also replayed solely to fill the missing port telemetry; its timing matches the existing checkpoints. Old E2E reference values are read from the preceding diagnostic. No Prefill, HBM, NMP, CPA optimizer, or thermal benchmark run occurs. Related regression tests exercise the existing physical/NMP implementations.

Optional energy observation was enabled for early completed cases, then disabled because this task requires performance/port telemetry only. Three unfinished energy-observed B8 attempts (70B/20K/B8, 405B/64K/B8, 405B/126K/B8) were stopped and restarted without that observer. Completed cases were reused. The observer does not enter any service equation; old replay timings match their original checkpoints in both modes. No energy comparison is claimed in the final table.

## Root cause

Old small weight operators occupy consecutive die-fastest lanes, concentrating complete rows in a few physical groups and their nearest ports. The replay preserves the original 28.41% worst-case loss. The new permutation changes ownership only. The 35 reachable GROUP_DIRECT read ports per slab are unchanged; the other 15 are not made reachable for reads by an invented route. KV writes retain existing REGION_DIRECT routing across region ports.

Remaining small-operator limits are real within the atom definition: an 8B K/V projection has 1,024 indivisible 8,192-byte rows. Even assigning each row a distinct port needs at least 8.192 microseconds at the frozen 1 GB/s port rate, while its aggregate thermal-cap time is only about 2.470 microseconds. Full utilization is therefore not a valid requirement for this operator without changing atom granularity, which this diagnostic does not do.

## 18-case results

| Model | H | B | Reference TPS | Old TPS | New TPS | Old drop % | New drop % | Old util. % | New util. % | Gain pp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8B | 20K | 1 | 186.242 | 133.321 | 171.384 | 28.415 | 7.978 | 71.755 | 93.133 | 21.377 |
| 8B | 20K | 8 | 680.286 | 573.171 | 650.848 | 15.746 | 4.327 | 83.875 | 96.526 | 12.651 |
| 8B | 64K | 1 | 139.517 | 107.647 | 131.168 | 22.843 | 5.984 | 77.120 | 94.735 | 17.615 |
| 8B | 64K | 8 | 303.019 | 276.826 | 293.732 | 8.644 | 3.065 | 92.214 | 98.434 | 6.220 |
| 8B | 126K | 1 | 102.601 | 84.001 | 97.668 | 18.129 | 4.808 | 81.951 | 96.037 | 14.087 |
| 8B | 126K | 8 | 170.098 | 160.150 | 165.666 | 5.848 | 2.605 | 95.493 | 99.119 | 3.626 |
| 70B | 20K | 1 | 22.591 | 20.417 | 21.567 | 9.625 | 4.532 | 90.551 | 95.822 | 5.271 |
| 70B | 20K | 8 | 122.796 | 113.681 | 118.064 | 7.423 | 3.853 | 92.661 | 96.797 | 4.136 |
| 70B | 64K | 1 | 20.464 | 18.647 | 19.602 | 8.881 | 4.214 | 91.329 | 96.184 | 4.855 |
| 70B | 64K | 8 | 76.637 | 71.998 | 73.728 | 6.053 | 3.796 | 95.287 | 97.969 | 2.682 |
| 70B | 126K | 1 | 17.928 | 16.446 | 17.185 | 8.268 | 4.149 | 92.230 | 96.599 | 4.370 |
| 70B | 126K | 8 | 50.100 | 47.472 | 48.217 | 5.247 | 3.758 | 96.865 | 98.663 | 1.798 |
| 405B | 20K | 1 | 4.024 | 3.798 | 3.920 | 5.624 | 2.574 | 94.482 | 97.635 | 3.153 |
| 405B | 20K | 8 | 26.094 | 24.768 | 25.417 | 5.080 | 2.594 | 94.910 | 97.824 | 2.914 |
| 405B | 64K | 1 | 3.915 | 3.693 | 3.808 | 5.687 | 2.725 | 94.623 | 97.697 | 3.074 |
| 405B | 64K | 8 | 21.238 | 20.109 | 20.534 | 5.314 | 3.313 | 95.736 | 98.183 | 2.447 |
| 405B | 126K | 1 | 3.738 | 3.530 | 3.636 | 5.575 | 2.745 | 94.810 | 97.780 | 2.970 |
| 405B | 126K | 8 | 16.825 | 15.896 | 16.160 | 5.525 | 3.953 | 96.528 | 98.528 | 2.000 |

All comparison fields, including old/new achieved TB/s and bottlenecks, are in comparison.csv. Achieved bandwidth is Decode boundary payload / summed corresponding active boundary service time; it excludes compute gaps and Prefill. Both E2E timings include the same reused Prefill.

## Aggregate

| Statistic | Old | New |
|---|---:|---:|
| utilization_min (%) | 71.7554 | 93.1325 |
| utilization_max (%) | 96.8647 | 99.1191 |
| utilization_geomean (%) | 90.3960 | 97.0811 |
| TPS_drop_min (%) | 5.0802 | 2.5736 |
| TPS_drop_max (%) | 28.4148 | 7.9778 |
| TPS_drop_geomean (%) | 8.3796 | 3.7604 |
| drop_from_TPS_ratio_geomean (%) | 10.1522 | 3.9523 |
| Utilization geometric mean: batch=1 (%) | 87.2557 | 96.1693 |
| Utilization geometric mean: batch=8 (%) | 93.6493 | 98.0014 |
| Utilization geometric mean: model=Llama-3.1-8B (%) | 83.3354 | 96.3089 |
| Utilization geometric mean: model=Llama-3.1-70B (%) | 93.1275 | 97.0007 |
| Utilization geometric mean: model=Llama-3.1-405B (%) | 95.1788 | 97.9406 |
| Utilization geometric mean: context=LC20K (%) | 87.6308 | 96.2765 |
| Utilization geometric mean: context=LC64K (%) | 90.8056 | 97.1914 |
| Utilization geometric mean: context=LC126K (%) | 92.8278 | 97.7812 |

The geometric mean of percentage drops differs from 1 minus the geometric mean of throughput ratios; both are reported explicitly.

## Port telemetry

| Metric across all Decode transfers | Old | New |
|---|---:|---:|
| transfer_count | 503808.000000 | 503808.000000 |
| PORT_SERIALIZATION | 201792.000000 | 183360.000000 |
| GLOBAL_THERMAL_CAP | 302016.000000 | 320448.000000 |
| ROUTE_STARTUP | 0.000000 | 0.000000 |
| boundary_s | 75.715342 | 73.316462 |
| cap_only_s | 71.762584 | 71.762584 |
| PORT_SERIALIZATION_boundary_s | 4.947148 | 2.214363 |
| active_ports_transfer_mean | 5833.058300 | 6757.338994 |
| max_port_bytes_transfer_mean | 65960.315249 | 54161.788331 |
| mean_active_port_bytes_transfer_mean | 58001.137745 | 48516.281649 |
| max_mean_load_ratio_transfer_mean | 1.160169 | 1.173107 |
| max_port_utilization_transfer_mean | 0.618196 | 0.599998 |
| mean_active_port_utilization_transfer_mean | 0.535730 | 0.524222 |

The transfer-averaged max/mean load ratio does not improve globally (1.160169 to 1.173107). It equally weights large reads and tiny appends and excludes inactive ports from each mean; it is not the optimized maximum completion time. The report therefore does not claim every fairness statistic improves. Actual boundary time and port-limited service time decrease, while the append tradeoff is shown below.

Port-count/load/utilization means are arithmetic means over individual nonempty transfers, not utilization of a hypothetical aggregated concurrent workload. Limiting-resource counts and boundary durations are sums. Per-case statistics are in port_comparison.csv, including phase-wide active-port unions and cumulative port-load ratios separately. Phase port utilization is that port’s cumulative bytes / port_Bps / Decode elapsed time; transfer means and whole-phase utilization are explicitly separate columns. Per-operator first-step atom counts, groups, active ports and peak/mean load ratios are in first_step_operator_ports.csv.gz. Full 32-step resident/service per-port distributions are in the compressed case telemetry.

## Worst-case analysis: 8B / 20K / B1

| Operator, summed across 32 Decode steps and all layers | Old boundary ms | New boundary ms | Cap-only ms |
|---|---:|---:|---:|
| TOKEN_EMBED_LOOKUP | 0.262165 | 0.262165 | 0.000077 |
| Q | 16.782578 | 10.118353 | 10.118353 |
| K | 16.782420 | 8.391846 | 2.529588 |
| V | 16.782419 | 8.391846 | 2.529588 |
| KV_APPEND | 0.061519 | 0.278916 | 0.001235 |
| ATTENTION_QK | 12.440128 | 12.440128 | 12.440128 |
| ATTENTION_AV | 12.440128 | 12.440128 | 12.440128 |
| O | 16.782575 | 10.118353 | 10.118353 |
| FFN_GATE | 35.414237 | 35.414237 | 35.414237 |
| FFN_UP | 35.414237 | 35.414237 | 35.414237 |
| FFN_DOWN | 58.725616 | 35.414237 | 35.414237 |
| LM_HEAD | 9.900967 | 9.900967 | 9.900967 |

KV_APPEND is a local tradeoff: its cumulative boundary time increases from 0.061519 to 0.278916 ms. The same token/head atoms now occupy fewer slabs and a wider set of read ports; writes still use the unchanged REGION_DIRECT region-port striping, so a small append can have more bytes per active write port. This is retained and reported, not corrected away. The weight-read service savings exceed that small append penalty. The policy is not claimed to improve every individual transfer.

## Safety and verdict

18/18 workloads improve throughput; 0 regressions. All 18 have legal slot/group/die capacity, conserved semantic traffic and no NMP stages. The physical-lane bijection also proves capacity conservation relative to the original Uniform allocation. Existing Uniform/Balanced/CPA placement signatures are checked against pre-change captures; the frozen Uniform execution tests cover unchanged NMP execution.

Related placement, Decode, physical service and energy/traffic tests are recorded in targeted_tests.log. No full pytest is run. Protected formal files and the prior diagnostic are unchanged; see preservation.json.

**PORT_BALANCED_PLACEMENT_EFFECTIVE**. The result supports considering this GPU-only mapping for a subsequent formal integration review. It is not automatically adopted: the formal GPU baseline and Prefill remain unchanged, and a future integration must explicitly decide how the GPU physical resident mapping becomes part of the architecture semantics.

Reproduce: `conda run --no-capture-output -n om3dthermal python scripts/diagnose_m3d_gpu_port_balanced.py`; then `conda run --no-capture-output -n om3dthermal python scripts/report_m3d_gpu_port_balanced.py`.
