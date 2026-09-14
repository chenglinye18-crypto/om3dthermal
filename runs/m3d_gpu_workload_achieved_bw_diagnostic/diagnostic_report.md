# M3D-GPU workload-achieved bandwidth diagnostic

Verdict: **EXISTING_ROUTING_CONDITIONAL_DIAGNOSTIC__NOT_BASELINE_REPLACEMENT**.

All 18 results reuse the exact GPU physical checkpoints selected by each canonical candidate’s legacy_fingerprint (including the repaired 405B/64K/B8 checkpoint). No reference was recomputed. 576 Decode steps and all 18 original GPU-only Prefill totals are reused. One 8B/20K/B1 first-step GPU replay provides operator/port attribution; no HBM, DNS, CPA, thermal, or full-workload simulation is run.

## Equations and accounting audit

* Reference: primary_execution.gpu_memory_closure closes thermal, internal, coil and GPU ceilings. The effective minimum is the frozen unrounded 3.3957835581187483 TB/s (displayed as 3.395784), not an unconstrained thermal-only assignment. Full-cycle MAT/MIV/FEOL is already included once.
* Diagnostic: each existing GPU operator uses max(Array+MIV service, external service, GPU service). Its external service is max(total bytes / thermal cap, max_port(load / port bandwidth + route startup)). There is no additional bytes/Bcap term added to these times. Operators follow the existing dependency schedule.
* Physical array service sums 32-byte MAT/MIV cycles within each group/layer lane, then takes the maximum across groups/dies. It does not add the aggregate reference cycle again. FEOL route startup is separate here because physical floorplan.service_ns contains MAT+MIV only.
* Concurrent requests merge demand on each shared group/port before taking resource maxima; startup takes a maximum. They are not globally serialized B1 executions. The GPU request terms sum on the shared GPU, and equal-shape requests have the same compute/memory ratio. GPU_COMPUTE is a legacy label containing a GPU compute/memory maximum and small-op memory service.
* MAC, Fabric, inter-region NoC, reduction, NMP SRAM/pipeline and NMP background are zero. The existing physical mapper is UNIFORM_STRIPING for memory ownership; no NMP operator is executed and no placement algorithm is changed or optimized.
* Read/write/interface traffic agrees with the independent semantic ledger for all 576 growing-context steps. Each context appears exactly once. Array and boundary byte totals describe successive physical locations of the same traffic and are not added as extra logical bytes.
* Prefill uses the saved GPU-only incremental ledger and original data-chain timing max(compute, aggregate array, aggregate boundary), with unchanged H/P/G. Its component durations were not saved, so their CSV fields are null. MIV duration is also not independently checkpointed and is marked included in Array, never invented as a separate additive delay.
* Energy uses identical frozen event counts and coefficients; only 74 W GPU static energy scales with the changed E2E time. This agrees with saved physical-checkpoint energy. No thermal reclosure is performed.

Achieved bandwidth = sum of Decode boundary payload bytes / sum of the corresponding operator boundary active-service durations. It excludes GPU idle/compute gaps and Prefill; it is **not** bytes/E2E wall time or a measured device utilization. Prefill and total traffic are separate CSV fields. Bottleneck labels compare saved aggregate Decode resource-service totals.

## Large-degradation diagnosis and limitation

The worst case first step has 5.196937 ms cap-only service versus 7.242808 ms actual boundary service. Total route startup is only 2.122307 microseconds. 225 transfers are port-limited and 129 cap-limited.

Across its 32 layers, K and V each need 0.524451 ms boundary service versus 0.079050 ms cap-only; FFN_DOWN needs 1.835175 versus 1.106695 ms. These are small/row-sharded operator port-demand concentration effects. The saved resident mapper assigns indivisible row atoms to physical groups; GROUP_DIRECT chooses the nearest physical port. Small row counts cannot uniformly occupy all resources.

Although only 35 of 50 ports per slab are used by equal group traffic, their aggregate 11.13 TB/s still exceeds the thermal cap. Therefore the 35/50 count alone cannot explain the loss: **per-operator load imbalance**, not a global 70% bandwidth coefficient, is decisive. The detailed audit files expose actual port serialization and independent startup.

No double counting or lost group/request concurrency was found in this replay. However, this physical path inherits a resident row-atom mapping shared with the NMP framework, unlike the formal placement-free GPU aggregate closure. Accounting consistency does not prove that a GPU memory controller must use that mapping or cannot stripe rows more finely. Hence these are conditional results for the existing routing/mapping, not proof of achievable real-hardware bandwidth or sufficient evidence to replace the formal baseline. No mapping redesign is made.

## Results

| Model | H | B | Reference tok/s | Data-chain tok/s | Drop % | BW TB/s | Util. % | Bottleneck |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 8B | 20K | 1 | 186.242 | 133.321 | 28.41 | 2.4367 | 71.76 | BOUNDARY |
| 8B | 20K | 8 | 680.286 | 573.171 | 15.75 | 2.8482 | 83.88 | BOUNDARY |
| 8B | 64K | 1 | 139.517 | 107.647 | 22.84 | 2.6188 | 77.12 | BOUNDARY |
| 8B | 64K | 8 | 303.019 | 276.826 | 8.64 | 3.1314 | 92.21 | BOUNDARY |
| 8B | 126K | 1 | 102.601 | 84.001 | 18.13 | 2.7829 | 81.95 | BOUNDARY |
| 8B | 126K | 8 | 170.098 | 160.150 | 5.85 | 3.2427 | 95.49 | BOUNDARY |
| 70B | 20K | 1 | 22.591 | 20.417 | 9.63 | 3.0749 | 90.55 | BOUNDARY |
| 70B | 20K | 8 | 122.796 | 113.681 | 7.42 | 3.1466 | 92.66 | BOUNDARY |
| 70B | 64K | 1 | 20.464 | 18.647 | 8.88 | 3.1013 | 91.33 | BOUNDARY |
| 70B | 64K | 8 | 76.637 | 71.998 | 6.05 | 3.2357 | 95.29 | BOUNDARY |
| 70B | 126K | 1 | 17.928 | 16.446 | 8.27 | 3.1319 | 92.23 | BOUNDARY |
| 70B | 126K | 8 | 50.100 | 47.472 | 5.25 | 3.2893 | 96.86 | BOUNDARY |
| 405B | 20K | 1 | 4.024 | 3.798 | 5.62 | 3.2084 | 94.48 | BOUNDARY |
| 405B | 20K | 8 | 26.094 | 24.768 | 5.08 | 3.2229 | 94.91 | BOUNDARY |
| 405B | 64K | 1 | 3.915 | 3.693 | 5.69 | 3.2132 | 94.62 | BOUNDARY |
| 405B | 64K | 8 | 21.238 | 20.109 | 5.31 | 3.2510 | 95.74 | BOUNDARY |
| 405B | 126K | 1 | 3.738 | 3.530 | 5.58 | 3.2195 | 94.81 | BOUNDARY |
| 405B | 126K | 8 | 16.825 | 15.896 | 5.52 | 3.2779 | 96.53 | BOUNDARY |

## Aggregate and trends

TPS degradation min/max/geomean = 5.080% / 28.415% / 8.380%. For clarity, 1 minus the geometric mean of TPS ratios is 10.152%; these are different statistics. Bandwidth utilization min/max/geomean = 71.755% / 96.865% / 90.396%.

* B1/B8 geometric-mean utilization: 87.256% / 93.649%. B8 improves utilization in all nine pairs.
* 8B/70B/405B geometric-mean utilization: 83.335% / 93.128% / 95.179%. Model-size improvement is not universal: at B8/126K, 70B is 96.865% versus 405B 96.528%.
* 20K/64K/126K geometric-mean utilization: 87.631% / 90.806% / 92.828%. All six model/batch combinations improve boundary utilization with context. E2E degradation is not uniformly monotonic: 405B/B8 worsens from 5.080% to 5.314% to 5.522% because E2E includes Prefill and GPU operator scheduling as well as boundary service.

Formal 72-row CSV and all canonical candidate files remain byte-identical. No HBM/DNS/CPA source, workload, hardware, placement or thermal setting changed. No pytest run.

Run: `conda run --no-capture-output -n om3dthermal python scripts/diagnose_m3d_gpu_achieved_bandwidth.py`. Outputs are isolated here; the two SVG/PDF figures are diagnostic only.
