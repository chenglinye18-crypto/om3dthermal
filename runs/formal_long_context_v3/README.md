# Formal long-context v3

18 operating points: Llama-3.1-8B / Qwen2.5-32B base; H=20000/64000/126000; B=1/8/32; P=128; G=32. Each of B requests arrives at t=0. The output numerator is B*G. Historical H is cached KV, never H-token Prefill. Incremental linear/FFN computation covers P only; each Decode step j uses H+P+j and appends local KV.

## HBM resident waves

Persistent weights must fit. The maximum resident batch includes weights, final KV(H+P+G), and the larger of canonical Prefill(P) and final-step Decode workspaces, rounded to 32-byte slots. Canonical workspace includes live activations and runtime buffers; no additional unspecified runtime reserve is invented. Wave 1's historical KV is initially local; all waiting histories are host-valid. Each later wave admits H-token KV once, through the canonical Grace/C2C pipeline bounded also by HBM write bandwidth. Waves execute serially to completion and release KV. There is no weight streaming and no post-completion archival cost in completion latency or primary energy. These results do not use legacy fixed-S wave numbers.

HOST_OFFLOAD preserves the existing traffic-minimizing static extent allocation and optimistic max(local GPU/memory, recurring remote service) overlap. As in v2, external host capacity is sufficiently provisioned; this is not a claim that every point fits one physical 480 GB Grace memory. The canonical Grace/C2C bandwidth and energy coefficients remain unchanged. HBM_BEST selects the entire policy by E2E throughput (ties choose HOST_OFFLOAD); all its energy, traffic and latency fields come from that policy. No per-metric selection.

## Latency and energy definitions

TTFT = queue delay + admission delay + incremental Prefill + first Decode step. Completion latency is arrival-to-final-token. TPOT = active Decode duration / G. P95 uses the linear sample percentile across all B individual requests, including duplicate per-wave values. Decode throughput uses B*G divided by summed active wave Decode durations; E2E throughput uses full batch makespan including admission and Prefill.

GPU static energy is counted once over the serial active system interval, including admission. Waiting requests do not receive additional copies of server static energy. There is no added per-request queue idle-power model; the GPU serves the current wave while others wait. Separate HBM/Grace static power remains zero in the canonical baseline accounting. HBM reads use 3.00 pJ/bit and writes retain 1.9955 pJ/bit. Admission counts host-memory reads, C2C transfer and HBM writes once. All flat energy components close to E2E_J; tokens/J = B*G/E2E_J.

## Model provenance and boundaries

Llama-8B retains the exact v2 matrix-derived parameter count (8,030,261,248), not the rounded legacy registry value. Qwen2.5-32B is the base checkpoint: 64 layers, hidden 5120, FFN 27648, 40 Q heads, 8 KV heads, head dimension 128, vocabulary 152064, untied embeddings, BF16 weights/KV, maximum context 131072, no sliding-window attention. Its exact 32,763,876,352 parameters include QKV biases and RMSNorm; the official safetensors index lists 65,527,752,704 bytes. Parameter formula: 64*(2*5120^2+2*5120*1024+3*5120*27648+2*5120+5120+2*1024)+2*5120*152064+5120.

Sources: https://huggingface.co/Qwen/Qwen2.5-32B/raw/main/config.json and https://huggingface.co/Qwen/Qwen2.5-32B/raw/main/model.safetensors.index.json . The frozen dense analytical evaluator does not separately time projection-bias additions; all parameters occupy capacity, while active execution follows its existing matrix and small-operator accounting. These are model estimates, not measured inference.

## Physical and thermal semantics

The three M3D paths are fully local and use unchanged physical execution/hardware. GPU uses the established GPU_PORT_BALANCED placement, Uniform and CPA use their respective frozen NMP placement. Prefill reuses the canonical cached-prefix ledger and existing v2 path-specific timing: physical GPU memory service for M3D_GPU, corrected aggregate GPU Prefill for NMP systems. No adaptive execution is introduced.

Tmax is peak Decode steady-state temperature, not E2E duty-cycle average. HBM retains the 85 C active-service thermally closed design point. Every M3D workload receives its own GPU/per-die power RHS and FP64 GPU-PCG steady-state solve, using the existing operator cache; memory power is die-grouped and uniform in BEOL. Old v2 workload power solutions are not reused. Hardware, mesh and solver tolerances are unchanged.

## Capacity

| model | workload | B | peak_bytes | safe_resident_batch | wave_sizes | M3D_status |
|---|---|---|---|---|---|---|
| Llama-3.1-8B | LC20K | 1 | 18713944064 | 48 | [1] | PASS |
| Llama-3.1-8B | LC20K | 8 | 37287895040 | 48 | [8] | PASS |
| Llama-3.1-8B | LC20K | 32 | 100970012672 | 48 | [32] | PASS |
| Llama-3.1-8B | LC64K | 1 | 24481112064 | 15 | [1] | PASS |
| Llama-3.1-8B | LC64K | 8 | 83425239040 | 15 | [8] | PASS |
| Llama-3.1-8B | LC64K | 32 | 285519388672 | 15 | [15, 15, 2] | PASS |
| Llama-3.1-8B | LC126K | 1 | 32609858496 | 7 | [1] | PASS |
| Llama-3.1-8B | LC126K | 8 | 148455210496 | 7 | [7, 1] | PASS |
| Llama-3.1-8B | LC126K | 32 | 545639274496 | 7 | [7, 7, 7, 7, 4] | PASS |
| Qwen2.5-32B | LC20K | 1 | 70833809408 | 14 | [1] | PASS |
| Qwen2.5-32B | LC20K | 8 | 107976206336 | 14 | [8] | PASS |
| Qwen2.5-32B | LC20K | 32 | 235321567232 | 14 | [14, 14, 4] | PASS |
| Qwen2.5-32B | LC64K | 1 | 82368145408 | 4 | [1] | PASS |
| Qwen2.5-32B | LC64K | 8 | 200250894336 | 4 | [4, 4] | PASS |
| Qwen2.5-32B | LC64K | 32 | 604420319232 | 4 | [4, 4, 4, 4, 4, 4, 4, 4] | PASS |
| Qwen2.5-32B | LC126K | 1 | 98621073408 | 2 | [1] | PASS |
| Qwen2.5-32B | LC126K | 8 | 330274318336 | 2 | [2, 2, 2, 2] | PASS |
| Qwen2.5-32B | LC126K | 32 | 1124514015232 | 2 | [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2] | PASS |

## HBM overflow policy comparison

| model | context | batch | safe_resident_batch | wave_sizes | host_tokens_per_s | wave_tokens_per_s | selected_HBM_policy | host_P95_completion_latency | wave_P95_completion_latency |
|---|---|---|---|---|---|---|---|---|---|
| Llama-3.1-8B | LC64K | 32 | 15 | [15, 15, 2] | 92.6462 | 264.1278 | HBM_RESIDENT_WAVE | 11.0528 | 3.6637 |
| Llama-3.1-8B | LC126K | 8 | 7 | [7, 1] | 158.1195 | 139.1806 | HBM_HOST_OFFLOAD | 1.6190 | 1.7059 |
| Llama-3.1-8B | LC126K | 32 | 7 | [7, 7, 7, 7, 4] | 32.3389 | 132.5818 | HBM_RESIDENT_WAVE | 31.6647 | 7.7235 |
| Qwen2.5-32B | LC20K | 32 | 14 | [14, 14, 4] | 139.6062 | 228.1969 | HBM_RESIDENT_WAVE | 7.3349 | 4.4874 |
| Qwen2.5-32B | LC64K | 8 | 4 | [4, 4] | 58.9610 | 82.6773 | HBM_RESIDENT_WAVE | 4.3419 | 3.0964 |
| Qwen2.5-32B | LC64K | 32 | 4 | [4, 4, 4, 4, 4, 4, 4, 4] | 28.2218 | 79.5707 | HBM_RESIDENT_WAVE | 36.2840 | 12.8691 |
| Qwen2.5-32B | LC126K | 8 | 2 | [2, 2, 2, 2] | 17.5793 | 41.2605 | HBM_RESIDENT_WAVE | 14.5625 | 6.2045 |
| Qwen2.5-32B | LC126K | 32 | 2 | [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2] | 13.2116 | 40.4840 | HBM_RESIDENT_WAVE | 77.5077 | 24.4190 |

## Four-path results

| model | context | batch | path | tokens_per_s | tokens_per_J | Tmax_C |
|---|---|---|---|---|---|---|
| Llama-3.1-8B | LC20K | 1 | HBM_BEST | 169.2825 | 0.3880 | 85.0000 |
| Llama-3.1-8B | LC20K | 1 | M3D_GPU | 172.9299 | 0.4423 | 81.1888 |
| Llama-3.1-8B | LC20K | 1 | M3D_NMP_UNIFORM | 554.9303 | 1.6464 | 62.5049 |
| Llama-3.1-8B | LC20K | 1 | M3D_NMP_CPA | 619.9605 | 1.9224 | 60.7933 |
| Llama-3.1-8B | LC20K | 8 | HBM_BEST | 623.7459 | 1.4156 | 85.0000 |
| Llama-3.1-8B | LC20K | 8 | M3D_GPU | 650.8478 | 1.6063 | 82.4572 |
| Llama-3.1-8B | LC20K | 8 | M3D_NMP_UNIFORM | 811.5222 | 2.0880 | 69.2587 |
| Llama-3.1-8B | LC20K | 8 | M3D_NMP_CPA | 971.6609 | 2.4758 | 70.5142 |
| Llama-3.1-8B | LC20K | 32 | HBM_BEST | 871.5886 | 1.9748 | 85.0000 |
| Llama-3.1-8B | LC20K | 32 | M3D_GPU | 920.6861 | 2.2352 | 83.2838 |
| Llama-3.1-8B | LC20K | 32 | M3D_NMP_UNIFORM | 834.5002 | 2.1375 | 69.5740 |
| Llama-3.1-8B | LC20K | 32 | M3D_NMP_CPA | 995.4791 | 2.4838 | 71.7269 |
| Llama-3.1-8B | LC64K | 1 | HBM_BEST | 127.2515 | 0.2896 | 85.0000 |
| Llama-3.1-8B | LC64K | 1 | M3D_GPU | 131.4578 | 0.3303 | 81.8209 |
| Llama-3.1-8B | LC64K | 1 | M3D_NMP_UNIFORM | 483.9912 | 1.3243 | 64.8504 |
| Llama-3.1-8B | LC64K | 1 | M3D_NMP_CPA | 519.0980 | 1.4702 | 63.2105 |
| Llama-3.1-8B | LC64K | 8 | HBM_BEST | 277.5070 | 0.6304 | 85.0000 |
| Llama-3.1-8B | LC64K | 8 | M3D_GPU | 293.7319 | 0.7182 | 83.2334 |
| Llama-3.1-8B | LC64K | 8 | M3D_NMP_UNIFORM | 652.3308 | 1.5892 | 71.5368 |
| Llama-3.1-8B | LC64K | 8 | M3D_NMP_CPA | 723.4521 | 1.7731 | 71.4358 |
| Llama-3.1-8B | LC64K | 32 | HBM_BEST | 264.1278 | 0.6441 | 85.0000 |
| Llama-3.1-8B | LC64K | 32 | M3D_GPU | 338.5329 | 0.8215 | 83.6485 |
| Llama-3.1-8B | LC64K | 32 | M3D_NMP_UNIFORM | 670.3866 | 1.6221 | 72.0084 |
| Llama-3.1-8B | LC64K | 32 | M3D_NMP_CPA | 741.2719 | 1.7835 | 72.6177 |
| Llama-3.1-8B | LC126K | 1 | HBM_BEST | 93.6588 | 0.2131 | 85.0000 |
| Llama-3.1-8B | LC126K | 1 | M3D_GPU | 97.6678 | 0.2432 | 82.3216 |
| Llama-3.1-8B | LC126K | 1 | M3D_NMP_UNIFORM | 402.3124 | 1.0353 | 67.4119 |
| Llama-3.1-8B | LC126K | 1 | M3D_NMP_CPA | 420.4347 | 1.1194 | 65.5665 |
| Llama-3.1-8B | LC126K | 8 | HBM_BEST | 158.1195 | 0.3540 | 85.0000 |
| Llama-3.1-8B | LC126K | 8 | M3D_GPU | 165.6665 | 0.4037 | 83.5026 |
| Llama-3.1-8B | LC126K | 8 | M3D_NMP_UNIFORM | 511.9114 | 1.1896 | 73.9794 |
| Llama-3.1-8B | LC126K | 8 | M3D_NMP_CPA | 553.6433 | 1.2884 | 74.1026 |
| Llama-3.1-8B | LC126K | 32 | HBM_BEST | 132.5818 | 0.3347 | 85.0000 |
| Llama-3.1-8B | LC126K | 32 | M3D_GPU | 179.0292 | 0.4344 | 83.7446 |
| Llama-3.1-8B | LC126K | 32 | M3D_NMP_UNIFORM | 523.8890 | 1.2077 | 74.5797 |
| Llama-3.1-8B | LC126K | 32 | M3D_NMP_CPA | 562.1058 | 1.2937 | 74.9037 |
| Qwen2.5-32B | LC20K | 1 | HBM_BEST | 43.1550 | 0.0991 | 85.0000 |
| Qwen2.5-32B | LC20K | 1 | M3D_GPU | 45.2867 | 0.1135 | 82.5302 |
| Qwen2.5-32B | LC20K | 1 | M3D_NMP_UNIFORM | 176.8545 | 0.5050 | 64.2238 |
| Qwen2.5-32B | LC20K | 1 | M3D_NMP_CPA | 191.9953 | 0.5576 | 63.5747 |
| Qwen2.5-32B | LC20K | 8 | HBM_BEST | 207.9663 | 0.4716 | 85.0000 |
| Qwen2.5-32B | LC20K | 8 | M3D_GPU | 218.2519 | 0.5343 | 82.8530 |
| Qwen2.5-32B | LC20K | 8 | M3D_NMP_UNIFORM | 275.0577 | 0.6775 | 71.2737 |
| Qwen2.5-32B | LC20K | 8 | M3D_NMP_CPA | 322.4659 | 0.7920 | 71.9369 |
| Qwen2.5-32B | LC20K | 32 | HBM_BEST | 228.1969 | 0.5382 | 85.0000 |
| Qwen2.5-32B | LC20K | 32 | M3D_GPU | 366.6519 | 0.8852 | 83.1671 |
| Qwen2.5-32B | LC20K | 32 | M3D_NMP_UNIFORM | 286.2149 | 0.7003 | 71.7611 |
| Qwen2.5-32B | LC20K | 32 | M3D_NMP_CPA | 331.4724 | 0.7876 | 73.8998 |
| Qwen2.5-32B | LC64K | 1 | HBM_BEST | 37.0014 | 0.0843 | 85.0000 |
| Qwen2.5-32B | LC64K | 1 | M3D_GPU | 38.8947 | 0.0965 | 82.6825 |
| Qwen2.5-32B | LC64K | 1 | M3D_NMP_UNIFORM | 161.2514 | 0.4306 | 65.8114 |
| Qwen2.5-32B | LC64K | 1 | M3D_NMP_CPA | 169.7676 | 0.4629 | 64.7967 |
| Qwen2.5-32B | LC64K | 8 | HBM_BEST | 82.6773 | 0.1957 | 85.0000 |
| Qwen2.5-32B | LC64K | 8 | M3D_GPU | 119.0112 | 0.2905 | 83.1639 |
| Qwen2.5-32B | LC64K | 8 | M3D_NMP_UNIFORM | 230.6905 | 0.5469 | 73.0771 |
| Qwen2.5-32B | LC64K | 8 | M3D_NMP_CPA | 255.2072 | 0.6068 | 73.1774 |
| Qwen2.5-32B | LC64K | 32 | HBM_BEST | 79.5707 | 0.1940 | 85.0000 |
| Qwen2.5-32B | LC64K | 32 | M3D_GPU | 152.7275 | 0.3702 | 83.4186 |
| Qwen2.5-32B | LC64K | 32 | M3D_NMP_UNIFORM | 237.3308 | 0.5583 | 73.6637 |
| Qwen2.5-32B | LC64K | 32 | M3D_NMP_CPA | 259.9348 | 0.6045 | 74.6690 |
| Qwen2.5-32B | LC126K | 1 | HBM_BEST | 30.5444 | 0.0695 | 85.0000 |
| Qwen2.5-32B | LC126K | 1 | M3D_GPU | 32.1612 | 0.0795 | 82.8329 |
| Qwen2.5-32B | LC126K | 1 | M3D_NMP_UNIFORM | 137.8265 | 0.3537 | 67.4516 |
| Qwen2.5-32B | LC126K | 1 | M3D_NMP_CPA | 142.8667 | 0.3732 | 66.4814 |
| Qwen2.5-32B | LC126K | 8 | HBM_BEST | 41.2605 | 0.0997 | 85.0000 |
| Qwen2.5-32B | LC126K | 8 | M3D_GPU | 72.5405 | 0.1768 | 83.3321 |
| Qwen2.5-32B | LC126K | 8 | M3D_NMP_UNIFORM | 185.3352 | 0.4278 | 74.4657 |
| Qwen2.5-32B | LC126K | 8 | M3D_NMP_CPA | 199.5536 | 0.4619 | 74.4766 |
| Qwen2.5-32B | LC126K | 32 | HBM_BEST | 40.4840 | 0.0993 | 85.0000 |
| Qwen2.5-32B | LC126K | 32 | M3D_GPU | 83.8190 | 0.2035 | 83.4850 |
| Qwen2.5-32B | LC126K | 32 | M3D_NMP_UNIFORM | 190.6149 | 0.4363 | 75.1142 |
| Qwen2.5-32B | LC126K | 32 | M3D_NMP_CPA | 204.0744 | 0.4608 | 76.2862 |

## Geometric means, normalized to HBM_BEST per case

| group | path | cases | normalized_throughput | normalized_energy_efficiency |
|---|---|---|---|---|
| Llama-3.1-8B | HBM_BEST | 9 | 1.0000 | 1.0000 |
| Llama-3.1-8B | M3D_GPU | 9 | 1.0984 | 1.1697 |
| Llama-3.1-8B | M3D_NMP_UNIFORM | 9 | 2.5815 | 2.8285 |
| Llama-3.1-8B | M3D_NMP_CPA | 9 | 2.8639 | 3.1647 |
| Qwen2.5-32B | HBM_BEST | 9 | 1.0000 | 1.0000 |
| Qwen2.5-32B | M3D_GPU | 9 | 1.3926 | 1.4525 |
| Qwen2.5-32B | M3D_NMP_UNIFORM | 9 | 3.0601 | 3.2233 |
| Qwen2.5-32B | M3D_NMP_CPA | 9 | 3.3481 | 3.5283 |
| ALL | HBM_BEST | 18 | 1.0000 | 1.0000 |
| ALL | M3D_GPU | 18 | 1.2368 | 1.3034 |
| ALL | M3D_NMP_UNIFORM | 18 | 2.8106 | 3.0194 |
| ALL | M3D_NMP_CPA | 18 | 3.0965 | 3.3416 |

## Wave-policy effects relative to HOST_OFFLOAD

| group | policy | cases | tokens_per_s_geomean_ratio_vs_HOST_OFFLOAD | tokens_per_J_geomean_ratio_vs_HOST_OFFLOAD | mean_TTFT_geomean_ratio_vs_HOST_OFFLOAD | mean_TPOT_geomean_ratio_vs_HOST_OFFLOAD | P95_completion_latency_geomean_ratio_vs_HOST_OFFLOAD |
|---|---|---|---|---|---|---|---|
| Llama-3.1-8B | RESIDENT_WAVE | 9 | 1.2956 | 1.0725 | 1.2264 | 0.5481 | 0.7606 |
| Llama-3.1-8B | HBM_BEST | 9 | 1.3141 | 1.0840 | 1.1398 | 0.5605 | 0.7562 |
| Qwen2.5-32B | RESIDENT_WAVE | 9 | 1.5320 | 0.9889 | 1.6884 | 0.2538 | 0.6502 |
| Qwen2.5-32B | HBM_BEST | 9 | 1.5320 | 0.9889 | 1.6884 | 0.2538 | 0.6502 |
| ALL | RESIDENT_WAVE | 18 | 1.4089 | 1.0299 | 1.4390 | 0.3729 | 0.7032 |
| ALL | HBM_BEST | 18 | 1.4189 | 1.0353 | 1.3872 | 0.3771 | 0.7012 |

Ratios below 1 improve latency; ratios above 1 improve throughput/energy efficiency. P95 completion need not worsen under waves: eliminating recurring remote traffic can outweigh serialization, while queueing can still delay the first token. No result is adjusted to enforce a narrative.

## Status

```json
{
  "benchmark_id": "formal_long_context_v3",
  "operating_points": 18,
  "formal_rows": 72,
  "old_HEAD": "2b693b8a91da5e7a16efcfb04354bc4983832104",
  "M3D_capacity_infeasible": 0,
  "B1_wave_infeasible": 0,
  "thermal_over_85": [],
  "wave_selected": 7,
  "HBM_thermal_bandwidth_TBps": 3.0865643306417803,
  "M3D_GPU_thermal_cap_TBps": 3.3957835581187483,
  "NMP_physical_boundary_cap_status": "UNCHANGED_FROZEN_IMPLEMENTATION"
}
```

## Running and tests

Run these commands from the repository in the native Windows Conda environment `om3dthermal`:

```powershell
conda activate om3dthermal
python scripts/preflight_formal_long_context_v3.py
python scripts/run_formal_long_context_v3.py
python scripts/thermal_formal_long_context_v3.py
python scripts/finalize_formal_long_context_v3.py
python scripts/plot_formal_long_context_v3.py
```

The runner reuses completed v3 candidate files and uses F:/om3dthermal_cache/formal_long_context_v3_shared for temporary read-only execution maps. It never consumes legacy wave numbers or writes v2. Separate runners may own independent cases; wait for all runners before thermal/finalization. The thermal operator cache remains local and is not committed. Detailed candidate/checkpoint archives may be compressed without changing numeric content.

The required related serving/workload run returned 313 passed, 11 failed, 8 skipped in 271.01 seconds. Nine failures require previously deleted legacy result files, one exposes an existing v1/v2 configuration import collision (W64), and one asserts an obsolete HBM thermal bandwidth. Those old files, tests and physical settings were not changed to mask failures. The final new targeted run returned 6 passed in 6.54 seconds, including exact Qwen per-step/per-die event conservation. Full pytest was not run: no shared/public evaluator was modified. See test_report.json and test logs for exact results.

The additional direct physical/decode dependency suite returned 344 passed, 0 failed, 0 skipped in 615.98 seconds. Combined related suites: 657 passed, 11 failed, 8 skipped.
