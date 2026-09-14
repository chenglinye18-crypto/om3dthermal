# Final long-context benchmark v2



Canonical source: final_e2e_metrics.csv (72 rows). H=20000/64000/126000; P=128 incremental Prefill exactly once; G=32; B=1/8; Llama-3.1 8B/70B/405B. Decode contexts H+128 through H+159. Full E2E performance/energy includes Prefill and Decode; generated tokens=B*32.

All workload executions are fresh v2. GPU raw event-observation timing is superseded by the unchanged corrected package-level aggregate GPU timing; physical NMP executes every step. Uniform and CPA are raw placements, not adaptive clipping.

HBM/M3D-GPU Tmax=85 C is a THERMAL_CLOSED_DESIGN_POINT at frozen 3.297387/3.395784 TB/s, not a workload thermal solve. Uniform/CPA temperatures are WORKLOAD_SPECIFIC_STEADY_STATE, DECODE only, using E_decode/T_decode. These metrics are deliberately distinct and are not full-request transient peaks.

NMP power uses actual per-die energy distributed uniformly in each memory die BEOL, with GPU FEOL source. Existing background, interface and all physical energy terms are retained. No Prefill energy is mapped. One existing M3D operator is reused for all feasible NMP RHS solves, with unchanged FP64 PCG tolerances.

HBM external capacity is sufficiently provisioned without an invented TB capacity. Frozen local/external max overlap, 416.34 GB/s C2C, local read+write 1.9955 pJ/bit and external 5.3 pJ/bit remain unchanged. External energy is included in system tokens/J.

## Throughput (tokens/s)

| model | context | batch | HBM_GPU | M3D_GPU | M3D_NMP_UNIFORM | M3D_NMP_CPA |
|---|---|---|---|---|---|---|
| Llama-3.1-8B | LC20K | 1 | 180.85 | 186.24 | 554.93 | 619.96 |
| Llama-3.1-8B | LC20K | 8 | 662.4 | 680.29 | 811.52 | 971.66 |
| Llama-3.1-8B | LC64K | 1 | 135.62 | 139.52 | 483.99 | 519.1 |
| Llama-3.1-8B | LC64K | 8 | 294.94 | 303.02 | 652.33 | 723.45 |
| Llama-3.1-8B | LC126K | 1 | 99.764 | 102.6 | 402.31 | 420.43 |
| Llama-3.1-8B | LC126K | 8 | 168.09 | 170.1 | 511.91 | 553.64 |
| Llama-3.1-70B | LC20K | 1 | 22.038 | 22.591 | 95.173 | 107.01 |
| Llama-3.1-70B | LC20K | 8 | 65.774 | 122.8 | 149.87 | 171.76 |
| Llama-3.1-70B | LC64K | 1 | 21.884 | 20.464 | 86.452 | 93.044 |
| Llama-3.1-70B | LC64K | 8 | 19.805 | 76.637 | 124.57 | 134.5 |
| Llama-3.1-70B | LC126K | 1 | 11.401 | 17.928 | 72.995 | 77.062 |
| Llama-3.1-70B | LC126K | 8 | 9.9408 | 50.1 | 98.404 | 104.25 |
| Llama-3.1-405B | LC20K | 1 | 0.59991 | 4.0238 | 25.845 | 28.785 |
| Llama-3.1-405B | LC20K | 8 | 4.3293 | 26.094 | 42.762 | 47.535 |
| Llama-3.1-405B | LC64K | 1 | 0.58032 | 3.9152 | 24.382 | 26.191 |
| Llama-3.1-405B | LC64K | 8 | 3.4815 | 21.238 | 36.242 | 38.387 |
| Llama-3.1-405B | LC126K | 1 | 0.55481 | 3.7383 | 20.956 | 22.202 |
| Llama-3.1-405B | LC126K | 8 | 2.7286 | 16.825 | 28.963 | 30.327 |

## Energy efficiency (tokens/J)

| model | context | batch | HBM_GPU | M3D_GPU | M3D_NMP_UNIFORM | M3D_NMP_CPA |
|---|---|---|---|---|---|---|
| Llama-3.1-8B | LC20K | 1 | 0.41619 | 0.44839 | 1.6464 | 1.9224 |
| Llama-3.1-8B | LC20K | 8 | 1.5108 | 1.619 | 2.088 | 2.4758 |
| Llama-3.1-8B | LC64K | 1 | 0.31025 | 0.3339 | 1.3243 | 1.4702 |
| Llama-3.1-8B | LC64K | 8 | 0.67334 | 0.72217 | 1.5892 | 1.7731 |
| Llama-3.1-8B | LC126K | 1 | 0.22814 | 0.24535 | 1.0353 | 1.1194 |
| Llama-3.1-8B | LC126K | 8 | 0.37784 | 0.40559 | 1.1896 | 1.2884 |
| Llama-3.1-70B | LC20K | 1 | 0.050522 | 0.054442 | 0.25813 | 0.30284 |
| Llama-3.1-70B | LC20K | 8 | 0.23065 | 0.29125 | 0.3524 | 0.40225 |
| Llama-3.1-70B | LC64K | 1 | 0.04536 | 0.048988 | 0.22163 | 0.24731 |
| Llama-3.1-70B | LC64K | 8 | 0.10898 | 0.18178 | 0.28592 | 0.31108 |
| Llama-3.1-70B | LC126K | 1 | 0.035341 | 0.042869 | 0.18293 | 0.19984 |
| Llama-3.1-70B | LC126K | 8 | 0.062408 | 0.11884 | 0.22466 | 0.23941 |
| Llama-3.1-405B | LC20K | 1 | 0.0042789 | 0.0097127 | 0.060794 | 0.071229 |
| Llama-3.1-405B | LC20K | 8 | 0.029422 | 0.061832 | 0.089513 | 0.099734 |
| Llama-3.1-405B | LC64K | 1 | 0.0041361 | 0.0093854 | 0.054755 | 0.061814 |
| Llama-3.1-405B | LC64K | 8 | 0.023778 | 0.050247 | 0.076179 | 0.082082 |
| Llama-3.1-405B | LC126K | 1 | 0.0039504 | 0.0089458 | 0.047378 | 0.052587 |
| Llama-3.1-405B | LC126K | 8 | 0.018718 | 0.039753 | 0.06255 | 0.066361 |

## Temperature (C; metric definitions above)

| model | context | batch | HBM_GPU | M3D_GPU | M3D_NMP_UNIFORM | M3D_NMP_CPA |
|---|---|---|---|---|---|---|
| Llama-3.1-8B | LC20K | 1 | 85 | 85 | 62.505 | 60.793 |
| Llama-3.1-8B | LC20K | 8 | 85 | 85 | 69.259 | 70.514 |
| Llama-3.1-8B | LC64K | 1 | 85 | 85 | 64.85 | 63.21 |
| Llama-3.1-8B | LC64K | 8 | 85 | 85 | 71.537 | 71.436 |
| Llama-3.1-8B | LC126K | 1 | 85 | 85 | 67.412 | 65.566 |
| Llama-3.1-8B | LC126K | 8 | 85 | 85 | 73.979 | 74.103 |
| Llama-3.1-70B | LC20K | 1 | 85 | 85 | 66.313 | 64.436 |
| Llama-3.1-70B | LC20K | 8 | 85 | 85 | 73.69 | 74.345 |
| Llama-3.1-70B | LC64K | 1 | 85 | 85 | 67.648 | 65.773 |
| Llama-3.1-70B | LC64K | 8 | 85 | 85 | 74.982 | 74.735 |
| Llama-3.1-70B | LC126K | 1 | 85 | 85 | 68.558 | 66.632 |
| Llama-3.1-70B | LC126K | 8 | 85 | 85 | 75.13 | 74.912 |
| Llama-3.1-405B | LC20K | 1 | 85 | 85 | 74.58 | 71.936 |
| Llama-3.1-405B | LC20K | 8 | 85 | 85 | 81.033 | 81.41 |
| Llama-3.1-405B | LC64K | 1 | 85 | 85 | 74.988 | 71.983 |
| Llama-3.1-405B | LC64K | 8 | 85 | 85 | 81.016 | 80.025 |
| Llama-3.1-405B | LC126K | 1 | 85 | 85 | 74.327 | 71.386 |
| Llama-3.1-405B | LC126K | 8 | 85 | 85 | 79.044 | 78.256 |

## Normalized results

| metric | numerator | denominator | paired_cases | min | mean | geomean | max | min_case | max_case |
|---|---|---|---|---|---|---|---|---|---|
| tokens_per_s | M3D_GPU | HBM_GPU | 18 | 0.93513 | 3.2749 | 2.3397 | 6.7465 | Llama-3.1-70B/LC64K/1 | Llama-3.1-405B/LC64K/1 |
| tokens_per_s | M3D_NMP_UNIFORM | M3D_GPU | 18 | 1.1929 | 3.1871 | 2.7754 | 6.4231 | Llama-3.1-8B/LC20K/8 | Llama-3.1-405B/LC20K/1 |
| tokens_per_s | M3D_NMP_CPA | M3D_GPU | 18 | 1.3987 | 3.4583 | 3.0248 | 7.1536 | Llama-3.1-70B/LC20K/8 | Llama-3.1-405B/LC20K/1 |
| tokens_per_s | M3D_NMP_CPA | HBM_GPU | 18 | 1.4669 | 12.261 | 7.0771 | 47.982 | Llama-3.1-8B/LC20K/8 | Llama-3.1-405B/LC20K/1 |
| tokens_per_s | M3D_NMP_CPA | M3D_NMP_UNIFORM | 18 | 1.045 | 1.0905 | 1.0899 | 1.1973 | Llama-3.1-8B/LC126K/1 | Llama-3.1-8B/LC20K/8 |
| tokens_per_J | M3D_GPU | HBM_GPU | 18 | 1.0717 | 1.5441 | 1.4643 | 2.2699 | Llama-3.1-8B/LC20K/8 | Llama-3.1-405B/LC20K/1 |
| tokens_per_J | M3D_NMP_UNIFORM | M3D_GPU | 18 | 1.21 | 3.2452 | 2.8008 | 6.2593 | Llama-3.1-70B/LC20K/8 | Llama-3.1-405B/LC20K/1 |
| tokens_per_J | M3D_NMP_CPA | M3D_GPU | 18 | 1.3811 | 3.6393 | 3.1232 | 7.3336 | Llama-3.1-70B/LC20K/8 | Llama-3.1-405B/LC20K/1 |
| tokens_per_J | M3D_NMP_CPA | HBM_GPU | 18 | 1.6387 | 5.7096 | 4.5734 | 16.647 | Llama-3.1-8B/LC20K/8 | Llama-3.1-405B/LC20K/1 |
| tokens_per_J | M3D_NMP_CPA | M3D_NMP_UNIFORM | 18 | 1.0609 | 1.1157 | 1.1151 | 1.1857 | Llama-3.1-405B/LC126K/8 | Llama-3.1-8B/LC20K/8 |

## Runtime

{
  "config": {
    "benchmark_id": "formal_long_context_v2",
    "models": [
      "Llama-3.1-8B",
      "Llama-3.1-70B",
      "Llama-3.1-405B"
    ],
    "contexts": {
      "LC20K": 20000,
      "LC64K": 64000,
      "LC126K": 126000
    },
    "incremental_prefill_tokens": 128,
    "generated_tokens": 32,
    "batches": [
      1,
      8
    ],
    "paths": [
      "HBM_GPU",
      "M3D_GPU",
      "M3D_NMP_UNIFORM",
      "M3D_NMP_CPA"
    ],
    "thermal": {
      "design_point_C": 85.0,
      "GPU_metric": "THERMAL_CLOSED_DESIGN_POINT",
      "NMP_metric": "WORKLOAD_SPECIFIC_STEADY_STATE",
      "NMP_phase": "DECODE",
      "mapping": "DIE_GROUPED_BEOL_UNIFORM",
      "operator_cache": "runs/formal_long_context_v2/thermal_setup.pkl"
    },
    "operating_point_source": "configs/architecture/m3d_feol_execution.yaml"
  },
  "base_HEAD": "158e7c323dbb40366621a0ee8768aacb96ada257",
  "operating_points": 18,
  "final_rows": 72,
  "NMP_physical_checkpoints": 36,
  "NMP_Decode_steps": 1152,
  "GPU_observation_steps": 576,
  "v1_workload_reuse": false,
  "adaptive_clipping": false,
  "pytest_run": false,
  "sanity_checks": "PASS",
  "capacity_exceptions": [],
  "original_requirement_all_72_metrics_finite": true,
  "total_wall_clock_s": 3870.4153435230255,
  "runtime_target": "TARGET_1H_NOT_MET",
  "outer_processes": 4,
  "inner_workers": 1,
  "cpu_count": 24,
  "maximum_outer_processes": 6,
  "priority_runtime": {
    "start_epoch": 1789395814.2295566,
    "additional_case_processes": 2,
    "maximum_outer_processes": 6,
    "inner_workers": 1,
    "cases": [
      "Llama-3.1-405B/LC20K/B8/CPA",
      "Llama-3.1-405B/LC126K/B8/CPA"
    ],
    "reason": "Longest CPA planning first; atomic case locks prevent duplicate execution",
    "available_RAM_GB_at_start": 12.43
  },
  "runtime_configurations": [
    {
      "start_epoch": 1789395168.7497923,
      "outer_processes": 4,
      "inner_workers": 1,
      "available_RAM_bytes": 17783910400
    }
  ],
  "calibration": [
    {
      "model": "Llama-3.1-8B",
      "context": "LC64K",
      "batch": 1,
      "path": "M3D_NMP_UNIFORM",
      "elapsed_s": 7.036054400028661,
      "inner_workers": 4,
      "execution_checkpoint_reused": false
    },
    {
      "model": "Llama-3.1-8B",
      "context": "LC64K",
      "batch": 1,
      "path": "M3D_NMP_CPA",
      "elapsed_s": 53.489519499999005,
      "inner_workers": 4,
      "execution_checkpoint_reused": false
    }
  ],
  "initial_available_RAM_bytes": 17783910400,
  "summed_path_runtime_s": {
    "HBM_GPU": 8.242662299948279,
    "M3D_GPU": 826.5093924000976,
    "M3D_NMP_UNIFORM": 2118.2006138999714,
    "M3D_NMP_CPA": 11133.838599900133
  },
  "runtime_accounting": "Total wall includes orchestration and disk recovery. Path sums are recorded successful attempts inclusive of projection; projection is a subset, not additive. Interrupted pre-recovery attempt costs remain in total wall, not reconstructed. Capacity repair wall-clock is reported separately in capacity_repair.",
  "summed_spatial_projection_s": 538.3811344998539,
  "thermal_operator": {
    "operator_reused": true,
    "operator_builds": 0,
    "physical_signature": "2fcecd86ccbe1599b61790bab556d55d334ecb78086df9888d84bc638a674dcd",
    "cache_status": "HIT",
    "cache_load_s": 44.96409319998929,
    "cache": "E:\\BaiduSyncdisk\\study\\PAPER\\DAC 2026\\Project\\runs\\formal_long_context_v2\\thermal_setup.pkl",
    "elapsed_s": 1278.454397201538,
    "cache_file_unchanged": true,
    "wait_for_candidates": true,
    "repair_only": true
  },
  "thermal_solve_s": 261.2891585000325,
  "trend_changes": 0,
  "thermal_semantics": "GPU baselines: 85C closed design point; NMP: workload-specific sustained Decode steady state. Not transient request peaks.",
  "physical_fingerprint": "97b6c65bc3565153e1aa1fed0b34225de846aafed7d6eb360f2e68e29b41976e",
  "capacity_repair": {
    "existing_rows_unchanged": 69,
    "valid_rows": 72,
    "pytest_run": false,
    "wall_clock_s": 2045.8326568603516,
    "base_HEAD": "158e7c323dbb40366621a0ee8768aacb96ada257",
    "new_execution_paths": 3,
    "new_NMP_Decode_steps": 64,
    "new_thermal_solves": 2,
    "inner_workers": 4,
    "source_change": "Capacity-only layer legalization; physical equations and CPA unchanged"
  },
  "fingerprint_provenance": {
    "existing_69": "726008d5174adbbf190ee45d1e7074a3e33b02db5bcebaa1a4f8057098492edf",
    "repaired_3": "97b6c65bc3565153e1aa1fed0b34225de846aafed7d6eb360f2e68e29b41976e"
  },
  "initial_sweep_wall_clock_s": 3870.4153435230255
}

## Trend audit

[]

Lightweight conservation and integrity assertions PASS. No pytest was run. The capacity repair preserves all 69 existing result values and their physical artifacts; only the three missing paths are newly executed. Frozen physical equations and CPA are unchanged. Legacy results may have been removed separately by the user during workspace cleanup.

## Capacity exceptions

[]

## Capacity repair

LC64K/B8 previously failed because cyclic start-layer phases caused local slot overflow, despite more total free memory than LC126K/B8. GPU feasibility now checks global persistent state plus workspace. Only when original Uniform initialization fails, a capacity-only layer reassignment retains die/group/atom ownership and uses no route, latency or CPA objective. CPA then runs unchanged from that legal initial placement. This Uniform is explicitly UNIFORM_STRIPING_CAPACITY_LEGALIZED.

[
  {
    "path": "M3D_GPU",
    "capacity": "GLOBAL_CAPACITY_GATE"
  },
  {
    "path": "M3D_NMP_UNIFORM",
    "capacity": "UNIFORM_STRIPING_CAPACITY_LEGALIZED"
  },
  {
    "path": "M3D_NMP_CPA",
    "capacity": "UNIFORM_STRIPING_CAPACITY_LEGALIZED"
  }
]
