# Formal long-context v2 GPU refresh

72 canonical rows; H=20K/64K/126K, P=128, G=32, B=1/8. Only HBM_GPU and M3D_GPU were refreshed. NMP results and physical artifacts are frozen.

HBM: thermal cap 3.2973872924850354 -> 3.0865643306417803 TB/s from runs/no_nmp_geometry_sensitivity_v2/thermal_limits.csv. Read energy 1.9955 -> 3.00 pJ/bit. The existing write coefficient remains 1.9955 pJ/bit; the read-only rebaseline does not invent a write model. C2C and all logical traffic/compute/capacity remain unchanged. This is an optimistic full-utilization thermally closed design point.

M3D_GPU: GPU_PORT_BALANCED physical operator-level data-chain and contention-aware external-port service; all 576 saved Decode steps reused. Exact atom-birth integration regenerates route-sensitive events without reexecuting timing. Frozen event coefficients account for real FEOL route changes and new static duration. No CPA or NMP MAC activity.

Prefill scope: M3D_GPU uses the diagnostic's reused physical Prefill time, rather than the old formal aggregate Prefill time. Its Prefill energy is recomputed with port-balanced routing. The two NMP paths retain their frozen aggregate Prefill times and energy. Therefore this formal comparison contains that explicitly documented Prefill model difference.

GPU temperatures remain 85 C THERMAL_CLOSED_DESIGN_POINT, not new workload solves. NMP temperatures remain their unchanged Decode-only steady-state results. No thermal operator was loaded or thermal sweep rerun.

Old formal M3D bandwidth was an aggregate closure, not a measured achieved bandwidth. The table separately labels old Uniform achieved bandwidth. NMP absolute values are unchanged; normalized NMP bars necessarily change when their HBM denominator changes.

## Updated HBM

| model | context | batch | tokens_per_s | tokens_per_J | J_per_token |
|---|---|---|---|---|---|
| Llama-3.1-8B | LC20K | 1 | 169.283 | 0.388029 | 2.57713 |
| Llama-3.1-8B | LC20K | 8 | 623.746 | 1.41558 | 0.706422 |
| Llama-3.1-8B | LC64K | 1 | 127.251 | 0.289573 | 3.45336 |
| Llama-3.1-8B | LC64K | 8 | 277.507 | 0.630405 | 1.58628 |
| Llama-3.1-8B | LC126K | 1 | 93.6588 | 0.213076 | 4.69316 |
| Llama-3.1-8B | LC126K | 8 | 158.119 | 0.354035 | 2.82458 |
| Llama-3.1-70B | LC20K | 1 | 20.6292 | 0.0471148 | 21.2247 |
| Llama-3.1-70B | LC20K | 8 | 65.7736 | 0.222902 | 4.48627 |
| Llama-3.1-70B | LC64K | 1 | 20.5346 | 0.0426076 | 23.47 |
| Llama-3.1-70B | LC64K | 8 | 19.8048 | 0.107223 | 9.3264 |
| Llama-3.1-70B | LC126K | 1 | 11.4012 | 0.0339014 | 29.4973 |
| Llama-3.1-70B | LC126K | 8 | 9.94076 | 0.0618258 | 16.1745 |
| Llama-3.1-405B | LC20K | 1 | 0.599906 | 0.00425698 | 234.908 |
| Llama-3.1-405B | LC20K | 8 | 4.32926 | 0.0292909 | 34.1404 |
| Llama-3.1-405B | LC64K | 1 | 0.580324 | 0.00411563 | 242.976 |
| Llama-3.1-405B | LC64K | 8 | 3.48149 | 0.023692 | 42.2083 |
| Llama-3.1-405B | LC126K | 1 | 0.554806 | 0.00393167 | 254.345 |
| Llama-3.1-405B | LC126K | 8 | 2.72858 | 0.0186648 | 53.5768 |

## M3D GPU comparison

| model | context | batch | old_formal_tokens_per_s | new_tokens_per_s | old_formal_bandwidth_TBps | old_bandwidth_semantics | old_uniform_achieved_TBps | new_achieved_TBps | new_utilization_pct | tokens_per_J |
|---|---|---|---|---|---|---|---|---|---|---|
| Llama-3.1-8B | LC20K | 1 | 186.242 | 171.384 | 3.39578 | AGGREGATE_CLOSURE_NOT_MEASURED | 2.43666 | 3.16258 | 93.1325 | 0.441567 |
| Llama-3.1-8B | LC20K | 8 | 680.286 | 650.848 | 3.39578 | AGGREGATE_CLOSURE_NOT_MEASURED | 2.84822 | 3.2778 | 96.5256 | 1.60628 |
| Llama-3.1-8B | LC64K | 1 | 139.517 | 131.168 | 3.39578 | AGGREGATE_CLOSURE_NOT_MEASURED | 2.61883 | 3.217 | 94.7351 | 0.330161 |
| Llama-3.1-8B | LC64K | 8 | 303.019 | 293.732 | 3.39578 | AGGREGATE_CLOSURE_NOT_MEASURED | 3.13139 | 3.3426 | 98.4339 | 0.718169 |
| Llama-3.1-8B | LC126K | 1 | 102.601 | 97.6678 | 3.39578 | AGGREGATE_CLOSURE_NOT_MEASURED | 2.78287 | 3.26122 | 96.0375 | 0.24318 |
| Llama-3.1-8B | LC126K | 8 | 170.098 | 165.666 | 3.39578 | AGGREGATE_CLOSURE_NOT_MEASURED | 3.24274 | 3.36587 | 99.1191 | 0.403681 |
| Llama-3.1-70B | LC20K | 1 | 22.5914 | 21.5675 | 3.39578 | AGGREGATE_CLOSURE_NOT_MEASURED | 3.07492 | 3.25391 | 95.822 | 0.0539847 |
| Llama-3.1-70B | LC20K | 8 | 122.796 | 118.064 | 3.39578 | AGGREGATE_CLOSURE_NOT_MEASURED | 3.14657 | 3.28703 | 96.7974 | 0.289214 |
| Llama-3.1-70B | LC64K | 1 | 20.4642 | 19.6017 | 3.39578 | AGGREGATE_CLOSURE_NOT_MEASURED | 3.10132 | 3.2662 | 96.1841 | 0.0486085 |
| Llama-3.1-70B | LC64K | 8 | 76.637 | 73.7278 | 3.39578 | AGGREGATE_CLOSURE_NOT_MEASURED | 3.23575 | 3.32682 | 97.9692 | 0.180532 |
| Llama-3.1-70B | LC126K | 1 | 17.9284 | 17.1845 | 3.39578 | AGGREGATE_CLOSURE_NOT_MEASURED | 3.13192 | 3.2803 | 96.5994 | 0.0425431 |
| Llama-3.1-70B | LC126K | 8 | 50.1001 | 48.2173 | 3.39578 | AGGREGATE_CLOSURE_NOT_MEASURED | 3.28932 | 3.35037 | 98.6625 | 0.118033 |
| Llama-3.1-405B | LC20K | 1 | 4.0238 | 3.92025 | 3.39578 | AGGREGATE_CLOSURE_NOT_MEASURED | 3.20841 | 3.31546 | 97.6346 | 0.00966704 |
| Llama-3.1-405B | LC20K | 8 | 26.094 | 25.4171 | 3.39578 | AGGREGATE_CLOSURE_NOT_MEASURED | 3.22295 | 3.32189 | 97.8241 | 0.0615451 |
| Llama-3.1-405B | LC64K | 1 | 3.91516 | 3.80846 | 3.39578 | AGGREGATE_CLOSURE_NOT_MEASURED | 3.2132 | 3.31758 | 97.697 | 0.00933887 |
| Llama-3.1-405B | LC64K | 8 | 21.2376 | 20.5341 | 3.39578 | AGGREGATE_CLOSURE_NOT_MEASURED | 3.25098 | 3.33408 | 98.183 | 0.0499468 |
| Llama-3.1-405B | LC126K | 1 | 3.73826 | 3.63563 | 3.39578 | AGGREGATE_CLOSURE_NOT_MEASURED | 3.21954 | 3.32038 | 97.7796 | 0.00890136 |
| Llama-3.1-405B | LC126K | 8 | 16.8252 | 16.16 | 3.39578 | AGGREGATE_CLOSURE_NOT_MEASURED | 3.27788 | 3.3458 | 98.528 | 0.0394685 |

## Four-path aggregates

Geometric means across paired operating points (not aggregate serving throughput).

| model | path | cases | aggregation | decode_tokens_per_s | E2E_tokens_per_s | E2E_tokens_per_J | speedup_vs_HBM |
|---|---|---|---|---|---|---|---|
| ALL | HBM_GPU | 18 | GEOMETRIC_MEAN | 18.4241 | 17.6429 | 0.0666778 | 1 |
| ALL | M3D_GPU | 18 | GEOMETRIC_MEAN | 44.3732 | 40.775 | 0.100569 | 2.31113 |
| ALL | M3D_NMP_UNIFORM | 18 | GEOMETRIC_MEAN | 146.461 | 117.824 | 0.283729 | 6.67826 |
| ALL | M3D_NMP_CPA | 18 | GEOMETRIC_MEAN | 162.932 | 128.412 | 0.316389 | 7.27841 |
| Llama-3.1-8B | HBM_GPU | 6 | GEOMETRIC_MEAN | 207.081 | 195.141 | 0.443057 | 1 |
| Llama-3.1-8B | M3D_GPU | 6 | GEOMETRIC_MEAN | 216.15 | 202.785 | 0.50461 | 1.03917 |
| Llama-3.1-8B | M3D_NMP_UNIFORM | 6 | GEOMETRIC_MEAN | 658.838 | 555.18 | 1.43985 | 2.84502 |
| Llama-3.1-8B | M3D_NMP_CPA | 6 | GEOMETRIC_MEAN | 739.768 | 612.225 | 1.61728 | 3.13735 |
| Llama-3.1-70B | HBM_GPU | 6 | GEOMETRIC_MEAN | 20.7327 | 19.9232 | 0.0681928 | 1 |
| Llama-3.1-70B | M3D_GPU | 6 | GEOMETRIC_MEAN | 41.5782 | 38.08 | 0.0939576 | 1.91134 |
| Llama-3.1-70B | M3D_NMP_UNIFORM | 6 | GEOMETRIC_MEAN | 125.162 | 101.653 | 0.248748 | 5.10222 |
| Llama-3.1-70B | M3D_NMP_CPA | 6 | GEOMETRIC_MEAN | 139.123 | 110.777 | 0.276659 | 5.56021 |
| Llama-3.1-405B | HBM_GPU | 6 | GEOMETRIC_MEAN | 1.45668 | 1.41254 | 0.00981172 | 1 |
| Llama-3.1-405B | M3D_GPU | 6 | GEOMETRIC_MEAN | 9.72163 | 8.77905 | 0.0214536 | 6.21506 |
| Llama-3.1-405B | M3D_NMP_UNIFORM | 6 | GEOMETRIC_MEAN | 38.0991 | 28.9833 | 0.0637727 | 20.5185 |
| Llama-3.1-405B | M3D_NMP_CPA | 6 | GEOMETRIC_MEAN | 42.0263 | 31.2218 | 0.0707843 | 22.1032 |

## Preservation

{'old_HEAD': 'd4a6cd5f0e47eede5175c82315365e35ee63952b', 'HBM_GPU': '18/18 updated', 'M3D_GPU': '18/18 updated', 'M3D_NMP_UNIFORM': '18/18 unchanged', 'M3D_NMP_CPA': '18/18 unchanged', 'protected_artifact_count': 250, 'candidate_bytes': 'IDENTICAL', 'all_NMP_row_fields': 'IDENTICAL', 'status': 'PASS'}

## Reproduction

In the om3dthermal Conda environment: python scripts/refresh_formal_gpu_results.py; python scripts/finalize_formal_long_context_v2.py; python scripts/plot_formal_long_context_v2.py. The refresh reuses completed candidates on subsequent invocations. No thermal sweep or NMP execution is dispatched.

## Validation and plots

31 related tests passed. No full pytest and no thermal solver. Executed python scripts/plot_formal_long_context_v2.py: 72 rows; HBM normalization all 1; three SVG/PDF pairs generated.
