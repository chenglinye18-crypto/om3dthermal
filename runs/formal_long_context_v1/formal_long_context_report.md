# Formal long-context four-path E2E evaluation



Canonical source: `final_e2e_metrics.csv`. Exactly 18 points; no adaptive clipping. All temperatures are workload-specific E2E-equivalent steady-state solves. Throughput, energy efficiency and their ratios are nominal; per-path thermal-feasibility flags remain binding.

LC20K=(20000,512,256); LC64K=(64000,512,512); LC126K=(126000,512,512). Models 8B/70B/405B, batches 1/8. W1 is retained as short-context boundary evidence; B32 remains archived stress. Historical artifacts are preserved.



## Throughput (tokens/s)

| model | context_label | B | HBM_GPU_tok_s | M3D_GPU_tok_s | M3D_NMP_UNIFORM_tok_s | M3D_NMP_CPA_tok_s |
|---|---|---|---|---|---|---|
| Llama-3.1-8B | LC20K | 1 | 183.736 | 189.146 | 587.448 | 660.068 |
| Llama-3.1-8B | LC20K | 8 | 685.08 | 704.511 | 856.148 | 1039.91 |
| Llama-3.1-8B | LC64K | 1 | 138.995 | 143.102 | 536.207 | 580.236 |
| Llama-3.1-8B | LC64K | 8 | 311.369 | 320.458 | 750.155 | 847.757 |
| Llama-3.1-8B | LC126K | 1 | 103.035 | 106.072 | 466.441 | 490.966 |
| Llama-3.1-8B | LC126K | 8 | 178.692 | 179.854 | 619.987 | 682.145 |
| Llama-3.1-70B | LC20K | 1 | 22.4836 | 23.005 | 103.402 | 117.399 |
| Llama-3.1-70B | LC20K | 8 | 65.7348 | 130.695 | 162.887 | 188.861 |
| Llama-3.1-70B | LC64K | 1 | 22.5377 | 20.9995 | 97.335 | 105.73 |
| Llama-3.1-70B | LC64K | 8 | 20.0655 | 84.725 | 148.546 | 162.742 |
| Llama-3.1-70B | LC126K | 1 | 11.5797 | 18.5752 | 85.344 | 90.9944 |
| Llama-3.1-70B | LC126K | 8 | 10.1147 | 55.5005 | 122.311 | 131.314 |
| Llama-3.1-405B | LC20K | 1 | 0.6159 | 4.10334 | 29.5739 | 33.4528 |
| Llama-3.1-405B | LC20K | 8 | 4.39232 | 28.0629 | 48.4533 | 54.5984 |
| Llama-3.1-405B | LC64K | 1 | 0.596911 | 4.00733 | 28.4961 | 30.9748 |
| Llama-3.1-405B | LC64K | 8 | 3.55235 | 24.1118 | 45.6244 | 48.9925 |
| Llama-3.1-405B | LC126K | 1 | 0.57068 | 3.85064 | 25.0883 | 26.8999 |
| Llama-3.1-405B | LC126K | 8 | 2.78481 | 19.3695 | 37.6161 | 39.8432 |



## Energy efficiency (tokens/J)

| model | context_label | B | HBM_GPU_tok_per_J | M3D_GPU_tok_per_J | M3D_NMP_UNIFORM_tok_per_J | M3D_NMP_CPA_tok_per_J |
|---|---|---|---|---|---|---|
| Llama-3.1-8B | LC20K | 1 | 0.422132 | 0.454435 | 1.75294 | 2.06257 |
| Llama-3.1-8B | LC20K | 8 | 1.57096 | 1.68662 | 2.23373 | 2.68622 |
| Llama-3.1-8B | LC64K | 1 | 0.319485 | 0.343977 | 1.52113 | 1.72502 |
| Llama-3.1-8B | LC64K | 8 | 0.715272 | 0.769392 | 1.86908 | 2.12888 |
| Llama-3.1-8B | LC126K | 1 | 0.236807 | 0.25491 | 1.24337 | 1.3702 |
| Llama-3.1-8B | LC126K | 8 | 0.401191 | 0.431872 | 1.46669 | 1.61853 |
| Llama-3.1-70B | LC20K | 1 | 0.0513352 | 0.0552763 | 0.27922 | 0.331746 |
| Llama-3.1-70B | LC20K | 8 | 0.241645 | 0.312263 | 0.385963 | 0.446511 |
| Llama-3.1-70B | LC64K | 1 | 0.0467476 | 0.0504798 | 0.2572 | 0.291396 |
| Llama-3.1-70B | LC64K | 8 | 0.115961 | 0.203124 | 0.344891 | 0.382086 |
| Llama-3.1-70B | LC126K | 1 | 0.0365652 | 0.0446386 | 0.221134 | 0.246097 |
| Llama-3.1-70B | LC126K | 8 | 0.066218 | 0.133061 | 0.283075 | 0.306298 |
| Llama-3.1-405B | LC20K | 1 | 0.00437369 | 0.00986067 | 0.0672014 | 0.0799927 |
| Llama-3.1-405B | LC20K | 8 | 0.0306673 | 0.0670104 | 0.100962 | 0.114156 |
| Llama-3.1-405B | LC64K | 1 | 0.00425234 | 0.00963452 | 0.0645923 | 0.0746213 |
| Llama-3.1-405B | LC64K | 8 | 0.0254152 | 0.0577586 | 0.0950437 | 0.104368 |
| Llama-3.1-405B | LC126K | 1 | 0.00407347 | 0.0092556 | 0.0576789 | 0.0655724 |
| Llama-3.1-405B | LC126K | 8 | 0.0201143 | 0.0463766 | 0.081245 | 0.0875272 |



## Temperature (C)

| model | context_label | B | HBM_GPU_Tmax_C | M3D_GPU_Tmax_C | M3D_NMP_UNIFORM_Tmax_C | M3D_NMP_CPA_Tmax_C | metric_status |
|---|---|---|---|---|---|---|---|
| Llama-3.1-8B | LC20K | 1 | 85.0823 | 85.1133 | 63.5824 | 62.4712 | NOMINAL_WITH_EXPLICIT_THERMAL_CLOSURE_REQUIRED |
| Llama-3.1-8B | LC20K | 8 | 85.2291 | 85.3954 | 70.4429 | 72.1886 | NOMINAL_WITH_EXPLICIT_THERMAL_CLOSURE_REQUIRED |
| Llama-3.1-8B | LC64K | 1 | 85.0515 | 85.0797 | 65.6331 | 64.0065 | NOMINAL_WITH_EXPLICIT_THERMAL_CLOSURE_REQUIRED |
| Llama-3.1-8B | LC64K | 8 | 85.0978 | 85.173 | 72.2908 | 72.6614 | NOMINAL_WITH_EXPLICIT_THERMAL_CLOSURE_REQUIRED |
| Llama-3.1-8B | LC126K | 1 | 85.0595 | 85.0982 | 68.5489 | 66.7642 | NOMINAL_WITH_EXPLICIT_THERMAL_CLOSURE_REQUIRED |
| Llama-3.1-8B | LC126K | 8 | 86.1431 | 85.1628 | 74.9249 | 75.306 | NOMINAL_WITH_EXPLICIT_THERMAL_CLOSURE_REQUIRED |
| Llama-3.1-70B | LC20K | 1 | 85.3627 | 85.1064 | 67.576 | 66.0246 | NOMINAL_WITH_EXPLICIT_THERMAL_CLOSURE_REQUIRED |
| Llama-3.1-70B | LC20K | 8 | 58.2456 | 85.5578 | 74.9269 | 75.9797 | NOMINAL_WITH_EXPLICIT_THERMAL_CLOSURE_REQUIRED |
| Llama-3.1-70B | LC64K | 1 | 89.9366 | 85.0754 | 68.499 | 66.9152 | NOMINAL_WITH_EXPLICIT_THERMAL_CLOSURE_REQUIRED |
| Llama-3.1-70B | LC64K | 8 | 43.3973 | 85.2888 | 75.8008 | 75.624 | NOMINAL_WITH_EXPLICIT_THERMAL_CLOSURE_REQUIRED |
| Llama-3.1-70B | LC126K | 1 | 64.8204 | 85.0999 | 69.6232 | 67.9682 | NOMINAL_WITH_EXPLICIT_THERMAL_CLOSURE_REQUIRED |
| Llama-3.1-70B | LC126K | 8 | 40.3666 | 85.2889 | 76.1223 | 76.0627 | NOMINAL_WITH_EXPLICIT_THERMAL_CLOSURE_REQUIRED |
| Llama-3.1-405B | LC20K | 1 | 38.5723 | 85.0973 | 75.6949 | 73.5191 | NOMINAL_WITH_EXPLICIT_THERMAL_CLOSURE_REQUIRED |
| Llama-3.1-405B | LC20K | 8 | 38.9596 | 85.6069 | 81.9274 | 82.2653 | NOMINAL_WITH_EXPLICIT_THERMAL_CLOSURE_REQUIRED |
| Llama-3.1-405B | LC64K | 1 | 38.5055 | 85.0634 | 75.6643 | 72.8909 | NOMINAL_WITH_EXPLICIT_THERMAL_CLOSURE_REQUIRED |
| Llama-3.1-405B | LC64K | 8 | 38.4325 | 85.3568 | 81.6221 | 80.6893 | NOMINAL_WITH_EXPLICIT_THERMAL_CLOSURE_REQUIRED |
| Llama-3.1-405B | LC126K | 1 | 38.4647 | 85.0824 | 75.1922 | 72.5412 | NOMINAL_WITH_EXPLICIT_THERMAL_CLOSURE_REQUIRED |
| Llama-3.1-405B | LC126K | 8 | 38.2371 | 85.3954 | 79.6962 | 79.0546 | NOMINAL_WITH_EXPLICIT_THERMAL_CLOSURE_REQUIRED |



## Normalized results

| metric | ratio | min | mean | geomean | max | min_case | max_case |
|---|---|---|---|---|---|---|---|
| throughput | M3D_GPU_over_HBM | 0.931749 | 3.42581 | 2.40879 | 6.9554 | Llama-3.1-70B LC64K B1 | Llama-3.1-405B LC126K B8 |
| throughput | UNIFORM_over_M3D_GPU | 1.21524 | 3.53198 | 3.03943 | 7.20728 | Llama-3.1-8B LC20K B8 | Llama-3.1-405B LC20K B1 |
| throughput | CPA_over_M3D_GPU | 1.44505 | 3.87547 | 3.35305 | 8.15258 | Llama-3.1-70B LC20K B8 | Llama-3.1-405B LC20K B1 |
| throughput | CPA_over_UNIFORM | 1.05258 | 1.10389 | 1.10318 | 1.21464 | Llama-3.1-8B LC126K B1 | Llama-3.1-8B LC20K B8 |
| throughput | CPA_over_HBM | 1.51794 | 14.2333 | 8.07678 | 54.3154 | Llama-3.1-8B LC20K B8 | Llama-3.1-405B LC20K B1 |
| energy_efficiency | M3D_GPU_over_HBM | 1.07362 | 1.5801 | 1.49129 | 2.30565 | Llama-3.1-8B LC20K B8 | Llama-3.1-405B LC126K B8 |
| energy_efficiency | UNIFORM_over_M3D_GPU | 1.23602 | 3.618 | 3.08484 | 6.81509 | Llama-3.1-70B LC20K B8 | Llama-3.1-405B LC20K B1 |
| energy_efficiency | CPA_over_M3D_GPU | 1.42992 | 4.12646 | 3.49899 | 8.11229 | Llama-3.1-70B LC20K B8 | Llama-3.1-405B LC20K B1 |
| energy_efficiency | CPA_over_UNIFORM | 1.07732 | 1.13484 | 1.13425 | 1.20257 | Llama-3.1-405B LC126K B8 | Llama-3.1-8B LC20K B8 |
| energy_efficiency | CPA_over_HBM | 1.70992 | 6.56124 | 5.218 | 18.2895 | Llama-3.1-8B LC20K B8 | Llama-3.1-405B LC20K B1 |



## Thermal statistics

| path | max_Tmax_C | mean_Tmax_C | over_85C | max_case | hotspot_statistics |
|---|---|---|---|---|---|
| HBM_GPU | 89.9366 | 62.498 | 8 | Llama-3.1-70B LC64K B1 | {"gpu/FEOL": 18} |
| M3D_GPU | 85.6069 | 85.2245 | 18 | Llama-3.1-405B LC20K B8 | {"gpu/FEOL": 18} |
| M3D_NMP_UNIFORM | 81.9274 | 73.2093 | 0 | Llama-3.1-405B LC20K B8 | {"gpu/FEOL": 18} |
| M3D_NMP_CPA | 82.2653 | 72.3851 | 0 | Llama-3.1-405B LC20K B8 | {"gpu/FEOL": 18} |



## Scope and accounting

HBM external storage is sufficiently provisioned, without inventing a physical TB capacity. The frozen persistent-state allocator and optimistic max(local, external) overlap are retained. HBM access energy is 1.9955 pJ/bit for reads+writes; Grace+C2C is 5.3 pJ/bit; C2C is 416.34 GB/s. HBM and Grace static memory power stay zero under the requested system-energy boundary. GPU and all M3D coefficients are unchanged, including the existing 0.1 W/slab active-NMP FEOL budget.

HBM local and external service can overlap. Their aggregate delivered traffic is therefore not bounded by the local HBM bandwidth alone. In 70B/LC64K/B1 this retained optimistic overlap makes HBM nominal throughput exceed M3D-GPU; it is not an HBM-local paired case. The workload GPU power proxy accounts for the delivered traffic. Its thermal RHS has 415.1704 W GPU, 52.2380 W local memory, and 14.7062 W excluded external power, yielding 89.9366 C at GPU FEOL. A bandwidth-characterization cap is not a guarantee that every E2E workload power map stays below 85 C.

Energy terms in energy_closure_audit.csv are disjoint. Memory includes array/peripheral/MIV/local routes and the existing FEOL budget. For GPU-only execution its active memory interface is included in local memory, with all NMP columns zero. For NMP paths, Fabric/NoC include their actual wire and router events; interface includes the shared memory-die interface and edge-port routes over the E2E window. Every component sum closes to E2E energy. Generated tokens are B*G.

User clarification explicitly selected per-die BEOL-uniform memory power, retaining the GPU FEOL source. This is not tile-resolved FEOL temperature. Uniform/CPA use their own actual per-die events. Old physical Decode latencies are never reexecuted: resident plans are restored, deterministic CPA traces are checked, and memory events are integrated over the exact atom birth contexts. All integrated global events and Prefill energies are checked against saved results.

In the thermal table, memory_power_W includes every memory-die component. NMP_power_W is the MAC+SRAM+Fabric+NoC+reduction subset, not an additional source; the existing FEOL budget and shared interface remain included in memory_power_W.

CPA still minimizes the modeled critical path. Faster execution can increase E2E-average power and Tmax even when total joules fall. Temperature and energy were not used to retune placement or select a different execution path.

M3D operator signature must match the existing cached geometry/mesh/material/BC. HBM original cache was absent (including no_nmp_geometry_sensitivity_v2); the user authorized one identical rebuild. No dense factorization, mesh alteration, solver tolerance change, or transient model is introduced. FP64 GPU-PCG/Jacobi, true KCL residual and no full-vector D2H during iteration are retained.

HBM package temperature excludes external Grace and off-package link energy. The full system-energy denominator still includes both. No separate GPU-side C2C thermal mapper exists, so none is invented.



## Execution verdict

CPA faster than GPU: 18/18; GPU faster: 0/18; ties: 0.

Within the formal long-context benchmark, the NMP+CPA path is consistently selected; GPU fallback remains only a short-context boundary mechanism.



## Thermal closure verdict

All nominal rows have finite workload-specific Tmax. Any >85 C row is explicitly THERMAL_CLOSURE_REQUIRED; its throughput/energy remain nominal and must not be labelled thermally feasible. There is no existing f_NMP closure mechanism, and no DVFS model has been added.

All paths thermal-feasible: False. See per-path thermal statuses in canonical data.



## Reuse and new work

12 existing points reused; 24 NMP case checkpoints / 9216 physical Decode steps reused. LC64K adds 6 points, 12 NMP case checkpoints / 6144 Decode steps, plus GPU memory-event observation checkpoints. No partial Decode extrapolation. Raw checkpoints and placement caches remain local.

| model | context_label | B | path | Decode_steps | elapsed_including_placement_s |
|---|---|---|---|---|---|
| Llama-3.1-8B | LC64K | 1 | M3D_GPU | 512 | 39.1 |
| Llama-3.1-8B | LC64K | 1 | M3D_NMP_UNIFORM | 512 | 47.6 |
| Llama-3.1-8B | LC64K | 1 | M3D_NMP_CPA | 512 | 143.2 |
| Llama-3.1-8B | LC64K | 8 | M3D_GPU | 512 | 347.3 |
| Llama-3.1-8B | LC64K | 8 | M3D_NMP_UNIFORM | 512 | 407.5 |
| Llama-3.1-8B | LC64K | 8 | M3D_NMP_CPA | 512 | 1242.2 |
| Llama-3.1-70B | LC64K | 1 | M3D_GPU | 512 | 203.5 |
| Llama-3.1-70B | LC64K | 1 | M3D_NMP_UNIFORM | 512 | 259.3 |
| Llama-3.1-70B | LC64K | 1 | M3D_NMP_CPA | 512 | 431.4 |
| Llama-3.1-70B | LC64K | 8 | M3D_GPU | 512 | 954.2 |
| Llama-3.1-70B | LC64K | 8 | M3D_NMP_UNIFORM | 512 | 2410.4 |
| Llama-3.1-70B | LC64K | 8 | M3D_NMP_CPA | 512 | 2484.5 |
| Llama-3.1-405B | LC64K | 1 | M3D_GPU | 512 | 319.5 |
| Llama-3.1-405B | LC64K | 1 | M3D_NMP_UNIFORM | 512 | 465.5 |
| Llama-3.1-405B | LC64K | 1 | M3D_NMP_CPA | 512 | 808.3 |
| Llama-3.1-405B | LC64K | 8 | M3D_GPU | 512 | 992.4 |
| Llama-3.1-405B | LC64K | 8 | M3D_NMP_UNIFORM | 512 | 1438.6 |
| Llama-3.1-405B | LC64K | 8 | M3D_NMP_CPA | 512 | 3576.4 |



Validation: targeted tests 29 passed (27.97 s); complete pytest 1100 passed (1154.27 s), no failures. Frozen source/config and historical-result integrity gates passed, including B32. Base HEAD: b5f98baa692eacdc0897c094c4d25611a4d38856. Commit subject: Close formal long-context E2E evaluation. Final Git identity is reported with the delivery.

Raw GPU observation checkpoints/logs retain legacy per-stage timing for energy-event observation only. Canonical GPU timing and static energy use the corrected aggregate GPU streaming helper.

Matrix, nominal throughput, and energy-efficiency datasets are complete. Temperature computation is complete (72/72), but 85 C feasibility is not closed: 8 HBM and 18 M3D-GPU rows require thermal closure. No frozen bandwidth or hardware parameter was tuned.
