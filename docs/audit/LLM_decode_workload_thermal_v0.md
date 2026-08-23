# E6 Workload-Dependent Decode Thermal Integration Audit

## Research Question

Can the frozen E5 `LLMDecodeWorkloadPowerMetrics` be mapped without residual
fixed-bandwidth power into the existing thermal source selectors, and can the
result be solved with the frozen GPU-PCG path to report memory, GPU, and package
Tmax? Yes, for the conditional sensitivity scenario documented below.

## Starting Commit

`40df90e85a8fe76157de29a22f674c5677c0653f` on `main`, equal to
`origin/main`, with a clean tracked working tree.

## Files Modified

- `src/om3dthermal/evaluator/llm_decode_workload_thermal.py` (new)
- `src/om3dthermal/evaluator/__init__.py`
- `tests/test_llm_decode_workload_thermal.py` (new)
- `docs/audit/LLM_decode_workload_thermal_v0.md` (new)

No thermal, power, workload, configuration, or expected-value file was changed.

## Frozen Scenario and Provenance

The workload is B=1, S=131072, 16-bit weights/KV, runtime bytes=0, and
reserved capacity=0. The E4 performance inputs are 39.2e12 bit/s matched
reference bandwidth and 100e12 FLOP/s numerical compute. The bandwidth is not
a validated capability. `rho` in {0, 1, 100, 1000} is a write/read energy
sensitivity parameter, not a physical write-energy claim. The existing case
compiler supplies geometry, mesh, boundaries, and selectors; E6 replaces every
power-source value.

## Mapping Equations

GPU power is mapped exactly once to the existing GPU source. For HBM,

`E_dram = E_memory_internal + E_vertical + E_feol_route + E_interface`

`E_base = E_base_route`

`P_dynamic,dram = P_dynamic * E_dram / (E_dram + E_base)` and
`P_dynamic,base = P_dynamic * E_base / (E_dram + E_base)`.

Refresh and memory background are added to the DRAM share; effective logic
background is added to the base share; both are divided equally over visible
groups. Orthogonal Si maps the complete memory workload total to its existing
single memory selector. M3D maps the complete conditional memory lower bound to
the existing merged bitcell/BEOL selector and is never spatially split.

All write-dependent power uses
`WRITE_SPATIAL_DISTRIBUTION_READ_SHAPE_SENSITIVITY_ONLY`.

## Source Breakdown (W)

| Architecture | rho | GPU | Memory thermal sources |
|---|---:|---:|---|
| Conventional HBM | 0 | 300 | dram0=25.635798177706; base0=2.155837440692; dram1=25.635798177706; base1=2.155837440692 |
| Conventional HBM | 1 | 300 | dram0=25.635897833803; base0=2.155845956998; dram1=25.635897833803; base1=2.155845956998 |
| Conventional HBM | 100 | 300 | dram0=25.645763787422; base0=2.156689071310; dram1=25.645763787422; base1=2.156689071310 |
| Conventional HBM | 1000 | 300 | dram0=25.735454274864; base0=2.164353746873; dram1=25.735454274864; base1=2.164353746873 |
| Orthogonal Si | 0 | 300 | memory=55.280440006936 |
| Orthogonal Si | 1 | 300 | memory=55.280651792530 |
| Orthogonal Si | 100 | 300 | memory=55.301618566294 |
| Orthogonal Si | 1000 | 300 | memory=55.492225600515 |
| Orthogonal M3D-IGZO | 0 | 300 | merged memory=33.560232136479 |
| Orthogonal M3D-IGZO | 1 | 300 | merged memory=33.560364576140 |
| Orthogonal M3D-IGZO | 100 | 300 | merged memory=33.573476102609 |
| Orthogonal M3D-IGZO | 1000 | 300 | merged memory=33.692671797773 |

## Power Closure and 12-Scenario Thermal Table

Temperatures are degC and powers are W. `Pdyn` is memory dynamic access power.

| Architecture | rho | Pdyn | Memory total | Package power | Closure abs. | Memory Tmax | GPU Tmax | Package Tmax | Iterations |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Conventional HBM | 0 | 54.7660246600 | 55.5832712368 | 355.5832712368 | 4.15e-12 | 81.505699 | 81.933449 | 81.933449 | 930 |
| Conventional HBM | 1 | 54.7662410048 | 55.5834875816 | 355.5834875816 | 4.09e-12 | 81.505736 | 81.933485 | 81.933485 | 930 |
| Conventional HBM | 100 | 54.7876591407 | 55.6049057175 | 355.6049057175 | 4.15e-12 | 81.509323 | 81.937072 | 81.937072 | 930 |
| Conventional HBM | 1000 | 54.9823694667 | 55.7996160435 | 355.7996160435 | 4.21e-12 | 81.541937 | 81.969679 | 81.969679 | 930 |
| Orthogonal Si | 0 | 53.6118949126 | 55.2804400069 | 355.2804400069 | 7.46e-11 | 84.177096 | 84.621099 | 84.621099 | 820 |
| Orthogonal Si | 1 | 53.6121066982 | 55.2806517925 | 355.2806517925 | 7.46e-11 | 84.177120 | 84.621122 | 84.621122 | 820 |
| Orthogonal Si | 100 | 53.6330734720 | 55.3016185663 | 355.3016185663 | 7.46e-11 | 84.179448 | 84.623445 | 84.623445 | 820 |
| Orthogonal Si | 1000 | 53.8236805062 | 55.4922256005 | 355.4922256005 | 7.48e-11 | 84.200620 | 84.644562 | 84.644562 | 820 |
| Orthogonal M3D-IGZO | 0 | 33.5260821267 | 33.5602321365 | 333.5602321365 | 1.32e-10 | 81.839920 | 82.290987 | 82.290987 | 1090 |
| Orthogonal M3D-IGZO | 1 | 33.5262145664 | 33.5603645761 | 333.5603645761 | 1.32e-10 | 81.839935 | 82.291002 | 82.291002 | 1090 |
| Orthogonal M3D-IGZO | 100 | 33.5393260929 | 33.5734761026 | 333.5734761026 | 1.32e-10 | 81.841393 | 82.292456 | 82.292456 | 1090 |
| Orthogonal M3D-IGZO | 1000 | 33.6585217880 | 33.6926717978 | 333.6926717978 | 1.33e-10 | 81.854644 | 82.305672 | 82.305672 | 1090 |

All closure errors are below 1e-9 W.

## GPU-PCG Diagnostics

All 12 workload solves and all three old-reference solves converged. Every solve
used backend `gpu_pcg`, FP64, matrix-free operation, Jacobi diagonal
preconditioning, 293.15 K fresh initialization, relative residual tolerance
1e-3, maximum temperature-update tolerance 1e-2 K, maximum 100000 iterations,
and check interval 10. Full-vector D2H transfers during iteration were zero.
Across workload solves, final relative residual was 4.59e-5 to 3.76e-4,
maximum temperature update was 0.00680 to 0.00924 K, and relative power
imbalance was 7.29e-6 to 1.62e-4.

## rho=1 Regression

| Architecture | Cells | Edges | Mapped power delta (W) | Max full-field delta (K) | Memory Tmax delta (K) | GPU Tmax delta (K) | Package Tmax delta (K) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Conventional HBM | 859596 | 2531340 | 0 | 1.65e-12 | 3.41e-13 | 5.68e-14 | 5.68e-14 |
| Orthogonal Si | 1518468 | 4466056 | 0 | 0 | 0 | 0 | 0 |
| Orthogonal M3D-IGZO | 1953392 | 5748778 | 5.68e-14 | 2.10e-12 | 7.96e-13 | 3.98e-13 | 3.98e-13 |

Cell and edge counts were identical on both sides. All mapped-power deltas are
below 1e-9 W and package Tmax deltas are below 0.05 K.

## Monotonic Checks

For every architecture over rho 0, 1, 100, 1000, memory dynamic power, memory
total power, package power, and package Tmax are monotonically non-decreasing.

## Validation

The seven required test files passed together: 140 tests passed. The run also
completed 12 workload GPU-PCG solves and three old-reference GPU-PCG solves.
One long-lived validation process retained host memory after a `MemoryError`
during repeated remeshing; terminating only the two task-owned Python processes
and rerunning the remaining solves in bounded processes resolved the operational
issue without code, solver, tolerance, or data changes.

## Scientific Interpretation

E6 validates the software integration and conditional thermal sensitivity for
the frozen matched-reference scenario. It does not validate the 39.2 Tb/s
bandwidth as a capability, `rho` as a physical write-energy model, GPU energy,
system J/token, or an architecture ranking. The M3D values remain conditional
lower bounds because logic background is unresolved.

## Status and Open Questions

- `E6_WORKLOAD_THERMAL = PASS`
- `BANDWIDTH_CAPABILITY = NOT_VALIDATED`
- `WRITE_ENERGY_MODEL = NOT_VALIDATED`
- `M3D_LOGIC_BACKGROUND = CONDITIONAL_LOWER_BOUND`
- `GPU_ENERGY_MODEL = NOT_AVAILABLE`
- `SYSTEM_J_TOKEN = NOT_AVAILABLE`
- `E7_FINAL_TABLE = NOT_STARTED`

Open evidence questions are the architecture write-energy values, bandwidth
capability, M3D logic background, and GPU energy. The next recommended step is
Research Lead review of this E6 integration and its conditional claim boundary.
Do not enter E7 without a separate task.
