# E7 Conditional LLM Decode E2E Final Table v0

## Research Question

Can the committed workload, aggregate-capacity, matched-reference performance,
conditional memory-energy, workload-power, and E6 thermal typed results be
assembled into one semantically guarded 3-architecture by 4-rho table without
introducing a new physical model? Yes, within the claim boundaries below.

## Starting Commit

`498a7c63be363681ccc47835a955443cef961e94` on `main`, equal to
`origin/main`, with a clean tracked working tree.

## Files Inspected

- E4, E5, and E6 evaluator typed models and tests
- workload and architecture-capacity typed models
- committed E4/E5/E6 audit evidence
- canonical case capacity inputs through the existing adapter

No thermal solver, remeshing, thermal sweep, or rho=1 reference regression was
run in E7.

## Files Modified

- `src/om3dthermal/evaluator/llm_decode_e2e.py` (new)
- `src/om3dthermal/evaluator/__init__.py`
- `tests/test_llm_decode_e2e.py` (new)
- `docs/audit/LLM_decode_conditional_E2E_final_v0.md` (new)

## Frozen Workload

Identifier: `LLaMA-3.1-8B-class-B1-S131072-v0`. B=1, S=131072,
16-bit weights/KV, runtime bytes=0, and reserved capacity=0. Required capacity
is 33,179,869,184 B. Read traffic is 33,179,869,184 B/token, write traffic is
131,072 B/token, and compute is 83,728,793,600 FLOP/token. These are existing
workload outputs; footprint was not reused to derive traffic in E7.

## Scenario Definition and Performance Results

All rows use 39.2e12 bit/s with status
`MATCHED_REFERENCE_NOT_CAPABILITY_VALIDATED` and 100e12 FLOP/s with status
`NUMERICAL_CHOICE_NOT_HARDWARE_VALIDATED`. Under the existing shared-payload,
roofline-max evaluator, memory time is 0.006771428623673469 s/token-equivalent,
compute time is 0.000837287936 s/token-equivalent, aggregate and per-sequence
throughput are both 147.67932375509625 token/s for B=1, and the bottleneck is
`MEMORY`. These are matched-reference scenario results, not capabilities.

## Capacity Results

| Architecture | Usable B | Usable GiB | Required B | Margin B | Utilization | Feasible | Status |
|---|---:|---:|---:|---:|---:|---|---|
| Conventional HBM | 123211874304 | 114.75 | 33179869184 | 90032005120 | 0.2692911651 | true | AGGREGATE_CAPACITY_FEASIBILITY_ONLY |
| Orthogonal Si | 251557576704 | 234.28125 | 33179869184 | 218377707520 | 0.1318977135 | true | AGGREGATE_CAPACITY_FEASIBILITY_ONLY |
| Orthogonal M3D-IGZO | 460366807040 | 428.75 | 33179869184 | 427186937856 | 0.0720726792 | true | AGGREGATE_CAPACITY_FEASIBILITY_ONLY |

## Conditional Memory Energy, Workload Power, and Thermal Results

The aggregator forwards E4 `memory_dynamic_energy_j_per_token`, E5 workload
power, and committed E6 Tmax; it does not recompute these quantities. E5 old
fixed-bandwidth access power remains regression-only and is not added. GPU power
is the existing fixed 300 W baseline, not a workload GPU-energy model. All
thermal results are converged frozen FP64 GPU-PCG results with source-power
closure below 1e-9 W.

## Final 12-Row Conditional E2E Table

All rows share: B=1; S=131072; required capacity=33,179,869,184 B;
read/write traffic=33,179,869,184/131,072 B/token; aggregate/per-sequence
throughput=147.6793237551/147.6793237551 token/s; memory/compute time=
0.006771428624/0.000837287936 s; bottleneck=`MEMORY`; capacity feasible=true.
`Ewrite` is pJ/bit, `Emem` is conditional memory dynamic J/token, power is W,
and temperature is degC.

| Architecture | rho | Ewrite | Emem | Pmem,dyn | Pmem,total | Pgpu | Ppackage | Memory Tmax | GPU Tmax | Package Tmax | Completeness |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Conventional HBM | 0 | 0 | 0.370844226987 | 54.7660246600 | 55.5832712368 | 300 | 355.5832712368 | 81.505699 | 81.933449 | 81.933449 | RESOLVED_EXISTING_STATIC_COMPONENTS |
| Conventional HBM | 1 | 1.39709798482 | 0.370845691951 | 54.7662410048 | 55.5834875816 | 300 | 355.5834875816 | 81.505736 | 81.933485 | 81.933485 | RESOLVED_EXISTING_STATIC_COMPONENTS |
| Conventional HBM | 100 | 139.709798482 | 0.370990723329 | 54.7876591407 | 55.6049057175 | 300 | 355.6049057175 | 81.509323 | 81.937072 | 81.937072 | RESOLVED_EXISTING_STATIC_COMPONENTS |
| Conventional HBM | 1000 | 1397.09798482 | 0.372309190404 | 54.9823694667 | 55.7996160435 | 300 | 355.7996160435 | 81.541937 | 81.969679 | 81.969679 | RESOLVED_EXISTING_STATIC_COMPONENTS |
| Orthogonal Si | 0 | 0 | 0.363029119781 | 53.6118949126 | 55.2804400069 | 300 | 355.2804400069 | 84.177096 | 84.621099 | 84.621099 | RESOLVED_EXISTING_STATIC_COMPONENTS |
| Orthogonal Si | 1 | 1.36765578312 | 0.363030553872 | 53.6121066982 | 55.2806517925 | 300 | 355.2806517925 | 84.177120 | 84.621122 | 84.621122 | RESOLVED_EXISTING_STATIC_COMPONENTS |
| Orthogonal Si | 100 | 136.765578312 | 0.363172528884 | 53.6330734720 | 55.3016185663 | 300 | 355.3016185663 | 84.179448 | 84.623445 | 84.623445 | RESOLVED_EXISTING_STATIC_COMPONENTS |
| Orthogonal Si | 1000 | 1367.65578312 | 0.364463210811 | 53.8236805062 | 55.4922256005 | 300 | 355.4922256005 | 84.200620 | 84.644562 | 84.644562 | RESOLVED_EXISTING_STATIC_COMPONENTS |
| Orthogonal M3D-IGZO | 0 | 0 | 0.227019472153 | 33.5260821267 | 33.5602321365 | 300 | 333.5602321365 | 81.839920 | 82.290987 | 82.290987 | CONDITIONAL_LOWER_BOUND_UNRESOLVED_LOGIC_BACKGROUND |
| Orthogonal M3D-IGZO | 1 | 0.855260575673 | 0.227020368958 | 33.5262145664 | 33.5603645761 | 300 | 333.5603645761 | 81.839935 | 82.291002 | 82.291002 | CONDITIONAL_LOWER_BOUND_UNRESOLVED_LOGIC_BACKGROUND |
| Orthogonal M3D-IGZO | 100 | 85.5260575673 | 0.227109152724 | 33.5393260929 | 33.5734761026 | 300 | 333.5734761026 | 81.841393 | 82.292456 | 82.292456 | CONDITIONAL_LOWER_BOUND_UNRESOLVED_LOGIC_BACKGROUND |
| Orthogonal M3D-IGZO | 1000 | 855.260575673 | 0.227916277866 | 33.6585217880 | 33.6926717978 | 300 | 333.6926717978 | 81.854644 | 82.305672 | 82.305672 | CONDITIONAL_LOWER_BOUND_UNRESOLVED_LOGIC_BACKGROUND |

Every rho=0 row is `MATHEMATICAL_WRITE_ENERGY_LOWER_BOUND`; zero is a
mathematical sensitivity boundary, not a physical zero-write-energy claim.
Every row retains
`WRITE_SPATIAL_DISTRIBUTION_READ_SHAPE_SENSITIVITY_ONLY`.

## Cross-Stage Consistency Checks

The typed aggregator rejects architecture, workload traffic/FLOPs/model,
capacity, feasibility, batch, rho, performance/energy traffic,
energy/power dynamic energy, performance/power throughput, power/thermal power,
thermal closure, convergence, scenario, completeness, and write-spatial status
mismatches. A capacity-infeasible row must retain blocked performance, energy,
and power outputs and cannot contain thermal/Tmax data. No blocked or unresolved
upstream state is promoted to evaluated status.

## Monotonic and rho-Invariant Checks

For each architecture over rho 0, 1, 100, 1000, write energy, conditional
memory dynamic energy/token, memory dynamic power, memory total power, package
power, and package Tmax are monotonically non-decreasing. Workload identifier,
B, S, required capacity, feasibility, read/write traffic, FLOPs/token, and
matched-reference aggregate tokens/s remain invariant.

## Claim Boundary

```text
BANDWIDTH_CAPABILITY = NOT_VALIDATED
WRITE_ENERGY_MODEL = NOT_VALIDATED
GPU_ENERGY_MODEL = NOT_AVAILABLE
SYSTEM_J_TOKEN = NOT_AVAILABLE
```

For M3D rows additionally:

```text
M3D_LOGIC_BACKGROUND = CONDITIONAL_LOWER_BOUND
CONDITIONAL_LOWER_BOUND_UNRESOLVED_LOGIC_BACKGROUND
```

The table contains `memory_dynamic_energy_j_per_token`; it does not contain or
imply system J/token. It is not a complete architecture ranking.

## Assumptions / Provenance

- Workload, capacity, E4, and E5 values are forwarded from existing typed
  evaluators under their frozen scenario.
- Thermal values are transcribed from committed E6 typed/audit evidence; E7
  does not rerun thermal work.
- Rho is `RHO_SENSITIVITY_NOT_PHYSICAL_CLAIM`.
- Capacity is aggregate feasibility only, not placement feasibility.
- Existing static components are added once; old fixed-bandwidth power remains
  regression-only.

## PASS / FAIL

`E7_CONDITIONAL_E2E_FINAL_TABLE = PASS` requires all specified tests,
cross-stage gates, exact 12-row shape/order, monotonic and invariant checks,
claim guards, clean final tree, and successful commit/push.

## Open Questions

Bandwidth capability, physical architecture write energy, M3D logic-background
power, GPU workload energy, and complete system J/token remain unresolved or
unavailable.

## Next Recommended Step

Research Lead review of the E7 conditional table and its claim boundaries.
Any bandwidth validation, write-energy refinement, system-energy completion,
paper writing, or visualization requires a separately authorized task.
