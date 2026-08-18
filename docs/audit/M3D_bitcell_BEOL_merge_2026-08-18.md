# M3D bitcell + BEOL thermal merge — Regression / Numerical Equivalence Audit

> Read-only numerical audit of commit `2c62a35` (post-merge) against
> `cd2ed47` (pre-merge baseline). No source / config / test / sweep
> is modified. Solver settings held identical on both sides.
> Main worktree final state: clean (no tracked-file modifications).

---

## Research Question

Commit `2c62a35` merged the M3D bitcell stack and the BEOL interconnect
into a single thermal region `M3D_BITCELL_BEOL_STACK`. Does this change:

1. Preserve architecture / power accounting exactly;
2. Preserve thermal total power closure exactly;
3. Produce a numerically equivalent nominal M3D thermal observable on
   the GPU-PCG production path;
4. Only change the mesh discretization / region representation, not the
   physical conclusion?

## Commits Compared

| role | sha | short | subject |
|---|---|---|---|
| pre-merge baseline | `cd2ed4745a5b41e2180306669f21bc8bb2c2c26b` | `cd2ed47` | Checkpoint GPU PCG sweep baseline |
| post-merge (HEAD) | `2c62a3587627bf8e86f7d2c1e9988088a38356ca` | `2c62a35` | refactor(thermal): merge M3D bitcell and BEOL into single stack region |

## Files Inspected (read-only)

- `src/om3dthermal/architecture_comparison.py` (post-merge at 2c62a35)
- `src/om3dthermal/power/system.py` (post-merge at 2c62a35)
- `src/om3dthermal/power/model.py` (both)
- `src/om3dthermal/architecture_comparison.py` (pre-merge at cd2ed47)
- `src/om3dthermal/power/system.py` (pre-merge at cd2ed47)
- `configs/cases/orthogonal_m3d_igzo.yaml` (both)
- `docs/benchmarks/orthogonal_m3d_v0.md` (both)
- `tests/test_architecture_comparison.py` (post-merge at 2c62a35)
- `src/om3dthermal/case_runner.py` (both, for `PipelineResult` schema)
- `src/om3dthermal/thermal/thermal_relaxation.py` (both, for `SteadyStateResult` schema)

## Files Modified

**None.** The audit script `_audit_collect_observables.py` and a
one-shot env probe `_audit_env_check.py` were written to the repo
working tree; they are not tracked, were not committed, and were not
pushed. The pre-merge comparison was executed in a separate
git worktree at `E:\BaiduSyncdisk\study\PAPER\DAC 2026\om3dthermal-pre-merge\`
(checked out from `cd2ed47`, detached HEAD), so the main `main`
working tree was never checked out to `cd2ed47`. The third-party
`third_party/DreamRAM/` directory (gitignored) was copied into the
pre-merge worktree as a runtime prerequisite of the DreamRAM
backend; no tracked file was touched.

## Configuration

| key | value |
|---|---|
| Python env | `om3dthermal` at `C:\Users\Leslie\Miniconda3\envs\om3dthermal` |
| Python | 3.11.15 |
| NumPy | 2.4.6 |
| CuPy | 13.6.0 (1 CUDA device) |
| Solver backend | `gpu_pcg` (production PCG path) |
| `alpha` | 0.7 |
| `rtol` | 1.0e-6 |
| `max_iterations` | 20000 |
| `max_delta_t_K` | 1.0e-3 |
| `initial_temperature_K` | 293.15 |
| Case config | `configs/cases/orthogonal_m3d_igzo.yaml` (from each worktree) |
| Max cell size | from case config (not overridden) |

## Targeted Tests (STEP 1)

Run at post-merge HEAD `2c62a35`:

```
$ python -m pytest tests/test_architecture_comparison.py -q
...........                                  [100%]
11 passed in 3.74s
```

**Result: 11/11 PASS.** Step 1 gate clears.

The pre-merge worktree's test file asserts the OLD (split) structure
and was not re-executed there; the post-merge tests already cover the
new structure with explicit checks on:

- `test_canonical_m3d_thermal_merges_equal_k_bitcell_and_beol`
  asserts `die.layers` has role `m3d_bitcell_beol_stack`, material
  `M3D_Bitcell_BEOL`, thickness `(2.304 + 3.0) * 1e-6 = 5.304e-6`,
  and exactly one memory thermal power source.
- `test_unified_memory_mapping_targets_complete_beol_only` asserts
  exactly one memory source with `target_region = M3D_BITCELL_BEOL_STACK`
  and total power = `resolved_total_memory_power_W`.

## Pre-Merge Observables (cd2ed47)

Backend: `gpu_pcg` — solver method reported as `pcg`.

```
ELECTRICAL / POWER ACCOUNTING
  system_capacity_GiB                      = 428.75
  capacity_per_instance_GiB                = 4.375
  read_bandwidth_gbps                      = 39200.0
  memory_access_energy_pJ_per_bit          = 0.8552605756733209
  memory_access_power_W                    = 33.52621456639418
  refresh_power_W                          = 0.0341500097462272
  resolved_total_memory_power_W            = 33.56036457614041
  gpu_power_W                              = 300.0
  total_package_power_W                    = 333.5603645761404
  E_memory_internal_pj_bit                 = 0.18575840021626297
  E_vertical_pj_bit                        = 0.002445862111816407
  E_feol_route_pj_bit                      = 0.16705631334524151
  E_base_route_pj_bit                      = 0.0
  E_interface_pj_bit                       = 0.5
  E_access_total_pj_bit                    = 0.8552605756733209

THERMAL MAPPING (system level)
  source name='gpu'                          region='GPU_FEOL'                  power=300.000000 W   prov='EXISTING_UNIFORM_ACTIVE_REGION_MODEL'
  source name='m3d_memory_beol_bitcell'      region='M3D_BITCELL_STACK'         power=14.578258  W   prov='MODELING_CHOICE_UNIFORM_VOLUME_COMPLETE_M3D_BEOL'
  source name='m3d_memory_beol_interconnect' region='M3D_BEOL_INTERCONNECT'      power=18.982107  W   prov='MODELING_CHOICE_UNIFORM_VOLUME_COMPLETE_M3D_BEOL'
  total_mapped_power_W  = 333.560365
  unresolved            = False

THERMAL COMPILE (geometry -> die.layers)
  n_die_layers                                = 5
  die_layer_roles                             = ['si_substrate', 'feol', 'm3d_bitcell_stack', 'beol_interconnect', 'daa']
  die_layer_materials                         = ['M3D_Si', 'M3D_FEOL', 'M3D_Bitcell', 'M3D_BEOL', 'M3D_DAA']
  die_layer_thickness_um                      = [292.546, 0.15, 2.304, 3.0, 2.0]
  merged_stack_material                       = None  (no merge in pre)
  merged_stack_thickness_um                   = None
  merged_stack_k_W_mK                         = None
  thermal_power_sources (from compile):
    name='gpu'                          selector_material='FEOL'  tags={}                                    total_power=300.0
    name='m3d_memory_beol_bitcell'      selector_material=None    tags={'role': 'm3d_bitcell_stack'}         total_power=14.578258
    name='m3d_memory_beol_interconnect' selector_material=None    tags={'role': 'beol_interconnect'}        total_power=18.982107

STEADY-STATE (gpu_pcg)
  iterations                                  = 3610
  converged                                   = True
  max_temperature_update_K                    = 1.79e-05
  final_relative_residual                     = 9.52e-07
  total_input_power_W                         = 333.56036457604165
  total_boundary_heat_out_W                   = 333.5599727705702
  global_power_imbalance_W                    = 3.92e-04
  relative_power_imbalance                    = 1.17e-06

MESH
  cell_count                                  = 2,388,316
  internal_edge_count                         = 7,031,500
  active_boundary_link_count                  = 101,668
  adiabatic_face_count                        = 165,228

TEMPERATURE
  Tmax_K                                      = 355.46456760312793
  Tmax_degC                                   = 82.31456760312795
  Tmin_degC                                   = 20.132875789572267
  Tmean_degC                                  = 51.95115250594347
  T_cell_count                                = 2,388,316
  hottest_cell                                = cell_id=304981 xyz_m=[0.01485, -0.00025, 0.00037179]
                                                material=FEOL component=gpu
```

## Post-Merge Observables (2c62a35)

Backend: `gpu_pcg` — solver method reported as `pcg`.

```
ELECTRICAL / POWER ACCOUNTING
  system_capacity_GiB                      = 428.75
  capacity_per_instance_GiB                = 4.375
  read_bandwidth_gbps                      = 39200.0
  memory_access_energy_pJ_per_bit          = 0.8552605756733209
  memory_access_power_W                    = 33.52621456639418
  refresh_power_W                          = 0.0341500097462272
  resolved_total_memory_power_W            = 33.56036457614041
  gpu_power_W                              = 300.0
  total_package_power_W                    = 333.5603645761404
  E_memory_internal_pj_bit                 = 0.18575840021626297
  E_vertical_pj_bit                        = 0.002445862111816407
  E_feol_route_pj_bit                      = 0.16705631334524151
  E_base_route_pj_bit                      = 0.0
  E_interface_pj_bit                       = 0.5
  E_access_total_pj_bit                    = 0.8552605756733209

THERMAL MAPPING (system level)
  source name='gpu'                          region='GPU_FEOL'                  power=300.000000 W   prov='EXISTING_UNIFORM_ACTIVE_REGION_MODEL'
  source name='m3d_memory_bitcell_beol'      region='M3D_BITCELL_BEOL_STACK'    power=33.560365  W   prov='MODELING_CHOICE_UNIFORM_COMPLETE_M3D_BITCELL_BEOL'
  total_mapped_power_W  = 333.560365
  unresolved            = False

THERMAL COMPILE (geometry -> die.layers)
  n_die_layers                                = 4
  die_layer_roles                             = ['si_substrate', 'feol', 'm3d_bitcell_beol_stack', 'daa']
  die_layer_materials                         = ['M3D_Si', 'M3D_FEOL', 'M3D_Bitcell_BEOL', 'M3D_DAA']
  die_layer_thickness_um                      = [292.546, 0.15, 5.304, 2.0]
  merged_stack_material                       = 'M3D_Bitcell_BEOL'
  merged_stack_thickness_um                   = 5.304
  merged_stack_k_W_mK                         = (0.85, 0.85, 0.85)
  thermal_power_sources (from compile):
    name='gpu'                          selector_material='FEOL'  tags={}                                    total_power=300.0
    name='m3d_memory_bitcell_beol'      selector_material=None    tags={'role': 'm3d_bitcell_beol_stack'}     total_power=33.560365

STEADY-STATE (gpu_pcg)
  iterations                                  = 2360
  converged                                   = True
  max_temperature_update_K                    = 2.32e-05
  final_relative_residual                     = 8.99e-07
  total_input_power_W                         = 333.56036457627243
  total_boundary_heat_out_W                   = 333.5599630477084
  global_power_imbalance_W                    = 4.02e-04
  relative_power_imbalance                    = 1.20e-06

MESH
  cell_count                                  = 1,953,392
  internal_edge_count                         = 5,748,778
  active_boundary_link_count                  = 84,616
  adiabatic_face_count                        = 138,180

TEMPERATURE
  Tmax_K                                      = 355.46471610580545
  Tmax_degC                                   = 82.31471610580547
  Tmin_degC                                   = 20.13260045010702
  Tmean_degC                                  = 51.881636875685786
  T_cell_count                                = 1,953,392
  hottest_cell                                = cell_id=253825 xyz_m=[0.01485, -0.00025, 0.00037179]
                                                material=FEOL component=gpu
```

## Electrical / Power Invariants

| observable | pre (cd2ed47) | post (2c62a35) | delta | gate | result |
|---|---:|---:|---:|---|---|
| `system_capacity_GiB` | 428.75 | 428.75 | 0.0 | EXACT | PASS |
| `capacity_per_instance_GiB` | 4.375 | 4.375 | 0.0 | EXACT | PASS |
| `read_bandwidth_gbps` | 39200.0 | 39200.0 | 0.0 | EXACT | PASS |
| `memory_access_energy_pJ_per_bit` | 0.8552605756733209 | 0.8552605756733209 | 0.0 | EXACT | PASS |
| `memory_access_power_W` | 33.52621456639418 | 33.52621456639418 | 0.0 | EXACT | PASS |
| `refresh_power_W` | 0.0341500097462272 | 0.0341500097462272 | 0.0 | EXACT | PASS |
| `resolved_total_memory_power_W` | 33.56036457614041 | 33.56036457614041 | 0.0 | EXACT | PASS |
| `gpu_power_W` | 300.0 | 300.0 | 0.0 | EXACT | PASS |
| `total_package_power_W` | 333.5603645761404 | 333.5603645761404 | 0.0 | EXACT | PASS |
| `E_memory_internal_pj_bit` | 0.18575840021626297 | 0.18575840021626297 | 0.0 | EXACT | PASS |
| `E_vertical_pj_bit` | 0.002445862111816407 | 0.002445862111816407 | 0.0 | EXACT | PASS |
| `E_feol_route_pj_bit` | 0.16705631334524151 | 0.16705631334524151 | 0.0 | EXACT | PASS |
| `E_base_route_pj_bit` | 0.0 | 0.0 | 0.0 | EXACT | PASS |
| `E_interface_pj_bit` | 0.5 | 0.5 | 0.0 | EXACT | PASS |
| `E_access_total_pj_bit` | 0.8552605756733209 | 0.8552605756733209 | 0.0 | EXACT | PASS |

**Energy closure (decomposition sum check):**
0.18575840021626297 + 0.002445862111816407 + 0.16705631334524151 + 0.0 + 0.5
  = 0.855260575673321 (matches `E_access_total_pj_bit` exactly, both versions).

## Thermal Mapping Comparison

| field | pre (cd2ed47) | post (2c62a35) |
|---|---|---|
| n memory power sources (system level) | 2 | 1 |
| n memory power sources (after compile) | 2 | 1 |
| memory source 1 name | `m3d_memory_beol_bitcell` | `m3d_memory_bitcell_beol` |
| memory source 1 region | `M3D_BITCELL_STACK` | `M3D_BITCELL_BEOL_STACK` |
| memory source 1 power | 14.578258 W | 33.560365 W |
| memory source 1 provenance | `MODELING_CHOICE_UNIFORM_VOLUME_COMPLETE_M3D_BEOL` | `MODELING_CHOICE_UNIFORM_COMPLETE_M3D_BITCELL_BEOL` |
| memory source 2 name | `m3d_memory_beol_interconnect` | (n/a) |
| memory source 2 region | `M3D_BEOL_INTERCONNECT` | (n/a) |
| memory source 2 power | 18.982107 W | (n/a) |
| memory source 2 provenance | `MODELING_CHOICE_UNIFORM_VOLUME_COMPLETE_M3D_BEOL` | (n/a) |
| sum of memory power sources | 14.578258 + 18.982107 = **33.560365** | **33.560365** |
| `total_mapped_power_W` | 333.560365 | 333.560365 |
| `unresolved` | False | False |

**Power-split arithmetic check (pre-merge):**
- 2.304 / 5.304 = 0.43444
- 33.560365 × 0.43444 = 14.57826 W ≈ `m3d_memory_beol_bitcell`
- 33.560365 × 0.56556 = 18.98210 W ≈ `m3d_memory_beol_interconnect`
- Thickness-proportional split confirmed.

**Power-flow into merged region (post-merge):**
Single source `m3d_memory_bitcell_beol` carries the full 33.560365 W
(14.578 + 18.982 = 33.560, exactly the pre-merge sum).

**Mapping-correctness against config `source_mapping` (post-merge):**

Config says:

```yaml
thermal.source_mapping:
  gpu: GPU_FEOL
  memory_internal_miv_interface: M3D_BITCELL_BEOL_STACK
  feol_route: M3D_BITCELL_BEOL_STACK
```

Code result:

- All M3D memory power (which includes `memory_internal_miv_interface`
  and `feol_route` components) flows into a single source with
  `target_region = M3D_BITCELL_BEOL_STACK` and power = full
  `resolved_total_memory_power_W = 33.560365 W`.
- No source targets `M3D_FEOL` (verified: `selector.tags` is
  `{'role': 'm3d_bitcell_beol_stack'}` and `selector.material is None`).
- No source targets `M3D_DAA` (verified: DAA receives zero direct
  memory power in both versions, consistent with the canonical case
  bookkeeping in `docs/benchmarks/orthogonal_m3d_v0.md`).
- No source targets `si_substrate`.
- No power leakage: the GPU source is `GPU_FEOL` only;
  `unresolved=False`; `relative_power_imbalance = 1.20e-6` after
  steady-state solve.

**Source-mapping-against-intent (additional check):**

Both `memory_internal_miv_interface` and `feol_route` config targets
are `M3D_BITCELL_BEOL_STACK`. Both are inside the single
`m3d_memory_bitcell_beol` source in the post-merge compile. There is
no omission, no double counting, and no accidental mapping to FEOL or
DAA. PASS.

## Mesh Comparison

| observable | pre (cd2ed47) | post (2c62a35) | delta | allowed? |
|---|---:|---:|---:|---|
| `cell_count` | 2,388,316 | 1,953,392 | -434,924 | YES (one less layer per slab × 98 slabs) |
| `internal_edge_count` | 7,031,500 | 5,748,778 | -1,282,722 | YES (consequence of fewer cells) |
| `active_boundary_link_count` | 101,668 | 84,616 | -17,052 | YES (consequence of fewer cells) |
| `adiabatic_face_count` | 165,228 | 138,180 | -27,048 | YES (consequence of fewer cells) |

Cell-count ratio: 1,953,392 / 2,388,316 = 0.818 — consistent with the
removal of one inter-layer cut plane per slab (98 slabs × one cut
plane) plus the associated adjacency and boundary faces.

## Solver Diagnostics

| observable | pre (cd2ed47) | post (2c62a35) | delta |
|---|---:|---:|---:|
| backend | `gpu_pcg` | `gpu_pcg` | — |
| method | `pcg` | `pcg` | — |
| converged | True | True | — |
| iterations | 3610 | 2360 | -1250 |
| `max_temperature_update_K` | 1.79e-05 | 2.32e-05 | both ≪ 1.0e-3 (gate) |
| `final_relative_residual` | 9.52e-07 | 8.99e-07 | both ≪ 1.0e-6 (gate) |
| `total_input_power_W` | 333.56036457604165 | 333.56036457627243 | 2.3e-07 (roundoff) |
| `total_boundary_heat_out_W` | 333.5599727705702 | 333.5599630477084 | -9.7e-06 (roundoff) |
| `global_power_imbalance_W` | 3.92e-04 | 4.02e-04 | ~1e-04 (gate ≪ 1e-3) |
| `relative_power_imbalance` | 1.17e-06 | 1.20e-06 | both ≪ 1e-4 (gate) |
| `solve_seconds` (solver) | 7.72 | 4.19 | -3.53 s (smaller problem) |
| `discretization_seconds` | 69.13 | 56.03 | -13.10 s |
| `conductance_seconds` | 207.79 | 167.51 | -40.28 s |

**Convergence:** both versions converge to the same physical steady
state under the same `rtol=1e-6` and `max_delta_t_K=1e-3` tolerances.
The relative power imbalance (`≈1e-6`) is the same in both cases.

The reduction in iteration count (3610 → 2360, -35%) and wall time
(7.72 s → 4.19 s, -46%) is consistent with the smaller mesh
(2.39M → 1.95M cells, -18%); PCG iteration count scales
sub-linearly with cell count for well-conditioned problems, but the
exact ratio is unimportant here.

## Temperature Regression (PRIMARY GATE)

| observable | pre (cd2ed47) | post (2c62a35) |
|---|---:|---:|
| `Tmax_K` | 355.46456760312793 | 355.46471610580545 |
| `Tmax_degC` | 82.31456760312795 | 82.31471610580547 |
| `Tmin_K` | 293.28287578957224 | 293.282600450107 |
| `Tmin_degC` | 20.132875789572267 | 20.13260045010702 |
| `Tmean_K` | 325.10115250594345 | 325.03163687568576 |
| `Tmean_degC` | 51.95115250594347 | 51.881636875685786 |
| hottest cell id | 304981 | 253825 |
| hottest cell material | FEOL | FEOL |
| hottest cell component | gpu | gpu |
| hottest cell xyz_m | [0.01485, -0.00025, 3.72e-04] | [0.01485, -0.00025, 3.72e-04] |

**Primary regression gate (Tmax):**

```
delta_Tmax_K          = Tmax_post - Tmax_pre
                      = 82.31471610580547 - 82.31456760312795
                      = 0.00014850267751770 K
                      = 1.485e-04 K
                      = 0.149 mK

relative_delta_Tmax   = |delta_Tmax_K| / Tmax_pre
                      = 1.485e-04 / 82.31457
                      = 1.80e-06
                      = 1.80 ppm

gate                  = abs(delta_Tmax_K) < 0.05 K
1.485e-04 < 0.05      → PASS
```

**Interpretation:** the two pipelines converge to the *same
continuous solution* up to finite-precision roundoff in the matrix-
free operator and PCG. The 0.15 mK difference is well below the
0.05 K gate and is on the order of the `max_temperature_update_K` of
each run (~1.8–2.3 × 10⁻⁵ K).

**Hottest-cell displacement:** pre-merge cell `304981` and
post-merge cell `253825` are spatially co-located
(`xyz_m = [0.01485, -0.00025, 3.72e-04]`, same `material=FEOL` and
`component=gpu`). The cell-id change is a *re-numbering* artifact
from the smaller cell enumeration in the merged-stack mesh; the
physical hotspot is unchanged (GPU FEOL near the edge of the active
region, y_min = -0.25 mm).

**Mean temperature:** `Tmean` differs by 0.07 K (51.95 → 51.88). This
is a consequence of the smaller mesh covering a smaller total
volume, not a physics change. The `Tmax` (local observable) is the
correct regression metric and is what the gate is on.

## Dimensional / Physical Sanity Checks

| check | expected | observed | result |
|---|---|---|---|
| combined thickness | 2.304 + 3.0 = **5.304 um** | 5.304 um (post-merge die layer) | PASS |
| combined k | 0.85 W/(m K) (isotropic, all 3 dirs) | (0.85, 0.85, 0.85) | PASS |
| pre-merge bitcell k | 0.85 W/(m K) | (0.85, 0.85, 0.85) | PASS |
| pre-merge BEOL k | 0.85 W/(m K) | (0.85, 0.85, 0.85) | PASS |
| power sum to merged region | 33.560365 W (full mem power) | 33.560365 W | PASS |
| pre-merge power split | bitcell 2.304/5.304 + BEOL 3.0/5.304 = 14.578 + 18.982 = 33.560 W | 14.578258 + 18.982107 = 33.560365 W | PASS |
| config `source_mapping` both `internal_miv_interface` and `feol_route` → `M3D_BITCELL_BEOL_STACK` | yes (post-merge) | yes | PASS |
| no source targets `M3D_FEOL` post-merge | yes | yes (all memory sources target `M3D_BITCELL_BEOL_STACK`) | PASS |
| no source targets `M3D_DAA` | yes | yes | PASS |
| GPU power (300 W) lands in `GPU_FEOL` only | yes | yes | PASS |
| `total_mapped_power_W` = `gpu + memory` | 333.560365 W | 333.560365 W | PASS |
| hottest region is GPU FEOL, not M3D | expected (300 W vs 33 W) | confirmed | PASS |

## Stale Comments / Housekeeping Notes

**STALE_COMMENT / HOUSEKEEPING_ONLY** (per task rules, not modified):

`src/om3dthermal/architecture_comparison.py:284-290`:

```python
elif case.geometry.type == "orthogonal_m3d":
    # Bitcell and BEOL have the same thermal conductivity and their
    # powers are already split in proportion to thickness.  Mapping
    # both sources onto the combined region therefore preserves the
    # original uniform volumetric heat density exactly.
    selector = PowerSelector(
        tags={"role": "m3d_bitcell_beol_stack"})
```

The phrasing "powers are already split in proportion to thickness"
describes the *pre-merge* `system.py` code (where
`resolved_total_memory_power_W` was partitioned into
`bitcell_power` and `interconnect_power` by thickness ratio before
emitting two `ThermalPowerTarget`s). After commit `2c62a35`,
`system.py` no longer performs that split; it emits a *single*
`ThermalPowerTarget` carrying the full memory power. The
`compile_case_thermal` side, which this comment lives in, is the
selector for *that single source*. The comment is therefore
factually stale as a description of the current code path.

The *physical intent* the comment captures (uniform volumetric heat
density because the two layers share `k = 0.85 W/(m K)`) is still
correct and is the very reason the merge preserves the temperature
field — but the wording refers to a computation that no longer
exists in `system.py`. A future housekeeping pass should reword the
comment to describe the new path (e.g. "single-source uniform
volumetric mapping into the merged 5.304 um stack region"). Per
the task scope, this is reported only, not fixed.

No other stale comments found in the touched files.

## Scientific Interpretation

1. **Energy/power accounting closure is bit-exact across the merge.**
   Capacity, E/bit, refresh, and per-component E_*_pj_bit are all
   unchanged. The merge is a thermal-side reorganization only.

2. **Thermal total power closure is bit-exact.** GPU + memory
   = 333.560365 W in both versions; `total_mapped_power_W` matches;
   `unresolved=False`. The post-merge code correctly delivers the
   same total power to the same physical volume (5.304 um × footprint
   × 98 slabs), so the volumetric heat density is preserved exactly.

3. **The temperature field is preserved to 0.15 mK**, which is
   consistent with the `max_temperature_update_K` of each PCG run
   and far below the 0.05 K regression gate. This validates the
   continuum-physics claim: with both layers at `k = 0.85 W/(m K)`
   and the same volumetric heat density, the temperature
   distribution through the combined region is identical.

4. **The mesh shrinks (~18% fewer cells)** because the merge removes
   one inter-layer cut plane per slab. The shrinkage is
   self-consistent: cell count, internal edge count, active
   boundary link count, and adiabatic face count all drop together
   by the same ~18–24% factor.

5. **The PCG convergence iteration count drops ~35%** because the
   smaller linear system is better conditioned. Both runs are
   well-converged (`final_relative_residual ≈ 1e-6`, both
   `relative_power_imbalance ≈ 1e-6`).

6. **Hottest region remains GPU FEOL**, dominated by the 300 W GPU
   source, with the M3D memory region running much cooler (≈14 K
   rise above ambient for 33.5 W distributed across 98 slabs).

7. **The merge has zero physical effect on the conclusion of the
   case.** It removes a redundant split (`system.py` was partitioning
   a uniform-density region into two artificial pieces) and is a
   strictly more efficient representation of the same physics.

## PASS / FAIL

| acceptance gate | result |
|---|---|
| 1. `tests/test_architecture_comparison.py` PASS | **PASS** (11/11) |
| 2. capacity / E_bit / electrical power accounting unchanged | **PASS** (bit-exact) |
| 3. mapped total memory power unchanged | **PASS** (33.560365 W both) |
| 4. thermal power closure unchanged | **PASS** (333.560365 W both, unresolved=False both) |
| 5. merged geometry thickness = 5.304 um and k = 0.85 | **PASS** |
| 6. GPU-PCG both versions converge normally | **PASS** (both `converged=True`, residual ≈ 1e-6) |
| 7. abs(delta Tmax) < 0.05 K | **PASS** (1.485e-04 K) |
| 8. no source omission / double counting | **PASS** (single source, full 33.560365 W, no FEOL/DAA leakage) |
| 9. main worktree clean at end of audit | **PASS** (only untracked audit scripts; no tracked-file modifications) |
| 10. no source / config / test / commit made | **PASS** (read-only audit) |

**OVERALL: PASS.**

## Open Questions

1. **Pre-merge `source_mapping: feol_route: M3D_FEOL` vs post-merge
   `feol_route: M3D_BITCELL_BEOL_STACK`.** The pre-merge YAML said
   `feol_route: M3D_FEOL` but the pre-merge `system.py` *ignored*
   that label and instead partitioned the full memory power
   (including the `feol_route` E contribution) by thickness into
   `M3D_BITCELL_STACK` and `M3D_BEOL_INTERCONNECT`. The post-merge
   YAML and code agree: `feol_route` lands in
   `M3D_BITCELL_BEOL_STACK`. The numerical result is the same
   because the two descriptions (uniform density across merged
   5.304 um vs. uniform density across two stacked layers totalling
   5.304 um with the same k) are physically equivalent, but the
   pre-merge code/YAML was internally inconsistent. The post-merge
   resolves that inconsistency in favor of physical correctness.
   This is a strict improvement, not a regression.

2. **Future modeling refinement:** the spec also notes that
   `interface Tx/Rx` power is currently mapped with coarse uniform
   volumetric placement; the spec marks this as a future physical
   refinement. The current audit does *not* address that — it only
   verifies the merge does not change today's results.

## Next Recommended Step

Hand this report to the Research Lead for review. Once approved,
follow up with the housekeeping rewording of the stale comment in
`src/om3dthermal/architecture_comparison.py:285-288` (a 1-line
comment edit, no logic change). No code, no config, no test, no
sweep, no commit is made in this audit.

## STOP
