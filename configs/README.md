# Configuration map

## Main research entry point

The current primary study is workload-aware LLM decode inference centered on
the Orthogonal M3D-IGZO proposed architecture:

```powershell
python -m om3dthermal experiment configs\experiment\m3d_igzo_llama31_8b_decode_conditional_v0.yaml
```

The experiment compares exactly three active architectures:

1. `orthogonal_m3d_igzo` — proposed design and research focus;
2. `conventional_hbm_2x1` — main comparison baseline;
3. `orthogonal_si` — mechanism/technology ablation.

The table output remains ordered Conventional HBM, Orthogonal Si, then M3D
for compatibility with the frozen E7 audit. Ordering does not imply research
priority.

## Directory ownership

| Directory | Responsibility | User-facing? |
|---|---|---|
| `experiment/` | composes architectures, platform, workload, scenario, sweep, and output policy | yes; start here |
| `architecture/` | architecture identity, role, canonical-case reference, provenance | normally read-only |
| `workload/` | LLM model/decode semantics such as batch, context, precision, and runtime footprint | yes for workload studies |
| `platform/` | facts shared across architecture comparisons, currently fixed GPU/package policy | yes when platform changes |
| `cases/` | three validated compatibility cases supplying existing geometry/power/thermal physics | internal source of truth |
| `sweeps/` | targeted primitive sensitivity definitions; not automatically current E2E evidence | only for authorized sensitivity work |
| `legacy/` | historical, fixed-power, fixture, or explicitly unvalidated configs | no |

## Important compatibility boundary

The files under `cases/` predate the new layered experiment interface. Their
top-level `workload` block is an analytical memory-activity operating point
used by the existing power backend; it is not the LLM workload definition.
The LLM workload source of truth is `workload/llama31_8b_decode_b1_s131072.yaml`.

Do not copy the 39.2 Tb/s value from a case into an architecture capability
claim. In the formal experiment it remains an explicit matched-reference
scenario with status `MATCHED_REFERENCE_NOT_CAPABILITY_VALIDATED`.

## M3D-IGZO claim boundary

The active M3D case is `cases/orthogonal_m3d_igzo.yaml`. It retains:

- Zhu operation-energy primitives with 64-layer to 8-layer transferability
  marked `NOT_VALIDATED`;
- unresolved logic-background power handled only through the experiment's
  explicit conditional-lower-bound policy;
- rho-based write-energy sensitivity, not a validated write-energy model;
- frozen merged bitcell/BEOL thermal geometry and frozen GPU-PCG physics.

`legacy/unvalidated/orthogonal_m3d_si.yaml` is not an active fourth design.

## Sensitivity policy

`sweeps/memory_internal_v0.yaml` preserves the old OFAT question definition.
Its retired numerical results must not be reused as current paper data. Any
future MAT/RD-per-ACT study should write to `results/sensitivity/` and be rerun
against a stated current baseline after the E2E claim target is fixed.
