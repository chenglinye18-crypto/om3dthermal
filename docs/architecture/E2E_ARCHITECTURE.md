# Workload-aware E2E architecture

## Scope and frozen baseline

This document describes the compatibility architecture introduced after the
validated `c6ecf418` conditional E2E baseline.  It changes ownership,
interfaces, configuration, orchestration, and output structure; it does not
change validated formulas, physical parameters, thermal mapping, or GPU-PCG.

The formal flow is:

```text
ArchitectureSpec + PlatformSpec + WorkloadSpec
                  |
                  v
       existing canonical case resolution
                  |
                  v
workload demand -> capacity -> performance -> conditional memory energy
                                      -> workload power
                  |
                  v
     public compatibility thermal adapter -> frozen GPU-PCG -> Tmax
                  |
                  v
          typed E2E rows + result bundle
```

## Ownership

| Package | Owns | Must not own |
|---|---|---|
| `architecture/` | architecture identity, packing, energy/static fact contracts | workload or sweep choices |
| `platform/` | shared package/GPU facts | memory-architecture primitives |
| `workload/` | model shape, precision, batch/context, architecture-independent demand | hardware feasibility or thermal semantics |
| `evaluation/` | generic workload-to-hardware gates, beginning with capacity | application formulas or thermal solving |
| `evaluator/` | compatibility home for conditional decode stage calculators during migration | YAML loading or report formatting |
| `thermal/` | mesh/operator/solver and public case adapter | LLM accounting |
| `experiment/` | configuration composition and orchestration | scientific formulas or serialization |
| `result/` | serialization, checksummed bundles, and manifests | scientific calculation |
| `provenance/` | source/status and run-environment records | inferred scientific claims |
| `reporting/` | presentation-only tables | recalculation |

Legacy `architecture_comparison.py`, canonical case YAML, power resolution,
and thermal implementations remain numerical sources of truth.  New modules
wrap them through explicit adapters while migration is in progress.

## Configuration boundary

- `configs/architecture/`: identity, role, canonical-case reference, and
  provenance only.
- `configs/platform/`: facts shared across the compared memory architectures.
- `configs/workload/`: application/model semantics only.
- `configs/experiment/`: references to the above plus matched scenario,
  sweep, policy, and output choices.

The formal GPU-PCG backend, convergence tolerances, iteration limit, check
interval, and fresh initial temperature are a frozen internal solver contract,
not experiment inputs.  They are persisted in typed thermal results for
reproducibility but intentionally omitted from user-facing YAML.

Architecture configs must not contain batch/context or matched bandwidth.
Workload configs must not contain capacity, pJ/bit, power, or thermal fields.
An experiment may combine layers but may not silently manufacture a missing
architecture fact.

## Formal output contract

Each run writes a non-overwriting, checksummed directory containing:

```text
resolved_config.yaml
architecture.json
workload.json
capacity.json
performance.json
energy.json
power.json
thermal.json
provenance.json
summary.json
summary.csv
manifest.json
```

`resolved_config.yaml` records the composed inputs.  Stage files retain typed
outputs and statuses.  `provenance.json` records Git/environment/input hashes.
`manifest.json` marks bundle completeness and checksums every other artifact.
The CSV is a presentation view of the typed summary, not a second calculator.

## Claim boundaries

The formal scenario remains conditional:

- bandwidth is `MATCHED_REFERENCE_NOT_CAPABILITY_VALIDATED`;
- write energy is a rho sensitivity, not a validated physical model;
- the 0.5 pJ/bit M3D contactless-interface term is
  `PAPER_REPORTED_INDUCTIVE_LINK_ENERGY` (Shiba SSC-L 2023, 7 nm FinFET,
  12.8 Gb/s; the same value adopted by MOSAIC, Mitarai VLSI 2026); its
  clock/SerDes coverage boundary is unconfirmed, so it remains swept in
  sensitivity analyses and is not a complete-PHY validation;
- M3D logic-background power is either an explicit conditional lower bound or
  a `PARAMETRIC_SENSITIVITY` value, never a validated nominal;
- GPU decode energy and system J/token exist only as the optional E8
  affine-utilization stage (`ANALYTICAL_AFFINE_UTILIZATION_MODEL` with
  `PARAMETRIC_NOMINAL_WITHIN_MEASURED_REFERENCE_RANGE`); the frozen E7 row
  keeps `gpu_energy_model_status = NOT_AVAILABLE` and the thermal path keeps
  the fixed 300 W GPU power source — at u = 1 the affine model reproduces
  that fixed baseline exactly;
- the read-shaped write spatial distribution is sensitivity-only.

No directory structure or PASS status upgrades those scientific claims.

## Compatibility and migration rule

New work should enter through the domain specs, workload-demand contract, and
formal experiment runner.  `workload.capacity` and
`workload.architecture_capacity` remain compatibility import locations; new
callers use `evaluation`.  Result serialization is owned by `result/`, while
`experiment.result_bundle` remains a compatibility import.

Validated legacy calculators remain in place until a replacement produces
bit-equivalent typed outputs and passes the existing regression gates.  Remove
compatibility adapters only in a separately reviewed cleanup after callers and
tests have migrated.
