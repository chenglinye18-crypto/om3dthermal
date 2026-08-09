# AGENTS.md

## Project scope

om3dthermal is a steady-state 3D thermal simulation framework for
HBM-on-GPU and future M3D / orthogonal-M3D research.

The current stable baseline is `v0.1.0-steady`.

## Hard constraints

1. This project is STEADY-STATE ONLY.
2. Do NOT implement transient thermal simulation.
3. Do NOT add heat-capacitance time stepping, Crank-Nicolson, BDF,
   explicit transient integration, or other transient solvers.
4. Do NOT introduce transient-related configuration or dependencies.
5. Do NOT change physical equations or benchmark assumptions unless
   explicitly requested.
6. Do NOT claim strict reproduction of literature results when required
   inputs, such as the original non-uniform power map, are unavailable.

## Current numerical model

The main pipeline is:

Compact YAML
→ geometry
→ ThermalCell discretization
→ face adjacency
→ anisotropic face conductance
→ boundary / power mapping
→ matrix-free thermal operator
→ PCG steady-state solution

The production solver is matrix-free.
Do not introduce dense NxN matrices or matrix inversion.

Weighted Jacobi is retained only as a reference/debugging solver.

## Configuration

`configs/hbm_on_gpu_12hi.yaml` is the user-facing configuration.

Keep it compact and hand-editable.

Do not expose internal selector/tag/priority/metadata plumbing in the
main benchmark YAML unless absolutely necessary.

Paper provenance and modeling assumptions belong in:
`docs/benchmarks/`

## Research priorities

After v0.1.0-steady, prioritize validation of steady-state results:

- mesh convergence;
- solver-tolerance convergence;
- geometry/inset sensitivity;
- power-distribution sensitivity;
- comparison against analytical or literature references;
- temperature-dependent steady-state material models when justified.

Do not add new physics merely because it is technically possible.

## Benchmark invariants

For changes that are not intended to alter physics, verify against the
v0.1.0-steady HBM benchmark:

- 272460 ThermalCells
- 790964 internal edges
- total input power = 574 W
- current baseline Tmax ≈ 136.03 °C

If a refactor changes these values unexpectedly, investigate the cause.
Do not update expected values simply to make tests pass.

## Development workflow

1. Inspect only files relevant to the requested task.
2. Make the smallest coherent change.
3. Keep unrelated refactors out of the PR.
4. Add targeted tests for changed behavior.
5. Run targeted tests first, then the full test suite when appropriate.
6. Keep generated `runs/` outputs untracked.
7. Do not add PDFs or large simulation artifacts to Git history.
8. Do not merge a PR unless explicitly requested.

## Scientific discipline

Clearly distinguish:

- PAPER_REPORTED
- DERIVED_FROM_PAPER_FIGURE
- MODELING_CHOICE
- NUMERICAL_CHOICE

Do not tune modeling assumptions merely to reproduce a target
temperature from a paper.

Prefer physically interpretable validation over adding solver features.