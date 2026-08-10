# AGENTS.md

## Project scope

om3dthermal is a steady-state 3D thermal simulation framework for HBM-on-GPU
and future M3D / orthogonal-M3D research. The consolidated baseline is the
canonical Son23-powered Conventional 2x2 HBM case.

## Hard constraints

1. This project is STEADY-STATE ONLY.
2. Do not implement transient simulation, heat-capacitance time stepping, AMR,
   dense matrices, or matrix inversion.
3. Do not change physical equations or benchmark assumptions unless explicitly
   requested.
4. Do not claim strict literature reproduction when required inputs such as
   original non-uniform power maps are unavailable.

## Current numerical model

Compact YAML -> geometry -> ThermalCell discretization -> face adjacency ->
anisotropic face conductance -> boundary/power mapping -> matrix-free thermal
operator -> PCG steady-state solution.

The CPU reference solver and CuPy GPU backend are both matrix-free. Weighted
Jacobi is retained only as a reference/debugging solver.

## Configuration

`configs/exp_conv_2x2_g414_m160.yaml` is the canonical Conventional 2x2
reference configuration. Orthogonal MOSAIC and M3D experiments remain separate
configs and must not overwrite it. Keep configs compact and hand-editable;
paper provenance and modeling assumptions belong in `docs/benchmarks/`.

## Research priorities

- mesh convergence;
- solver-tolerance convergence;
- geometry/inset sensitivity;
- power-distribution sensitivity;
- comparison against analytical or literature references;
- temperature-dependent steady-state materials only when justified.

Do not add new physics merely because it is technically possible.

## Benchmark invariants

For non-physics changes, verify the current canonical Conventional 2x2 case:

- 859596 ThermalCells;
- 2531340 internal edges;
- total input power = 574 W;
- rtol=1e-6 baseline Tmax approximately 122.9713 degC.

Investigate unexpected changes; do not update expected values merely to make
tests pass.

## Development workflow

1. Inspect only files relevant to the requested task.
2. Make the smallest coherent change.
3. Keep unrelated refactors out of commits.
4. Add and run targeted tests for changed behavior.
5. Keep generated `runs/` outputs untracked.
6. Do not add PDFs or large simulation artifacts to Git history.
7. Do not merge unless explicitly requested.

## Scientific discipline

Clearly distinguish `PAPER_REPORTED`, `DERIVED_FROM_PAPER_FIGURE`,
`MODELING_CHOICE`, and `NUMERICAL_CHOICE`. Do not tune assumptions merely to
reproduce a paper temperature; prefer physically interpretable validation.
