# AGENTS.md

## Execution environment

This repository uses **one canonical native-Windows Conda environment**.
Every agent, every test, every CLI invocation, every sweep run must use it.

- Environment name: `om3dthermal`
- Platform: Windows (`win32`)
- Python: 3.11
- Conda base: `C:\Users\Leslie\Miniconda3`
- Project root: `E:\BaiduSyncdisk\study\PAPER\DAC 2026\Project`
- GPU: `NVIDIA GeForce RTX 4070 SUPER` (Compute Capability 8.9, CUDA 13.0 driver)

Before running Python, tests, power models, thermal simulations, or sweeps:

```powershell
conda activate om3dthermal
cd "E:\BaiduSyncdisk\study\PAPER\DAC 2026\Project"
```

Required invocation pattern:

```powershell
python -m om3dthermal ...
python -m pytest tests/test_sweep.py -q
```

Hard bans for any agent on this repository:

- **No WSL / Ubuntu / apt / sudo** for project Python. WSL Python was used
  for early exploration and is now retired.
- **No `D:\anaconda3`** or any other historical Conda root. The stale
  registrations under `D:\anaconda3\envs\*` were intentionally removed from
  `%USERPROFILE%\.conda\environments.txt` (see "Conda state" below).
- **No `C:\Users\Leslie\.cache\codex-runtimes\...`** (Codex-managed Python
  is not part of the canonical project environment).
- **No `cupy` 14.x** and **no separate `cupy-cuda12x` pip distribution**.
  CuPy 13.x is installed once via `conda-forge`; the project `pyproject.toml`
  `[gpu]` extra must not be re-installed on top of the Conda cupy.
- **No temporary `venv`**. The Conda env is the environment.
- **No `pip install cupy-*` and no `pip install nvidia-cuda-*`**. Those
  wheel sets duplicate the Conda `cudatoolkit` packages and break the GPU
  stack on Windows.

### Conda state

- The active Conda root is `C:\Users\Leslie\Miniconda3` (official Windows
  Miniconda3, JustMe install, no system Python registration).
- `C:\Users\Leslie\.conda\environments.txt` was cleaned of stale
  `D:\anaconda3\envs\{dacs2024,EMspice,PROTON,finance-analyse}` entries
  during the environment standardization pass. A timestamped backup was
  left next to it as `environments.txt.bak.YYYYMMDD_HHMMSS`.
- The file is currently empty (UTF-8, no BOM) so future `conda create` /
  `conda env remove` calls can rewrite it without UnicodeDecodeError.

### Smoke / verification recipe

After creating or restoring the environment, run the following gates; all
must PASS before any benchmark sweep is launched:

```powershell
conda activate om3dthermal
cd "E:\BaiduSyncdisk\study\PAPER\DAC 2026\Project"

REM Gate 6: native Windows Python + CPU imports
python -c "import sys; print(sys.executable, sys.platform)"
python -c "import numpy, scipy, yaml, pydantic; print('CPU_IMPORT_PASS')"

REM Gate 7: CuPy GPU probe (device >= 1, kernel exec)
python -c "import cupy as cp; print('CuPy', cp.__version__, cp.cuda.runtime.getDeviceCount()); x=cp.arange(10, dtype=cp.float64); print((x*x).tolist()); print('CUPY_GPU_PASS')"

REM Gate 8: project import + targeted tests
python -c "import om3dthermal; print(om3dthermal.__file__)"
python -m pytest tests/test_sweep.py -q

REM Gate 9: one GPU thermal smoke (Conventional HBM baseline, backend=gpu)
python -m om3dthermal.cli solve-steady configs\cases\conventional_hbm_2x1.yaml --out runs\_gpu_smoke --alpha 0.7 --backend gpu
```

If any gate fails, the failure must be diagnosed inside the `om3dthermal`
Conda environment; do not fall back to WSL, system Python, or a temporary
venv to "make it work".

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
operator -> thermal-resistance-network relaxation (CPU or GPU).

The CPU reference solver and the CuPy GPU backend are both matrix-free
and both implement the same relaxation equation.

## Formal solver (do not change)

The only production steady-state thermal solver is the
**thermal-resistance-network relaxation**:

```
delta_Q_i = P_i - sum_j G_ij (T_i - T_j) - sum_b G_ib (T_i - T_b)
R_eff_i  = 1 / ( sum_j G_ij + sum_b G_ib )
delta_T_i = alpha * delta_Q_i * R_eff_i        alpha in (0, 1]
T_new_i   = T_old_i + delta_T_i                (simultaneous update)
```

Convergence requires both `relative_heat_flow_residual < tol` and
`max_abs_delta_T < tol` to be satisfied at the same `check_interval`
boundary. CPU and GPU backends must implement the same relaxation
equation (FP64 end-to-end; no fast-math, no mixed precision).

Do not introduce PCG, CG, sparse linear solvers, matrix inversion,
or alternative steady-state solvers unless explicitly requested by the
user. The GPU and CPU relaxation kernels must produce numerically
equivalent temperature fields.

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
