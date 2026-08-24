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

REM Gate 9: production GPU-PCG kernel/operator smoke
python -m pytest tests/test_thermal_relaxation.py -q -k gpu_pcg
```

If any gate fails, the failure must be diagnosed inside the `om3dthermal`
Conda environment; do not fall back to WSL, system Python, or a temporary
venv to "make it work".

## Project scope

om3dthermal is a workload-aware architecture/power/steady-state-thermal
framework for HBM-on-GPU and M3D / orthogonal-memory research.  The canonical
DAC E2E Conventional baseline is `configs/cases/conventional_hbm_2x1.yaml`.
Conventional 2x2 cases under `configs/legacy/` are historical thermal
validation benchmarks, not the main E2E baseline.

## Hard constraints

1. This project is STEADY-STATE ONLY.
2. Do not implement transient simulation, heat-capacitance time stepping, AMR,
   dense matrices, or matrix inversion.
3. Do not change physical equations or benchmark assumptions unless explicitly
   requested.
4. Do not claim strict literature reproduction when required inputs such as
   original non-uniform power maps are unavailable.

## Current numerical model

Formal experiment YAML -> architecture/platform/workload descriptors ->
capacity/traffic/FLOPs -> matched-reference performance -> conditional memory
energy -> workload power -> existing thermal mapping -> FP64 matrix-free
GPU-PCG with Jacobi preconditioning -> typed Tmax observations/result bundle.

The legacy standalone thermal CLI retains CPU/GPU relaxation paths for
validation and historical commands.  It is not the production DAC E2E path.

## Formal solver (frozen; do not change)

The production DAC E2E steady-state solver is the existing **FP64,
matrix-free GPU-PCG solver with Jacobi diagonal preconditioning**.  It checks
the true KCL residual and performs no full-vector device-to-host transfer
during iteration.  Its physical operator, tolerances, mapping, and numerical
implementation are frozen.

Do not optimize or replace GPU-PCG, change the thermal equations, or migrate
additional public solver paths unless explicitly requested.  The historical
relaxation implementations remain compatibility/reference code; their old
policy is not the current production-solver policy.

## Configuration

Formal configuration is split across `configs/architecture/`,
`configs/platform/`, `configs/workload/`, and `configs/experiment/`.  The
architecture descriptors reference validated canonical cases instead of
copying their physical values.  Keep provenance and claim status explicit.

## Research priorities

- mesh convergence;
- solver-tolerance convergence;
- geometry/inset sensitivity;
- power-distribution sensitivity;
- comparison against analytical or literature references;
- temperature-dependent steady-state materials only when justified.

Do not add new physics merely because it is technically possible.

## Benchmark invariants

For E2E non-physics changes, preserve the validated Conventional 2x1 GPU-PCG
baseline (859596 cells, 2531340 edges, analytical package input about
355.58349 W, Tmax about 81.93349 degC at the canonical tolerances).  The
574 W / ~122.97 degC Conventional 2x2 result is a legacy thermal benchmark.
Investigate unexpected changes; never update expected values merely to make
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
