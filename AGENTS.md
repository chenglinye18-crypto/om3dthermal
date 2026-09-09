# AGENTS.md

## Environment

Native Windows Conda environment `om3dthermal`, Python 3.11.
Conda root: `C:\Users\Leslie\Miniconda3`.
Project: `E:\BaiduSyncdisk\study\PAPER\DAC 2026\Project`.
GPU: NVIDIA GeForce RTX 4070 SUPER.

```powershell
conda activate om3dthermal
python -m om3dthermal --help
python -m pytest -q
```

Use this environment for every Python command, test and simulation. No WSL,
system Python, Codex runtime Python, other Conda root or temporary venv.
Use Conda CuPy 13.x; do not install cupy-* or nvidia-cuda-* pip wheels.

## Research constraints

Steady-state only: no transient time stepping, heat capacitance, AMR, dense
matrices or inversion. Preserve physical equations, frozen parameters, mesh,
solver tolerances and cache validation unless explicitly requested.
Production is FP64 matrix-free GPU-PCG with Jacobi preconditioning and true
KCL residual checks; no full-vector device-to-host copy during iteration.
Keep geometry/capacity, workload/traffic/performance, memory/GPU power,
thermal mapping/cache, serving, host offload, placement and existing NMP.
Preserve slab thickness/count and stack-layer parameterization.
Conventional HBM 2x1 is the paper baseline; orthogonal Si is an ablation.
Keep units, sources and claim boundaries explicit; do not claim strict
literature reproduction without the original required inputs.

## Work rules

Inspect relevant dependencies, make coherent changes and run scientific
behavior tests. Investigate regressions; never loosen tolerances or alter
expected values to hide a failure. Protect untracked figures, raw data,
results and caches. Do not modify third_party or commit large artifacts.
Stage exact task files. Do not merge or run sweeps without authorization.
