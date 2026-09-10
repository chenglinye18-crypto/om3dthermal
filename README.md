# om3dthermal

Workload-aware HBM/M3D architecture, power and steady-state thermal model.

## Install

Use native Windows Python 3.11 in `C:\Users\Leslie\Miniconda3`:

```powershell
conda activate om3dthermal
python -m pip install -e ".[test]"
```

CuPy 13.x and CUDA come from the Conda environment; do not install additional
CuPy/CUDA pip wheels. DreamRAM must be available at its pinned local path;
see [third-party sources](third_party/README.md). Retain upstream licenses.

## Configuration and commands

`configs/cases/` owns physical parameters; `architecture/` references those
cases, `platform/` owns GPU/host inputs, and `workload/` owns model inputs.
`experiment/` composes them. HBM 2x1 is the Conventional paper baseline;
orthogonal Si remains an ablation. [Units, sources and boundaries](docs/research/frozen_model_parameters.md).

```powershell
python -m om3dthermal experiment configs/experiment/m3d_igzo_llama31_8b_decode_conditional_v0.yaml
python -m om3dthermal prefill
python -m om3dthermal nmp-attention --output-dir runs/nmp_attention
python -m scripts.run_final_dense_e2e_matrix
python -m scripts.run_gh200_host_offload_rebaseline
python -m scripts.evaluate_die_local_placement --help
python -m pytest -q
```

Serving configuration `capacity_aware_serving_v0.yaml` uses
`om3dthermal.experiment.run_serving_experiment`; placement and NMP evaluation
scripts remain in `scripts/`. `bandwidth-thermal-sweep` retains the existing
paper sweep entry; do not run it for a single-point regression.

The separate No-NMP geometry sensitivity keeps the frozen cases unchanged:

```powershell
python -m om3dthermal thermal-sensitivity --output-dir runs/no_nmp_geometry_sensitivity --nominal-cache-dir runs/no_nmp_bandwidth_thermal_sweep_frozen_v2/cache
```

`*_sensitivity.yaml` adds hypothetical 24-high HBM and 300 µm / 106-slab M3D.
This entry recomputes DreamRAM full/closed-row energies at 24-high and uses
their arithmetic mean; nominal HBM retains its frozen energy. Memory static
and refresh power are zero throughout. Bandwidth is a prescribed thermal
load, not an achievable-throughput claim. The four curves use independent
geometry caches, a common 0–4.8 TB/s slope fit, and a 2.2–2.6 TB/s local slope.
CSV, JSON (including hotspots and row energies), PNG and SVG go to the output
directory. Omit `--nominal-cache-dir` to build all four caches there.

## Outputs

Formal experiments write stage JSON, tables, resolved inputs and a checksummed
manifest under `results/`; nonempty output directories are rejected.
Evaluation scripts write JSON/CSV under their specified output directory.
Thermal setup caches contain the fixed operator, never workload power.
`runs/`, `results/`, local figures and raw data are not committed.
