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
python scripts/compare_decode_policies.py
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

The canonical **physical FEOL Decode comparison** runs with
`python scripts/compare_decode_policies.py` in Conda `om3dthermal`. It prints
the floorplan audit before the fixed Llama 3.1 8B/70B/405B x
NO_NMP/ATTENTION_NMP/MAC_NMP matrix. B1 uses resident 125000-token history,
1000-token GPU incremental Prefill and contexts 126000..126999 for Decode.
Weights/KV/activations use 2 bytes; AV partials use FP32. All data are resident.

`configs/architecture/m3d_feol_execution.yaml` freezes 318 x 100-um slabs,
35x8 clusters per layer, 8 layers, 70 physical four-cluster service groups,
70 distributed FEOL SA banks, region group counts 18/18/18/16, and 32
16-MAC tiles per slab. Every region has 128 KiB staging, 2x256-bit @ 1GHz
shared fabric (64 GB/s), and a root in its geometric center. A bidirectional
linear root backbone uses 32 GB/s per direction, 8 FP32 adds/cycle at each
root and geometry-derived, clock-rounded wire pipeline latency.

Atomic rows/vectors stripe die-first across actual groups, then across
capacity-checked layers. A group-layer bit-stripes its data across its four
existing cluster slots. Each group sums its 32-byte services across active
layers at the existing 10-ns sensing/read primitive plus layer MIV delay.
The FEOL SA is colocated with the group MIV landing; sensing is counted once.
SA-to-tile routes use physical Manhattan lengths and the existing distributed
Elmore RC. Region time is max(array, shared fabric, assigned tile compute).
No whole-die bandwidth pool or fixed 1-ns route shortcut remains in execution.

Activation ingress and structured gather use region root 0 on each active
slab. Multicast forwards along shortest linear paths; AV first reduces tile
partials at region roots and then uses the pairwise region tree. Only one
partial per active slab reaches the GPU for cross-slab reduction. Ordinary
GPU memory reads route directly SA-to-edge, bypassing the MAC fabric.
Every source uses its nearest port on the single y_min edge. Per-port
serialization (8 Gb/s each) can constrain a stage below the shared external
cap: min(raw 15.9 TB/s, GPU 4.8 TB/s, frozen thermal 3.4 TB/s). Root/port
concentration is explicit, not a utilization factor. No cross-slab NoC exists.
Large streams tile through the 128-KiB buffers; full weights do not stage there.

Only `runs/decode_policy_318_feol_v1/` is canonical. It contains summary,
normalized, traffic, bottleneck CSVs and floorplan/resident-group audits.
Four independent CPU workers evaluate disjoint exact context ranges; a fresh
resident-layout rerun must reproduce every per-step result hash. There is
no interpolation and no second output directory. Active group/region/tile
counts are across all slabs per memory stage. Realized array/external rates
are bytes divided by the entire Decode duration. Fabric realized rate is the
time-weighted mean per active region; bottleneck fractions attribute each
stage to its largest time component. These definitions are not peak rates.

The old die-only NMP execution/energy API, quadratic horizon approximation,
thermal carrier producer and contradictory regression gates were removed,
without compatibility forwarding. Legacy mixed/persistent/formal serving
rejects its retired NMP path; GPU/capacity functionality remains separate.
The existing M3D write-energy primitive remains available to unrelated users.
This experiment reports performance only, with no system-energy columns or
thermal solve. Prefill retains the existing optimistic tiled historical-KV
single-read ledger and GPU roofline; physical bulk group/port service is
included. These are analytical estimates, not measured silicon performance.

Tests: `python -m pytest -q tests/test_decode_policy.py tests/test_feol_latency.py tests/test_physical_capacity.py tests/test_memory_bandwidth.py tests/test_m3d_100um_slab_architecture.py tests/test_llm_prefill.py tests/test_mixed_phase_e2e.py`.

Formal experiments write stage JSON, tables, resolved inputs and a checksummed
manifest under `results/`; nonempty output directories are rejected.
Evaluation scripts write JSON/CSV under their specified output directory.
Thermal setup caches contain the fixed operator, never workload power.
`runs/`, `results/`, local figures and raw data are not committed.
