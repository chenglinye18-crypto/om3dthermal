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

The fixed **318-slab Decode placement comparison** runs with
`python scripts/compare_decode_policies.py` (Conda `om3dthermal`). It prints
the resolved audit before evaluating Llama 3.1 8B/70B/405B, B1, resident 125K
history + 1K GPU incremental Prefill + 1K Decode steps at contexts 126000..126999.
All policies use 2-byte weights/KV/activations and the same H200/M3D resources;
405B uses [Meta's BF16-mp8, 8-KV-head definition](https://github.com/meta-llama/llama-models/blob/main/models/sku_list.py). Parameter capacity includes
untied embedding/output matrices and RMSNorm, rather than rounded model labels.

`ExecutionPolicy` in `serving/decode_policy.py` compiles the existing canonical
MAC-NMP stage/handoff ledger against one maximum-context resident placement.
`ATTENTION_NMP` keeps only QK/AV near memory and local KV writes; projections,
FFN, LM Head, RoPE and Softmax stay GPU. Canonical FP32 AV partials return to
GPU for reduction; no physical K transpose is materialized. `NO_NMP` keeps
all compute GPU, still using M3D memory. Serial dependent stages and within-stage
roofline overlap are retained; integer KV shards are evaluated at every step.

`runs/decode_policy_318/` contains the audit, Prefill ledger/energy envelope,
9-case summary, normalized ratios, mean per-token traffic, and all 9000 step
latencies/energies. Decode throughput uses 1000/output-Decode-seconds; E2E
uses 1000/(Prefill+Decode seconds). Energy efficiencies are **modeled tokens/J**.
Prefill/E2E main energy columns use the existing upper-energy endpoint;
`*_J_min` retains the lower endpoint (no new nominal coefficient).
GPU Decode retains 11.68 pJ/bit + 74 W idle; NMP MAC uses 0.604 pJ/MAC.
Memory static/refresh is excluded for all policies under the frozen comparison
policy, including the raw diagnostic refresh term in canonical NMP.
This preserves MAC stages/dynamic energy, not the excluded refresh energy.
Controller/clock-tree/PHY idle and non-MAC NMP overhead are not fully modeled;
GPU bit-based Decode energy is not a per-operator silicon measurement.
Incremental Prefill retains the existing optimistic tiled, single-pass historical
KV read assumption. No host offload, thermal solve, or hardware measurement is run.

Regression command:
`python -m pytest -q tests/test_decode_policy.py tests/test_nmp_die_activity.py tests/test_llm_prefill.py`.
The three pre-existing `test_nmp_decode_batch.py` failures reference old timing,
capacity and 2.4 TB/s bandwidth; their expectations are unchanged.

Fixed-run results (2026-09-11; same 318-slab memory for every row):

| Llama 3.1 | Policy | Decode tok/s | E2E generated tok/s | Decode modeled tok/J | E2E modeled tok/J |
|---|---|---:|---:|---:|---:|
| 8B | NO_NMP | 147.062 | 144.638 | 0.264606 | 0.259118 |
| 8B | ATTENTION_NMP | 271.101 | 262.977 | 0.525629 | 0.504409 |
| 8B | MAC_NMP | 1046.170 | 934.748 | 4.429591 | 3.270206 |
| 70B | NO_NMP | 25.848 | 25.411 | 0.046497 | 0.045508 |
| 70B | ATTENTION_NMP | 31.742 | 31.085 | 0.059393 | 0.057788 |
| 70B | MAC_NMP | 181.574 | 161.997 | 0.804565 | 0.584673 |
| 405B | NO_NMP | 5.398 | 5.322 | 0.009704 | 0.009533 |
| 405B | ATTENTION_NMP | 5.674 | 5.591 | 0.010438 | 0.010241 |
| 405B | MAC_NMP | 43.473 | 39.015 | 0.200059 | 0.146088 |

Attention-only speedups over NO_NMP are 1.843/1.228/1.051 for 8B/70B/405B;
MAC-NMP adds 3.859/5.720/7.662 over Attention-only. Attention-only keeps
15.009/139.003/807.496 GB of weight boundary traffic per generated token;
both NMP scopes eliminate historical KV boundary traffic. MAC-NMP's extra
benefit grows as weight traffic dominates. These are analytical estimates
under the energy and Prefill caveats above, not silicon measurements.

Formal experiments write stage JSON, tables, resolved inputs and a checksummed
manifest under `results/`; nonempty output directories are rejected.
Evaluation scripts write JSON/CSV under their specified output directory.
Thermal setup caches contain the fixed operator, never workload power.
`runs/`, `results/`, local figures and raw data are not committed.
