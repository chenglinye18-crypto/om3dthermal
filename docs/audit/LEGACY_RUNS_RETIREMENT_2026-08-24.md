# Legacy `runs/` retirement audit — 2026-08-24

## Decision

The pre-refactor `runs/` tree was retired after a read-only inventory.  Its
scientific questions remain reproducible from versioned configs, tests, and
current audit reports, but its numerical artifacts mix incompatible project
generations and are not a consistent current three-architecture result set.

This retirement does not delete or modify canonical case configs, formulas,
power models, thermal code, expected values, or committed audit evidence.

## Inventory before retirement

- Total local size: approximately 4.3 GiB.
- Recorded thermal-solver outcomes: 71.
- Outcomes after collapsing three obvious duplicate records: 68.
- Full temperature-field runs (`temperature_cells.csv/.npz`): 22.
- Summary-only sweep/comparison/audit outcomes: 49.

| Group | Solver outcomes | Purpose | Retirement reason |
|---|---:|---|---|
| Standalone full-field runs | 22 | early thermal benchmarks and diagnostics | legacy 2x2/2x1, old power maps, superseded diagnostics |
| Memory OFAT sweep | 34 | RD-per-ACT and MAT/subarray sensitivity | main commit `44164311`, before M3D merge and workload-aware E2E |
| Mesh sweep | 5 | Conventional 2x2 mesh convergence | legacy 574 W baseline, not canonical E2E 2x1 |
| Architecture comparison and adhesive reruns | 6 | early analytical three-architecture comparison | two exact duplicates; remaining results use mixed pre-current numerics/mesh |
| Relaxation/PCG sanity | 2 | historical numerical comparison | relaxation is not the production E2E solver |
| Successful post-merge audit records | 2 | M3D merge regression | duplicate local JSON evidence; committed regression report is authoritative |

Empty smoke/audit directories, one failed dry audit, a partial resolved config,
and a standalone log were not counted as thermal outcomes.

## Retired full-field families

- Conventional Son23 2x2 and 2x1, GPU 414 W and 300 W.
- Conventional legacy-uniform 2x2 and 2x1 variants.
- Conventional no-base variants and four `superseded_diagnostic` variants.
- Orthogonal Si/MOSAIC 98-slab, GPU 414 W and 300 W.
- M3D 8-layer matched-39.2-Tb/s all-read, GPU 414 W and 300 W.
- CPU/GPU backend comparison for the legacy Conventional 2x2 case.

The bulk temperature/power fields were generated artifacts ignored by Git and
cannot be recovered from Git after deletion. Twenty small legacy
`solver_history.csv` / `steady_state_summary.json` files had been tracked; the
retirement commit removes them from the active tree, so those summaries remain
recoverable from Git history. Reproducing full fields requires rerunning a
selected, versioned experiment under the current environment.

## Preserved MAT/sensitivity question

The sweep definition remains versioned at
`configs/sweeps/memory_internal_v0.yaml`.  It is an OFAT design, not a
Cartesian sweep:

- Conventional HBM: 8 RD-per-ACT points, 3 MAT-row points, 3 MAT-column points.
- Orthogonal Si: 8 RD-per-ACT points, 3 MAT-row points, 3 MAT-column points.
- Orthogonal M3D-IGZO: 3 subarray-row points, 3 subarray-column points.

The retired 34-point result ranges are retained only as research-direction
evidence, not as current paper values:

| Architecture / axis | Points | Parameter range | Eaccess range (pJ/bit) | Package Tmax range (degC) |
|---|---:|---:|---:|---:|
| Conventional HBM / MAT columns | 3 | 256–1024 | 1.37–1.48 | 81.77–82.50 |
| Conventional HBM / MAT rows | 3 | 256–1024 | 1.09–2.01 | 79.82–86.15 |
| Conventional HBM / RD-per-ACT | 8 | 1/64–1 | 1.09–3.25 | 79.92–93.99 |
| Orthogonal Si / MAT columns | 3 | 256–1024 | 1.37–1.38 | 84.62–84.67 |
| Orthogonal Si / MAT rows | 3 | 256–1024 | 1.09–1.92 | 83.32–87.23 |
| Orthogonal Si / RD-per-ACT | 8 | 1/64–1 | 1.13–2.80 | 83.58–90.85 |
| M3D-IGZO / subarray columns | 3 | 256–1024 | 0.85–0.86 | 82.29–82.30 |
| M3D-IGZO / subarray rows | 3 | 256–1024 | 0.86–0.87 | 82.29–82.34 |

The old data suggest that row activation/utilization is more load-bearing than
column size, while the old M3D MAT-size response is small. This is a hypothesis
for later targeted reruns, not a validated current sensitivity claim.

## Current source of truth retained

- `configs/cases/conventional_hbm_2x1.yaml`
- `configs/cases/orthogonal_si.yaml`
- `configs/cases/orthogonal_m3d_igzo.yaml`
- `docs/audit/M3D_bitcell_BEOL_merge_2026-08-18.md`
- `docs/audit/LLM_decode_workload_thermal_v0.md`
- `docs/audit/LLM_decode_conditional_E2E_final_v0.md`

Future unified runs belong under `results/` and must record a resolved config,
Git/environment provenance, stage-separated typed outputs, and a checksum
manifest. MAT refinement should be rerun only after its workload/E2E parameter
ownership and claim target are fixed.
