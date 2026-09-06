# ref_iedm25_conv_2x2_g414_m160

This document records the origin and modelling status of every parameter
used by `configs/legacy/exp_conv_2x2_g414_m160.yaml`. The compact YAML itself only
declares the numerical values the simulation uses; everything that
explains *why* a value was chosen (paper citation, figure caption,
modelling assumption) lives here.

The benchmark is **paper-parameter-aligned** with the IEDM 2025
3D-HBM-on-GPU stack shown in Fig. 3(b). It is **not** a strict
temperature reproduction of any reported 141.7 °C number; the paper does
not publish the 0.5 mm non-uniform power map. See the
`benchmark_label` field in `runs/hbm12_steady/steady_state_summary.json`
for the current label.

## Status tags

- `PAPER_REPORTED` — value taken directly from the cited IEDM 2025
  figure / table.
- `DERIVED_FROM_PAPER_FIGURE` — value measured off a figure or inferred
  from the published text; not numerically tabulated.
- `MODELING_CHOICE` — value chosen by us to make the simulation
  well-posed; not paper-derived.

## Material conductivities

| Material         | k (W/m·K)            | Status                  | Note |
|------------------|----------------------|-------------------------|------|
| Lid              | 400                  | MODELING_CHOICE         | Copper-class lid |
| TIM              | 9.71                 | DERIVED_FROM_PAPER_FIGURE | Indium / solder TIM |
| Silicon          | 140                  | PAPER_REPORTED          | Substrate and DRAM dies |
| Thermal_Silicon  | 140 (= Silicon)      | MODELING_CHOICE         | The thermal-Si interposer uses bulk Si k |
| DRAM_BEOL        | 0.85                 | DERIVED_FROM_PAPER_FIGURE | Cu-low-k stack effective k |
| HBM_Base_BEOL    | 1.5                  | DERIVED_FROM_PAPER_FIGURE | Same family as DRAM_BEOL |
| BEOL_MXY         | 1.5                  | DERIVED_FROM_PAPER_FIGURE | GPU BEOL effective k |
| Oxide            | 1.5                  | MODELING_CHOICE         | SiO₂ |
| FEOL             | 7.9                  | DERIVED_FROM_PAPER_FIGURE | Si active layer effective k |
| BSPDN            | [83, 83, 71]         | MODELING_CHOICE         | Backside power delivery; vertical k boosted to capture TSV array |
| Cu_Pillar_Bump   | [0.54, 0.54, 13.25]  | DERIVED_FROM_PAPER_FIGURE | Solder / Cu hybrid bump effective k |
| GPU_HBM_uBump    | [0.59, 0.59, 19.28]  | DERIVED_FROM_PAPER_FIGURE | uBump array effective k |
| Hybrid_Bonding   | 4.8                  | DERIVED_FROM_PAPER_FIGURE | Cu-Cu hybrid bonding dielectric effective k |
| Laminate         | 13                   | MODELING_CHOICE         | Package substrate effective k |
| Mold             | 3                    | PAPER_REPORTED          | EMC mold between HBM columns; isotropic |

## Geometry

- **Package**: 65 mm × 65 mm (IEDM 2025 Fig. 3 envelope).
- **GPU die**: 30 mm × 22 mm (IEDM 2025 Fig. 3).
- **Memory zone**: 30 mm × 22 mm, centred (four HBM columns arranged
  in a 2×2 pattern inside this envelope).
- **Thermal silicon**: 8 mm × 22 mm, on the package between the GPU
  and the HBM array.
- **HBM column footprint**: 11 mm × 11 mm per stack.
- **DRAM die footprint**: 10.8 mm × 10.8 mm — every DRAM die in
  the 11 repeats and the top die uses this size. The 0.1 mm
  per-side ring (between the DRAM die edge and the 11 × 11 mm
  HBM column footprint) is automatically filled with Mold.
- **HBM column centres**:
  - `hbm_left_top`:     (-9.5 mm,  +5.5 mm)
  - `hbm_left_bottom`:  (-9.5 mm,  -5.5 mm)
  - `hbm_right_top`:    (+9.5 mm,  +5.5 mm)
  - `hbm_right_bottom`: (+9.5 mm,  -5.5 mm)

## Stack templates

### Foundation (Laminate)
- Laminate, 300 µm.

### GPU (Cu pillar → BEOL, bottom-to-top)
- Cu_Pillar_Bump, 70 µm
- BSPDN,        1.715 µm
- FEOL,         0.15 µm
- BEOL_MXY,     1.4 µm

Total ≈ 73.265 µm.

### HBM-12hi (base → 11× DRAM → top die, bottom-to-top)
- gpu_hbm_ubump, 40 µm (role: `gpu_hbm_interface`)
- hbm_base_beol, 5 µm  (role: `hbm_base`)
- hbm_base_si,   50 µm (role: `hbm_base`)
- × 11 of:
  - hybrid_bonding_NN, 2 µm  (role: `hybrid_bonding`)
  - dram_beol_NN,      3 µm  (role: `dram_beol`)
  - dram_si_NN,        41 µm (role: `dram_si`)
- top_hybrid_bonding, 2 µm   (role: `hybrid_bonding`)
- top_dram_beol,      3 µm   (role: `dram_beol`)
- top_dram_si,        169 µm (role: `dram_si`, top_die)

Total = 95 + 11 × 46 + 174 = **775 µm** (matches Fig. 3(b)).

### Thermal-silicon interposer
- Oxide,           1 µm
- Thermal_Silicon, 774 µm

Total = **775 µm**, matching the HBM-12hi height so the GPU/TIM/Lid
stack can sit on a flat top.

### Top (TIM → Lid, bottom-to-top)
- TIM, 200 µm
- Lid, 3000 µm

## Mesh

- `dx = dy = 0.5 mm` (lateral)
- `dz_max = 100 µm` (vertical; the discretiser subdivides each
  layer's thickness to honour layer boundaries while keeping cell
  height ≤ 100 µm).

With these settings the discretiser produces **741 AxisAlignedBox**
primitives, **272 460 ThermalCell** entries and **790 964**
internal conductance edges (see `runs/hbm12_mesh/mesh_summary.json`).

## Boundary conditions

- **Lid top convection**: 30 000 W/m²/K, ambient 20 °C — status
  `PAPER_REPORTED` (forced-air cold plate).
- **Laminate bottom convection**: 200 W/m²/K, ambient 20 °C —
  status `PAPER_REPORTED` (board-level natural convection).
- All other outer faces are adiabatic by default.

## Power sources

| Source                | Selector                                  | Power   | Status           |
|-----------------------|-------------------------------------------|---------|------------------|
| `gpu_total`           | `component: gpu, material: FEOL`          | 414 W   | `PAPER_REPORTED` |
| `hbm_left_top`        | `component: memory_column:hbm_left_top, tags: {role: dram_beol}`  | 40 W    | `PAPER_REPORTED` (per-HBM uniform) |
| `hbm_left_bottom`     | `component: memory_column:hbm_left_bottom, tags: {role: dram_beol}` | 40 W  | `PAPER_REPORTED` |
| `hbm_right_top`       | `component: memory_column:hbm_right_top, tags: {role: dram_beol}` | 40 W    | `PAPER_REPORTED` |
| `hbm_right_bottom`    | `component: memory_column:hbm_right_bottom, tags: {role: dram_beol}` | 40 W | `PAPER_REPORTED` |

Total input: **574 W** (414 W GPU + 4 × 40 W HBM).

### Distributing GPU power on the FEOL layer — `MODELING_CHOICE`

The paper reports only a top-level GPU power of 414 W. It does not
publish the sub-die 0.5 mm non-uniform power map. We therefore place
all 414 W on the FEOL layer with `uniform_volume` distribution — a
well-posed modelling choice that does not require any non-public
information.

### Distributing HBM power on each column's DRAM_BEOL — `MODELING_CHOICE`

The paper reports 40 W per HBM column. We distribute each 40 W on the
column's `DRAM_BEOL` cells with `uniform_volume` distribution. The
`top_die`'s DRAM_BEOL is included (it shares the `role: dram_beol`
tag); the per-column totals therefore cover all 12 dies.

## Solver

- `alpha: 0.7`, `rtol: 1e-6` in the compact YAML; the CLI accepts
  `--alpha 0.7` and `--rtol 1e-8` for the published benchmark.
- Solver: thermal-resistance-network relaxation (CPU or GPU).
- Default `max_iterations = 10 000`.
- Initial temperature: 293.15 K.

## What this benchmark is *not*

- **Not a strict reproduction of 141.7 °C.** The paper does not
  publish the 0.5 mm non-uniform power map; we therefore label the
  result as a `paper-parameter-aligned uniform-power baseline`.
- **Not a thermal-only or electrical-only benchmark.** The whole
  stack is solved in steady state with a single 0.5 mm lateral mesh.
- **Not a multi-physics model.** No transient, no temperature-
  dependent k, no radiation, no two-phase cooling. Those will arrive
  in later PRs; see the project README.
