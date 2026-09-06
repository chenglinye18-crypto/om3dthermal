# Retired README benchmark sections

> HISTORICAL RECORD — retired from DAC baselines on 2026-09-05. Retained only for provenance and test-fixture interpretation. Old commands and result claims are not current experiment instructions. Relative links below retain their historical context.

## IEDM 2025 HBM-on-GPU benchmark configuration

The archived `configs/legacy/exp_conv_2x2_g414_m160.yaml` is a paper-anchored benchmark for
3D HBM-on-GPU integration. It is sourced from:

> Yukai Chen, Melina Lofrano, Diksha Moolchandani, Herman Oprins, Geert Van
> Der Plas, Julien Ryckaert, Dwaipayan Biswas, James Myers,
> *"Breaking Thermal Bottleneck in 3D HBM-on-GPU Integration via
> System-Technology Co-Optimization"*, **IEDM 2025**,
> DOI: [10.1109/IEDM50572.2025.11353711](https://doi.org/10.1109/IEDM50572.2025.11353711).

The IEDM proceedings paper itself is **not** redistributed in this
repository; only the citation above is included.

Every value below is tagged as `PAPER_REPORTED`, `DERIVED_FROM_PAPER_*` or
`MODELING_ASSUMPTION` so it is always clear which numbers come from the
paper, which ones follow directly from the Fig. 3 layout, and which are
modelling choices this front-end has made because the paper does not specify
them.

### Configuration format

The archived `configs/legacy/exp_conv_2x2_g414_m160.yaml` is a **compact** form
(≈87 lines) optimised for hand-editing. The top-level blocks are
`materials`, `geometry`, `stacks`, `mesh`, `boundary`, `power`,
`solver`. Paper citations and per-parameter modelling notes are
recorded separately in
[`docs/benchmarks/ref_iedm25_conv_2x2_g414_m160.md`](docs/benchmarks/ref_iedm25_conv_2x2_g414_m160.md)
so the YAML itself stays small.

`load_config()` auto-detects the compact form and compiles it into
the legacy `SimulationConfig` shape before validation. Old hand-written
legacy YAMLs (the long form with explicit `footprints`,
`stack_templates`, `horizontal`, `thermal_boundary_conditions`,
`thermal_power_sources` blocks) still pass through unchanged and are
used as fixtures in `tests/`.

### Lateral layout — Fig. 3(a) (`DERIVED_FROM_PAPER_GEOMETRY`)

| Region             | Centre (mm)     | Size (mm)     |
|--------------------|-----------------|---------------|
| Package            | (0, 0)          | 65 × 65       |
| GPU                | (0, 0)          | 30 × 22       |
| HBM × 4            | see below       | 11 × 11 each  |
| Thermal silicon    | (0, 0)          |  8 × 22       |
| TIM + Lid          | (0, 0)          | 30 × 22       |

The four HBM stack footprints are placed at

- `hbm_left_top`     (−9.5, +5.5) mm
- `hbm_left_bottom`  (−9.5, −5.5) mm
- `hbm_right_top`    (+9.5, +5.5) mm
- `hbm_right_bottom` (+9.5, −5.5) mm

The four HBM base footprints and the central 8 × 22 mm thermal-silicon
footprint together form a **30 × 22 mm nominal placement envelope** at the
memory-zone centre:

- x ∈ [−15, −4]: left HBM  | x ∈ [−4, +4]: thermal silicon | x ∈ [+4, +15]: right HBM
- y ∈ [−11,  0]: bottom HBM pair | y ∈ [0, +11]: top HBM pair

This is a *placement* envelope, not a zero-gap tiling: per-layer lateral
insets and the HBM-base / DRAM-die footprint mismatch with mold-filled
cavities shown in Fig. 3(a) are not yet modelled — see
"Known limitations" below.

The paper text reports the GPU as ≈ 32 × 20 mm, but Fig. 3(a) shows an
explicit HBM-on-GPU effective region of 30 × 22 mm (`11 + 8 + 11` by
`11 + 11`). The config uses the Fig. 3(a) tiling; the text value is recorded
in `metadata.geometry_choice`.

### Vertical stack — Fig. 3(b)/(c) (`PAPER_REPORTED` thicknesses)

The HBM-on-GPU stack is built bottom-to-top, starting from the GPU's signal
routing layer `BEOL_MXY` and finishing at the lid. Within each regular DRAM
die, the layer order per Fig. 3(b) is **Hybrid Bonding → DRAM BEOL → DRAM Si
substrate** (not HB → Si → BEOL); the same pattern holds for the top die.

GPU and package layers (bottom to top):

| Layer                | Material        | Thickness | Source     |
|----------------------|-----------------|-----------|------------|
| Laminate             | Laminate        | 300 µm    | Fig. 3(c)  |
| Cu pillar bump       | Cu_Pillar_Bump  | 70 µm     | Fig. 3(c)  |
| BSPDN                | BSPDN           | 1.715 µm  | Fig. 3(c)  |
| FEOL                 | FEOL            | 0.15 µm   | Fig. 3(c)  |
| BEOL_MXY             | BEOL_MXY        | 1.4 µm    | Fig. 3(c)  |
| TIM                  | TIM             | 200 µm    | Fig. 3(c)  |
| Lid                  | Lid             | 3000 µm   | Fig. 3(c)  |

12-Hi HBM stack on top of `BEOL_MXY` (bottom to top):

| Position | Layer               | Material        | Thickness |
|----------|---------------------|-----------------|-----------|
| 1        | GPU-HBM uBump       | GPU_HBM_uBump   | 40 µm     |
| 2        | HBM base BEOL       | HBM_Base_BEOL   | 5 µm      |
| 3        | HBM base Si         | Silicon         | 50 µm     |
| 4 – 36   | 11 × (HB → BEOL → Si, with each Si being 41 µm) | as below | 2 + 3 + 41 µm |
| 37       | Top hybrid bonding  | Hybrid_Bonding  | 2 µm      |
| 38       | Top DRAM BEOL       | DRAM_BEOL       | 3 µm      |
| 39       | Top DRAM Si         | Silicon         | 169 µm    |

Sum:

```
40 + 5 + 50 + 11 × (2 + 3 + 41) + 2 + 3 + 169
= 40 + 5 + 50 + 506 + 174
= 775 µm
```

`hbm_12hi.total_thickness == 775e-6 m` and every `memory_column:*` box
spans exactly 775 µm. The top DRAM die is the 169 µm one; the other 11
DRAM dies are 41 µm each. The expanded layer order is enforced by a strict
adjacency test in `tests/test_stack_expansion.py`.

### Central thermal silicon column (`DERIVED_FROM_PAPER_FIGURE`)

The paper inserts a high-conductivity silicon block in the void between the
HBM stacks; per Fig. 3(b) the block interfaces with `BEOL_MXY` through a thin
oxide layer. The config implements the column as a two-layer stack
`thermal_silicon_stack` so the structure is fully visible in `regions.csv`
and the section plots:

| Layer                | Material        | Thickness |
|----------------------|-----------------|-----------|
| Thermal silicon oxide interface | Oxide | 1 µm |
| Thermal silicon body | Thermal_Silicon | 774 µm   |

Total = 1 + 774 = 775 µm, matching the HBM reference height. The body
material is `Thermal_Silicon` (k = 140 W/mK, isotropic), distinct from the
plain `Silicon` material only by its `metadata` and tag usage.

### Material conductivities (`PAPER_REPORTED`)

All values come from Fig. 3(c), except `Mold`, which is taken from the
prose of Section II-A. Each material records
`source: "Chen et al., IEDM 2025"` and a `source_location`.

| Material        | kx     | ky     | kz     | Source            |
|-----------------|--------|--------|--------|-------------------|
| Lid             | 400    | 400    | 400    | Fig. 3(c)         |
| TIM             | 9.71   | 9.71   | 9.71   | Fig. 3(c)         |
| Silicon         | 140    | 140    | 140    | Fig. 3(c)         |
| DRAM_BEOL       | 0.85   | 0.85   | 0.85   | Fig. 3(c)         |
| Hybrid_Bonding  | 4.8    | 4.8    | 4.8    | Fig. 3(c)         |
| HBM_Base_BEOL   | 1.5    | 1.5    | 1.5    | Fig. 3(c)         |
| GPU_HBM_uBump   | 0.59   | 0.59   | 19.28  | Fig. 3(c)         |
| Oxide           | 1.5    | 1.5    | 1.5    | Fig. 3(c)         |
| BEOL_MXY        | 1.5    | 1.5    | 1.5    | Fig. 3(c)         |
| FEOL            | 7.9    | 7.9    | 7.9    | Fig. 3(c)         |
| BSPDN           | 83     | 83     | 71     | Fig. 3(c)         |
| Cu_Pillar_Bump  | 0.54   | 0.54   | 13.25  | Fig. 3(c)         |
| Laminate        | 13     | 13     | 13     | Fig. 3(c)         |
| Thermal_Silicon | 140    | 140    | 140    | Fig. 3(c)         |
| Mold            | 3      | 3      | 3      | Section II-A      |

`k_local` ordering is `[kx, ky, kz]`. For horizontal layers `kx` / `ky` are
the in-plane values and `kz` is the cross-plane value. The current boxes
are not rotated, so the tensor is not re-expressed in world coordinates.

### Reported but not yet applied (`PAPER_REPORTED`)

These values are stored under `metadata.reported_operating_conditions` and
flagged `status: stored_for_future_solver_not_yet_applied`. The current
schema has no power or boundary-condition field, so they are kept as data
only and not interpreted:

- GPU power: 414 W total
- HBM stacks: 4 stacks × 40 W per stack
- Ambient temperature: 20 °C
- Lid-side HTC: 30 000 W/m²K
- Laminate-side HTC: 200 W/m²K
- All other boundaries: adiabatic
- Power map resolution: 0.5 mm, non-uniform commercial profile

The actual per-cell power values are not publicly reported in the paper and
are not invented here.

### Modelling assumptions (`MODELING_ASSUMPTION`)

- **GPU-vs-figure size mismatch (`DERIVED_FROM_PAPER_GEOMETRY`).** As noted
  above, the paper text says ≈ 32 × 20 mm but Fig. 3(a) shows a 30 × 22 mm
  tiled region. The config follows Fig. 3(a).

## Known limitations

- The DRAM die size `10.8 x 10.8 mm` is `DERIVED_FROM_PAPER_FIGURE`
  (see `docs/benchmarks/ref_iedm25_conv_2x2_g414_m160.md`). Fig. 3(a) of the
  IEDM 2025 paper indicates approximately 1 mm of total mold-filled
  width around each HBM stack, so the locked value leaves 0.1 mm
  per side for the mold ring. The paper does not report a complete
  per-edge DRAM footprint.

### HBM benchmark (paper-parameter-aligned uniform-power baseline)

`om3dthermal.cli solve-steady configs/legacy/exp_conv_2x2_g414_m160.yaml --out runs/hbm12_steady --alpha 0.7 --backend gpu --rtol 1e-8 --max-iterations 100000`

| Quantity | Value |
|----------|-------|
| Cells | 272,460 |
| Internal edges | 790,964 (x: 268,374 / y: 267,030 / z: 255,560) |
| Active boundary links | 19,540 |
| Adiabatic boundary faces | 33,292 |
| Total input power | **574.0 W** (414 W GPU + 4 × 40 W HBM) |
| GPU power | 414.0 W |
| HBM power (4 columns) | 160.0 W |
| Lid top HTC | 30 000 W/m²·K at 293.15 K (PAPER_REPORTED) |
| Laminate bottom HTC | 200 W/m²·K at 293.15 K (PAPER_REPORTED) |
| Solver | thermal-resistance-network relaxation, alpha 0.7 |
| Iteration cap | 100 000 |
| Final relative residual | ~ 1e-8 (target) |
| Relative power imbalance | ~ 1e-8 (target) |
| Discretisation time | ~6.5 s |
| Boundary build time | ~16.3 s |
| Operator build time | ~0.22 s |
| Relaxation solve time | GPU < 5 s on RTX 4070 SUPER |
| Total wall time | ~25 s |

`benchmark_label = "paper-parameter-aligned uniform-power baseline"`,
`strict_paper_temperature_reproduction = false`. The result is
explicitly **not** a claim that the code reproduces the paper's
141.7 °C number: the paper does not publish the per-layer 0.5 mm
non-uniform power map, and the GPU power is allocated to the FEOL
layer and the HBM power to the DRAM_BEOL layer as a modelling
choice.

