# om3dthermal

`om3dthermal` is a geometry front-end for axis-aligned 3D thermal models.
It reads a unit-aware YAML config, validates and expands layer stacks, and
emits a uniform set of `AxisAlignedBox` regions that a later thermal solver
can consume. The current version does **not** include thermal resistance /
FEM / FVM, temperature solving, power maps, boundary conditions, meshing,
the orthogonal MOSAIC builder, COMSOL or Icepak integration, or a GUI.

## Install and run

Python 3.10+ is required:

```bash
python -m pip install -e ".[test]"
python -m pytest
python -m om3dthermal.cli build configs/hbm_on_gpu_12hi.yaml --out runs/hbm12_iedm2025
```

The `build` command writes (under the directory given by `--out`, which
is git-ignored by default — see "Build artifacts" below):

- `regions.csv` — SI coordinates, material, tags, source path and reserved
  rotation for every box;
- `geometry_summary.json` — box / material counts, stack heights, component
  bounds and extremal dimensions;
- `top_view.png` — footprint names and mm coordinates;
- `xz_section.png`, `yz_section.png` — z-stacked material layers in µm.

## Configuration structure

All input lengths can be written as Pint-recognised strings (e.g. `65 mm`,
`41 um`); bare numbers are SI metres. Material names and footprint names
must match their dictionary keys, and every footprint must lie inside the
package footprint.

- `materials`: material name, optional local conductivity `k_local: [kx, ky, kz]`
  (in-plane first, cross-plane last), and metadata. Box rotation is still the
  identity matrix, so the tensor is not rotated yet.
- `footprints`: 2D rectangle centre and size. Every footprint must lie inside
  `package_footprint`.
- `stack_templates`: bottom-to-top `items`. Plain entries are
  `kind: layer`; repeated entries are `kind: repeat` with a positive integer
  `count` and a list of inner layers. Expanded layer names are suffixed with
  `_01`, `_02`, ... so they remain unique. A layer can carry an optional
  `lateral_inset` (see "Per-layer lateral footprints" below).
- `horizontal.foundation`: footprint and stack of the package foundation.
- `horizontal.gpu`: GPU footprint and stack sitting on top of the foundation.
- `horizontal.memory_zone`: GPU-top memory zone, with a reference stack
  height, a low-priority background slab, and any number of columns. Columns
  can either reference a `stack`, or be a single-material slab via
  `match_height_of`. Shorter stacks must supply a `fill_above` material that
  pads them to the reference height.
- `horizontal.top`: TIM/Lid stack on top of the memory zone. The current
  config places it on the 30 × 22 mm `memory_zone` footprint (Fig. 3(b)),
  not on the full 65 × 65 mm package.

Absolute z coordinates are never written to the config — the builder derives
them in SI metres from the stack templates.

## IEDM 2025 HBM-on-GPU benchmark configuration

The shipped `configs/hbm_on_gpu_12hi.yaml` is a paper-anchored benchmark for
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

- The lateral inset value `0.5 mm per side` for DRAM dies is
  `DERIVED_FROM_PAPER_FIGURE` (see `metadata.dram_lateral_inset` in the
  YAML). Fig. 3(a) indicates approximately 1 mm of total mold-filled
  width around each HBM stack, but the paper does not report a complete
  per-edge DRAM footprint. A sensitivity sweep over the inset value is
  required before any thermal-solver work. The current 0.5 mm value is a
  first-pass benchmark assumption.

## Current boundaries

`HorizontalColumnsBuilder` is a dedicated builder, not a general-purpose
CAD / boolean engine. The future `OrthogonalBladesBuilder` should emit the
same `AxisAlignedBox` set, but it is not implemented yet.

## Per-layer lateral footprints

Per Fig. 3(a), the HBM DRAM dies are slightly smaller than the HBM base
die; the lateral cavity between the DRAM die and the base-die sidewall
is filled with mold compound. `HorizontalColumnsBuilder` represents this
by giving each `Layer` an optional `lateral_inset` against the parent
column footprint:

```yaml
- kind: layer
  name: dram_si
  material: Silicon
  thickness: 41 um
  lateral_inset:
    x: 0.5 mm   # shorthand for x_minus = x_plus = 0.5 mm
    y: 0.5 mm
```

The explicit four-edge form `lateral_inset: {x_minus, x_plus, y_minus,
y_plus}` is also accepted; all four edges must be `>= 0`, and a builder-side
check rejects insets whose per-axis sum is greater than or equal to the
parent extent (which would erase the central entity).

When a layer carries a non-zero `lateral_inset`, the builder emits a
single central entity box at the inset coordinates plus up to four
lateral fill boxes (left / right / bottom / top strips) that share the
layer's z range. The fill material is the column's parent footprint
material, defaulting to `memory_zone.background_material` (currently
`Mold`). Every fill is tagged with `role: lateral_fill`, `fill_material`,
`parent_layer`, `parent_column` and `inset_side`; the central entity is
tagged with `lateral_inset_applied: true` and `parent_footprint`. A
`validate_layer_partition(...)` helper enforces: no 3D overlap between
central and any fill, no overlap between any two fills, complete area
coverage of the parent footprint at the layer's z range, and no fill
extending past the parent bounds.

The shipped benchmark uses `lateral_inset: {x: 0.5 mm, y: 0.5 mm}` on
the 11 regular DRAM layers and the 3 top DRAM layers. The HBM base die,
base BEOL and GPU-HBM uBump keep the full 11 × 11 mm footprint because
the base die is the outermost structure. The `memory_zone_background`
slab is no longer emitted: the four HBM parent footprints and the
central thermal-silicon footprint together tile the 30 × 22 mm memory
zone, and the per-layer mold fill is the only mold region in the scene.

## Build artifacts

`runs/` is git-ignored. The `build` command above regenerates the four
output files (`regions.csv`, `geometry_summary.json`, `top_view.png`,
`xz_section.png`, `yz_section.png`) locally on demand; they are not
committed.
