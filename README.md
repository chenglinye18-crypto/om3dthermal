# om3dthermal

`om3dthermal` is an axis-aligned 3D thermal modelling and steady-state solving
tool. It reads unit-aware YAML, builds conventional and orthogonal MOSAIC
geometry, generates a boundary-preserving block mesh, maps uniform power,
constructs conductance and boundary operators, and solves with matrix-free
PCG. It does not include transient solving, temperature-dependent materials,
AMR, non-uniform power-map calibration, COMSOL/Icepak integration, or a GUI.

The current six-case experiment matrix and the authoritative mapping from
configs to results are documented in
[`docs/benchmarks/experiment_matrix.md`](docs/benchmarks/experiment_matrix.md).

## Install and run

Python 3.10+ is required:

```bash
python -m pip install -e ".[test]"
python -m pytest
python -m om3dthermal.cli build configs/hbm_on_gpu_12hi.yaml --out runs/hbm12_iedm2025
python -m om3dthermal.cli discretize configs/hbm_on_gpu_12hi.yaml --out runs/hbm12_mesh
python -m om3dthermal.cli conductance configs/hbm_on_gpu_12hi.yaml --out runs/hbm12_conductance
python -m om3dthermal.cli solve-steady configs/hbm_on_gpu_12hi.yaml --out runs/hbm12_steady --method pcg
```

The `build` command writes (under the directory given by `--out`, which
is git-ignored by default — see "Build artifacts" below):

- `regions.csv` — SI coordinates, material, tags, source path and reserved
  rotation for every box;
- `geometry_summary.json` — box / material counts, stack heights, component
  bounds and extremal dimensions;
- `top_view.png` — footprint names and mm coordinates;
- `xz_section.png`, `yz_section.png` — z-stacked material layers in µm.

The `discretize` command writes the block-structured mesh artifacts (see
"Block-structured thermal discretisation" below):

- `thermal_cells.csv` — one row per cell, with grid indices, SI coordinates,
  material, parent-box provenance and tags;
- `adjacency_edges.csv` — face-shared neighbour pairs with geometric
  quantities (face area, half distances, interface coordinate, material
  interface flag);
- `boundary_faces.csv` — cell faces with no neighbour, classified as
  `scene_outer_boundary` or `exposed_internal_boundary`;
- `mesh_summary.json` — cut / cell / edge / boundary counts, volume and
  surface-area conservation, material / component / axis breakdowns, and
  build / adjacency timings.

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

### Configuration format

The shipped `configs/hbm_on_gpu_12hi.yaml` is a **compact** form
(≈87 lines) optimised for hand-editing. The top-level blocks are
`materials`, `geometry`, `stacks`, `mesh`, `boundary`, `power`,
`solver`. Paper citations and per-parameter modelling notes are
recorded separately in
[`docs/benchmarks/hbm_on_gpu_12hi.md`](docs/benchmarks/hbm_on_gpu_12hi.md)
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
  (see `docs/benchmarks/hbm_on_gpu_12hi.md`). Fig. 3(a) of the
  IEDM 2025 paper indicates approximately 1 mm of total mold-filled
  width around each HBM stack, so the locked value leaves 0.1 mm
  per side for the mold ring. The paper does not report a complete
  per-edge DRAM footprint.

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

## Block-structured thermal discretisation

Once the `AxisAlignedBox` scene is built, `discretize` partitions the
material regions into a regular, conformally meshed set of `ThermalCell`
nodes and emits the face adjacency graph and the boundary face
inventory. **No thermal state (temperature, power, conductance,
resistance) is computed here** — only the geometric input a future
KCL / steady-state solver would consume.

### Data flow

```
YAML
  -> AxisAlignedBox Scene
  -> Global Cut Planes  (per-axis union of every box boundary
                         + per-box uniform subdivisions <= max_cell_size)
  -> ThermalCell        (one per voxel inside a box)
  -> AdjacencyEdge      (face-shared neighbour pairs)
  -> BoundaryFace       (cell faces with no neighbour)
  -> future conductance matrix
  -> future steady-state solver
```

### Why global cut planes

Each axis keeps a single strictly-increasing array of cut positions.
Every `AxisAlignedBox` boundary (`x0/x1/y0/y1/z0/z1`) is a cut. For each
box interior, a uniform subdivision is added so no resulting cell
exceeds `max_cell_size`. Cuts within `_LENGTH_TOL` are merged (so a
real material boundary that happens to coincide with a subdivision
plane is not duplicated). Because the cuts contain every material
boundary, the mesh is **conformal**: a T-junction where one large face
meets two smaller ones is automatically split into two smaller
adjacency edges whose `face_area`s sum to the original big face.

### What gets exported

| File | Row = | Carries |
|------|-------|---------|
| `thermal_cells.csv`    | one cell        | `(ix, iy, iz)`, SI extents, `material`, `parent_box_id`/`_name`, `component`, `source_path`, `rotation`, `tags` (JSON) |
| `adjacency_edges.csv`  | one shared face | `axis`, `interface_coordinate`, `face_area`, `half_distance_a`/`_b`, `center_distance`, `material_a`/`_b`, `is_material_interface` |
| `boundary_faces.csv`   | one open face   | `axis`, `side`, `coordinate`, `area`, `normal`, `component`, `material`, `classification` |
| `mesh_summary.json`    | — (single object) | counts, conservation metrics, timings, breakdowns by material / component / axis |

### Configuration

`SimulationConfig.discretization` is optional. When present, the
discretiser refuses `preserve_box_boundaries: false` rather than
silently merging real material boundaries. The shipped benchmark uses:

```yaml
discretization:
  max_cell_size:
    x: 0.5 mm   # PAPER_REPORTED_POWER_MAP_RESOLUTION (IEDM Fig. 3 power map grid)
    y: 0.5 mm
    z: 100 um   # MODELING_CHOICE
  preserve_box_boundaries: true
```

The 0.5 mm x/y is the published IEDM power-map grid; the 100 µm z is a
modelling choice that ensures every layer thicker than 100 µm is
subdivided, while 0.15 µm FEOL, 1 µm oxide and 1.4 µm BEOL_MXY layers
are kept intact as single cells along z.

### Conservation guarantees

The discretiser enforces two hard invariants at write time:

1. **Volume conservation** — for every `AxisAlignedBox`, the sum of its
   child `ThermalCell` volumes equals the box volume (within a
   relative tolerance of `1e-9`); the scene total of cell volumes
   equals the scene total of box volumes.
2. **Surface area partition** — for every `ThermalCell`, the sum of
   shared face areas (counted on this cell's side) plus the sum of
   boundary face areas equals the analytic surface area
   `2 * (sx*sy + sx*sz + sy*sz)`.

A geometry overlap between two boxes is a hard
`GeometryOverlapError` that names both boxes and the offending voxel
— never a silent priority override.

### Algorithm complexity

Cell generation walks the global cut grid once per box:
`O(sum_over_boxes(voxels_in_box))`. Adjacency and boundary face
construction walk the integer grid once per cell: `O(N_cells)`. The
worst case is bounded by the per-axis cut count; no `O(N²)` pairwise
cell comparison is ever performed.

### Not implemented yet

The discretiser emits the geometry only. The following are explicit
non-goals of this stage and will live in a downstream thermal-solver
module:

- material conductivity tensors in world coordinates;
- face / cell conductance `G_ij`, normal conductance, interface
  resistance;
- power mapping (the paper's 0.5 mm power map is recorded in
  `metadata` but not yet assigned to cells);
- boundary conditions (adiabatic, HTC, fixed temperature);
- KCL / steady-state temperature solve;
- adaptive mesh refinement.

## Internal face thermal conductance

The `conductance` CLI command runs the discretiser then computes the
per-edge two-point face thermal conductance and writes a columnar
`ConductanceTable`. The mesh is still purely geometric from
`discretize`; `conductance` is the first place material physics
enters the picture. **No boundary conditions, power, KCL matrix or
temperature solve is implemented here** — only the per-edge numbers
that a future steady-state solver would consume.

### Data flow

```
YAML
  -> AxisAlignedBox Scene
  -> ThermalCell mesh
  -> AdjacencyEdge
  -> Material tensor projection
  -> ConductanceTable
  -> future KCL assembly
  -> future steady-state temperature
```

### Physics

For each material the local conductivity tensor is diagonal:

```
K_local = diag(kx, ky, ky)    # units W/(m*K)
```

The cell's rotation matrix `R` carries the material to world
coordinates:

```
K_global = R K_local R^T
```

For an axis-aligned face with normal `n`, the per-cell normal
conductivity collapses to a one-line expression

```
k_n = sum_m k_local[m] * R[n, m]^2
```

For a shared face between cells `a` and `b` with face area `A`,
half-extents `d_a` / `d_b` along the normal, and an optional per-pair
areal interface resistance `R''` (m²·K/W):

```
G_ab = A / ( d_a / k_na + R''_ab + d_b / k_nb )   [W/K]
R_ab = 1 / G_ab                                  [K/W]
```

Hard invariants enforced at write time:

- `G_ab > 0`, finite, symmetric under `(a, b)` swap;
- `A > 0`, `d_a > 0`, `d_b > 0`, `k_na > 0`, `k_nb > 0`,
  `R''_ab >= 0`;
- materials with `k_local is None` raise
  `MissingThermalConductivityError` naming the cell, parent box and
  edge id.

### Rotation support

This stage supports **signed axis permutations only**: identity and
0/90/180/270-degree rotations about any axis. Arbitrary-angle
rotations are rejected with
`UnsupportedMaterialRotationError` because the two-point face
conductance

```
k_n = n^T K_global n
```

captures the normal flux but not the tangential coupling terms a full
anisotropic FVM would include. Supporting general rotations requires
a non-orthogonal / multi-point flux scheme that is explicitly out of
scope of this stage. The shipped benchmark uses the identity
rotation everywhere.

### Interface resistance rules

Per-material-pair `R''` is configured under
`thermal_conductance.interfaces`. The pair `(A, B)` is treated as
**unordered**; `[A, B]` and `[B, A]` are the same rule and a
duplicate is rejected. The `default_interface_areal_resistance`
field is the fallback when no explicit pair rule matches. The
shipped HBM benchmark uses an empty `interfaces` list and
`default_interface_areal_resistance: 0 m^2*K/W` because Hybrid
Bonding, TIM, BEOL, Oxide and Mold are already modelled as
finite-thickness material layers and an additional contact `R''`
would double-count them.

### Configuration

```yaml
thermal_conductance:
  rotation_policy: axis_aligned_only
  default_interface_areal_resistance: 0 m^2*K/W
  interfaces: []   # list of { materials: [A, B], areal_resistance: <value>, metadata: ... }
```

`ArealThermalResistance` accepts strings in SI m²·K/W (e.g.
`"0 m^2*K/W"`, `"1e-8 m^2*K/W"`, `"2 mm^2*K/W"`) or bare numbers
(SI). Negative, NaN, infinite and non-`m²·K/W` values are rejected
at parse time.

### Output

`om3dthermal.cli conductance configs/hbm_on_gpu_12hi.yaml --out runs/hbm12_conductance`
writes (under git-ignored `runs/`):

- the three discretisation CSVs and the mesh summary (re-emitted for
  self-containment);
- `conductance_edges.npz` — columnar NumPy arrays, one entry per
  edge (the canonical machine-readable form);
- `conductance_summary.json` — counts, k_n / G / R ranges, the
  cache entry count, default / per-pair `R''` usage and timings;
- `conductance_edges.csv` — only when `--write-conductance-csv` is
  passed (the benchmark is ~790 k rows; CSV is wasteful when NPZ is
  available).

### ConductanceTable columns

| Column | Type | Meaning |
|--------|------|---------|
| `edge_id` | int64 | index in the adjacency edge list |
| `cell_a` / `cell_b` | int64 | cell ids |
| `axis` | int8 (0=x, 1=y, 2=z) | normal axis of the shared face |
| `face_area_m2` | float64 | shared face area |
| `half_distance_a_m` / `half_distance_b_m` | float64 | half-extent along the normal |
| `k_normal_a_W_mK` / `k_normal_b_W_mK` | float64 | per-cell normal conductivity |
| `interface_areal_resistance_m2K_W` | float64 | `R''` for the (a, b) material pair |
| `resistance_K_W` | float64 | `R_ab` in K/W |
| `conductance_W_K` | float64 | `G_ab` in W/K |
| `material_interface` | bool | `material_a != material_b` |
| `interface_rule_index` | int32 | which rule was used (`-1` for default) |

### Algorithm complexity

`build_conductance_table` walks the edge list once. Per-edge cost is
`O(1)` thanks to a per-`(material, canonical_rotation, axis)` `k_n`
cache (15 materials × 4 distinct rotation signatures × 3 axes = at
most 45 cache slots for the benchmark) and a pre-built unordered-pair
`R''` dict. The conductance arithmetic is done in vectorised NumPy
over a Python loop of one iteration per edge.

## Matrix-free steady-state solver

The `solve-steady` CLI command runs the discretiser + conductance
+ boundary links + power + matrix-free steady-state solver, all
without ever materialising a dense or sparse matrix on the
production path. **No boundary conditions, power mapping, KCL
matrix, or temperature solve is implemented here** without
explicit physics; the output is the per-cell temperature vector
a future nonlinear steady-state solver would consume.

### Data flow

```
YAML
  -> AxisAlignedBox Scene
  -> ThermalCell mesh
  -> Internal Conductance
  -> BoundaryLink / PowerVector
  -> MatrixFreeThermalOperator
  -> Weighted Jacobi or PCG
  -> Steady-State Temperature
```

### Matrix-free operator

The per-edge conductance table is consumed directly. For each
internal edge ``(a, b)`` we accumulate ``G_ab * (T_a - T_b)``
into ``(A T)_a`` and subtract it from ``(A T)_b``; for each active
boundary link on cell ``i`` we add ``G_ib * T_i``. The
implementation uses `numpy.bincount` / `numpy.add.at`, so a single
``apply(T)`` is ``O(N_edges + N_boundary_links)`` with no Python
per-edge object creation. The diagonal ``D = diag(A)`` and the
right-hand side ``b = P + sum_b G_ib * T_ref`` are precomputed once
when the operator is built, so an iteration is one ``apply`` plus
trivial vector arithmetic.

### Solvers

Two solvers share the same matrix-free operator:

- **Weighted Jacobi** with the textbook local-residual step

  ```
  T_new = T + omega * (b - A T) / D        omega in (0, 1]
  ```

  Convergence requires **both** ``||b - A T|| < rtol * ||b||`` and
  ``max |delta T| < temperature_update_tolerance`` to be satisfied
  in the same check window. NaN / inf and temperatures below 0 K
  are divergence signals that abort the iteration.

- **Matrix-free PCG** (default) is the standard
  `scipy.sparse.linalg.cg` wrapped around the same
  ``apply`` as a `LinearOperator.matvec`, with a Jacobi
  preconditioner ``M^-1 x = x / D``. PCG is what the shipped HBM
  benchmark uses because the convergence rate is far better than
  Jacobi on 272 k cells.

Jacobi iteration count has **no time meaning**: it is the number
of linear-algebra refinements, not seconds of physical time. A
transient / capacity-based extension is explicitly out of scope
of this stage.

### Boundary heat link

The face temperature is unknown (it sits at the cell boundary, not
the cell centre), so the per-link resistance includes the half-cell
bulk:

```
R_cell_to_ambient = d_i / (k_n A) + R'' / A + 1 / (h A)   [convection]
R_cell_to_face    = d_i / (k_n A) + R'' / A               [fixed T]
G_i               = A / R                                  [W/K]
```

`R''` defaults to `0 m^2*K/W`; adiabatic faces contribute
`G = 0` and are excluded from the active `BoundaryLinkTable`.

### Power source mapping

Only `uniform_volume` is supported. ``P_i = P_total * V_i /
sum V_selected``; multiple sources covering the same cell are
additive. Total mapped power must equal configured total within
`1e-9` relative tolerance.

### Anchored-component check

Before the solver runs, `validate_anchored_components` runs a
union-find over the internal edge list and flags every connected
component that has at least one non-adiabatic boundary link.
Components with only adiabatic faces are rejected with
`UnanchoredThermalComponentError` because their temperature is
not uniquely defined. The error names the offending component
indices, cell counts, representative cell ids, and total power
(even if the power is zero — the singularity remains).

### Global energy balance

After the solve, the summary reports

- `Q_input = sum P_i` (must equal the configured power total);
- `Q_boundary = sum_b G_ib * (T_i - T_ref)` (the heat leaving
  through active links);
- `global_power_imbalance = Q_input - Q_boundary`;
- `relative_power_imbalance = |imbalance| / max(|Q_input|, eps)`.

A converged steady state drives this to floating-point precision;
the shipped HBM benchmark's `relative_power_imbalance` is below
`1e-7` once PCG reaches its tolerance.

### HBM benchmark (paper-parameter-aligned uniform-power baseline)

`om3dthermal.cli solve-steady configs/hbm_on_gpu_12hi.yaml --out runs/hbm12_steady --method pcg --rtol 1e-8 --max-iterations 2000`

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
| Solver | PCG with Jacobi preconditioner |
| Iteration cap | 2 000 |
| Final relative residual | 5.41 × 10⁻⁸ |
| Final absolute residual | 6.11 × 10⁻⁶ W |
| Relative power imbalance | 1.07 × 10⁻⁸ |
| Min / max / mean T | 293.40 / 409.18 / 344.12 K |
| Min / max T in °C | 20.25 / 136.03 °C |
| Hottest cell | id 58 553 at 409.18 K |
| Discretisation time | ~6.5 s |
| Boundary build time | ~16.3 s |
| Operator build time | ~0.22 s |
| PCG solve time | ~30 s |
| Total wall time | ~46 s |

`benchmark_label = "paper-parameter-aligned uniform-power baseline"`,
`strict_paper_temperature_reproduction = false`. The result is
explicitly **not** a claim that the code reproduces the paper's
141.7 °C number: the paper does not publish the per-layer 0.5 mm
non-uniform power map, and the GPU power is allocated to the FEOL
layer and the HBM power to the DRAM_BEOL layer as a modelling
choice.

### Not implemented yet

The matrix-free steady-state solver emits the per-cell temperature
vector. The following are explicit non-goals of this stage and
will live in a downstream thermal-solver module:

- transient heat capacity / time stepping;
- temperature-dependent `k(T)` and `P(T)` (would require an
  outer Picard / Newton loop);
- radiation;
- arbitrary-angle non-orthogonal FVM with full off-diagonal
  flux coupling;
- adaptive mesh refinement;
- inlet / outlet fluid loops for two-phase cooling;
- multi-physics coupling (mechanical, electrical).

## Build artifacts

`runs/` is git-ignored. The `build` command above regenerates the four
output files (`regions.csv`, `geometry_summary.json`, `top_view.png`,
`xz_section.png`, `yz_section.png`) locally on demand; they are not
committed. The `discretize` command regenerates the four mesh files
(`thermal_cells.csv`, `adjacency_edges.csv`, `boundary_faces.csv`,
`mesh_summary.json`) likewise locally. The `conductance` command
re-emits the four mesh files plus `conductance_edges.npz` and
`conductance_summary.json` (and `conductance_edges.csv` only when
`--write-conductance-csv` is set). The `solve-steady` command
runs the matrix-free steady-state thermal solver and writes
`temperature_cells.npz`, `temperature_cells.csv`,
`boundary_heat_flows.csv`, `power_cells.npz`,
`solver_history.csv` and `steady_state_summary.json`.
