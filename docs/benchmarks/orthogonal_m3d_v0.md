# Orthogonal M3D-eDRAM v0

Status: **geometry bookkeeping closed; effective thermal conductivity
unresolved; no formal solve**.

## Slab cross-section

The 98 slabs reuse the Orthogonal MOSAIC placement and signed-axis
permutation: each slab lies in global y-z, spans 22 x 5.5 mm, and has its
300 um thickness/pitch along global x. The array length is 29.4 mm inside the
30 mm cube direction.

| Region order | Thickness | Provenance |
|---|---:|---|
| Si substrate | 292.546 um | `DERIVED_FROM_GEOMETRY_CLOSURE` |
| FEOL | 0.150 um | `MODELING_CHOICE`; reuses current GPU FEOL thickness |
| 8-layer M3D bit-cell stack | 2.304 um | 288 nm/layer is `DERIVED_FROM_PAPER_FIGURE` from Tang et al. Fig. 4 |
| BEOL interconnect | 3.0 um | `MODELING_CHOICE` |
| DAA | 2.0 um | inherited from current Orthogonal MOSAIC model |

Closure:

`292.546 + 0.150 + (8 x 0.288) + 3.0 + 2.0 = 300.000 um`.

Eight layers are a `PAPER_SUPPORTED_MODELING_CHOICE`: the paper evaluates
2/4/6/8-layer stacking, and v0 selects eight as nominal. The measured 288 nm
pitch is not labeled paper-reported.

DAA occurs once in the orthogonal slab-pitch cross-section and is not inserted
between M3D bit-cell layers. The M3D-BEOL internal order is bit-cell stack then
interconnect.

## Power target

The default thermal mode remains `iso_total = 156.8 W`. Its declared carrier
is only the eight-layer `m3d_bitcell_stack`, uniform across those layers. Si
substrate, FEOL, 3 um interconnect, and DAA receive zero direct memory power.

An optional workload-dependent 2T0C operation-energy model records these
`PAPER_REPORTED` values without averaging states or transitions:

| Operation | Energy |
|---|---:|
| read 0 | 0.60 fJ/bit |
| read 1 | 368.0 fJ/bit |
| write 0->0 | 0.30 fJ/bit |
| write 0->1 | 0.37 fJ/bit |
| write 1->0 | 0.58 fJ/bit |
| write 1->1 | 0.24 fJ/bit |
| refresh 0 | 0.90 fJ/bit |
| refresh 1 | 370.0 fJ/bit |
| hold | 4.26e-15 W/active row |

Operation energy is not power. The helper multiplies each state/transition
operation rate by its energy and converts fJ to J using 1e-15. Refresh rate is
the modeled capacity divided by the resolved refresh period; hold power is
active rows times the reported per-row value. Read/write rates, state and
transition probabilities, refresh period, refresh-state probabilities, and
active-row count must all be supplied by a workload before this mode can
produce watts. No 50/50 states, bandwidth, read/write ratio, refresh period, or
active-row default is assumed.

Both power modes map only to the M3D bit-cell stack and distribute the result
uniformly across its eight layers. The reported 256 TOPS/W and 50 TOPS/mm2 CiM
metrics are not used to derive thermal power or per-bit power.

## Unresolved thermal property

The combined bit-cell/interconnect region is heterogeneous. A standalone IGZO
conductivity is therefore not used as the M3D-BEOL property. The template
requires an effective anisotropic model with both values unresolved:

- `m3d_beol.thermal.k_in_plane_W_mK`
- `m3d_beol.thermal.k_cross_plane_W_mK`

`load_orthogonal_m3d_template()` accepts the geometry bookkeeping, while the
production `load_config()` path lists these parameters and refuses geometry,
mesh, or solve.

## Capacity bookkeeping

Capacity uses the reported 30 Mb/mm2/layer density and a nominal
`slab_array_fill_factor = 1.0`; it does not invert the 0.023 um2 cell area.

- slab area: **121 mm2**
- 8-layer capacity/slab: **29,040 Mb = 29.04 Gb**
- 98-slab cube: **2,845.92 Gb = 355.74 GB**

This is upper-bound architecture bookkeeping and does not affect the thermal
operator. No M3D thermal solve has been run.
