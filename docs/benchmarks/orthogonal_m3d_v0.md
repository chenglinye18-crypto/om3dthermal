# Orthogonal M3D-eDRAM v0

Status: **geometry and thermal-property bookkeeping closed; the formal
steady-state cases use a homogenized bit-cell stack**.

## Slab cross-section

The 98 slabs reuse the Orthogonal MOSAIC placement and signed-axis
permutation: each slab lies in global y-z, spans 22 x 5.5 mm, and has its
300 um thickness/pitch along global x. The array length is 29.4 mm inside the
30 mm cube direction.

| Region order | Thickness | Provenance |
|---|---:|---|
| Si substrate | 292.546 um | `DERIVED_FROM_GEOMETRY_CLOSURE` |
| FEOL | 0.150 um | `MODELING_CHOICE`; reuses current GPU FEOL thickness |
| 8-layer M3D bit-cell stack | 2.304 um | One homogenized thermal region; 288 nm/layer is `DERIVED_FROM_PAPER_FIGURE` from Tang et al. Fig. 4 |
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

For the canonical thermal mesh, the eight 288 nm physical layers and the
3.0 um BEOL interconnect are combined into one 5.304 um
`M3D_Bitcell_BEOL` box per slab. Both source regions use the same modeled
thermal conductivity, 0.85 W/(m K). Their powers are summed before thermal
mapping and distributed as one uniform volumetric source over the merged
region, matching the uniform active-memory-region assumption used by the
other canonical architectures. The merge preserves total thickness,
conductivity, volume, and total power while removing one global cut plane per
slab. Electrical bitcell and interconnect accounting remains separate for
capacity and operation-energy decomposition.

## Power target

The formal default thermal mode is operation-energy based. At the matched
39.2 Tb/s delivered-bandwidth reference, an all-1 read uses 368 fJ/bit and
produces 14.4256 W of array-core power. Its only carrier is the homogenized
eight-layer `m3d_bitcell_stack`. Si substrate, FEOL, 3 um interconnect, and DAA
receive zero direct memory power. The former `iso_total = 156.8 W` mode remains
available only as a legacy/isopower control.

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

Both power modes map only to the homogenized M3D bit-cell stack. This is
equivalent to uniform average power across its eight constituent layers. The
reported 256 TOPS/W and 50 TOPS/mm2 CiM
metrics are not used to derive thermal power or per-bit power.

## Thermal-property modeling choice

The heterogeneous bit-cell stack and the 3 um BEOL interconnect are each
modeled isotropically at 0.85 W/(m K). This is a `MODELING_CHOICE`; it is not
identified as an IGZO-only material property.

## Capacity bookkeeping

Capacity uses the reported 30 Mb/mm2/layer density and a nominal
`slab_array_fill_factor = 1.0`; it does not invert the 0.023 um2 cell area.

- slab area: **121 mm2**
- 8-layer capacity/slab: **29,040 Mb = 29.04 Gb**
- 98-slab cube: **2,845.92 Gb = 355.74 GB**

This is upper-bound architecture bookkeeping and does not affect the thermal
operator.

## Formal M3D-v1 steady-state results

The formal cases use the single homogenized 2.304 um bit-cell region described
above. Temperatures are therefore stack-region values, not resolved physical
sub-layer values.

| Case | GPU | Array core | Package | Global Tmax | Bit-cell stack Tmax |
|---|---:|---:|---:|---:|---:|
| `exp_orth_m3d8_g414_bw39p2_all1read` | 414 W | 14.4256 W | 428.4256 W | 107.0909 degC | 87.2895 degC |
| `exp_orth_m3d8_g300_bw39p2_all1read` | 300 W | 14.4256 W | 314.4256 W | 83.5425 degC | 69.2263 degC |

Both global hotspots are in GPU FEOL near (14.85, 0.25, 0.37179) mm. The
GPU-power delta is 23.5484 K. These are first-order array-core thermal
baselines; peripheral, controller, interconnect-direct, refresh, and hold
power remain excluded.
