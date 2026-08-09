# Orthogonal HBM (MOSAIC) steady-state baseline provenance

This benchmark is the first paper-aligned geometry/material baseline for a
MOSAIC memory cube above the canonical GPU package. It is not a calibrated
temperature reproduction. The package, GPU stack, TIM, lid, ambient, HTC,
boundary conditions, and matrix-free PCG path are unchanged from the canonical
`hbm_on_gpu_12hi` baseline.

## PAPER_REPORTED

| Quantity | Value |
|---|---:|
| MOSAIC cube | 22 x 30 x 5.5 mm |
| Memory-die count | 98 |
| Die dimensions | 22 mm x 5.5 mm x 300 um |
| Si substrate | 293 um, 140 W/(m K) |
| BEOL | 5 um, 0.85 W/(m K) |
| DAA | 2 um, 0.2 W/(m K) |
| Adhesive | 3 um, 0.2 W/(m K) |
| Memory power | 1.6 W/die |

The die stack is exactly 300 um, and 98 adjacent dies occupy 29.4 mm of the
30 mm cube direction. The die planes are perpendicular to the GPU plane.

## MODELING_CHOICE

- Scalar conductivity is represented isotropically.
- Each die's 1.6 W is uniform by volume within its BEOL region.
- The 29.4 mm die array is centered in the 30 mm cube direction. The remaining
  0.6 mm is Mold, split into two 0.3 mm end caps.
- Fig. 2's 30 x 22 mm top-view footprint is mapped to global `x=30 mm`,
  `y=22 mm`, matching the canonical GPU. The local die-thickness direction is
  rotated onto global `+x` with the
  existing signed-axis-permutation material-frame mechanism. No alternate
  conductance implementation is introduced.
- There is no air gap or internal adiabatic cavity; unoccupied cube volume is
  explicit isotropic Mold at 3 W/(m K).
- A single 30 x 22 mm isotropic Adhesive layer lies between the GPU top and
  the complete MOSAIC cube bottom. Its thickness is 3 um and conductivity is
  0.2 W/(m K), as reported in Fig. 2 / Fig. 4. It is distinct from the 2 um
  DAA layers internal to every vertical memory die.

## UNRESOLVED

The paper also does not uniquely specify whether the 29.4 mm die array is
offset within the 30 mm cube direction. Centering is therefore an explicit
first-baseline modeling choice, not a paper-reported placement.
