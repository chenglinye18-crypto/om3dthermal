# Current six-case thermal experiment matrix

This file is the authoritative context for discussions about the current
`om3dthermal` experiments. The six official cases cover three structures at
two GPU power levels. All use the same full 65 x 65 mm package domain,
30 x 22 mm GPU, uniform GPU FEOL power, current TIM/Lid/cooling boundaries,
and the matrix-free steady-state PCG solver. No power map, mesh sweep, or
parameter fitting is part of these results.

## Official cases and config/result mapping

| ID | Structure | Config | Run directory | GPU | Memory | Total |
|---|---|---|---|---:|---:|---:|
| O-414 | Orthogonal MOSAIC + Adhesive | `configs/orthogonal_hbm.yaml` | `runs/orthogonal_hbm_with_adhesive` | 414 W | 156.8 W | 570.8 W |
| O-300 | Orthogonal MOSAIC + Adhesive | `configs/orthogonal_hbm_paper_check.yaml` | `runs/orthogonal_hbm_paper_check` | 300 W | 156.8 W | 456.8 W |
| C22-414 | Conventional 2x2 HBM | `configs/hbm_on_gpu_12hi.yaml` | `runs/baseline` | 414 W | 160 W | 574 W |
| C22-300 | Conventional 2x2 HBM | `configs/vlsi_12hi_hbm_paper_check.yaml` | `runs/canonical_conventional_gpu300_check` | 300 W | 160 W | 460 W |
| C21-414 | Conventional y-merged 2x1 HBM | `configs/hbm_on_gpu_12hi_2x1.yaml` | `runs/hbm_on_gpu_12hi_2x1` | 414 W | 160 W | 574 W |
| C21-300 | Conventional y-merged 2x1 HBM | `configs/hbm_on_gpu_12hi_2x1_gpu300.yaml` | `runs/hbm_on_gpu_12hi_2x1_gpu300` | 300 W | 160 W | 460 W |

`runs/` is git-ignored; the configs, validation tests, and the consolidated
numbers below are committed.

## Structure definitions

### Orthogonal MOSAIC + Adhesive (O)

- MOSAIC footprint: 30 x 22 mm; height: 5.5 mm.
- 98 vertical memory dies arranged along global x over 29.4 mm.
- Each die spans global y=22 mm and z=5.5 mm; thickness direction is x and
  die plane is y-z.
- Per-die thickness: 293 um Si + 5 um BEOL + 2 um DAA = 300 um.
- A single 30 x 22 mm x 3 um isotropic Adhesive layer lies between GPU and
  MOSAIC.
- Remaining cube cavity is isotropic Mold. The die-array end margins are
  0.3 mm on each x side.
- Memory power: 98 x 1.6 W = 156.8 W, uniform in each die BEOL.

### Conventional 2x2 HBM (C22)

- Four 11 x 11 mm parent HBM footprints at centres
  (-9.5,+5.5), (-9.5,-5.5), (+9.5,+5.5), (+9.5,-5.5) mm.
- Each DRAM footprint is 10.8 x 10.8 mm, leaving a 0.1 mm Mold ring.
- A central 8 x 22 mm Thermal Silicon column separates left/right HBM pairs.
- Conventional 12Hi stack height: 775 um.
- Memory power: four equal 40 W sources = 160 W total, uniform over the
  twelve DRAM BEOL planes in each HBM.
- Layout metadata: `2x2_hbm`.

### Conventional y-merged 2x1 HBM (C21)

- Two groups at centres (-9.5,0) and (+9.5,0) mm.
- Each parent footprint is 11 x 22 mm; each continuous DRAM footprint is
  10.8 x 21.8 mm.
- The former top/bottom HBM pair on each side is connected across y=0. There
  is no internal y-direction Mold seam; the 0.1 mm Mold ring remains only on
  the outer perimeter.
- The central 8 x 22 mm Thermal Silicon column and 775 um vertical stack are
  unchanged from C22.
- Memory power: two equal 80 W sources = 160 W total, with the same uniform
  DRAM-BEOL placement semantics.
- Layout metadata: `2x1_hbm`.

## Final thermal results

| ID | Tmin (degC) | Global Tmax (degC) | Memory Tmax (degC) | Cells | Edges | Iterations | Relative residual | Relative power imbalance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| O-414 | 20.2240 | 122.6727 | 105.8671 | 1,518,468 | 4,466,056 | 1,679 | 9.7524e-7 | 1.9850e-7 |
| O-300 | 20.1739 | 99.1471 | 87.2945 | 1,518,468 | 4,466,056 | 1,656 | 9.9908e-7 | 2.6203e-7 |
| C22-414 | 20.2288 | 120.9801 | 120.4354 | 859,596 | 2,531,340 | 2,041 | 9.9517e-9 | 7.8868e-10 |
| C22-300 | 20.1818 | 100.3105 | 99.9167 | 859,596 | 2,531,340 | 2,022 | 9.9131e-9 | 8.9945e-10 |
| C21-414 | 20.2284 | 118.7773 | 118.1897 | 859,596 | 2,531,340 | 2,021 | 9.9065e-9 | 6.0280e-10 |
| C21-300 | 20.1814 | 98.7769 | 98.3527 | 859,596 | 2,531,340 | 2,004 | 9.8629e-9 | 6.5386e-10 |

All six configs declare `rtol=1e-6`. The table reports the residual actually
reached by each solve. All six converged and have finite temperatures and
closed power balance.

## Hotspots and boundary heat flow

| ID | Global hotspot (component/material; x,y,z mm) | Lid out | Laminate out |
|---|---|---:|---:|
| O-414 | GPU/FEOL; (14.850,-0.250,0.37179) | 550.2951 W | 20.5047 W |
| O-300 | GPU/FEOL; (14.850,+0.250,0.37179) | 440.8118 W | 15.9881 W |
| C22-414 | GPU/FEOL; (-11.727,-0.050,0.37179) | 554.7392 W | 19.2608 W |
| C22-300 | GPU/FEOL; (-11.977,+0.050,0.37179) | 444.7563 W | 15.2437 W |
| C21-414 | GPU/FEOL; (-11.977,+0.248,0.37179) | 554.7887 W | 19.2113 W |
| C21-300 | GPU/FEOL; (-12.223,+0.248,0.37179) | 444.7965 W | 15.2035 W |

## Direct comparisons

- O-300 minus O-414 Tmax: -23.5256 degC.
- C22-300 minus C22-414 Tmax: -20.6696 degC.
- C21-300 minus C21-414 Tmax: -20.0004 degC.
- C21-414 minus C22-414 Tmax: -2.2028 degC.
- C21-300 minus C22-300 Tmax: -1.5336 degC.
- O-300 versus the approximately 81.3 degC MOSAIC reference: +17.8471 degC.
- C22-300 versus the approximately 80.0 degC conventional reference:
  +20.3105 degC. This C22 comparison is intentionally the canonical geometry
  with only GPU power changed; it is not a separate two-cube 660 um geometry.

## Excluded/stale artifacts: do not use as official cases

- `runs/paper_check_mosaic`: Minimax legacy run using the wrong conventional
  2x2 geometry; its 99.792 degC result is invalid for Orthogonal MOSAIC.
- `runs/orthogonal_hbm_orientation_fixed`: orientation-corrected run before
  the paper-reported 3 um Adhesive was added.
- `runs/vlsi_12hi_hbm_paper_check`: abandoned two-cube 22 x 11 mm / 660 um
  modelling attempt. The user replaced it with the canonical C22 power-only
  comparison represented by `runs/canonical_conventional_gpu300_check`.

When discussing results, identify the case by ID and config path, not merely
by the labels MOSAIC, VLSI, or conventional, to avoid case/config mixing.
