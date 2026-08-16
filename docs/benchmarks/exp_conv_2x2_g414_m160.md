# exp_conv_2x2_g414_m160

This experiment keeps the geometry, materials, mesh, boundary conditions,
solver, GPU power, and total HBM power of
`exp_conv_2x2_g414_m160_legacy_uniform`. Son23 component-aware placement is
the canonical conventional power model; geometry and total power are unchanged.

## Provenance

- `PAPER_REPORTED`: Son et al., EDAPS 2023, 30 W per-stack partition:
  logic PHY 4 W, logic TSV 2 W, and each of 12 DRAM dies bank 1.5 W plus
  TSV 0.5 W.
- `DERIVED_FROM_REFERENCE`: scale every component by 40/30 for the current
  40 W stack.
- `MODELING_CHOICE`: map PHY and logic TSV to the existing `HBM_Base_BEOL`;
  map each bank and DRAM TSV source to that die's existing `DRAM_BEOL`.
- `UNRESOLVED_DO_NOT_GUESS`: lateral PHY, TSV, and bank dimensions. No lateral
  floorplan is introduced in this phase.

## Resolved 40 W per-stack accounting

| Component | Power |
|---|---:|
| Logic PHY | 5.333333333 W |
| Logic TSV | 2.666666667 W |
| Logic total | 8 W |
| DRAM bank per die | 2 W |
| DRAM TSV per die | 0.666666667 W |
| DRAM total per die | 2.666666667 W |
| 12 DRAM total | 32 W |
| Stack total | 40 W |

Across four stacks, logic is 32 W, DRAM is 128 W, and HBM is 160 W. With
the unchanged 414 W GPU, package input is 574 W.

## Nominal steady-state result

- Cells / internal edges: 859,596 / 2,531,340.
- Tmin / Tmax: 20.2332 / 122.9715 degC.
- Global hotspot: GPU FEOL at (11.9773,-0.0500,0.37179) mm.
- HBM Tmax: 122.4325 degC.
- DRAM-region Tmax: 117.8021 degC; DRAM_BEOL Tmax: 117.3315 degC.
- Lid / laminate heat out: 554.4279 / 19.5721 W.
- Thermal-resistance-network relaxation: 2,040 iterations; relative residual 9.8698e-9.
- Relative power imbalance: 8.0487e-10.
- Delta Tmax versus total-power-matched uniform case: +1.9914 degC.
