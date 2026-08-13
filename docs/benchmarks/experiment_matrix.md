# Canonical thermal experiment matrix

The six cases below are the only formal thermal results. Conventional HBM uses
Son23 component-aware vertical placement by default; Orthogonal MOSAIC retains
its 1.6 W/die uniform-BEOL model.

| Case ID | Architecture | Config | Run | GPU | Memory | Total | Tmax |
|---|---|---|---|---:|---:|---:|---:|
| `exp_conv_2x2_g414_m160` | Conventional 2x2 | `configs/legacy/exp_conv_2x2_g414_m160.yaml` | `runs/exp_conv_2x2_g414_m160` | 414 W | 160 W | 574 W | 122.9715 degC |
| `exp_conv_2x2_g300_m160` | Conventional 2x2 | `configs/legacy/exp_conv_2x2_g300_m160.yaml` | `runs/exp_conv_2x2_g300_m160` | 300 W | 160 W | 460 W | 102.3078 degC |
| `exp_conv_2x1_g414_m160` | Conventional 2x1 | `configs/legacy/exp_conv_2x1_g414_m160.yaml` | `runs/exp_conv_2x1_g414_m160` | 414 W | 160 W | 574 W | 120.4741 degC |
| `exp_conv_2x1_g300_m160` | Conventional 2x1 | `configs/legacy/exp_conv_2x1_g300_m160.yaml` | `runs/exp_conv_2x1_g300_m160` | 300 W | 160 W | 460 W | 100.4761 degC |
| `exp_orth_mosaic98_g414_m156p8_uniform` | Orthogonal MOSAIC | `configs/legacy/exp_orth_mosaic98_g414_m156p8_uniform.yaml` | `runs/exp_orth_mosaic98_g414_m156p8_uniform` | 414 W | 156.8 W | 570.8 W | 122.6727 degC |
| `exp_orth_mosaic98_g300_m156p8_uniform` | Orthogonal MOSAIC | `configs/legacy/exp_orth_mosaic98_g300_m156p8_uniform.yaml` | `runs/exp_orth_mosaic98_g300_m156p8_uniform` | 300 W | 156.8 W | 456.8 W | 99.1471 degC |

The result-first paper page is
[`thermal_results_overview.md`](thermal_results_overview.md).

## Conventional power accounting

Each physical 40 W stack uses 8 W logic and 32 W DRAM. Across four physical
stacks: logic = 32 W, DRAM = 128 W, HBM = 160 W. PHY and logic TSV/I/O map to
`HBM_Base_BEOL`; bank and DRAM TSV/I/O map to the corresponding `DRAM_BEOL`.
There is no direct Hybrid-Bonding or silicon-bulk heat generation.

## Base-die removal optimization cases

These cases supplement rather than replace the six formal baseline results.
Each physical stack retains its 40 um GPU-HBM uBump and 12 DRAM dies, removes
the 5 um base BEOL plus 50 um base silicon, and carries 32 W of DRAM power with
zero logic power.

| Case ID | Architecture | GPU | HBM | Total | Tmax |
|---|---|---:|---:|---:|---:|
| `exp_conv_2x2_nobase_g414_m128` | Conventional 2x2 no-base | 414 W | 128 W | 542 W | 116.9883 degC |
| `exp_conv_2x2_nobase_g300_m128` | Conventional 2x2 no-base | 300 W | 128 W | 428 W | 95.9839 degC |
| `exp_conv_2x1_nobase_g414_m128` | Conventional 2x1 no-base | 414 W | 128 W | 542 W | 111.6267 degC |
| `exp_conv_2x1_nobase_g300_m128` | Conventional 2x1 no-base | 300 W | 128 W | 428 W | 92.1853 degC |

The earlier 55 um top-Mold-closure outputs are retained only as
`superseded_diagnostic` runs and are not formal optimization results.

## Legacy uniform history

These four cases remain available only for regression/history and are excluded
from the formal table, figures, and paper comparison.

| Legacy ID | Run | Legacy Tmax |
|---|---|---:|
| `exp_conv_2x2_g414_m160_legacy_uniform` | `runs/exp_conv_2x2_g414_m160_legacy_uniform` | 120.9801 degC |
| `exp_conv_2x2_g300_m160_legacy_uniform` | `runs/exp_conv_2x2_g300_m160_legacy_uniform` | 100.3105 degC |
| `exp_conv_2x1_g414_m160_legacy_uniform` | `runs/exp_conv_2x1_g414_m160_legacy_uniform` | 118.7773 degC |
| `exp_conv_2x1_g300_m160_legacy_uniform` | `runs/exp_conv_2x1_g300_m160_legacy_uniform` | 98.7769 degC |

The former intermediate/stale runs `paper_check_mosaic`,
`orthogonal_hbm_orientation_fixed`, and `vlsi_12hi_hbm_paper_check` were
removed after their status was confirmed. Former shorthand case IDs are no
longer used.
