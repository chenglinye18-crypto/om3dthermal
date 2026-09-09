# Frozen model parameters

This is the single active summary of power parameters used by the No-NMP
thermal comparison.

| Parameter | Frozen value | Status | Source |
|---|---:|---|---|
| GPU static power | 74 W | modeling choice | `configs/platform/gpu_package_h200_reference.yaml` |
| GPU decode dynamic energy | 11.68 pJ/bit | modeling choice | `configs/platform/gpu_package_h200_reference.yaml` |
| HBM full-row read | 0.978 pJ/bit | frozen input | `configs/cases/conventional_hbm_2x1.yaml` |
| HBM closed-row read | 3.013 pJ/bit | frozen input | `configs/cases/conventional_hbm_2x1.yaml` |
| HBM nominal read | 1.9955 pJ/bit | arithmetic mean | `(0.978 + 3.013) / 2` |
| M3D read | 0.8552605757 pJ/bit | analytical model | `configs/cases/orthogonal_m3d_igzo.yaml` |
| HBM/M3D memory static power | 0 W | modeling choice | canonical case files |
| M3D lateral sidebars | Cu, 400 W/(m K) | modeling choice | `configs/cases/orthogonal_m3d_igzo.yaml` |

The thermal cache is validated by geometry, mesh, materials, boundary
conditions, discretization, and schema version. Power and bandwidth are
solve-time RHS inputs and are not stored in the fixed operator cache.
