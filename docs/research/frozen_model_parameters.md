# Frozen model parameters

Canonical inputs live in configs/cases, configs/platform, and configs/workload.
Lengths are SI metres unless unit strings are supplied; bandwidth is decimal
byte/s (8 bit/byte), capacity GiB uses 2**30 bytes, energy is J/bit, and
conductivity is W/(m K). Temperatures are solved in K and reported in degC.

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

GPU peak bandwidth is 4.8 TB/s; BF16 dense peak is 989.5 TFLOP/s.
Prefill GEMM and causal attention each use a 700 TFLOP/s reference calibration,
not a local measurement. Compute power uses the 525–700 W reference range
and subtracts the 74 W static term before deriving J/FLOP.

The primary host path is GH200 Grace/NVLink-C2C: 416.34 GB/s measured-reference
H2D throughput, 4.0 pJ/bit memory subsystem (16 W / 500 GB/s / 8) plus
1.3 pJ/bit link energy, totaling 5.3 pJ/bit. Host idle/static remains
UNRESOLVED. The optional PCIe/DDR reference uses 56.2 GB/s and Zhao et al.,
“Quantifying Interconnect Energy Efficiency on Perlmutter”, Tables II–III:
167.9 ± 10.5 pJ/bit PCIe and 4.93 W / (24.94 GB/s * 8) DDR energy.
It is an A100/DDR4 cross-platform reference, not an H200 measurement.
Detailed source records and claim status remain in the platform YAML.

The scientific chain is geometry/capacity -> workload/traffic/performance ->
memory/GPU power -> thermal mapping -> FP64 matrix-free GPU-PCG with Jacobi
preconditioning. Slab thickness/count and stack layers remain configurable.
HBM 2x1 comprises two thermal groups representing four physical stacks.
M3D operation primitives use Zhu/Tang sources recorded in the case YAML;
64-to-8-layer transferability is not validated. Missing logic background is
an explicit conditional assumption. Missing original power maps preclude
strict literature temperature reproduction. Distinguish PAPER_REPORTED,
DERIVED_FROM_PAPER_FIGURE, MODELING_CHOICE, and NUMERICAL_CHOICE.

Serving retains request-state/KV conservation, host offload, placement and
existing NMP attention/expert execution. Scope, overlap policy, fixed versus
growing context and unresolved workspace must accompany comparisons.
No transient physics, new MAC/GPU partition, or inferred zero-cost path is implied.

The frozen No-NMP 2.4 TB/s check uses rtol=1e-3, max_delta_t_K=1e-2,
check_interval=10, initial_temperature_K=293.15 and at most 100000 iterations.
HBM: 336.5696 W, 70.32849904391713 degC; M3D: 314.6770030529277 W,
69.31261100456737 degC. M3D mesh is (0.5, 1.0, 0.25) mm; HBM uses its case
mesh. Other formal paths retain their own existing solver tolerances.
