# Conditional Architecture Decode Memory Energy Audit (v0)

**Workload:** Frozen LLaMA-3.1-8B-class (B=1, S=131072, 16-bit)
**Scenario:** Matched-reference sensitivity
**Scope:** Memory dynamic traffic energy only

| Architecture | rho | Capacity Feasible | Status | Eread (pJ/bit) | Ewrite (pJ/bit) | Read (J/token) | Write (J/token) | Total (J/token) |
|--------------|-----|-------------------|--------|----------------|-----------------|----------------|-----------------|-----------------|
| conventional_hbm_2x1 | 0 | True | EVALUATED_CONDITIONAL_ARCHITECTURE_MEMORY_ENERGY | 1.3971 | 0.0000 | 3.708442e-01 | 0.000000e+00 | 3.708442e-01 |
| conventional_hbm_2x1 | 1 | True | EVALUATED_CONDITIONAL_ARCHITECTURE_MEMORY_ENERGY | 1.3971 | 1.3971 | 3.708442e-01 | 1.464963e-06 | 3.708457e-01 |
| conventional_hbm_2x1 | 100 | True | EVALUATED_CONDITIONAL_ARCHITECTURE_MEMORY_ENERGY | 1.3971 | 139.7098 | 3.708442e-01 | 1.464963e-04 | 3.709907e-01 |
| conventional_hbm_2x1 | 1000 | True | EVALUATED_CONDITIONAL_ARCHITECTURE_MEMORY_ENERGY | 1.3971 | 1397.0980 | 3.708442e-01 | 1.464963e-03 | 3.723092e-01 |
| orthogonal_si | 0 | True | EVALUATED_CONDITIONAL_ARCHITECTURE_MEMORY_ENERGY | 1.3677 | 0.0000 | 3.630291e-01 | 0.000000e+00 | 3.630291e-01 |
| orthogonal_si | 1 | True | EVALUATED_CONDITIONAL_ARCHITECTURE_MEMORY_ENERGY | 1.3677 | 1.3677 | 3.630291e-01 | 1.434091e-06 | 3.630306e-01 |
| orthogonal_si | 100 | True | EVALUATED_CONDITIONAL_ARCHITECTURE_MEMORY_ENERGY | 1.3677 | 136.7656 | 3.630291e-01 | 1.434091e-04 | 3.631725e-01 |
| orthogonal_si | 1000 | True | EVALUATED_CONDITIONAL_ARCHITECTURE_MEMORY_ENERGY | 1.3677 | 1367.6558 | 3.630291e-01 | 1.434091e-03 | 3.644632e-01 |
| orthogonal_m3d_igzo | 0 | True | EVALUATED_CONDITIONAL_ARCHITECTURE_MEMORY_ENERGY | 0.8553 | 0.0000 | 2.270195e-01 | 0.000000e+00 | 2.270195e-01 |
| orthogonal_m3d_igzo | 1 | True | EVALUATED_CONDITIONAL_ARCHITECTURE_MEMORY_ENERGY | 0.8553 | 0.8553 | 2.270195e-01 | 8.968057e-07 | 2.270204e-01 |
| orthogonal_m3d_igzo | 100 | True | EVALUATED_CONDITIONAL_ARCHITECTURE_MEMORY_ENERGY | 0.8553 | 85.5261 | 2.270195e-01 | 8.968057e-05 | 2.271092e-01 |
| orthogonal_m3d_igzo | 1000 | True | EVALUATED_CONDITIONAL_ARCHITECTURE_MEMORY_ENERGY | 0.8553 | 855.2606 | 2.270195e-01 | 8.968057e-04 | 2.279163e-01 |

## Notes

- `rho = 0`: mathematical lower bound; write energy is zero (read-only traffic energy).
- `rho = 1`: write energy equals read energy.
- `rho = 100`: write energy is 100× read energy.
- `rho = 1000`: write energy is 1000× read energy.
- `NO_ARCHITECTURE_ENERGY_RESOLVED`: the architecture's power model does not expose a per-bit access energy (e.g., `reference_fixed` or `unresolved`).
- All energy results are `None` when capacity is infeasible or architecture energy is unresolved.

## Provenance

- `read_energy_status`: `CURRENT_NOMINAL_ANALYTICAL_MODEL`
- `write_energy_status`: `RHO_SENSITIVITY_NOT_PHYSICAL_CLAIM`
- `energy_scope_status`: `MEMORY_DYNAMIC_TRAFFIC_ENERGY_ONLY`
- `scenario_status`: `CONDITIONAL_MATCHED_REFERENCE_SENSITIVITY`
- `zhu_transferability_status`: `NOT_VALIDATED`