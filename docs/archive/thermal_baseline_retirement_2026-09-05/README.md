# Retired thermal comparison records

Retired from current DAC documentation on 2026-09-05 after the user selected
Conventional 2x1 as the sole Conventional baseline. This directory preserves
historical evidence; its tables, figures and commands are not current paper
results or recommended run examples.

- `experiment_matrix.md` and `thermal_results_overview.md`: superseded
  multi-layout, fixed-power comparison tables.
- `figures/`: their existing comparison plots, relocated without regeneration.
- `readme_retired_benchmark_sections.md`: old benchmark setup and solver result
  descriptions removed from the current README.
- The case-specific notes record historical geometry and power provenance.

The associated YAML files remain in `configs/legacy/` as test fixtures.
The authoritative current reference is
[the Conventional thermal baseline](../../benchmarks/thermal_results_overview.md).

No physical equations, canonical configuration values, solver implementation,
or regression expectations changed in this retirement. No new thermal results
were generated. Validation: 24 existing experiment-configuration and LLM decode
E2E tests passed in the canonical Windows `om3dthermal` Conda environment.
