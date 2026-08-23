# E5 LLM Decode Workload-Dependent Memory Power Audit

## Research Question

Connect the existing conditional architecture memory dynamic energy per
generated token, matched-scenario aggregate decode throughput, existing
refresh/background memory components, and fixed GPU baseline to a
workload-dependent memory/package power accounting result.

This is power accounting only. It is not thermal mapping, Tmax, GPU
energy/token, complete system J/token, bandwidth capability validation, or a
physical write-energy model.

## Starting Commit

`ba4a617cccbabe44904db019c22577130c4c2fa6` on `main`, equal to
`origin/main`, with a clean tracked working tree.

## Files Modified

- Added `src/om3dthermal/evaluator/llm_decode_workload_power.py`.
- Modified `src/om3dthermal/evaluator/__init__.py` only to export E5.
- Added `tests/test_llm_decode_workload_power.py`.
- Added this audit document.

No power, workload, thermal, architecture-comparison, configuration,
third-party, or existing expected-value file was modified.

## API Added

```python
evaluate_llm_decode_workload_power(
    energy: ArchitectureDecodeMemoryEnergyMetrics,
    performance: LLMDecodePerformanceMetrics,
    system: ResolvedSystemPower,
    *,
    unresolved_logic_background_policy: Literal[
        "REQUIRE_RESOLVED",
        "EXISTING_PLACEHOLDER_ZERO",
    ],
) -> LLMDecodeWorkloadPowerMetrics
```

The policy is mandatory and has no default.

## Equations

```text
Pdynamic = memory_dynamic_energy_j_per_token
           * aggregate_tokens_per_second

Pmemory = Pdynamic + Prefresh + Pmemory_background
          + Plogic_background_effective

Ppackage = Pfixed_gpu + Pmemory
```

Energy is already per generated token. Throughput is aggregate generated
tokens/s. No batch multiplier or divider is applied again.

## Consistency Gates

The evaluator rejects:

- energy architecture versus system case mismatch;
- performance architecture versus system case mismatch;
- energy/performance feasibility mismatch;
- read or write traffic mismatch;
- negative, non-finite, non-numeric, or boolean numeric inputs;
- placeholder-zero policy applied to an already numeric component;
- placeholder-zero use when the old total does not close.

Normal upstream/capacity blocking returns all derived power fields as `None`.
`REQUIRE_RESOLVED` with raw logic background `None` returns
`UNRESOLVED_STATIC_POWER` rather than silently substituting zero.

## Logic-Background Policy

Conventional HBM and Orthogonal Si use `REQUIRE_RESOLVED`. Their raw logic
background values are explicit numeric `0.0`, reported as
`RESOLVED_EXPLICIT_ZERO`.

Orthogonal M3D-IGZO uses `EXISTING_PLACEHOLDER_ZERO`. It is accepted only
because:

```text
raw logic background = None
old resolved memory total
  = old P_access + refresh + memory background + 0.0
```

The output preserves raw `None`, exposes effective `0.0`, and marks:

```text
logic_background_status
  = EXISTING_PLACEHOLDER_ZERO_NOT_SEPARATELY_MODELED

memory_total_completeness_status
  = CONDITIONAL_LOWER_BOUND_UNRESOLVED_LOGIC_BACKGROUND
```

It is not described as a resolved or validated zero and is not a complete
power model.

## Double-Counting Audit

The new sum does not consume any of:

- `system.memory_access_power_W`;
- `system.resolved_total_memory_power_W`;
- `system.memory_result.P_access_W`;
- `system.memory_result.P_total_W`.

The old access and total fields are used only for rho=1 regression checks and
the narrowly guarded M3D placeholder-closure check. Dynamic power is never
recomputed from configured bandwidth times Ebit.

## rho=1 Anchor

At rho=1, read and write Ebit are equal. The performance point is memory-bound,
so actual traffic/token times aggregate tokens/s equals the matched 39.2e12
payload bit/s. The new dynamic power therefore closes to the old read-access
power.

| Architecture | New dynamic W | Old access W | Difference W | New memory total minus old W |
|---|---:|---:|---:|---:|
| Conventional HBM | 54.76624100480178 | 54.76624100480178 | 0 | 0 |
| Orthogonal Si | 53.61210669822767 | 53.61210669822767 | 0 | 0 |
| Orthogonal M3D-IGZO | 33.52621456639419 | 33.52621456639418 | 7.11e-15 | 7.11e-15 |

All close within the required `1e-10 W` absolute tolerance.

## Frozen 12-Row Power Table

Scenario: B=1, S=131072, 16-bit weights/KV, zero reserved capacity,
39.2e12 bit/s matched-reference bandwidth, and 100e12 FLOP/s numerical compute
throughput. Aggregate throughput is 147.67932375509625 token/s and all rows are
memory-bound. Rho values are sensitivity points, not nominal assumptions.

| Architecture | rho | Dynamic J/token | Dynamic W | Refresh W | Logic raw/effective W | Memory total W | Package total W | Completeness |
|---|---:|---:|---:|---:|---|---:|---:|---|
| Conventional HBM | 0 | 0.370844226987498 | 54.7660246599951 | 0.817246576801 | 0 / 0 | 55.5832712367962 | 355.583271236796 | resolved existing static |
| Conventional HBM | 1 | 0.370845691950914 | 54.7662410048018 | 0.817246576801 | 0 / 0 | 55.5834875816029 | 355.583487581603 | resolved existing static |
| Conventional HBM | 100 | 0.370990723329151 | 54.7876591406630 | 0.817246576801 | 0 / 0 | 55.6049057174641 | 355.604905717464 | resolved existing static |
| Conventional HBM | 1000 | 0.372309190404025 | 54.9823694666737 | 0.817246576801 | 0 / 0 | 55.7996160434748 | 355.799616043475 | resolved existing static |
| Orthogonal Si | 0 | 0.363029119780785 | 53.6118949126341 | 1.668545094302 | 0 / 0 | 55.2804400069363 | 355.280440006936 | resolved existing static |
| Orthogonal Si | 1 | 0.363030553871815 | 53.6121066982277 | 1.668545094302 | 0 / 0 | 55.2806517925299 | 355.280651792530 | resolved existing static |
| Orthogonal Si | 100 | 0.363172528883828 | 53.6330734719919 | 1.668545094302 | 0 / 0 | 55.3016185662942 | 355.301618566294 | resolved existing static |
| Orthogonal Si | 1000 | 0.364463210811223 | 53.8236805062125 | 1.668545094302 | 0 / 0 | 55.4922256005148 | 355.492225600515 | resolved existing static |
| Orthogonal M3D-IGZO | 0 | 0.227019472152587 | 33.5260821267329 | 0.034150009746 | None / 0 | 33.5602321364791 | 333.560232136479 | conditional lower bound |
| Orthogonal M3D-IGZO | 1 | 0.227020368958300 | 33.5262145663942 | 0.034150009746 | None / 0 | 33.5603645761404 | 333.560364576140 | conditional lower bound |
| Orthogonal M3D-IGZO | 100 | 0.227109152723926 | 33.5393260928623 | 0.034150009746 | None / 0 | 33.5734761026085 | 333.573476102609 | conditional lower bound |
| Orthogonal M3D-IGZO | 1000 | 0.227916277865984 | 33.6585217880271 | 0.034150009746 | None / 0 | 33.6926717977733 | 333.692671797773 | conditional lower bound |

Memory background is explicit `0.0 W` for all rows. Fixed GPU power is
`300.0 W` for all rows.

## Status and Provenance

```text
dynamic_power_status
  = WORKLOAD_J_PER_TOKEN_TIMES_AGGREGATE_TOKENS_PER_SECOND
static_power_status
  = EXISTING_POWER_MODEL_COMPONENTS_ADDED_ONCE
gpu_power_status
  = FIXED_EXISTING_BASELINE_NOT_WORKLOAD_ENERGY_MODEL
system_energy_status
  = NOT_AVAILABLE_COMPUTE_ENERGY_EXCLUDED
scenario_status
  = CONDITIONAL_MATCHED_REFERENCE_SENSITIVITY
```

## Scientific Interpretation

The table is conditional workload-dependent power accounting under a matched
reference scenario. It is not an architecture bandwidth-capability result and
does not validate rho as physical write energy. M3D memory/package totals are
conditional lower bounds because logic-background power remains unresolved.

## Assumptions / Provenance

- 39.2 Tb/s remains `MATCHED_REFERENCE_NOT_CAPABILITY_VALIDATED`.
- 100 TFLOP/s remains `NUMERICAL_CHOICE_NOT_HARDWARE_VALIDATED`.
- Rho remains `RHO_SENSITIVITY_NOT_PHYSICAL_CLAIM`.
- Existing refresh/background values are reused once without re-modeling.
- GPU power is the fixed existing 300 W baseline, not workload GPU energy.
- Compute energy and complete system J/token are unavailable.
- No thermal integration was started.

## Validation

The targeted and required regression commands and final outcomes are recorded
in the final task report. No thermal solver is run for E5.

## PASS / FAIL

PASS requires every specified test and `git diff --check` to pass, the rho=1
anchors to close, and the final diff to contain only the four authorized E5
files.

## Open Questions

- M3D logic-background power remains unresolved; its total is a conditional
  lower bound.
- Architecture write energy remains a sensitivity parameter rather than a
  validated physical model.
- Matched bandwidth remains a scenario input rather than a capability.

## Next Recommended Step

Research Lead review of E5 accounting and conditional-lower-bound labeling.
Do not enter E6 thermal integration until separately authorized.
