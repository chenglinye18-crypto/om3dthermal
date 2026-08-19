# LLM Memory Dynamic Energy Primitive v0 Audit

## Research Question

Given an architecture-independent `LLMDecodeMetrics`, an already evaluated
`CapacityFeasibilityMetrics`, and independently supplied read and write access
energies in pJ/bit, what dynamic memory-access energy is incurred per generated
decode token?

This primitive answers only that question. Its final metric is
`memory_dynamic_energy_j_per_token`; it is not a complete system J/token model.

## Starting Commit

- Branch: `main`
- Commit: `8df61f0eef7a0b12f88b8a510e2c6a051403b47c`
- `HEAD == origin/main`: yes
- Starting working tree: clean

## Files Inspected

- `src/om3dthermal/workload/llm_decode.py`
- `src/om3dthermal/workload/capacity.py`
- `src/om3dthermal/evaluator/llm_decode_performance.py`
- `src/om3dthermal/evaluator/__init__.py`
- `tests/test_llm_decode.py`
- `tests/test_workload_capacity.py`
- `tests/test_llm_decode_performance.py`
- `tests/test_architecture_capacity_adapter.py`
- `tests/test_architecture_comparison.py`

## Files Modified

- Added `src/om3dthermal/evaluator/llm_decode_energy.py`.
- Modified `src/om3dthermal/evaluator/__init__.py` only to export the new API.
- Added `tests/test_llm_decode_energy.py`.
- Added this audit document.

No configuration, thermal, power, architecture, workload, benchmark, or
existing performance-evaluator file was modified.

## Existing Interfaces Reused

- `LLMDecodeMetrics` supplies the already normalized per-generated-token read
  and write traffic.
- `CapacityFeasibilityMetrics` supplies the existing aggregate capacity gate.
- The capacity calculation is not repeated or altered by this primitive.

## Accounting Boundary

Included: dynamic memory access energy caused by
`read_bytes_per_token` and `write_bytes_per_token`, using caller-supplied read
and write pJ/bit values.

Excluded: compute/GPU energy, memory refresh energy, background/static energy,
power derivation, workload-dependent power, thermal effects, temperature, and
bandwidth capability. Therefore the result must not be interpreted as complete
system energy per token or end-to-end energy efficiency.

## Equations and Units

For analytical traffic values `R_byte` and `W_byte` in byte-equivalents/token:

```text
R_bit = 8 bit/byte * R_byte
W_bit = 8 bit/byte * W_byte

E_read  [pJ/token] = R_bit [bit/token] * e_read  [pJ/bit]
E_write [pJ/token] = W_bit [bit/token] * e_write [pJ/bit]
E_total [pJ/token] = E_read + E_write
E_total [J/token]  = E_total [pJ/token] * 1e-12 [J/pJ]
```

The implementation preserves fractional byte-equivalents and performs no
floor, ceiling, integer conversion, or rounding. Output-model validation also
checks pJ contribution closure and the pJ-to-J conversion.

## Footprint vs Traffic Distinction

Only `LLMDecodeMetrics.read_bytes_per_token` and
`LLMDecodeMetrics.write_bytes_per_token` enter the energy equation.
`weight_footprint_bytes`, `kv_footprint_bytes`, and
`required_capacity_bytes` do not. The KV capacity equation is not reused as a
KV traffic equation.

The workload traffic is already expressed per generated token. The energy
primitive neither multiplies nor divides it by batch size again.

## Capacity Gate

The primitive consumes, but does not recompute,
`CapacityFeasibilityMetrics.capacity_feasible`.

- Feasible, including exact fit: numeric contributions and total are returned
  with `EVALUATED_MEMORY_DYNAMIC_TRAFFIC_ENERGY`.
- Infeasible: all four energy result fields are `None` and status is
  `CAPACITY_INFEASIBLE`; ordinary infeasibility is not raised as an exception.

The fixed scope marker is `MEMORY_DYNAMIC_TRAFFIC_ENERGY_ONLY`.

## Read vs Write Energy Semantics

`read_energy_pj_per_bit` and `write_energy_pj_per_bit` are independent,
mandatory keyword-only inputs with no defaults. The primitive does not infer
one from the other. In particular, existing read-access `Ebit` values are not
automatically used for writes, even if a caller eventually chooses equal
numbers explicitly.

There is currently no validated architecture-specific write-energy adapter.
Consequently, this task does not produce a three-architecture J/token table,
ranking, or paper claim.

## Hand Check

For 1 byte/token read, 1 byte/token write, 2 pJ/bit read, and 3 pJ/bit
write:

```text
read:  1 * 8 * 2 = 16 pJ/token
write: 1 * 8 * 3 = 24 pJ/token
total: 16 + 24    = 40 pJ/token
J/token: 40 * 1e-12 = 4.0e-11 J/token
```

The unit hand-check is encoded as a direct semantic test.

## Validation

Required native-Windows Conda environment: `om3dthermal`, Python 3.11 under
`C:\Users\Leslie\Miniconda3`.

The validation suite covers the hand check, dimensional closure, independent
read/write behavior, zero read/write/total traffic, fractional traffic,
footprint/traffic separation, absence of batch double accounting, infeasible
and exact-fit capacity gates, mandatory inputs, invalid inputs, finite large
and zero energies, and forbidden output fields. The final regression commands
and their outcomes are recorded in the task handoff report.

## Scope Exclusions

The output explicitly identifies the following as outside its boundary:

- compute energy;
- refresh energy;
- background/static energy;
- power derivation;
- thermal effects.

It contains no bandwidth, time, throughput, power, temperature, GPU-energy,
refresh-energy, background-energy, or generic/system J/token field.

## Assumptions / Provenance

- Traffic provenance is the existing architecture-independent
  `LLMDecodeMetrics` accounting model.
- Capacity provenance is the existing aggregate-only
  `CapacityFeasibilityMetrics`; no placement or fragmentation statement is
  implied.
- Read and write pJ/bit are explicit caller inputs. This primitive does not
  validate their device provenance or architectural transferability.
- Zhu 64-layer to the current 8-layer transferability remains
  `NOT_VALIDATED`.
- BWcoil architecture capability remains `NOT_VALIDATED`; the 39.2 Tb/s value
  remains `MATCHED_REFERENCE_NOT_CAPABILITY_VALIDATED` and is not used here.

## PASS / FAIL

PASS is contingent on all targeted and regression tests passing and the final
diff containing only the four authorized task files.

The implemented scientific boundary is memory dynamic access energy/token,
not complete system J/token.

## Open Questions

- What directly evidenced write-access energy boundary is appropriate for each
  architecture?
- Which read/write peripheral components are included in each candidate
  pJ/bit value, and are those boundaries mutually comparable?
- Is transfer from the cited device/layer stack to each evaluated architecture
  scientifically supportable?

## Next Recommended Step

Independently audit and establish an architecture-specific write-energy
accounting boundary. Only after that audit should a three-architecture memory
dynamic J/token table be generated.
