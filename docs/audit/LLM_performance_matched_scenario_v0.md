# LLM Decode Performance Primitive v0 — Matched-Bandwidth Scenario

> Read-only illustrative performance evaluation for the frozen
> LLaMA-3.1-8B-class workload under the matched-reference
> 39.2 Tb/s payload bandwidth and 100 TFLOP/s illustrative compute
> scenario.  Three canonical architectures are evaluated only at
> the **capacity-feasibility** stage; the performance numbers are
> therefore identical by design.  This document is **not** an
> architecture capability comparison.

---

## Research Question

In the capacity-feasible regime, given
- workload read/write traffic per token,
- workload FLOPs per token,
- a matched effective payload bandwidth,
- an effective GPU compute throughput,
- a batch size,

can a transparent roofline-style analytical model produce
- memory resource time per token-equivalent,
- compute resource time per token-equivalent,
- bottleneck,
- aggregate decode throughput,
- per-sequence decode throughput,
- per-sequence step latency,

while keeping
- aggregate / per-sequence semantics explicit,
- the matched scenario clearly **not** an architecture capability,
- infeasible workloads from emitting deceptive tokens/s numbers?

This is the v0 primitive only.  No bandwidth sweep, no per-architecture
capability input, no J/token, no power, no thermal, no Tmax.

## Starting Commit

```
e1fd7f1 feat(evaluator): connect workload to architecture capacity
```

Pre-task state:
- `git rev-parse HEAD       = e1fd7f1cdd4efcbf838711a9c1a283fcee56ab0c`
- `git rev-parse origin/main = e1fd7f1cdd4efcbf838711a9c1a283fcee56ab0c`
- `git status --short --branch = "## main...origin/main"` (no drift, clean)

## Evidence / Existing Interfaces

| existing primitive | role in v0 |
|---|---|
| `LLMDecodeInput`, `LLMDecodeMetrics`, `evaluate_llm_decode` | produces per-token read/write traffic and FLOPs |
| `evaluate_capacity_feasibility` (aggregate-only) | the shared capacity primitive |
| `ArchitectureCapacityFeasibility` | the gate consumed by the performance primitive |
| `evaluate_architecture_capacity_feasibility` | builds the gate for one canonical architecture |
| `ResolvedArchitectureCapacity` (capacity resolver) | supplies the architecture's physical capacity (used **only** to derive `capacity_feasible`; never enters the performance numbers) |

## Files Inspected (read-only)
- `src/om3dthermal/architecture_capacity.py`
- `src/om3dthermal/workload/llm_decode.py` (frozen B1-R1, not modified)
- `src/om3dthermal/workload/capacity.py`
- `src/om3dthermal/workload/architecture_capacity.py`
- `src/om3dthermal/workload/__init__.py`
- `tests/test_workload_capacity.py` (must still pass)
- `tests/test_architecture_capacity_adapter.py` (must still pass)
- `tests/test_llm_decode.py` (must still pass)
- `tests/test_architecture_comparison.py` (must still pass)

## Files Modified
| file | new / modified |
|---|---|
| `src/om3dthermal/evaluator/__init__.py` | new |
| `src/om3dthermal/evaluator/llm_decode_performance.py` | new |
| `tests/test_llm_decode_performance.py` | new |
| `docs/audit/LLM_performance_matched_scenario_v0.md` | new (this file) |

No source file outside the allowed set was modified.  The LLM
decode primitive, the capacity primitive, the architecture capacity
adapter, the power / thermal / config / CLI / sweep paths are
**untouched**.

## Configuration

| key | value |
|---|---|
| Python | 3.11.15 (om3dthermal env) |
| Pytest | 9.1.1 |
| `batch_size` | 1 |
| `context_length` | 131 072 |
| `weight_bits`, `kv_bits` | 16 |
| `runtime_bytes` | 0 |
| `reserved_capacity_bytes` | 0 |
| `matched_payload_bandwidth_bits_per_second` | 39.2 × 10¹² (39.2 Tb/s) |
| `effective_compute_flops_per_second` | 100 × 10¹² (100 TFLOP/s) |
| `memory_bandwidth_model` | `SHARED_READ_WRITE_PAYLOAD_BANDWIDTH` |
| `overlap_model` | `ROOFLINE_MAX` |
| `bandwidth_status` | `MATCHED_REFERENCE_NOT_CAPABILITY_VALIDATED` |
| `compute_throughput_status` | `NUMERICAL_CHOICE_NOT_HARDWARE_VALIDATED` |

## Validation

```
tests/test_llm_decode_performance.py            33 passed
tests/test_llm_decode.py                        24 passed
tests/test_workload_capacity.py                 18 passed
tests/test_architecture_capacity_adapter.py     11 passed
tests/test_architecture_comparison.py            11 passed
                                               ─────────
                                                97 passed
```

- `git diff --check` clean.
- `git diff --stat` only the 4 new / modified files above.
- `git status --short` shows no other modifications.

## Diagnostics

### Workload (frozen 8B-class, B=1, S=131072, 16-bit)

| metric | value |
|---|---:|
| `read_bytes_per_token` | 33 179 869 184 B = 30.90 GiB |
| `write_bytes_per_token` | 131 072 B = 128 KiB |
| `traffic_bytes_per_token` | 33 180 000 256 B |
| `traffic_bits_per_token` | 2.6544 × 10¹¹ bit |
| `flops_per_token` | 83 728 793 600 (= 83.73 GFLOP) |
| `flops_sanity_per_token` | 84 719 476 736 (= 2·Nparam + 4·L·Hq·S·Dhead) |
| `required_capacity_bytes` | 33 179 869 184 B = 30.90 GiB |

The `read_bytes_per_token` figure is dominated by `weight_footprint /
batch_size = 16e9 / 1 = 16 GiB` (the v0 `tile_reuse` model: one weight
tile services B inputs and is amortized over the batch).  The remaining
`17.18 GiB` of the read traffic is the historical KV-cache re-read for
context 131072.  The 128 KiB write is the new KV entry written each
step.

### Capacity Gate

| architecture | capacity_GiB | required_GiB | margin_GiB | feasible |
|---|---:|---:|---:|---|
| conventional_hbm_2x1 | 114.75 | 30.90 | 83.85 | **True** |
| orthogonal_si | 234.28 | 30.90 | 203.38 | **True** |
| orthogonal_m3d_igzo | 428.75 | 30.90 | 397.85 | **True** |

All three are capacity-feasible for the 30.90 GiB LLaMA-3.1-8B-class
footprint.  M3D has the largest headroom, but the v0 performance
primitive does **not** consume architecture-specific capability
bandwidth or compute; it consumes the matched-reference scenario
only.

### Bandwidth Scenario

```
matched_payload_bandwidth_bits_per_second = 39_200_000_000_000
                                          = 39.2 × 10¹² bit/s
                                          = 39.2 Tb/s
bandwidth_status  = MATCHED_REFERENCE_NOT_CAPABILITY_VALIDATED
```

This is the matched-reference delivered-bandwidth value already
pinned by the existing workload power config (read_bandwidth_gbps =
39 200).  It is **not** an M3D capability, an HBM capability, or any
architecture's measured peak.

### Compute Scenario

```
effective_compute_flops_per_second = 100_000_000_000_000
                                    = 100 × 10¹² FLOP/s
                                    = 100 TFLOP/s
compute_throughput_status  = NUMERICAL_CHOICE_NOT_HARDWARE_VALIDATED
```

This is the v0 illustrative value used only to wire up the
performance primitive and exercise the unit / batch-semantics tests.
It is **not** an RTX 4070 SUPER measurement, an LLM serving
benchmark, or any hardware datasheet number.  Future hardware-validated
scenarios must be added as a new task with a new compute throughput
status literal.

### Equation and Unit Audit

The full set of v0 equations, written in SI units (bit, FLOP, second)
unless otherwise stated:

```
traffic_bytes_per_token       = read_bytes_per_token
                                + write_bytes_per_token
traffic_bits_per_token        = traffic_bytes_per_token * 8
memory_time_per_token_equiv_s = traffic_bits_per_token
                                / matched_payload_bandwidth_bits_per_second
compute_time_per_token_eq_s   = flops_per_token
                                / effective_compute_flops_per_second
token_equivalent_time_s       = max(memory_time, compute_time)
                                [ROOFLINE_MAX]
aggregate_step_time_s         = batch_size * token_equivalent_time_s
aggregate_tokens_per_second   = batch_size / aggregate_step_time_s
                              = 1 / token_equivalent_time_s
per_sequence_step_latency_s   = aggregate_step_time_s
per_sequence_tokens_per_s     = 1 / aggregate_step_time_s
                              = aggregate_tokens_per_second / batch_size
crossover_FLOPs_per_second    = flops_per_token / memory_time
                                [if memory_time == 0: math.inf]
```

Unit audit:

| equation | numerator unit | denominator unit | result unit |
|---|---|---|---|
| `traffic_bits_per_token` | byte | — | bit |
| `memory_time` | bit | bit / s | **second** ✓ |
| `compute_time` | FLOP | FLOP / s | **second** ✓ |
| `aggregate_step_time` | (dimensionless) × s | — | **second** ✓ |
| `aggregate_tokens_per_s` | (dimensionless) | s | **1 / s = tok / s** ✓ |
| `per_sequence_tokens_per_s` | tok / s | (dimensionless) | **tok / s** ✓ |
| `crossover_FLOPs_per_s` | FLOP | s | **FLOP / s** ✓ |

The only explicit conversion in the primitive is `byte → bit` via the
factor of 8 on `traffic_bytes_per_token → traffic_bits_per_token`.  No
GiB / GB conversion, no per-second / per-millisecond mixing, no
`39.2 Tb/s` written as `39.2 TB/s`.

### Batch Semantics

The v0 model makes the aggregate / per-sequence distinction explicit
in three places:

1. The output model carries both `aggregate_tokens_per_second` and
   `per_sequence_tokens_per_second` as distinct fields, and they
   differ by exactly `batch_size`.
2. The `per_sequence_step_latency_s` field equals
   `aggregate_step_time_s` (a single aggregate decode step covers the
   whole batch).  The per-sequence throughput is the inverse of that
   step latency, not the inverse of `token_equivalent_time_s`.
3. `test_aggregate_and_per_sequence_batch_semantics_at_b8` pins the
   batch-8 identity:
   - `aggregate_step_time = 8 × token_equivalent_time`
   - `aggregate_tokens_per_second = 1 / token_equivalent_time`
   - `per_sequence_tokens_per_second = aggregate / 8`
   - `per_sequence_step_latency = aggregate_step_time`.

The single most common conflation the v0 design is rejecting is
**labelling `1 / token_equivalent_time` as both aggregate and
per-sequence throughput**.  It is aggregate only.  At `batch_size = 1`
the two values coincide, but they are still conceptually distinct and
the test pins the B=8 separation.

## Performance Table

The frozen 8B-class workload under the matched 39.2 Tb/s payload
bandwidth and 100 TFLOP/s illustrative compute scenario.  All
three architectures are capacity-feasible, so the performance row is
identical by design.  The table demonstrates that v0 is producing a
**matched-reference** illustrative scenario number, not an
architecture capability comparison.

| Architecture | Capacity Feasible | Memory Time (ms) | Compute Time (ms) | Bottleneck | Aggregate tok/s | Per-sequence tok/s | Status |
|---|:---:|---:|---:|---|---:|---:|---|
| conventional_hbm_2x1 | True | 6.7714 | 0.8373 | MEMORY | 147.6793 | 147.6793 | EVALUATED_MATCHED_REFERENCE_SCENARIO |
| orthogonal_si | True | 6.7714 | 0.8373 | MEMORY | 147.6793 | 147.6793 | EVALUATED_MATCHED_REFERENCE_SCENARIO |
| orthogonal_m3d_igzo | True | 6.7714 | 0.8373 | MEMORY | 147.6793 | 147.6793 | EVALUATED_MATCHED_REFERENCE_SCENARIO |

(Per-sequence equals aggregate only because `batch_size = 1`.  The
batch-8 separation is exercised in `tests/test_llm_decode_performance.py`.)

## Crossover Analysis

For this matched scenario:

```
memory_time                                  = 6.7714 ms / token
compute_time                                 = 0.8373 ms / token
crossover_compute_FLOPs_per_s = flops / Tmem = 8.3729e10 / 6.7714e-3
                                ≈ 1.2365 × 10¹³ FLOP/s ≈ 12.37 TFLOP/s
```

Interpretation:
- If the **actual effective compute throughput** is **higher** than
  ~12.4 TFLOP/s, the matched scenario is **memory-bound**.
- If it is **lower**, the matched scenario is **compute-bound**.
- The illustrative 100 TFLOP/s scenario is well above the crossover
  (about 8.1×), which is why the table reads `MEMORY`-bound at
  B=1, S=131072.

The crossover metric is **architecture-independent**: it does not pick
a GPU.  It is purely a workload-vs-bandwidth statement.  Any future
hardware-validated compute scenario can be compared against this
crossover without re-deriving the workload.

## Dimensional Sanity Checks

| band | spec | observed | result |
|---|---|---|---|
| `memory_time` | ~6.8 ms | 6.7714 ms | PASS |
| `compute_time` | ~0.84 ms | 0.8373 ms | PASS |
| `aggregate tok/s` | ~147 | 147.6793 | PASS |
| `bottleneck` | MEMORY | MEMORY | PASS |
| `crossover` | well below 100 TFLOP/s | 12.37 TFLOP/s | PASS (≈ 8.1× headroom) |
| `per_seq == aggregate` at B=1 | yes | 147.6793 == 147.6793 | PASS |

The single tight check is the `memory_time` band: the spec gives
"~6.8 ms" and the observed 6.7714 ms is within 0.5 % of that anchor.
The remaining bands are integer-token counts or orders of magnitude
and the observed values match exactly.

## Key Results

- All three architectures produce **identical** Tmemory (6.7714 ms),
  Tcompute (0.8373 ms), aggregate tok/s (147.6793), per-seq tok/s
  (147.6793 at B=1), and bottleneck (MEMORY) in this v0 scenario.
- This identity is **by design**: v0 consumes the matched-reference
  scenario, not architecture-specific capability numbers.
- The matched 8B-class workload is **memory-bound** under the v0
  scenario (Tmemory ≈ 8.1× Tcompute).  Compute is not the bottleneck
  until the effective compute throughput falls below ≈ 12.4 TFLOP/s.
- All three architectures are capacity-feasible for the 30.90 GiB
  workload, with `orthogonal_m3d_igzo` having the largest headroom
  (397.85 GiB margin).  The capacity headroom is the only
  architecture-dependent observable in v0; it does **not** enter
  the performance numbers.

## Scientific Interpretation

v0 establishes the simplest non-deceptive end-to-end performance
primitive that satisfies four scientific contracts at once:

1. **Aggregate / per-sequence semantics are explicit.**  The
   `token_equivalent_time_s` is a per-aggregate-step resource
   budget, not a per-sequence latency.  The per-sequence latency
   equals the aggregate step time; per-sequence throughput is the
   inverse of that step latency; aggregate throughput is the
   per-token-equivalent throughput times `batch_size`.
2. **The matched scenario is not an architecture capability.**  The
   `bandwidth_status` and `compute_throughput_status` labels are
   echoed verbatim and are part of the public output surface.  A
   future caller that wants to claim M3D or HBM capability must
   introduce a **new** status literal in a separate task; the v0
   labels cannot be silently re-interpreted.
3. **Infeasible workloads do not emit deceptive throughput.**  When
   the capacity gate is closed, the performance primitive returns
   `BLOCKED_BY_CAPACITY` with every numeric time / throughput field
   set to `None` and the bottleneck set to
   `NOT_EVALUATED_CAPACITY_INFEASIBLE`.  A Pydantic model validator
   double-checks the consistency of that state so a caller cannot
   construct a deceptive result by hand.
4. **Byte / bit / FLOP / second units are not silently mixed.**  The
   only conversion in the model is the explicit `× 8` from byte to
   bit on `traffic_bytes_per_token → traffic_bits_per_token`.  All
   denominators are SI seconds; all numerators are SI bit or FLOP.
   The `crossover_FLOPs_per_s` field uses `math.inf` as the
   divide-by-zero sentinel with the explicit policy documented in
   the source.

The v0 numbers are physically reasonable for the chosen scenario
only: 6.77 ms / token on a 30.9 GiB model with a 39.2 Tb/s payload
bandwidth is dominated by the read-side weight fetch (16 GiB at
39.2 Tb/s ≈ 3.3 ms) plus the KV-cache re-read (17.18 GiB at
39.2 Tb/s ≈ 3.5 ms) — together 6.77 ms — which is exactly what the
primitive computes.  This is consistent with a streaming decoder
that cannot overlap the weight fetch with the KV fetch in v0's
`ROOFLINE_MAX` model.

## Assumptions / Provenance

| label | value | role |
|---|---|---|
| `MATCHED_REFERENCE_NOT_CAPABILITY_VALIDATED` | echoed | bandwidth status |
| `NUMERICAL_CHOICE_NOT_HARDWARE_VALIDATED` | echoed | compute throughput status |
| `SHARED_READ_WRITE_PAYLOAD_BANDWIDTH` | echoed | memory bandwidth model |
| `ROOFLINE_MAX` | echoed | overlap model |
| `39.2 Tb/s` | scenario input | not a capability claim |
| `100 TFLOP/s` | scenario input | not a hardware measurement |
| LLM primitive (B1-R1) | frozen | unchanged |
| capacity primitive | frozen | unchanged |
| architecture capacity adapter | frozen | unchanged |

## Commit Hash

To be filled at commit time.

## Push Status

To be filled at push time.

## PASS / FAIL

| acceptance gate | result |
|---|---|
| capacity infeasible does not emit tokens/s | **PASS** (model validator enforces) |
| read + write traffic correctly merged | **PASS** (test_bit_byte_conversion) |
| byte → bit conversion correct | **PASS** (× 8 only) |
| Tmemory and Tcompute unit-correct | **PASS** (tested against spec band 6.8 ms / 0.84 ms) |
| ROOFLINE_MAX correct | **PASS** (test_memory_bound, test_compute_bound, test_balanced) |
| aggregate / per-sequence semantics correct | **PASS** (test_aggregate_and_per_sequence_batch_semantics_at_b8) |
| bottleneck classification correct | **PASS** (MEMORY / COMPUTE / BALANCED / NOT_EVALUATED) |
| crossover metric correct | **PASS** (incl. math.inf divide-by-zero policy) |
| matched / reference status preserved | **PASS** (test_status_provenance_labels_are_preserved) |
| compute scenario provenance preserved | **PASS** |
| three architectures same scenario same result | **PASS** (test_same_scenario_same_three_architectures) |
| first performance table hand-checkable | **PASS** (table values match spec bands) |
| all targeted tests PASS | **PASS** (97/97 across 5 suites) |
| no energy / power / thermal / J/token / Tmax scope creep | **PASS** (test_output_model_has_no_forbidden_metrics) |
| commit and push succeed | pending |
| tracked tree clean | pending |

**OVERALL: PASS** (preliminary — commit + push still pending).

## Open Questions

1. **Architecture-specific capability bandwidth.**  v0 cannot say
   "M3D is 2× faster" because it consumes the matched scenario, not
   per-architecture capability.  A future task must add a
   `memory_bandwidth_model: Literal["ARCHITECTURE_CAPABILITY_PAYLOAD_BANDWIDTH"]`
   (or similar) that sources the bandwidth from
   `ResolvedArchitectureCapacity` and re-runs the same evaluation
   per architecture.
2. **Compute throughput as a per-architecture input.**  Same story:
   today the 100 TFLOP/s is a single shared scenario; per-architecture
   compute scenario inputs are explicitly out of scope.
3. **`full_reread` KV assumption.**  The KV re-read traffic is
   17.18 GiB at S=131072 — the single largest contributor to
   `traffic_bits_per_token`.  Any change in the `kv_read_model`
   (`full_reread` / sliding window / paged attention) will move
   the memory time directly.  v0 only supports `full_reread`; the
   alternative is a new modeling choice in a future task.
4. **The `+inf` crossover sentinel.**  When `memory_time == 0`
   (zero-traffic workload), the crossover compute throughput is set
   to `math.inf`.  A future task may prefer a structured
   `crossover_status` field that is None / DEFINED / UNDEFINED; v0
   chose the simpler documented sentinel.

## Next Recommended Step

Hand the v0 performance primitive and the matched-scenario table to
the Research Lead for review.  The natural next task is to add a
**per-architecture capability** scenario that sources bandwidth and
compute from `ResolvedArchitectureCapacity` and produces a per-row
performance delta — but that must be a separate task, must introduce
a new `memory_bandwidth_model` literal, and must not silently reuse
the matched-reference status labels.

## STOP

Performance primitive, tests, audit document — all complete.
No J/token, power, or thermal scope entered.  No commit / push yet
in this document; they happen in the parent commit / push step.
