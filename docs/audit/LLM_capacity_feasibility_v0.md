# Canonical Architecture Capacity Feasibility v0

## Research Question

Can the same analytical LLM workload footprint fit within the aggregate
physical capacity resolved for each of the three canonical architectures?
This audit evaluates aggregate capacity only.

## Starting Commit

`8aea04e3bd12ef6bb199fbd6b424b8d6389bb8f8`

## Workload Configuration

| Field | Value |
|---|---:|
| `n_param` | 8,000,000,000 |
| `n_layers` | 32 |
| `n_heads_q` / `n_heads_kv` | 32 / 8 |
| `d_head` / `d_model` / `d_ff` | 128 / 4096 / 14336 |
| `vocab_size` | 128,256 |
| `batch_size` | 1 |
| `context_length` | 131,072 |
| `weight_bits` / `kv_bits` | 16 / 16 |
| `runtime_bytes` | 0 |

`evaluate_llm_decode()` produces:

| Footprint | Bytes | GiB |
|---|---:|---:|
| Weights | 16,000,000,000 | 14.901161 |
| KV | 17,179,869,184 | 16.000000 |
| Runtime | 0 | 0.000000 |
| **Required total** | **33,179,869,184** | **30.901161** |

## Workload Provenance

- Architecture parameters: LLaMA-3.1-8B-class analytical example.
- `n_param=8e9`: rounded architecture-class value, not an exact official
  checkpoint parameter count.
- Batch, context length, and precisions: `NUMERICAL_CHOICE`.
- `runtime_bytes=0`: `NUMERICAL_CHOICE / IDEALIZED_V0`.

## Architecture Capacity Provenance

Capacity is resolved from the existing canonical config, geometry resolver,
analytical memory resolver, and packing diagnostics. Exact integer
`total_bits` is the source of truth. No canonical capacity is hard-coded in
the resolver or adapter.

| Architecture | Instances | Exact total bits | Physical bytes | Physical GiB |
|---|---:|---:|---:|---:|
| Conventional HBM 2x1 | 2 | 985,694,994,432 | 123,211,874,304 | 114.750000 |
| Orthogonal Si | 98 | 2,012,460,613,632 | 251,557,576,704 | 234.281250 |
| Orthogonal M3D-IGZO | 98 | 3,682,934,456,320 | 460,366,807,040 | 428.750000 |

Capacity source status:
`ANALYTICAL_PACKING_DIAGNOSTICS_BIT_CLOSURE`.

## Equations

```text
system_capacity_bytes = total_bits / 8
system_capacity_GiB = system_capacity_bytes / 2**30
usable_capacity_bytes = physical_capacity_bytes - reserved_capacity_bytes
required_capacity_bytes = LLMDecodeMetrics.required_capacity_bytes
capacity_margin_bytes = usable_capacity_bytes - required_capacity_bytes
capacity_utilization = required_capacity_bytes / usable_capacity_bytes
capacity_feasible = required_capacity_bytes <= usable_capacity_bytes
```

The cross-layer adapter calls `evaluate_capacity_feasibility()`; it does not
reimplement these feasibility equations.

## Unit Conversion

All internal comparison quantities use bytes. GiB is a display conversion
only, using `bytes / 2**30`. Physical bytes are derived directly from exact
integer `total_bits`, not reconstructed from displayed GiB.

## Capacity Table

The hardware reserve is explicitly `0 bytes` for every row:
`NUMERICAL_CHOICE / IDEAL_RAW_CAPACITY_SCENARIO`.

| Architecture | Physical GiB | Reserved GiB | Usable GiB | Required GiB | Margin GiB | Utilization | Feasible | Scope |
|---|---:|---:|---:|---:|---:|---:|---|---|
| Conventional HBM 2x1 | 114.750000 | 0 | 114.750000 | 30.901161 | 83.848839 | 26.9291% | True | `AGGREGATE_CAPACITY_FEASIBILITY_ONLY` |
| Orthogonal Si | 234.281250 | 0 | 234.281250 | 30.901161 | 203.380089 | 13.1898% | True | `AGGREGATE_CAPACITY_FEASIBILITY_ONLY` |
| Orthogonal M3D-IGZO | 428.750000 | 0 | 428.750000 | 30.901161 | 397.848839 | 7.2073% | True | `AGGREGATE_CAPACITY_FEASIBILITY_ONLY` |

All feasibility comparisons use the unrounded byte values shown in the
preceding tables. Display rounding is not fed back into the calculation.

## Dimensional Sanity Checks

For every architecture:

```text
total_bits == bits_per_instance * instance_count
system_capacity_bytes == total_bits / 8
system_capacity_GiB == system_capacity_bytes / 2**30
```

For the workload:

```text
16,000,000,000 + 17,179,869,184 + 0
= 33,179,869,184 bytes
```

The same `LLMDecodeMetrics` object is consumed for all three rows, so required
capacity is identical. Physical capacity, margin, and utilization remain
architecture-specific.

## Scientific Interpretation

Under this 8B-class, batch-1, 128K-context, 16-bit, reserve-zero scenario, the
workload fits within the aggregate raw capacity of all three canonical
architectures. The result establishes only aggregate capacity feasibility.
It does not establish placement across banks, dies, slabs, or channels;
fragmentation; allocator/page-block effects; or production-serving capacity.

## Assumptions / Provenance

- Hardware reserve: explicit zero, `NUMERICAL_CHOICE / IDEAL_RAW_CAPACITY_SCENARIO`.
- Runtime footprint: explicit zero, `NUMERICAL_CHOICE / IDEALIZED_V0`.
- Workload required capacity already includes weight, KV, and runtime
  footprints; runtime is not added again by the adapter.
- Packing and physical-capacity assumptions are inherited unchanged from the
  canonical architecture resolvers.
- Capacity scope: `AGGREGATE_CAPACITY_FEASIBILITY_ONLY`.

## PASS / FAIL

**PASS** for the aggregate raw-capacity scenario.

## Open Questions

- Nonzero hardware reserve remains intentionally unevaluated.
- Physical placement, allocation granularity, and fragmentation remain
  outside this aggregate-only audit.

## STOP
