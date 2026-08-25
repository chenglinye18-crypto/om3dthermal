"""Targeted tests for the LLM decode performance primitive (matched scenario)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from om3dthermal.architecture_capacity import (
    ResolvedArchitectureCapacity,
    resolve_architecture_capacity,
)
from om3dthermal.evaluator import (
    LLMDecodePerformanceMetrics,
    evaluate_llm_decode_performance,
)
from om3dthermal.power import (
    load_case_config,
    resolve_case_geometry,
    resolve_system_power,
)
from om3dthermal.workload import (
    ArchitectureCapacityFeasibility,
    LLMDecodeInput,
    LLMDecodeMetrics,
    evaluate_architecture_capacity_feasibility,
    evaluate_llm_decode,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Matched-reference scenario used everywhere in v0.
MATCHED_BW_BITS_PER_S = 39.2e12            # 39.2 Tb/s
ILLUSTRATIVE_COMPUTE_FLOP_PER_S = 100e12   # 100 TFLOP/s

ARCHITECTURES = (
    "conventional_hbm_2x1",
    "orthogonal_si",
    "orthogonal_m3d_igzo",
)

ROOT = Path(__file__).parents[1]
CASES = ROOT / "configs" / "cases"


def _build_workload(**overrides) -> LLMDecodeMetrics:
    """Build a frozen LLM-3.1-8B-class workload (B=1, S=131072, 16-bit)."""
    defaults = dict(
        n_param=8_000_000_000,
        n_layers=32,
        n_heads_q=32,
        n_heads_kv=8,
        d_model=4096,
        d_ff=14336,
        vocab_size=128_256,
        batch_size=1,
        context_length=131_072,
        weight_bits=16,
        kv_bits=16,
        runtime_bytes=0,
    )
    defaults.update(overrides)
    return evaluate_llm_decode(LLMDecodeInput(**defaults))


def _build_workload_metrics(
    *,
    read_bytes_per_token: float,
    write_bytes_per_token: float,
    flops_per_token: int,
    required_capacity_bytes: float = 1.0,
) -> LLMDecodeMetrics:
    """Build a synthetic LLMDecodeMetrics for arithmetic / unit tests.

    Only the fields consumed by the performance primitive are populated
    with non-zero values; other fields are zero.  This is independent
    of the frozen workload and is the right tool for unit tests that
    want to exercise arithmetic without dragging in the full LLaMA
    architecture.
    """
    return LLMDecodeMetrics(
        weight_footprint_bytes=0.0,
        weight_active_per_step_bytes=0.0,
        kv_footprint_bytes=0.0,
        runtime_bytes=0.0,
        required_capacity_bytes=required_capacity_bytes,
        weight_read_bytes_per_token=0.0,
        kv_read_bytes_per_token=0.0,
        kv_write_bytes_per_token=0.0,
        read_bytes_per_token=read_bytes_per_token,
        write_bytes_per_token=write_bytes_per_token,
        flops_per_token=flops_per_token,
        flops_sanity_per_token=flops_per_token,
        weight_activity_model="full_footprint",
        weight_reuse_model="tile_reuse",
        kv_read_model="full_reread",
    )


def _capacity_feasibility(
    architecture: str,
    *,
    feasible: bool,
    required_capacity_bytes: float = 1.0,
    physical_capacity_bytes: float = 2.0,
) -> ArchitectureCapacityFeasibility:
    """Build a synthetic ArchitectureCapacityFeasibility for testing."""
    usable = physical_capacity_bytes
    return ArchitectureCapacityFeasibility(
        architecture=architecture,
        physical_capacity_bytes=physical_capacity_bytes,
        physical_capacity_GiB=physical_capacity_bytes / 2**30,
        reserved_capacity_bytes=0,
        usable_capacity_bytes=usable,
        required_capacity_bytes=required_capacity_bytes,
        capacity_margin_bytes=usable - required_capacity_bytes,
        capacity_utilization=(
            required_capacity_bytes / usable if usable > 0 else None),
        capacity_feasible=feasible,
        capacity_scope_status="AGGREGATE_CAPACITY_FEASIBILITY_ONLY",
        capacity_source_status="ANALYTICAL_PACKING_DIAGNOSTICS_BIT_CLOSURE",
    )


def _resolve_architecture_capacity(name: str) -> ResolvedArchitectureCapacity:
    case = load_case_config(CASES / f"{name}.yaml")
    geometry = resolve_case_geometry(case)
    system = resolve_system_power(case, project_root=ROOT, geometry=geometry)
    return resolve_architecture_capacity(case, geometry, system)


# ---------------------------------------------------------------------------
# 1. Bit/byte conversion
# ---------------------------------------------------------------------------

def test_bit_byte_conversion() -> None:
    """1 read byte + 1 write byte must give traffic_bits_per_token = 16."""
    workload = _build_workload_metrics(
        read_bytes_per_token=1.0,
        write_bytes_per_token=1.0,
        flops_per_token=0,
    )
    cap = _capacity_feasibility("test", feasible=True)
    r = evaluate_llm_decode_performance(
        workload, cap,
        batch_size=1,
        matched_payload_bandwidth_bits_per_second=1000.0,
        effective_compute_flops_per_second=1000.0,
    )
    assert r.traffic_bits_per_token == 16.0
    # 16 bits / 1000 bit/s = 0.016 s
    assert r.memory_time_per_token_equivalent_s == 16.0 / 1000.0


# ---------------------------------------------------------------------------
# 2. Memory time
# ---------------------------------------------------------------------------

def test_memory_time_arithmetic() -> None:
    """traffic=100 bits/token, bw=1000 bit/s -> Tmemory = 0.1 s."""
    workload = _build_workload_metrics(
        read_bytes_per_token=12.5,        # 12.5 * 8 = 100 bits
        write_bytes_per_token=0.0,
        flops_per_token=0,
    )
    cap = _capacity_feasibility("test", feasible=True)
    r = evaluate_llm_decode_performance(
        workload, cap, batch_size=1,
        matched_payload_bandwidth_bits_per_second=1000.0,
        effective_compute_flops_per_second=1e18,  # far above crossover
    )
    assert r.memory_time_per_token_equivalent_s == 0.1


# ---------------------------------------------------------------------------
# 3. Compute time
# ---------------------------------------------------------------------------

def test_compute_time_arithmetic() -> None:
    """FLOPs=200, throughput=1000 FLOP/s -> Tcompute = 0.2 s."""
    workload = _build_workload_metrics(
        read_bytes_per_token=0.0,         # zero memory time
        write_bytes_per_token=0.0,
        flops_per_token=200,
    )
    cap = _capacity_feasibility("test", feasible=True)
    r = evaluate_llm_decode_performance(
        workload, cap, batch_size=1,
        matched_payload_bandwidth_bits_per_second=1000.0,
        effective_compute_flops_per_second=1000.0,
    )
    assert r.compute_time_per_token_equivalent_s == 0.2


# ---------------------------------------------------------------------------
# 4. Memory-bound
# ---------------------------------------------------------------------------

def test_memory_bound() -> None:
    """When Tmemory > Tcompute, bottleneck must be MEMORY."""
    workload = _build_workload_metrics(
        read_bytes_per_token=1000.0,      # 8000 bits / 1000 bit/s = 8 s
        write_bytes_per_token=0.0,
        flops_per_token=100,              # 100 / 1000 = 0.1 s
    )
    cap = _capacity_feasibility("test", feasible=True)
    r = evaluate_llm_decode_performance(
        workload, cap, batch_size=1,
        matched_payload_bandwidth_bits_per_second=1000.0,
        effective_compute_flops_per_second=1000.0,
    )
    assert r.bottleneck == "MEMORY"


# ---------------------------------------------------------------------------
# 5. Compute-bound
# ---------------------------------------------------------------------------

def test_compute_bound() -> None:
    """When Tcompute > Tmemory, bottleneck must be COMPUTE."""
    workload = _build_workload_metrics(
        read_bytes_per_token=1.0,         # 8 bits / 1000 = 0.008 s
        write_bytes_per_token=0.0,
        flops_per_token=10_000,           # 10000 / 1000 = 10 s
    )
    cap = _capacity_feasibility("test", feasible=True)
    r = evaluate_llm_decode_performance(
        workload, cap, batch_size=1,
        matched_payload_bandwidth_bits_per_second=1000.0,
        effective_compute_flops_per_second=1000.0,
    )
    assert r.bottleneck == "COMPUTE"


# ---------------------------------------------------------------------------
# 6. Balanced
# ---------------------------------------------------------------------------

def test_balanced_when_memory_and_compute_time_equal() -> None:
    """When Tmemory == Tcompute, bottleneck must be BALANCED."""
    # traffic=1000 bits, bw=1000 -> Tmemory = 1.0 s
    # flops=1000, throughput=1000 -> Tcompute = 1.0 s
    workload = _build_workload_metrics(
        read_bytes_per_token=125.0,       # 1000 bits
        write_bytes_per_token=0.0,
        flops_per_token=1000,
    )
    cap = _capacity_feasibility("test", feasible=True)
    r = evaluate_llm_decode_performance(
        workload, cap, batch_size=1,
        matched_payload_bandwidth_bits_per_second=1000.0,
        effective_compute_flops_per_second=1000.0,
    )
    assert r.memory_time_per_token_equivalent_s == 1.0
    assert r.compute_time_per_token_equivalent_s == 1.0
    assert r.bottleneck == "BALANCED"


# ---------------------------------------------------------------------------
# 7. Capacity gate
# ---------------------------------------------------------------------------

def test_capacity_infeasible_blocks_performance() -> None:
    """When capacity_feasible=False, the evaluator must NOT emit a
    tokens/s number.  All time/throughput fields are None; status is
    BLOCKED_BY_CAPACITY; bottleneck is NOT_EVALUATED_CAPACITY_INFEASIBLE.
    """
    workload = _build_workload_metrics(
        read_bytes_per_token=1.0,
        write_bytes_per_token=1.0,
        flops_per_token=200,
    )
    cap = _capacity_feasibility(
        "test", feasible=False, required_capacity_bytes=10.0,
        physical_capacity_bytes=5.0,
    )
    r = evaluate_llm_decode_performance(
        workload, cap, batch_size=1,
        matched_payload_bandwidth_bits_per_second=39.2e12,
        effective_compute_flops_per_second=100e12,
    )
    assert r.capacity_feasible is False
    assert r.performance_status == "BLOCKED_BY_CAPACITY"
    assert r.bottleneck == "NOT_EVALUATED_CAPACITY_INFEASIBLE"
    assert r.memory_time_per_token_equivalent_s is None
    assert r.compute_time_per_token_equivalent_s is None
    assert r.token_equivalent_time_s is None
    assert r.aggregate_step_time_s is None
    assert r.aggregate_tokens_per_second is None
    assert r.per_sequence_tokens_per_second is None
    assert r.per_sequence_step_latency_s is None
    assert r.compute_throughput_required_to_match_memory_flops_per_second is None


# ---------------------------------------------------------------------------
# 8. Aggregate / per-sequence batch semantics
# ---------------------------------------------------------------------------

def test_aggregate_and_per_sequence_batch_semantics_at_b8() -> None:
    """With B=8, per-sequence throughput must equal aggregate / 8, and
    per-sequence step latency must equal aggregate step time.
    """
    workload = _build_workload_metrics(
        read_bytes_per_token=1000.0,      # 8000 bits / 1000 = 8 s memory
        write_bytes_per_token=0.0,
        flops_per_token=100,              # 0.1 s compute -> memory-bound
    )
    cap = _capacity_feasibility("test", feasible=True)
    r = evaluate_llm_decode_performance(
        workload, cap, batch_size=8,
        matched_payload_bandwidth_bits_per_second=1000.0,
        effective_compute_flops_per_second=1000.0,
    )
    tet = r.token_equivalent_time_s
    assert tet == 8.0
    assert r.aggregate_step_time_s == 8.0 * tet
    assert r.aggregate_tokens_per_second == 1.0 / tet
    assert r.per_sequence_tokens_per_second == r.aggregate_tokens_per_second / 8.0
    assert r.per_sequence_step_latency_s == r.aggregate_step_time_s


# ---------------------------------------------------------------------------
# 9. Same matched scenario across architectures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def capacities() -> dict[str, ResolvedArchitectureCapacity]:
    return {name: _resolve_architecture_capacity(name) for name in ARCHITECTURES}


def test_same_scenario_same_three_architectures(
    capacities: dict[str, ResolvedArchitectureCapacity],
) -> None:
    """Three capacity-feasible architectures using the same workload,
    bandwidth, and compute throughput must produce identical
    Tmemory, Tcompute, and tokens/s values.  Architecture-specific
    capability BW and compute are explicitly NOT used in v0.
    """
    workload = _build_workload()
    feasibilities = [
        evaluate_architecture_capacity_feasibility(
            workload, capacities[name], reserved_capacity_bytes=0,
        )
        for name in ARCHITECTURES
    ]
    assert all(f.capacity_feasible for f in feasibilities), (
        "all three architectures must be capacity-feasible for the "
        "frozen 8B-class workload")

    results = [
        evaluate_llm_decode_performance(
            workload, feasibilities[i], batch_size=1,
            matched_payload_bandwidth_bits_per_second=MATCHED_BW_BITS_PER_S,
            effective_compute_flops_per_second=ILLUSTRATIVE_COMPUTE_FLOP_PER_S,
        )
        for i in range(len(ARCHITECTURES))
    ]

    tmems = [r.memory_time_per_token_equivalent_s for r in results]
    tcomps = [r.compute_time_per_token_equivalent_s for r in results]
    agg = [r.aggregate_tokens_per_second for r in results]
    pst = [r.per_sequence_tokens_per_second for r in results]
    bn = [r.bottleneck for r in results]

    assert all(t == tmems[0] for t in tmems)
    assert all(t == tcomps[0] for t in tcomps)
    assert all(a == agg[0] for a in agg)
    assert all(p == pst[0] for p in pst)
    assert all(b == bn[0] for b in bn)


# ---------------------------------------------------------------------------
# 10. Crossover throughput
# ---------------------------------------------------------------------------

def test_crossover_compute_throughput_required_to_match_memory() -> None:
    """crossover_FLOPs_per_s = flops_per_token / memory_time."""
    workload = _build_workload_metrics(
        read_bytes_per_token=125.0,       # 1000 bits / 1000 bit/s = 1.0 s
        write_bytes_per_token=0.0,
        flops_per_token=2000,
    )
    cap = _capacity_feasibility("test", feasible=True)
    r = evaluate_llm_decode_performance(
        workload, cap, batch_size=1,
        matched_payload_bandwidth_bits_per_second=1000.0,
        effective_compute_flops_per_second=1000.0,
    )
    # memory_time = 1.0 s; required compute = 2000 / 1.0 = 2000 FLOPs/s
    assert r.memory_time_per_token_equivalent_s == 1.0
    assert r.compute_throughput_required_to_match_memory_flops_per_second == 2000.0


def test_crossover_throughput_is_infinity_when_memory_time_zero() -> None:
    """Divide-by-zero policy: when memory_time == 0, crossover = +inf.

    ``math.inf`` is the documented sentinel; callers can detect it
    with ``math.isinf``.
    """
    workload = _build_workload_metrics(
        read_bytes_per_token=0.0,
        write_bytes_per_token=0.0,
        flops_per_token=2000,
    )
    cap = _capacity_feasibility("test", feasible=True)
    r = evaluate_llm_decode_performance(
        workload, cap, batch_size=1,
        matched_payload_bandwidth_bits_per_second=1000.0,
        effective_compute_flops_per_second=1000.0,
    )
    assert r.memory_time_per_token_equivalent_s == 0.0
    assert math.isinf(
        r.compute_throughput_required_to_match_memory_flops_per_second
    )


# ---------------------------------------------------------------------------
# 11. Invalid inputs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_batch", [0, -1, -100])
def test_rejects_nonpositive_batch_size(bad_batch: int) -> None:
    workload = _build_workload_metrics(
        read_bytes_per_token=1.0, write_bytes_per_token=0.0, flops_per_token=0)
    cap = _capacity_feasibility("test", feasible=True)
    with pytest.raises(ValueError):
        evaluate_llm_decode_performance(
            workload, cap, batch_size=bad_batch,
            matched_payload_bandwidth_bits_per_second=1000.0,
            effective_compute_flops_per_second=1000.0,
        )


@pytest.mark.parametrize("bad_bw", [0.0, -1.0, -1e9])
def test_rejects_nonpositive_bandwidth(bad_bw: float) -> None:
    workload = _build_workload_metrics(
        read_bytes_per_token=1.0, write_bytes_per_token=0.0, flops_per_token=0)
    cap = _capacity_feasibility("test", feasible=True)
    with pytest.raises(ValueError):
        evaluate_llm_decode_performance(
            workload, cap, batch_size=1,
            matched_payload_bandwidth_bits_per_second=bad_bw,
            effective_compute_flops_per_second=1000.0,
        )


@pytest.mark.parametrize("bad_thr", [0.0, -1.0, -1e9])
def test_rejects_nonpositive_compute_throughput(bad_thr: float) -> None:
    workload = _build_workload_metrics(
        read_bytes_per_token=1.0, write_bytes_per_token=0.0, flops_per_token=0)
    cap = _capacity_feasibility("test", feasible=True)
    with pytest.raises(ValueError):
        evaluate_llm_decode_performance(
            workload, cap, batch_size=1,
            matched_payload_bandwidth_bits_per_second=1000.0,
            effective_compute_flops_per_second=bad_thr,
        )


@pytest.mark.parametrize("bad_value", [
    float("nan"), float("inf"), float("-inf"),
])
def test_rejects_nonfinite_bandwidth(bad_value: float) -> None:
    workload = _build_workload_metrics(
        read_bytes_per_token=1.0, write_bytes_per_token=0.0, flops_per_token=0)
    cap = _capacity_feasibility("test", feasible=True)
    with pytest.raises(ValueError):
        evaluate_llm_decode_performance(
            workload, cap, batch_size=1,
            matched_payload_bandwidth_bits_per_second=bad_value,
            effective_compute_flops_per_second=1000.0,
        )


@pytest.mark.parametrize("bad_value", [
    float("nan"), float("inf"), float("-inf"),
])
def test_rejects_nonfinite_compute_throughput(bad_value: float) -> None:
    workload = _build_workload_metrics(
        read_bytes_per_token=1.0, write_bytes_per_token=0.0, flops_per_token=0)
    cap = _capacity_feasibility("test", feasible=True)
    with pytest.raises(ValueError):
        evaluate_llm_decode_performance(
            workload, cap, batch_size=1,
            matched_payload_bandwidth_bits_per_second=1000.0,
            effective_compute_flops_per_second=bad_value,
        )


def test_rejects_negative_read_bytes_per_token() -> None:
    workload = _build_workload_metrics(
        read_bytes_per_token=-1.0,
        write_bytes_per_token=0.0,
        flops_per_token=0,
    )
    cap = _capacity_feasibility("test", feasible=True)
    with pytest.raises(ValueError):
        evaluate_llm_decode_performance(
            workload, cap, batch_size=1,
            matched_payload_bandwidth_bits_per_second=1000.0,
            effective_compute_flops_per_second=1000.0,
        )


def test_rejects_negative_write_bytes_per_token() -> None:
    workload = _build_workload_metrics(
        read_bytes_per_token=0.0,
        write_bytes_per_token=-1.0,
        flops_per_token=0,
    )
    cap = _capacity_feasibility("test", feasible=True)
    with pytest.raises(ValueError):
        evaluate_llm_decode_performance(
            workload, cap, batch_size=1,
            matched_payload_bandwidth_bits_per_second=1000.0,
            effective_compute_flops_per_second=1000.0,
        )


def test_rejects_negative_flops_per_token() -> None:
    # Pydantic would normally prevent this on the workload side via
    # the LLM primitive, but the evaluator double-checks defensively.
    workload = _build_workload_metrics(
        read_bytes_per_token=0.0, write_bytes_per_token=0.0, flops_per_token=-1)
    cap = _capacity_feasibility("test", feasible=True)
    with pytest.raises(ValueError):
        evaluate_llm_decode_performance(
            workload, cap, batch_size=1,
            matched_payload_bandwidth_bits_per_second=1000.0,
            effective_compute_flops_per_second=1000.0,
        )


# ---------------------------------------------------------------------------
# 12. Status provenance preservation
# ---------------------------------------------------------------------------

def test_status_provenance_labels_are_preserved() -> None:
    """The four frozen v0 labels must be echoed verbatim in the
    output regardless of whether the workload is feasible.
    """
    workload = _build_workload_metrics(
        read_bytes_per_token=1.0, write_bytes_per_token=0.0, flops_per_token=0)
    cap = _capacity_feasibility("test", feasible=True)
    r = evaluate_llm_decode_performance(
        workload, cap, batch_size=1,
        matched_payload_bandwidth_bits_per_second=1000.0,
        effective_compute_flops_per_second=1000.0,
    )
    assert r.bandwidth_status == "MATCHED_REFERENCE_NOT_CAPABILITY_VALIDATED"
    assert r.compute_throughput_status == "NUMERICAL_CHOICE_NOT_HARDWARE_VALIDATED"
    assert r.memory_bandwidth_model == "SHARED_READ_WRITE_PAYLOAD_BANDWIDTH"
    assert r.overlap_model == "ROOFLINE_MAX"


def test_status_provenance_preserved_even_when_blocked() -> None:
    """Provenance labels must also be preserved when the capacity
    gate is closed, so the audit trail does not lose scenario context
    just because the workload does not fit.
    """
    workload = _build_workload_metrics(
        read_bytes_per_token=1.0, write_bytes_per_token=0.0, flops_per_token=0)
    cap = _capacity_feasibility(
        "test", feasible=False, required_capacity_bytes=10.0,
        physical_capacity_bytes=5.0,
    )
    r = evaluate_llm_decode_performance(
        workload, cap, batch_size=1,
        matched_payload_bandwidth_bits_per_second=1000.0,
        effective_compute_flops_per_second=1000.0,
    )
    assert r.bandwidth_status == "MATCHED_REFERENCE_NOT_CAPABILITY_VALIDATED"
    assert r.compute_throughput_status == "NUMERICAL_CHOICE_NOT_HARDWARE_VALIDATED"
    assert r.memory_bandwidth_model == "SHARED_READ_WRITE_PAYLOAD_BANDWIDTH"
    assert r.overlap_model == "ROOFLINE_MAX"


# ---------------------------------------------------------------------------
# 13. No forbidden metrics in the output model
# ---------------------------------------------------------------------------

def test_output_model_has_no_forbidden_metrics() -> None:
    """The output model must NOT carry energy, J/token, power, Tmax,
    or thermal fields.  This test pins the public surface so a
    future refactor cannot silently add a forbidden metric.
    """
    forbidden = (
        "j_per_token",
        "energy",
        "power",
        "power_w",
        "tmax",
        "tmax_c",
        "tmax_k",
        "thermal",
        "watts",
    )
    field_names = set(LLMDecodePerformanceMetrics.model_fields.keys())
    for name in forbidden:
        assert name not in field_names, (
            f"LLMDecodePerformanceMetrics must not expose {name!r}")


# ---------------------------------------------------------------------------
# 14. Frozen LLaMA-3.1-8B-class scenario sanity
# ---------------------------------------------------------------------------

def test_frozen_llama_scenario_evaluates_memory_bound(
    capacities: dict[str, ResolvedArchitectureCapacity],
) -> None:
    """The frozen 8B-class workload under the matched-reference
    scenario (39.2 Tb/s payload, 100 TFLOP/s illustrative compute)
    is expected to be memory-bound.  Tmemory ~ 6.8 ms/token,
    Tcompute ~ 0.85 ms/token, aggregate throughput ~ 147 tok/s.
    These are sanity bands, not strict equalities; the gate here is
    that bottleneck == MEMORY and the order of magnitude matches.
    """
    workload = _build_workload()
    feasibilities = [
        evaluate_architecture_capacity_feasibility(
            workload, capacities[name], reserved_capacity_bytes=0,
        )
        for name in ARCHITECTURES
    ]
    r = evaluate_llm_decode_performance(
        workload, feasibilities[0], batch_size=1,
        matched_payload_bandwidth_bits_per_second=MATCHED_BW_BITS_PER_S,
        effective_compute_flops_per_second=ILLUSTRATIVE_COMPUTE_FLOP_PER_S,
    )
    assert r.capacity_feasible is True
    assert r.bottleneck == "MEMORY"
    # Order-of-magnitude bands.
    tmem_ms = r.memory_time_per_token_equivalent_s * 1e3
    tcmp_ms = r.compute_time_per_token_equivalent_s * 1e3
    agg = r.aggregate_tokens_per_second
    assert 5.0 < tmem_ms < 9.0
    assert 0.5 < tcmp_ms < 1.5
    assert 100 < agg < 200
    # And the architectural identity from test 9 carries through:
    # all three architectures are memory-bound and produce the same
    # numerical answer in this scenario.
    for f in feasibilities[1:]:
        r2 = evaluate_llm_decode_performance(
            workload, f, batch_size=1,
            matched_payload_bandwidth_bits_per_second=MATCHED_BW_BITS_PER_S,
            effective_compute_flops_per_second=ILLUSTRATIVE_COMPUTE_FLOP_PER_S,
        )
        assert r2.bottleneck == "MEMORY"
        assert r2.aggregate_tokens_per_second == agg
