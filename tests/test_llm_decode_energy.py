"""Semantic tests for LLM decode memory dynamic energy per token."""

from __future__ import annotations

import math

import pytest

from om3dthermal.evaluator import (
    LLMDecodeMemoryEnergyMetrics,
    evaluate_llm_decode_memory_energy,
)
from om3dthermal.workload import (
    CapacityFeasibilityMetrics,
    LLMDecodeInput,
    LLMDecodeMetrics,
    evaluate_capacity_feasibility,
    evaluate_llm_decode,
)


def _workload(
    *,
    read_bytes: float,
    write_bytes: float,
    weight_footprint: float = 1.0,
    required_capacity: float = 1.0,
) -> LLMDecodeMetrics:
    return LLMDecodeMetrics(
        weight_footprint_bytes=weight_footprint,
        weight_active_per_step_bytes=weight_footprint,
        kv_footprint_bytes=0.0,
        runtime_bytes=0.0,
        required_capacity_bytes=required_capacity,
        weight_read_bytes_per_token=read_bytes,
        kv_read_bytes_per_token=0.0,
        kv_write_bytes_per_token=write_bytes,
        read_bytes_per_token=read_bytes,
        write_bytes_per_token=write_bytes,
        flops_per_token=0,
        flops_sanity_per_token=0,
        weight_activity_model="full_footprint",
        weight_reuse_model="tile_reuse",
        kv_read_model="full_reread",
    )


def _capacity(*, feasible: bool = True) -> CapacityFeasibilityMetrics:
    return CapacityFeasibilityMetrics(
        physical_capacity_bytes=2,
        reserved_capacity_bytes=0,
        usable_capacity_bytes=2,
        required_capacity_bytes=1.0,
        capacity_margin_bytes=(1.0 if feasible else -1.0),
        capacity_utilization=(0.5 if feasible else 1.5),
        utilization_status="DEFINED",
        capacity_feasible=feasible,
        scope_status="AGGREGATE_CAPACITY_FEASIBILITY_ONLY",
    )


def _evaluate(
    workload: LLMDecodeMetrics,
    capacity: CapacityFeasibilityMetrics | None = None,
    *,
    read_energy: int | float = 2,
    write_energy: int | float = 3,
) -> LLMDecodeMemoryEnergyMetrics:
    return evaluate_llm_decode_memory_energy(
        workload,
        capacity or _capacity(),
        read_energy_pj_per_bit=read_energy,
        write_energy_pj_per_bit=write_energy,
    )


def test_unit_hand_check_and_dimensional_closure() -> None:
    result = _evaluate(_workload(read_bytes=1.0, write_bytes=1.0))
    assert result.read_bits_per_token == 8
    assert result.write_bits_per_token == 8
    assert result.read_dynamic_energy_pj_per_token == 16
    assert result.write_dynamic_energy_pj_per_token == 24
    assert result.memory_dynamic_energy_pj_per_token == 40
    assert result.memory_dynamic_energy_j_per_token == 4.0e-11
    assert result.memory_dynamic_energy_pj_per_token == (
        result.read_dynamic_energy_pj_per_token
        + result.write_dynamic_energy_pj_per_token)
    assert result.memory_dynamic_energy_j_per_token == (
        result.memory_dynamic_energy_pj_per_token * 1e-12)


def test_read_and_write_energy_inputs_are_separate() -> None:
    workload = _workload(read_bytes=1.0, write_bytes=1.0)
    baseline = _evaluate(workload, read_energy=2, write_energy=3)
    changed_write = _evaluate(workload, read_energy=2, write_energy=7)
    changed_read = _evaluate(workload, read_energy=5, write_energy=3)

    assert changed_write.read_dynamic_energy_pj_per_token == (
        baseline.read_dynamic_energy_pj_per_token)
    assert changed_write.write_dynamic_energy_pj_per_token != (
        baseline.write_dynamic_energy_pj_per_token)
    assert changed_read.write_dynamic_energy_pj_per_token == (
        baseline.write_dynamic_energy_pj_per_token)
    assert changed_read.read_dynamic_energy_pj_per_token != (
        baseline.read_dynamic_energy_pj_per_token)


def test_zero_write_traffic_has_no_hidden_write_energy() -> None:
    result = _evaluate(
        _workload(read_bytes=1.0, write_bytes=0.0), write_energy=1e12)
    assert result.write_bits_per_token == 0
    assert result.write_dynamic_energy_pj_per_token == 0
    assert result.memory_dynamic_energy_pj_per_token == (
        result.read_dynamic_energy_pj_per_token)


def test_zero_read_traffic_has_zero_read_contribution() -> None:
    result = _evaluate(
        _workload(read_bytes=0.0, write_bytes=1.0), read_energy=1e12)
    assert result.read_bits_per_token == 0
    assert result.read_dynamic_energy_pj_per_token == 0
    assert result.memory_dynamic_energy_pj_per_token == (
        result.write_dynamic_energy_pj_per_token)


def test_zero_total_traffic_is_well_defined() -> None:
    result = _evaluate(_workload(read_bytes=0.0, write_bytes=0.0))
    assert result.memory_dynamic_energy_pj_per_token == 0
    assert result.memory_dynamic_energy_j_per_token == 0
    assert result.evaluation_status == (
        "EVALUATED_MEMORY_DYNAMIC_TRAFFIC_ENERGY")


def test_fractional_analytical_bytes_are_not_truncated() -> None:
    result = _evaluate(
        _workload(read_bytes=0.125, write_bytes=0.125),
        read_energy=2,
        write_energy=3,
    )
    assert result.read_bits_per_token == 1.0
    assert result.write_bits_per_token == 1.0
    assert result.read_dynamic_energy_pj_per_token == 2.0
    assert result.write_dynamic_energy_pj_per_token == 3.0
    assert result.memory_dynamic_energy_pj_per_token == 5.0


def test_footprint_and_required_capacity_do_not_enter_energy_equation() -> None:
    small = _workload(
        read_bytes=4.0, write_bytes=2.0,
        weight_footprint=1.0, required_capacity=2.0)
    large = _workload(
        read_bytes=4.0, write_bytes=2.0,
        weight_footprint=1e15, required_capacity=2e15)
    small_result = _evaluate(small)
    large_result = _evaluate(large)
    assert small_result.memory_dynamic_energy_pj_per_token == (
        large_result.memory_dynamic_energy_pj_per_token)


def test_per_token_traffic_is_not_batch_scaled_again() -> None:
    workload = evaluate_llm_decode(LLMDecodeInput(
        n_param=8,
        n_layers=2,
        n_heads_q=2,
        n_heads_kv=1,
        d_head=2,
        d_model=4,
        d_ff=8,
        vocab_size=16,
        batch_size=8,
        context_length=4,
        weight_bits=8,
        kv_bits=8,
        runtime_bytes=0,
    ))
    result = _evaluate(workload, read_energy=2, write_energy=3)
    assert result.read_dynamic_energy_pj_per_token == (
        workload.read_bytes_per_token * 8 * 2)
    assert result.write_dynamic_energy_pj_per_token == (
        workload.write_bytes_per_token * 8 * 3)


def test_capacity_infeasible_blocks_all_energy_results() -> None:
    result = _evaluate(
        _workload(read_bytes=1.0, write_bytes=1.0),
        _capacity(feasible=False),
    )
    assert result.capacity_feasible is False
    assert result.evaluation_status == "CAPACITY_INFEASIBLE"
    assert result.read_dynamic_energy_pj_per_token is None
    assert result.write_dynamic_energy_pj_per_token is None
    assert result.memory_dynamic_energy_pj_per_token is None
    assert result.memory_dynamic_energy_j_per_token is None


def test_capacity_exact_fit_allows_evaluation() -> None:
    workload = _workload(
        read_bytes=1.0, write_bytes=1.0, required_capacity=10.0)
    capacity = evaluate_capacity_feasibility(
        workload,
        physical_capacity_bytes=10,
        reserved_capacity_bytes=0,
    )
    assert capacity.capacity_margin_bytes == 0
    assert capacity.capacity_feasible is True
    result = _evaluate(workload, capacity)
    assert result.memory_dynamic_energy_j_per_token == 4.0e-11


@pytest.mark.parametrize(
    "invalid",
    [-1, math.nan, math.inf, -math.inf, True, False, "1", None],
)
@pytest.mark.parametrize("field", ["read", "write"])
def test_invalid_energy_inputs_are_rejected(invalid, field: str) -> None:
    values = {"read": 2, "write": 3}
    values[field] = invalid
    with pytest.raises((TypeError, ValueError)):
        _evaluate(
            _workload(read_bytes=1.0, write_bytes=1.0),
            read_energy=values["read"],
            write_energy=values["write"],
        )


def test_zero_energy_inputs_are_allowed() -> None:
    result = _evaluate(
        _workload(read_bytes=10.0, write_bytes=10.0),
        read_energy=0,
        write_energy=0,
    )
    assert result.memory_dynamic_energy_pj_per_token == 0


def test_finite_large_energy_inputs_are_allowed() -> None:
    result = _evaluate(
        _workload(read_bytes=1.0, write_bytes=1.0),
        read_energy=1e200,
        write_energy=1e200,
    )
    assert math.isfinite(result.memory_dynamic_energy_pj_per_token)


def test_both_energy_parameters_are_mandatory() -> None:
    workload = _workload(read_bytes=1.0, write_bytes=1.0)
    capacity = _capacity()
    with pytest.raises(TypeError, match="read_energy_pj_per_bit"):
        evaluate_llm_decode_memory_energy(  # type: ignore[call-arg]
            workload, capacity, write_energy_pj_per_bit=3)
    with pytest.raises(TypeError, match="write_energy_pj_per_bit"):
        evaluate_llm_decode_memory_energy(  # type: ignore[call-arg]
            workload, capacity, read_energy_pj_per_bit=2)


def test_scope_status_and_exclusions_are_explicit() -> None:
    result = _evaluate(_workload(read_bytes=1.0, write_bytes=1.0))
    assert result.energy_scope_status == "MEMORY_DYNAMIC_TRAFFIC_ENERGY_ONLY"
    assert result.excluded_accounting_components == (
        "COMPUTE_ENERGY",
        "REFRESH_ENERGY",
        "BACKGROUND_STATIC_ENERGY",
        "POWER_DERIVATION",
        "THERMAL_EFFECTS",
    )


def test_output_model_has_no_forbidden_system_metrics() -> None:
    forbidden = {
        "bandwidth",
        "memory_time",
        "compute_time",
        "tokens_per_second",
        "power",
        "temperature",
        "tmax",
        "gpu_energy",
        "refresh_energy",
        "background_energy",
        "j_per_token",
        "total_j_per_token",
        "system_energy_j_per_token",
    }
    assert forbidden.isdisjoint(LLMDecodeMemoryEnergyMetrics.model_fields)
