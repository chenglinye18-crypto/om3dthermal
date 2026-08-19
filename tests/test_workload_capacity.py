"""Semantic tests for aggregate-only workload capacity feasibility."""

from __future__ import annotations

import math

import pytest

from om3dthermal.workload import (
    LLMDecodeInput,
    LLMDecodeMetrics,
    evaluate_capacity_feasibility,
    evaluate_llm_decode,
)


def _metrics_with_required(required: float) -> LLMDecodeMetrics:
    """Build an isolated workload result for capacity-boundary arithmetic."""
    return LLMDecodeMetrics(
        weight_footprint_bytes=required,
        weight_active_per_step_bytes=required,
        kv_footprint_bytes=0.0,
        runtime_bytes=0.0,
        required_capacity_bytes=required,
        weight_read_bytes_per_token=0.0,
        kv_read_bytes_per_token=0.0,
        kv_write_bytes_per_token=0.0,
        read_bytes_per_token=0.0,
        write_bytes_per_token=0.0,
        flops_per_token=0,
        flops_sanity_per_token=0,
        weight_activity_model="full_footprint",
        weight_reuse_model="tile_reuse",
        kv_read_model="full_reread",
    )


def _decode_input(**overrides: int) -> LLMDecodeInput:
    values = {
        "n_param": 8,
        "n_layers": 2,
        "n_heads_q": 2,
        "n_heads_kv": 1,
        "d_head": 2,
        "d_model": 4,
        "d_ff": 8,
        "vocab_size": 16,
        "batch_size": 1,
        "context_length": 1,
        "weight_bits": 8,
        "kv_bits": 8,
        "runtime_bytes": 0,
    }
    values.update(overrides)
    return LLMDecodeInput(**values)


def test_exact_fit_boundary_is_feasible() -> None:
    result = evaluate_capacity_feasibility(
        _metrics_with_required(80),
        physical_capacity_bytes=100,
        reserved_capacity_bytes=20,
    )
    assert result.usable_capacity_bytes == 80
    assert result.capacity_margin_bytes == 0
    assert result.capacity_utilization == 1.0
    assert result.capacity_feasible is True


def test_feasible_with_positive_margin() -> None:
    result = evaluate_capacity_feasibility(
        _metrics_with_required(60),
        physical_capacity_bytes=100,
        reserved_capacity_bytes=10,
    )
    assert result.usable_capacity_bytes == 90
    assert result.capacity_margin_bytes == 30
    assert result.capacity_utilization == 60 / 90
    assert result.capacity_feasible is True


def test_infeasible_with_negative_margin() -> None:
    result = evaluate_capacity_feasibility(
        _metrics_with_required(81),
        physical_capacity_bytes=100,
        reserved_capacity_bytes=20,
    )
    assert result.usable_capacity_bytes == 80
    assert result.capacity_margin_bytes == -1
    assert result.capacity_feasible is False


def test_reserve_larger_than_physical_is_rejected() -> None:
    with pytest.raises(ValueError, match="must not exceed"):
        evaluate_capacity_feasibility(
            _metrics_with_required(0),
            physical_capacity_bytes=100,
            reserved_capacity_bytes=101,
        )


@pytest.mark.parametrize(
    "physical,reserved",
    [(-1, 0), (100, -1)],
)
def test_negative_capacities_are_rejected(
    physical: int, reserved: int,
) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        evaluate_capacity_feasibility(
            _metrics_with_required(0),
            physical_capacity_bytes=physical,
            reserved_capacity_bytes=reserved,
        )


@pytest.mark.parametrize("invalid", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("field", ["physical", "reserved"])
def test_non_finite_capacities_are_rejected(
    invalid: float, field: str,
) -> None:
    values = {"physical": 100.0, "reserved": 0.0}
    values[field] = invalid
    with pytest.raises(ValueError, match="finite"):
        evaluate_capacity_feasibility(
            _metrics_with_required(0),
            physical_capacity_bytes=values["physical"],
            reserved_capacity_bytes=values["reserved"],
        )


@pytest.mark.parametrize("invalid", [math.nan, math.inf, -math.inf])
def test_non_finite_workload_requirement_is_rejected(invalid: float) -> None:
    with pytest.raises(ValueError, match="must be finite"):
        evaluate_capacity_feasibility(
            _metrics_with_required(invalid),
            physical_capacity_bytes=100,
            reserved_capacity_bytes=0,
        )


def test_negative_workload_requirement_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be non-negative"):
        evaluate_capacity_feasibility(
            _metrics_with_required(-1),
            physical_capacity_bytes=100,
            reserved_capacity_bytes=0,
        )


def test_fractional_workload_bytes_are_preserved() -> None:
    result = evaluate_capacity_feasibility(
        _metrics_with_required(0.125),
        physical_capacity_bytes=1,
        reserved_capacity_bytes=0,
    )
    assert result.required_capacity_bytes == 0.125
    assert result.capacity_margin_bytes == 0.875
    assert result.capacity_utilization == 0.125


def test_zero_required_and_zero_usable_is_empty_exact_fit() -> None:
    result = evaluate_capacity_feasibility(
        _metrics_with_required(0),
        physical_capacity_bytes=0,
        reserved_capacity_bytes=0,
    )
    assert result.capacity_feasible is True
    assert result.capacity_margin_bytes == 0
    assert result.capacity_utilization == 0.0
    assert result.utilization_status == "DEFINED_ZERO_REQUIRED_ZERO_USABLE"


def test_positive_required_and_zero_usable_has_undefined_utilization() -> None:
    result = evaluate_capacity_feasibility(
        _metrics_with_required(1),
        physical_capacity_bytes=10,
        reserved_capacity_bytes=10,
    )
    assert result.capacity_feasible is False
    assert result.capacity_margin_bytes == -1
    assert result.capacity_utilization is None
    assert result.utilization_status == "UNDEFINED_ZERO_USABLE_CAPACITY"


def test_runtime_is_already_in_required_and_is_not_added_again() -> None:
    workload = evaluate_llm_decode(_decode_input(runtime_bytes=7))
    expected = (
        workload.weight_footprint_bytes
        + workload.kv_footprint_bytes
        + workload.runtime_bytes
    )
    result = evaluate_capacity_feasibility(
        workload,
        physical_capacity_bytes=100,
        reserved_capacity_bytes=0,
    )
    assert workload.required_capacity_bytes == expected
    assert result.required_capacity_bytes == workload.required_capacity_bytes
    assert result.capacity_margin_bytes == 100 - expected


def test_batch_and_context_propagate_only_through_decode_metrics() -> None:
    base = evaluate_llm_decode(
        _decode_input(batch_size=1, context_length=1))
    larger_batch = evaluate_llm_decode(
        _decode_input(batch_size=2, context_length=1))
    larger_context = evaluate_llm_decode(
        _decode_input(batch_size=1, context_length=2))

    capacity = base.required_capacity_bytes
    base_result = evaluate_capacity_feasibility(
        base, physical_capacity_bytes=capacity, reserved_capacity_bytes=0)
    batch_result = evaluate_capacity_feasibility(
        larger_batch,
        physical_capacity_bytes=capacity,
        reserved_capacity_bytes=0,
    )
    context_result = evaluate_capacity_feasibility(
        larger_context,
        physical_capacity_bytes=capacity,
        reserved_capacity_bytes=0,
    )

    assert base_result.capacity_feasible is True
    assert batch_result.required_capacity_bytes == larger_batch.required_capacity_bytes
    assert context_result.required_capacity_bytes == larger_context.required_capacity_bytes
    assert larger_batch.kv_footprint_bytes > base.kv_footprint_bytes
    assert larger_context.kv_footprint_bytes > base.kv_footprint_bytes
    assert batch_result.capacity_feasible is False
    assert context_result.capacity_feasible is False


def test_scope_status_is_aggregate_only() -> None:
    result = evaluate_capacity_feasibility(
        _metrics_with_required(1),
        physical_capacity_bytes=1,
        reserved_capacity_bytes=0,
    )
    assert result.scope_status == "AGGREGATE_CAPACITY_FEASIBILITY_ONLY"
