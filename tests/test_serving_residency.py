"""Targeted capacity-residency tests for serving C1."""

from __future__ import annotations

import pytest

from om3dthermal.serving import ServingCapacitySource, evaluate_capacity_residency
from om3dthermal.workload import LLMDecodeInput, evaluate_llm_decode


def _input(**overrides) -> LLMDecodeInput:
    values = {
        "n_param": 8,
        "n_layers": 1,
        "n_heads_q": 1,
        "n_heads_kv": 1,
        "d_model": 1,
        "d_ff": 1,
        "vocab_size": 1,
        "batch_size": 4,
        "context_length": 4,
        "weight_bits": 8,
        "kv_bits": 8,
        "runtime_bytes": 0,
    }
    values.update(overrides)
    return LLMDecodeInput(**values)


def _capacity(value: float) -> ServingCapacitySource:
    return ServingCapacitySource(
        architecture="test",
        usable_capacity_bytes=value,
        capacity_source_status="TEST",
        provenance_status="SYNTHETIC_TEST_ONLY",
    )


def test_kv_per_request_closes_exactly_over_batch() -> None:
    metrics = evaluate_llm_decode(_input(batch_size=7))
    assert metrics.kv_footprint_bytes == 7 * metrics.kv_bytes_per_request


def test_legacy_runtime_is_fixed_and_explicit_split_is_opt_in() -> None:
    legacy = evaluate_llm_decode(_input(batch_size=3, runtime_bytes=11))
    assert legacy.runtime_fixed_bytes == 11
    assert legacy.runtime_per_request_bytes == 0
    assert legacy.runtime_bytes == 11
    assert legacy.runtime_capacity_semantics_status == (
        "LEGACY_RUNTIME_BYTES_AS_FIXED_MODELING_CHOICE")

    split = evaluate_llm_decode(_input(
        batch_size=3,
        runtime_bytes=0,
        runtime_fixed_bytes=5,
        runtime_per_request_bytes=2,
    ))
    assert split.runtime_bytes == 11
    assert split.runtime_capacity_semantics_status == (
        "EXPLICIT_FIXED_PLUS_PER_REQUEST_MODELING_CHOICE")


def test_explicit_runtime_split_rejects_legacy_double_counting() -> None:
    with pytest.raises(ValueError, match="double counting"):
        _input(runtime_bytes=1, runtime_fixed_bytes=1)


def test_exact_fit_and_one_byte_over_capacity_wall() -> None:
    metrics = evaluate_llm_decode(_input(batch_size=1))
    fixed = metrics.weight_footprint_bytes + metrics.runtime_fixed_bytes
    per_request = metrics.kv_bytes_per_request + metrics.runtime_per_request_bytes
    exact = evaluate_capacity_residency(
        metrics, _capacity(fixed + 3 * per_request), requested_requests=3)
    assert exact.max_resident_requests == 3
    assert exact.capacity_status == "FULLY_LOCAL"

    one_byte_short = evaluate_capacity_residency(
        metrics, _capacity(fixed + 3 * per_request - 1), requested_requests=3)
    assert one_byte_short.max_resident_requests == 2
    assert one_byte_short.capacity_status == "CAPACITY_PRESSURED"
    assert one_byte_short.spilled_requests == 1


def test_weights_not_fit_does_not_enter_normal_kv_residency() -> None:
    metrics = evaluate_llm_decode(_input(batch_size=1))
    result = evaluate_capacity_residency(
        metrics,
        _capacity(metrics.weight_footprint_bytes - 1),
        requested_requests=2,
    )
    assert result.capacity_status == "WEIGHTS_NOT_RESIDENT"
    assert result.max_resident_requests == 0
    assert result.local_resident_requests == 0


def test_fixed_and_per_request_runtime_affect_integer_floor() -> None:
    metrics = evaluate_llm_decode(_input(
        batch_size=1,
        runtime_fixed_bytes=5,
        runtime_per_request_bytes=3,
    ))
    fixed = metrics.weight_footprint_bytes + 5
    per_request = metrics.kv_bytes_per_request + 3
    result = evaluate_capacity_residency(
        metrics, _capacity(fixed + 2 * per_request + per_request - 1),
        requested_requests=4)
    assert result.max_resident_requests == 2
    assert result.local_resident_requests == 2
    assert result.spilled_requests == 2


def test_zero_kv_and_zero_runtime_per_request_is_unbounded() -> None:
    metrics = evaluate_llm_decode(_input(batch_size=1, context_length=0))
    result = evaluate_capacity_residency(
        metrics,
        _capacity(metrics.weight_footprint_bytes),
        requested_requests=1_000_000,
    )
    assert result.max_resident_requests is None
    assert result.capacity_status == "UNBOUNDED_PER_REQUEST_FOOTPRINT"
    assert result.local_resident_requests == 1_000_000
