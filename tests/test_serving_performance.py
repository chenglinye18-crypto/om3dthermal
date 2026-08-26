"""Targeted host-offload and GPU serving tests for C2-C7."""

from __future__ import annotations

import pytest

from om3dthermal.platform import HostOffloadSpec
from om3dthermal.provenance import ProvenanceRecord
from om3dthermal.serving import (
    AnalyticalRooflineGPUModel,
    HostOverlapSpec,
    MeasuredBatchCurveGPUModel,
    MeasuredBatchCurvePoint,
    ServingCapacitySource,
    evaluate_capacity_aware_serving,
)
from om3dthermal.workload import LLMDecodeInput


def _input() -> LLMDecodeInput:
    return LLMDecodeInput(
        n_param=8,
        n_layers=1,
        n_heads_q=1,
        n_heads_kv=1,
        d_model=1,
        d_ff=1,
        vocab_size=1,
        batch_size=1,
        context_length=4,
        weight_bits=8,
        kv_bits=8,
        runtime_bytes=0,
    )


def _provenance() -> tuple[ProvenanceRecord, ...]:
    return (ProvenanceRecord(
        record_id="synthetic_host_test",
        classification="NUMERICAL_CHOICE",
        source="unit test",
        status="SYNTHETIC_TEST_ONLY",
    ),)


def _host(memory: float, link: float, efficiency: float) -> HostOffloadSpec:
    return HostOffloadSpec(
        status="RESOLVED",
        host_memory_bandwidth_GBps=memory,
        host_device_link_bandwidth_GBps=link,
        host_offload_efficiency=efficiency,
        provenance=_provenance(),
    )


def _capacity(value: float) -> ServingCapacitySource:
    return ServingCapacitySource(
        architecture="test",
        usable_capacity_bytes=value,
        capacity_source_status="TEST",
        provenance_status="SYNTHETIC_TEST_ONLY",
    )


def _gpu() -> AnalyticalRooflineGPUModel:
    return AnalyticalRooflineGPUModel(
        matched_payload_bandwidth_bits_per_second=1e12,
        effective_compute_flops_per_second=1e12,
    )


def test_effective_host_bandwidth_selects_ddr_or_link_and_efficiency() -> None:
    assert _host(10, 20, 0.5).effective_bandwidth_bytes_per_second == 5e9
    assert _host(20, 10, 0.5).effective_bandwidth_bytes_per_second == 5e9
    assert _host(10, 20, 1.0).effective_bandwidth_bytes_per_second == 10e9


def test_zero_spill_has_zero_host_read_write_and_reproduces_gpu_baseline() -> None:
    workload = _input()
    gpu = _gpu()
    expected = gpu.evaluate(workload, batch_size=2)
    row = evaluate_capacity_aware_serving(
        architecture="test",
        workload_id="w",
        workload=workload,
        capacity=_capacity(1_000_000),
        requested_requests=2,
        host_offload=_host(10, 10, 1),
        overlap=HostOverlapSpec(policy="NO_OVERLAP", overlap_fraction=0),
        gpu_model=gpu,
    )
    assert row.host_read_bytes_per_step == 0
    assert row.host_write_bytes_per_step == 0
    assert row.host_transfer_bytes_per_step == 0
    assert row.gpu_decode_step_time_ms == expected.decode_step_time_ms
    assert row.total_decode_step_time_ms == expected.decode_step_time_ms
    assert row.aggregate_tokens_per_s == pytest.approx(
        expected.aggregate_tokens_per_s)


def test_spill_read_write_close_and_cannot_improve_performance() -> None:
    workload = _input()
    fixed_only = 8.0
    local = evaluate_capacity_aware_serving(
        architecture="test", workload_id="w", workload=workload,
        capacity=_capacity(1_000_000), requested_requests=2,
        host_offload=_host(1e-6, 1e-6, 1),
        overlap=HostOverlapSpec(policy="NO_OVERLAP", overlap_fraction=0),
        gpu_model=_gpu(),
    )
    spilled = evaluate_capacity_aware_serving(
        architecture="test", workload_id="w", workload=workload,
        capacity=_capacity(fixed_only), requested_requests=2,
        host_offload=_host(1e-6, 1e-6, 1),
        overlap=HostOverlapSpec(policy="NO_OVERLAP", overlap_fraction=0),
        gpu_model=_gpu(),
    )
    assert spilled.spilled_requests == 2
    assert spilled.host_transfer_bytes_per_step == (
        spilled.host_read_bytes_per_step + spilled.host_write_bytes_per_step)
    assert spilled.aggregate_tokens_per_s <= local.aggregate_tokens_per_s


def test_increasing_effective_host_bandwidth_cannot_worsen_performance() -> None:
    kwargs = dict(
        architecture="test", workload_id="w", workload=_input(),
        capacity=_capacity(8), requested_requests=2,
        overlap=HostOverlapSpec(policy="NO_OVERLAP", overlap_fraction=0),
        gpu_model=_gpu(),
    )
    slow = evaluate_capacity_aware_serving(
        **kwargs, host_offload=_host(1, 1, 0.5))
    fast = evaluate_capacity_aware_serving(
        **kwargs, host_offload=_host(1, 1, 1.0))
    assert fast.host_transfer_time_ms <= slow.host_transfer_time_ms
    assert fast.aggregate_tokens_per_s >= slow.aggregate_tokens_per_s


def test_overlap_penalty_ordering() -> None:
    kwargs = dict(
        architecture="test", workload_id="w", workload=_input(),
        capacity=_capacity(8), requested_requests=2,
        host_offload=_host(1e-6, 1e-6, 1), gpu_model=_gpu(),
    )
    no = evaluate_capacity_aware_serving(
        **kwargs, overlap=HostOverlapSpec(
            policy="NO_OVERLAP", overlap_fraction=0))
    partial = evaluate_capacity_aware_serving(
        **kwargs, overlap=HostOverlapSpec(
            policy="PARTIAL_OVERLAP", overlap_fraction=0.5))
    full = evaluate_capacity_aware_serving(
        **kwargs, overlap=HostOverlapSpec(
            policy="FULL_OVERLAP", overlap_fraction=1))
    assert full.host_penalty_time_ms <= partial.host_penalty_time_ms
    assert partial.host_penalty_time_ms <= no.host_penalty_time_ms
    assert full.total_decode_step_time_ms <= partial.total_decode_step_time_ms
    assert partial.total_decode_step_time_ms <= no.total_decode_step_time_ms


def test_unresolved_host_blocks_only_spilled_points() -> None:
    unresolved = HostOffloadSpec(
        status="UNRESOLVED", provenance=_provenance())
    all_local = evaluate_capacity_aware_serving(
        architecture="test", workload_id="w", workload=_input(),
        capacity=_capacity(1_000_000), requested_requests=1,
        host_offload=unresolved,
        overlap=HostOverlapSpec(policy="NO_OVERLAP", overlap_fraction=0),
        gpu_model=_gpu(),
    )
    spilled = evaluate_capacity_aware_serving(
        architecture="test", workload_id="w", workload=_input(),
        capacity=_capacity(8), requested_requests=1,
        host_offload=unresolved,
        overlap=HostOverlapSpec(policy="NO_OVERLAP", overlap_fraction=0),
        gpu_model=_gpu(),
    )
    assert all_local.evaluation_status == "EVALUATED"
    assert spilled.evaluation_status == "UNRESOLVED_HOST_BANDWIDTH"


def test_measured_curve_exact_interpolation_and_bounds_are_deterministic() -> None:
    points = (
        MeasuredBatchCurvePoint(
            batch_size=1, decode_step_ms=10, aggregate_tokens_per_s=100),
        MeasuredBatchCurvePoint(
            batch_size=3, decode_step_ms=30, aggregate_tokens_per_s=100),
    )
    model = MeasuredBatchCurveGPUModel(points)
    exact = model.evaluate(_input(), batch_size=1)
    interpolated = model.evaluate(_input(), batch_size=2)
    assert exact.decode_step_time_ms == 10
    assert interpolated.decode_step_time_ms == 20
    assert interpolated.aggregate_tokens_per_s == 100
    with pytest.raises(ValueError, match="outside"):
        model.evaluate(_input(), batch_size=4)
