"""Recurring historical-KV access and resident-only queue policy gates."""

from pathlib import Path

import pytest

from om3dthermal.serving import (
    PersistentMixedServiceCase,
    evaluate_conventional_overflow_policy,
    load_serving_e2e_closure_spec,
)
from om3dthermal.workload import load_dense_model_registry


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def inputs():
    registry = load_dense_model_registry(ROOT/"configs/workload/models")
    spec = load_serving_e2e_closure_spec(
        ROOT/"configs/experiment/serving_e2e_closure.yaml")
    return registry, spec.workspace


def _evaluate(inputs, *, S=131072, p=1, d=27, g=2,
              policy="HOST_KV_OFFLOAD"):
    registry, workspace = inputs
    case = PersistentMixedServiceCase(
        model_id="llama31_8b", context_length=S, prefill_requests=p,
        decode_requests=d, generated_decode_steps=g)
    return evaluate_conventional_overflow_policy(
        project_root=ROOT, model=registry["llama31_8b"], case=case,
        policy=policy, workspace_config=workspace)


def test_host_resident_active_history_is_read_every_step(inputs):
    result = _evaluate(inputs)
    assert result.status == "EVALUATED"
    assert len(result.steps) == 2
    assert all(step.host_historical_KV_bytes > 0 for step in result.steps)
    assert result.total_host_KV_read_bytes == sum(
        step.host_historical_KV_bytes for step in result.steps)


def test_growing_history_increases_second_step_and_conserves_bytes(inputs):
    result = _evaluate(inputs)
    first, second = result.steps
    assert second.required_historical_KV_bytes > first.required_historical_KV_bytes
    for step in result.steps:
        assert step.required_historical_KV_bytes == (
            step.local_historical_KV_bytes+step.host_historical_KV_bytes)
        assert step.stage_elapsed_time_s >= max(
            step.local_compute_memory_time_s, step.host_transfer_time_s)


def test_all_local_offload_backend_has_zero_host_read(inputs):
    result = _evaluate(inputs, S=32768, p=1, d=3, g=2)
    assert all(step.host_historical_KV_bytes == 0 for step in result.steps)
    assert result.total_host_KV_read_bytes == 0


def test_resident_only_wave_limit_is_horizon_safe(inputs):
    initial = _evaluate(
        inputs, g=0+1, policy="RESIDENT_ONLY_QUEUE_TO_FIT")
    grown = _evaluate(
        inputs, g=2048, policy="RESIDENT_ONLY_QUEUE_TO_FIT")
    assert grown.resident_limit_horizon_safe <= initial.resident_limit_initial
    assert grown.resident_wave_size <= grown.resident_limit_horizon_safe
    assert grown.peak_local_runtime_bytes <= grown.capacity_bytes
    assert grown.historical_host_read_bytes == 0
    assert grown.total_host_KV_read_bytes == grown.admission_host_read_bytes
    assert grown.num_waves == pytest.approx(
        -(-grown.D//grown.resident_wave_size))


def test_policy_labels_cannot_be_confused(inputs):
    queue = _evaluate(inputs, policy="RESIDENT_ONLY_QUEUE_TO_FIT")
    offload = _evaluate(inputs, policy="HOST_KV_OFFLOAD")
    assert queue.policy_role == "OPTIMISTIC_LOCAL_RESIDENCY_BASELINE"
    assert offload.policy_role == "OPTIMISTIC_RECURRING_HOST_ACCESS_BASELINE"
    assert queue.policy != offload.policy
