from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml

from om3dthermal.serving import (
    FormalInferenceWorkload,
    WorkspaceExecutionConfig,
    evaluate_formal_inference_workload,
    formal_capacity_limits,
)
from om3dthermal.workload import load_dense_model_registry
from scripts.run_formal_e2e_benchmark import _add_speedups, _ratio, _row


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def setup():
    raw = yaml.safe_load((
        ROOT/"configs/experiment/formal_e2e_benchmark.yaml"
    ).read_text(encoding="utf-8"))
    workspace = WorkspaceExecutionConfig.model_validate(raw["workspace"])
    models = load_dense_model_registry(ROOT/raw["model_registry_dir"])
    return models["llama31_8b"], workspace


def _evaluate(setup, *, B, G, system, policy):
    model, workspace = setup
    workload = FormalInferenceWorkload(
        model_id=model.model_id, batch_size=B,
        prompt_tokens=131072, generation_tokens=G)
    return evaluate_formal_inference_workload(
        project_root=ROOT, model=model, workload=workload,
        system=system, policy=policy, workspace_config=workspace)


def test_formal_schema_is_b_s_g_without_phase_ratio():
    fields = set(FormalInferenceWorkload.model_fields)
    assert fields == {
        "model_id", "batch_size", "prompt_tokens", "generation_tokens",
        "arrival_policy", "prefill_count_per_request"}
    assert not fields.intersection({"P", "D", "prefill_requests", "decode_requests"})


def test_b1_overflow_policies_are_equivalent_and_local(setup):
    resident = _evaluate(
        setup, B=1, G=4, system="CONVENTIONAL_HBM_GPU",
        policy="RESIDENT_ONLY_QUEUE_TO_FIT")
    offload = _evaluate(
        setup, B=1, G=4, system="CONVENTIONAL_HBM_GPU",
        policy="HOST_KV_OFFLOAD")
    assert resident.prefill_executions == offload.prefill_executions == 1
    assert resident.decode_steps_per_request == offload.decode_steps_per_request == 4
    assert resident.num_waves == offload.num_waves == 1
    assert resident.queue_time_s == 0
    assert offload.historical_host_read_bytes == 0
    assert offload.total_host_transfer_bytes == 0
    assert resident.batch_makespan_s == pytest.approx(offload.batch_makespan_s)
    assert resident.e2e_output_tokens_per_s == pytest.approx(
        offload.e2e_output_tokens_per_s)


def test_resident_waves_queue_but_do_not_add_queue_to_tpot(setup):
    model, workspace = setup
    safe = formal_capacity_limits(
        project_root=ROOT, model=model, S=131072, G=4,
        workspace_config=workspace)["B_max_HBM_safe"]
    result = _evaluate(
        setup, B=2*safe, G=4, system="CONVENTIONAL_HBM_GPU",
        policy="RESIDENT_ONLY_QUEUE_TO_FIT")
    assert result.num_waves == 2
    assert result.queue_time_s > 0
    assert result.p95_ttft_s > result.mean_ttft_s
    assert result.p95_completion_latency_s == result.batch_makespan_s
    assert result.mean_tpot_s*4 == pytest.approx(
        result.decode_active_time_s/result.num_waves)
    assert result.e2e_output_tokens_per_s == pytest.approx(
        result.batch_size*result.generation_tokens/result.batch_makespan_s)


def test_host_offload_reads_recurring_historical_kv(setup):
    model, workspace = setup
    safe = formal_capacity_limits(
        project_root=ROOT, model=model, S=131072, G=4,
        workspace_config=workspace)["B_max_HBM_safe"]
    result = _evaluate(
        setup, B=safe+1, G=4, system="CONVENTIONAL_HBM_GPU",
        policy="HOST_KV_OFFLOAD")
    assert result.historical_host_read_bytes > 0
    assert result.historical_host_read_bytes > result.migration_bytes
    assert result.peak_host_resident_bytes >= result.migration_bytes
    assert result.decode_tokens_per_s == pytest.approx(
        result.batch_size*result.generation_tokens/result.decode_active_time_s)
    assert result.decode_active_time_s == pytest.approx(
        result.mean_tpot_s*result.generation_tokens)


def test_safe_batch_uses_growing_kv_and_m3d_overflow_is_explicit(setup):
    model, workspace = setup
    short = formal_capacity_limits(
        project_root=ROOT, model=model, S=131072, G=1,
        workspace_config=workspace)
    long = formal_capacity_limits(
        project_root=ROOT, model=model, S=131072, G=512,
        workspace_config=workspace)
    assert long["B_max_HBM_safe"] <= short["B_max_HBM_safe"]
    assert long["B_max_M3D_safe"] <= short["B_max_M3D_safe"]
    result = _evaluate(
        setup, B=short["B_max_M3D_safe"]+1, G=1,
        system="ORTHOGONAL_M3D_IGZO_MEMORY_ONLY", policy="FULLY_LOCAL")
    assert result.status == "CAPACITY_INFEASIBLE"
    assert result.batch_makespan_s is None
    assert result.first_capacity_violation_step is not None


def test_capacity_and_nmp_gain_labels_are_separate(setup):
    rows = [_row(result) for result in (
        _evaluate(setup, B=1, G=2, system=system, policy=policy)
        for system, policy in (
            ("CONVENTIONAL_HBM_GPU", "RESIDENT_ONLY_QUEUE_TO_FIT"),
            ("CONVENTIONAL_HBM_GPU", "HOST_KV_OFFLOAD"),
            ("ORTHOGONAL_M3D_IGZO_MEMORY_ONLY", "FULLY_LOCAL"),
            ("IOM3D_FEOL_NMP", "FULLY_LOCAL"),
        ))]
    _add_speedups(rows)
    keys = set(rows[0])
    assert any(key.startswith("capacity_gain_vs_resident_only__") for key in keys)
    assert any(key.startswith("capacity_gain_vs_host_offload__") for key in keys)
    assert any(key.startswith("pure_nmp_gain_vs_m3d_only__") for key in keys)
    assert "capacity_gain_vs_resident_only__resident_concurrency_gain" in keys
    assert "capacity_gain_vs_host_offload__host_traffic_reduction_fraction" in keys
    assert not math.isnan(rows[0][
        "pure_nmp_gain_vs_m3d_only__decode_throughput_speedup"])
    assert _ratio("", "") is None
