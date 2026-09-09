from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml

from om3dthermal.serving import (
    CachedPrefixDiagnosticWorkload, WorkspaceExecutionConfig,
    evaluate_cached_prefix_diagnostic,
)
from om3dthermal.workload import (
    evaluate_cached_prefix_incremental_prefill, evaluate_llm_decode,
    load_dense_model_registry,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def diagnostic_setup():
    raw = yaml.safe_load((
        ROOT/"configs/experiment/cached_prefix_long_context_diagnostic.yaml"
    ).read_text(encoding="utf-8"))
    model = load_dense_model_registry(
        ROOT/raw["model_registry_dir"])[raw["model_id"]]
    workload = CachedPrefixDiagnosticWorkload.model_validate({
        key: raw[key] for key in (
            "model_id", "cached_history_tokens", "new_prompt_tokens",
            "generation_tokens", "final_context_tokens", "total_requests",
            "arrival_policy", "cached_kv_backing_store")})
    workspace = WorkspaceExecutionConfig.model_validate(raw["workspace"])
    return model, workload, workspace


@pytest.fixture(scope="module")
def result(diagnostic_setup):
    model, workload, workspace = diagnostic_setup
    return evaluate_cached_prefix_diagnostic(
        project_root=ROOT, model=model, workload=workload,
        workspace_config=workspace)


def test_cached_prefix_incremental_prefill_pair_count(diagnostic_setup):
    model, workload, _ = diagnostic_setup
    metrics = evaluate_cached_prefix_incremental_prefill(
        model.prefill_input(batch_size=1, prompt_length=workload.new_prompt_tokens),
        cached_history_tokens=workload.cached_history_tokens)
    assert metrics.attention_query_tokens_per_request == 1024
    assert metrics.historical_kv_tokens_attended_per_query == 128000
    assert metrics.attention_pairs_per_request == (
        1024*128000+1024*1025//2)


def test_cached_prefix_does_not_recompute_history(result):
    metrics = result.incremental_prefill_metrics_b1
    assert metrics.linear_ffn_tokens_computed_per_request == 1024
    assert metrics.prefill_input_tokens == 1024
    assert metrics.cached_kv_rewritten is False


def test_incremental_prefill_kv_append(result):
    metrics = result.incremental_prefill_metrics_b1
    assert metrics.final_kv_cache_bytes == pytest.approx(
        metrics.cached_kv_bytes_before+metrics.new_kv_write_bytes)
    assert metrics.new_kv_write_bytes/metrics.cached_kv_bytes_before == pytest.approx(
        1024/128000)


def test_final_context_exactly_128k(result):
    workload = result.workload
    assert workload.decode_start_context_tokens == 129024
    assert workload.decode_start_context_tokens+workload.generation_tokens == 131072


def test_safe_batch_uses_final_context(result, diagnostic_setup):
    model, workload, workspace = diagnostic_setup
    one = evaluate_llm_decode(model.decode_input(
        batch_size=1, context_length=workload.cached_history_tokens))
    weight = int(one.weight_footprint_bytes+one.runtime_fixed_bytes)
    kv_final = int(one.kv_write_bytes_per_token)*workload.final_context_tokens
    from om3dthermal.serving.mixed_phase_e2e import resolve_conventional_hbm_backend
    from om3dthermal.serving.nmp_decode import resolve_m3d_architecture_backend
    from om3dthermal.serving.workspace import evaluate_decode_workspace
    hbm = resolve_conventional_hbm_backend(ROOT)
    m3d = resolve_m3d_architecture_backend(ROOT)
    for point in result.capacities:
        batch = point.safe_active_batch+1
        workspace_bytes = evaluate_decode_workspace(
            model, batch_size=batch,
            context_length=workload.final_context_tokens-1,
            config=workspace).peak_bytes
        if point.system == "CONVENTIONAL_HBM_GPU":
            runtime = weight+batch*kv_final+workspace_bytes
            assert point.capacity_bytes == int(hbm.capacity_bytes)
        else:
            page = m3d.layout.slot_capacity_bytes
            physical = lambda value: math.ceil(value/page)*page
            runtime = physical(weight)+batch*physical(kv_final)+physical(workspace_bytes)
        assert runtime > point.capacity_bytes
        assert point.limiting_phase == "DECODE_FINAL"


def test_wave_queueing_128_requests(result):
    for summary in result.summaries:
        assert sum(summary.wave_sizes) == 128
        assert summary.num_waves == math.ceil(128/summary.safe_active_batch)
        assert max(summary.wave_sizes) <= summary.safe_active_batch


def test_ttft_includes_queueing(result):
    for summary in result.summaries:
        points = [item for item in result.requests if item.system == summary.system]
        first = points[0]
        first_queued = points[summary.safe_active_batch]
        assert first.queue_delay_s == 0
        assert first_queued.queue_delay_s > 0
        assert first_queued.ttft_s > first.ttft_s
        assert first_queued.ttft_s > first_queued.queue_delay_s


def test_completion_latency_p95(result):
    for summary in result.summaries:
        values = sorted(item.completion_latency_s for item in result.requests
                        if item.system == summary.system)
        assert summary.p95_completion_latency_s == values[math.ceil(.95*128)-1]
        assert summary.max_completion_latency_s == max(values)
        assert summary.batch_makespan_s == summary.max_completion_latency_s
