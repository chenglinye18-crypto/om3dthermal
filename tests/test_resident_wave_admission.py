"""Admission-aware Conventional-HBM resident-wave sensitivity tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from om3dthermal.serving import (
    MixedPhaseServingCase,
    evaluate_conventional_hbm_resident_wave_admission,
    evaluate_conventional_hbm_resident_wave_decode,
    evaluate_nmp_decode_batch,
    load_final_dense_e2e_matrix,
    resolve_conventional_hbm_backend,
)
from om3dthermal.workload import evaluate_llm_decode, load_dense_model_registry


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def registry():
    return load_dense_model_registry(ROOT / "configs/workload/models")


def _case(model_id: str, *, prefill: int, decode: int) -> MixedPhaseServingCase:
    return MixedPhaseServingCase(
        model_id=model_id, context_length=131072,
        batch_size=prefill + decode,
        prefill_requests=prefill, decode_requests=decode)


def _result(registry, model_id="llama31_8b", *, prefill=1, decode=27, G=128):
    return evaluate_conventional_hbm_resident_wave_admission(
        project_root=ROOT, model=registry[model_id],
        case=_case(model_id, prefill=prefill, decode=decode),
        generated_output_tokens_per_request=G)


def test_first_wave_zero_and_later_wave_admission_uses_canonical_kv(registry):
    result = _result(registry)
    metrics = evaluate_llm_decode(registry["llama31_8b"].decode_input(
        batch_size=27, context_length=131072))
    assert result.wave_sizes == (7, 7, 7, 6)
    assert result.admission_bytes_per_wave[0] == 0.0
    assert result.admission_time_ms_per_wave[0] == 0.0
    assert result.admission_bytes_per_wave[1:] == pytest.approx(tuple(
        size * metrics.kv_bytes_per_request for size in result.wave_sizes[1:]))
    assert result.first_wave_admission_status == "ZERO_ALREADY_RESIDENT"


def test_total_admission_bytes_and_canonical_bandwidth_time_close(registry):
    result = _result(registry)
    host_bw = resolve_conventional_hbm_backend(
        ROOT).host_offload.effective_bandwidth_bytes_per_second
    assert host_bw is not None
    assert result.total_admission_GB * 1e9 == pytest.approx(
        sum(result.admission_bytes_per_wave))
    assert result.total_admission_time_ms == pytest.approx(
        sum(result.admission_bytes_per_wave) / host_bw * 1e3)
    assert result.host_effective_bandwidth_bytes_per_s == host_bw


def test_compute_completion_tokens_and_throughput_close(registry):
    result = _result(registry, G=512)
    assert result.compute_time_ms == pytest.approx(
        512 * sum(result.wave_step_times_ms))
    assert result.total_completion_time_ms == pytest.approx(
        result.compute_time_ms + result.total_admission_time_ms)
    assert result.total_generated_tokens == 27 * 512
    assert result.admission_aware_tokens_per_s == pytest.approx(
        result.total_generated_tokens / (result.total_completion_time_ms * 1e-3))


def test_realistic_rate_is_monotonic_and_approaches_optimistic(registry):
    rows = [_result(registry, G=G) for G in (128, 256, 512, 1024, 2048)]
    rates = [row.admission_aware_tokens_per_s for row in rows]
    assert rates == sorted(rates)
    assert all(row.admission_aware_tokens_per_s
               <= row.optimistic_resident_wave_tokens_per_s for row in rows)
    assert rows[-1].throughput_retention_vs_optimistic > (
        rows[0].throughput_retention_vs_optimistic)
    assert rows[-1].throughput_loss_vs_optimistic < (
        rows[0].throughput_loss_vs_optimistic)


def test_qwen_d14_single_wave_has_exact_optimistic_rate(registry):
    result = _result(
        registry, "qwen25_7b", prefill=14, decode=14, G=128)
    assert result.wave_sizes == (14,)
    assert result.total_admission_GB == 0.0
    assert result.total_admission_time_ms == 0.0
    assert result.admission_aware_tokens_per_s == (
        result.optimistic_resident_wave_tokens_per_s)
    assert result.throughput_retention_vs_optimistic == 1.0
    assert result.admission_impact_classification == "LOW_IMPACT"


def test_existing_optimistic_resident_wave_is_unchanged(registry):
    result = evaluate_conventional_hbm_resident_wave_decode(
        project_root=ROOT, model=registry["llama31_8b"],
        case=_case("llama31_8b", prefill=1, decode=27))
    assert result.wave_sizes == (7, 7, 7, 6)
    assert result.aggregate_decode_tokens_per_s == pytest.approx(
        123.68831794418877)
    assert result.swap_admission_overhead_status == (
        "NOT_INCLUDED_OPTIMISTIC_UPPER_BOUND")


def test_final_matrix_manifest_remains_36_points():
    manifest = load_final_dense_e2e_matrix(
        ROOT / "configs/experiment/final_dense_e2e_matrix.yaml")
    assert len(manifest.models) * len(manifest.mixed_points) * len(
        manifest.systems) == 36


@pytest.mark.parametrize("batch,status,step_ms", [
    (1, "EVALUATED", 2.1167828881564006),
    (28, "EVALUATED", 43.54111957320394),
    (29, "CAPACITY_INFEASIBLE", None),
])
def test_nmp_frozen_regressions_unchanged(registry, batch, status, step_ms):
    result = evaluate_nmp_decode_batch(
        registry["llama31_8b"].decode_input(
            batch_size=batch, context_length=131072),
        project_root=ROOT)
    assert result.evaluation_status == status
    if step_ms is None:
        assert result.decode_step_time_ms is None
    else:
        assert result.decode_step_time_ms == pytest.approx(step_ms)
        assert result.capacity_violations == 0
