"""Conventional-HBM resident-wave scheduler sensitivity tests."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from om3dthermal.serving import (
    SYSTEM_CONFIGURATIONS,
    MixedPhaseServingCase,
    evaluate_conventional_hbm_mixed_phase,
    evaluate_conventional_hbm_resident_wave_decode,
    load_final_dense_e2e_matrix,
)
from om3dthermal.workload import load_dense_model_registry


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def registry():
    return load_dense_model_registry(ROOT / "configs/workload/models")


def _result(registry, model_id: str, *, prefill: int, decode: int):
    case = MixedPhaseServingCase(
        model_id=model_id, context_length=131072,
        batch_size=prefill + decode,
        prefill_requests=prefill, decode_requests=decode)
    return evaluate_conventional_hbm_resident_wave_decode(
        project_root=ROOT, model=registry[model_id], case=case)


@pytest.mark.parametrize("model_id,expected", [
    ("llama31_8b", 7), ("qwen25_7b", 17), ("llama2_7b", 1),
])
def test_resident_batch_limit_is_capacity_derived(registry, model_id, expected):
    result = _result(registry, model_id, prefill=1, decode=27)
    assert result.resident_batch_limit == expected
    assert result.capacity_status == "CAPACITY_PRESSURED"


@pytest.mark.parametrize("model_id,decode,expected", [
    ("llama31_8b", 27, (7, 7, 7, 6)),
    ("llama31_8b", 14, (7, 7)),
    ("qwen25_7b", 27, (17, 10)),
    ("qwen25_7b", 14, (14,)),
    ("llama2_7b", 27, (1,) * 27),
    ("llama2_7b", 14, (1,) * 14),
])
def test_wave_decomposition_closes(registry, model_id, decode, expected):
    result = _result(registry, model_id, prefill=28 - decode, decode=decode)
    assert result.wave_sizes == expected
    assert sum(result.wave_sizes) == decode
    assert all(size <= result.resident_batch_limit for size in result.wave_sizes)
    assert result.wave_count == math.ceil(decode / result.resident_batch_limit)


def test_matrix_weight_is_read_once_per_wave_not_per_request(registry):
    result = _result(registry, "llama31_8b", prefill=1, decode=27)
    assert len(set(result.wave_matrix_weight_read_bytes_per_step)) == 1
    assert result.wave_matrix_weight_read_bytes_per_step[0] > 0.0


def test_aggregate_throughput_and_equal_length_cancellation_close(registry):
    result = _result(registry, "qwen25_7b", prefill=1, decode=27)
    expected = 27 / (sum(result.wave_step_times_ms) * 1e-3)
    assert result.aggregate_decode_tokens_per_s == pytest.approx(expected)
    assert result.normalized_wave_decode_makespan_ms_per_output_token_depth == (
        pytest.approx(sum(result.wave_step_times_ms)))
    assert result.output_length_cancels_in_equal_length_wave_throughput == "YES"
    assert result.output_length_assumption_status == (
        "EQUAL_OUTPUT_LENGTH__CANCELS_FROM_AGGREGATE_THROUGHPUT")


def test_host_traffic_and_swap_overhead_are_excluded(registry):
    result = _result(registry, "llama2_7b", prefill=1, decode=27)
    assert result.host_read_bytes_per_decode_step == 0
    assert result.host_write_bytes_per_decode_step == 0
    assert result.swap_admission_overhead_status == (
        "NOT_INCLUDED_OPTIMISTIC_UPPER_BOUND")
    assert result.resident_wave_model == (
        "CAPACITY_CONSTRAINED_RUN_TO_COMPLETION_OPTIMISTIC_BOUND")
    assert result.total_nonresident_KV_GB > 0.0
    assert result.thermal is None


def test_existing_host_offload_baseline_is_unchanged(registry):
    case = MixedPhaseServingCase(
        model_id="llama31_8b", context_length=131072,
        batch_size=28, prefill_requests=1, decode_requests=27)
    result = evaluate_conventional_hbm_mixed_phase(
        project_root=ROOT, model=registry["llama31_8b"], case=case)
    assert result.decode_tokens_per_s == pytest.approx(4.276612519457736)
    assert result.decode_service_time_ms == pytest.approx(6313.408071728589)
    assert result.host_read_GB == pytest.approx(343.59738368)
    assert result.residency_policy == "DECODE_FIRST_LOCAL_RESIDENCY"


def test_resident_wave_is_not_a_fourth_final_matrix_architecture():
    manifest = load_final_dense_e2e_matrix(
        ROOT / "configs/experiment/final_dense_e2e_matrix.yaml")
    assert len(manifest.models) * len(manifest.mixed_points) * len(manifest.systems) == 36
    assert set(manifest.systems) == set(SYSTEM_CONFIGURATIONS)
    assert "CONVENTIONAL_HBM_GPU_RESIDENT_WAVE" not in SYSTEM_CONFIGURATIONS
