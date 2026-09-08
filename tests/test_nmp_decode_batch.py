"""Aggregate-batch FEOL-NMP Decode regression gates (non-thermal)."""

from pathlib import Path

import pytest

from om3dthermal.experiment import load_workload_spec
from om3dthermal.serving import evaluate_nmp_decode_batch


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def results():
    base = load_workload_spec(
        ROOT / "configs/workload/llama31_8b_decode_b1_s131072.yaml",
        project_root=ROOT,
    ).decode
    return {
        batch: evaluate_nmp_decode_batch(
            base.model_copy(update={"batch_size": batch}), project_root=ROOT)
        for batch in (1, 2, 3, 14, 27, 28, 29)
    }


def test_b1_exact_regression(results):
    row = results[1]
    assert row.decode_step_time_ms == 2.1167828881564006
    assert row.aggregate_decode_tokens_per_s == 472.41500561776746
    assert row.J_per_token == 0.3188179932437353
    assert row.tokens_per_J == 3.13658583013382
    assert row.boundary_bytes_per_step == 886_655_488
    assert row.score_bytes_per_step == 268_435_456
    assert row.probability_bytes_per_step == 268_435_456
    assert row.partial_bytes_per_step == 55_574_528


def test_batch_and_capacity_status_are_independent(results):
    for batch in (1, 2, 3, 14, 27, 28):
        row = results[batch]
        assert row.batch_model_status == "RESOLVED_ANALYTICAL_BATCH_MODEL"
        assert row.capacity_status == "FEASIBLE"
        assert row.evaluation_status == "EVALUATED"
        assert row.capacity_violations == 0
        assert row.no_cross_request_handoff_alias == "PASS"
        assert row.qk_av_paired_request_ownership == "PASS"
    failed = results[29]
    assert failed.batch_model_status == "RESOLVED_ANALYTICAL_BATCH_MODEL"
    assert failed.capacity_status == "CAPACITY_INFEASIBLE"
    assert failed.evaluation_status == "CAPACITY_INFEASIBLE"
    assert results[28].physical_page_rounded_capacity_GB == 497.037606912
    assert results[28].capacity_margin_GB > 0.0
    assert failed.physical_page_rounded_capacity_GB > failed.available_capacity_GB
    assert failed.capacity_margin_GB < 0.0


def test_shared_weights_and_request_private_scaling(results):
    b1 = results[1]
    for batch in (2, 3, 14, 27, 28):
        row = results[batch]
        assert row.matrix_weight_read_bytes_per_step == (
            b1.matrix_weight_read_bytes_per_step)
        assert row.embedding_weight_read_bytes_per_step == 8192 * batch
        assert row.kv_read_bytes_per_step == b1.kv_read_bytes_per_step * batch
        assert row.kv_write_bytes_per_step == b1.kv_write_bytes_per_step * batch
        assert row.qk_flops_per_step == b1.qk_flops_per_step * batch
        assert row.av_flops_per_step == b1.av_flops_per_step * batch
        assert row.score_bytes_per_step == b1.score_bytes_per_step * batch
        assert row.probability_bytes_per_step == (
            b1.probability_bytes_per_step * batch)
        assert row.weight_batch_reuse_status == "PASS"


def test_resident_and_active_weight_and_locality_boundaries_are_distinct(results):
    row = results[1]
    assert row.logical_required_capacity_GB * 1e9 > row.weight_read_bytes_per_step
    assert row.matrix_weight_read_bytes_per_step == 15_009_316_864
    assert row.weight_bulk_external_bytes_per_step == 0
    assert row.kv_bulk_external_bytes_per_step == 0
    assert row.direct_die_to_die_bytes_per_step == 0
    assert row.boundary_bandwidth_bytes_per_s == 2.4e12
    assert row.local_bandwidth_per_die_bytes_per_s > 2.4e12 / 106


def test_aggregate_timing_static_and_energy_closure(results):
    b1 = results[1]
    for batch in (1, 2, 3, 14, 27, 28):
        row = results[batch]
        assert row.decode_step_time_ms <= batch * b1.decode_step_time_ms
        seconds = row.decode_step_time_ms * 1e-3
        assert row.gpu_static_J_per_step == pytest.approx(74.0 * seconds)
        assert row.aggregate_decode_tokens_per_s == pytest.approx(batch / seconds)
        assert row.J_per_token == pytest.approx(row.total_J_per_step / batch)
        assert row.tokens_per_J == pytest.approx(batch / row.total_J_per_step)
