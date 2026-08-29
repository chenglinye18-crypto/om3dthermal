"""Targeted Fiddler published-profile extraction and demand tests."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from om3dthermal.experiment import load_moe_workload_spec, load_workload_spec
from om3dthermal.workload import (
    build_published_expert_demand,
    evaluate_llm_decode,
    evaluate_moe_decode,
    load_fiddler_published_profile,
)


ROOT = Path(__file__).parents[1]
CSV = (
    ROOT / "configs" / "workload" / "profiles"
    / "mixtral_8x7b_fiddler_iclr2025_sharegpt.csv"
)
METADATA = CSV.with_suffix(".metadata.json")
MIXTRAL = (
    ROOT / "configs" / "workload"
    / "mixtral_8x7b_v01_decode_b1_s32768.yaml"
)
DENSE = ROOT / "configs" / "workload" / "llama31_8b_decode_b1_s131072.yaml"


@pytest.fixture(scope="module")
def profile():
    return load_fiddler_published_profile(CSV, METADATA)


@pytest.fixture(scope="module")
def workload():
    return load_moe_workload_spec(MIXTRAL, project_root=ROOT).decode


def test_profile_shape_identity_and_provenance(profile) -> None:
    assert profile.model_id == "mistralai/Mixtral-8x7B-v0.1"
    assert (profile.num_layers, profile.num_experts, profile.top_k) == (32, 8, 2)
    assert len(profile.relative_popularity) == 32
    assert all(len(row) == 8 for row in profile.relative_popularity)
    assert sum(len(row) for row in profile.relative_popularity) == 256
    assert profile.profile_kind == "FIDDLER_PUBLISHED_ROUTING_PROFILE"
    assert profile.source_workload == "SHAREGPT"
    assert profile.classification == "DERIVED_FROM_PUBLISHED_ROUTING_PROFILE"
    assert profile.measurement_status == "NOT_MEASURED_BY_THIS_WORK"
    assert profile.synthetic_status == "NOT_SYNTHETIC"
    assert profile.extraction == "DIGITIZED_FROM_OFFICIAL_ARXIV_SOURCE_FIGURE"
    assert profile.extraction_gate == "PASS"


def test_relative_values_and_paper_statistical_closure(profile) -> None:
    values = tuple(
        value for row in profile.relative_popularity for value in row)
    assert all(math.isfinite(value) and value >= 0 for value in values)
    stats = profile.relative_statistics
    assert stats.mean == pytest.approx(0.7133249381484374)
    assert stats.std == pytest.approx(0.08328987188814552)
    assert stats.minimum == pytest.approx(0.224133048)
    assert stats.median == pytest.approx(0.710135097)
    assert stats.maximum == pytest.approx(0.994400447)
    assert stats.p25 == pytest.approx(0.669112084)
    assert stats.p75 == pytest.approx(0.760810584)
    assert stats.count_lt_0_6 == 15
    assert stats.count_gt_0_8 == 29
    assert abs(stats.mean - 0.71) <= 0.015
    assert abs(stats.std - 0.08) <= 0.015
    assert abs(stats.minimum - 0.22) <= 0.015
    assert abs(stats.maximum - 1.0) <= 0.015
    assert abs(stats.count_gt_0_8 - 27) <= 2


def test_metadata_preserves_deterministic_extraction_audit() -> None:
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    assert metadata["source_image_sha256"] == (
        "003b4c6ed2f20485b796cdb5591d871af8a1133900b5977bd3cb18b8c2ce0e29")
    assert metadata["raw_matrix_search"]["status"] == (
        "NO_RAW_32X8_ARTIFACT_FOUND")
    assert metadata["raw_matrix_search"]["official_repository_commit"] == (
        "227715bfd6e8c731b29548eab01d9919c4fe9564")
    assert metadata["geometry"]["source_size_pixels"] == [2700, 900]
    assert metadata["geometry"]["cell_sampling"] == "CENTER_PIXEL"
    assert metadata["geometry"]["maximum_rgb_distance_squared"] == 5
    assert metadata["extraction_gate"] == "PASS"
    assert all(metadata["validation_checks"].values())


def test_profile_loading_is_deterministic(profile) -> None:
    assert profile == load_fiddler_published_profile(CSV, METADATA)


def test_layer_top2_probability_closure_and_range(profile) -> None:
    assert len(profile.selection_probability) == 32
    for row in profile.selection_probability:
        assert len(row) == 8
        assert math.fsum(row) == pytest.approx(2.0, abs=1e-12)
        assert all(0 <= value <= 1 for value in row)


def test_stable_existing_object_mapping_and_demand_equation(
    profile, workload
) -> None:
    demand = build_published_expert_demand(profile, workload)
    assert len(demand.expert_objects) == 256
    assert demand.expert_objects[0].object_id == "expert.layer.00.expert.00"
    assert demand.expert_objects[83].object_id == "expert.layer.10.expert.03"
    assert demand.expert_objects[-1].object_id == "expert.layer.31.expert.07"
    assert demand.expert_footprint_bytes == 352_321_536
    for item in demand.expert_objects:
        assert item.expected_read_bytes_per_token == pytest.approx(
            item.selection_probability * demand.expert_footprint_bytes)


def test_per_layer_and_total_21_gib_traffic_closure(profile, workload) -> None:
    demand = build_published_expert_demand(profile, workload)
    for layer in range(32):
        layer_read = math.fsum(
            item.expected_read_bytes_per_token for item in demand.expert_objects
            if item.layer == layer)
        assert layer_read == pytest.approx(
            2 * demand.expert_footprint_bytes, abs=1e-6)
    assert demand.total_expert_read_bytes_per_token == 21 * 2**30
    assert demand.expected_active_expert_read_bytes_per_token == 21 * 2**30
    assert demand.total_closure_error_bytes == 0.0
    assert demand.maximum_per_layer_closure_error_bytes <= 1e-6
    assert demand.traffic_closure_status == (
        "ACTIVE_EXPERT_READ_TRAFFIC_CLOSURE_PASS")
    assert demand.shared_nonexpert_traffic_modified is False
    assert demand.kv_traffic_modified is False
    assert demand.placement_included is False


def test_published_skew_diagnostics_are_profile_derived(profile, workload) -> None:
    demand = build_published_expert_demand(profile, workload)
    assert demand.hottest_expert_object == "expert.layer.09.expert.05"
    assert demand.coldest_expert_object == "expert.layer.11.expert.03"
    assert demand.max_to_median_demand_ratio == pytest.approx(
        1.4024333551048247)
    assert demand.top_10_percent_selection_share == pytest.approx(
        0.12232719977564362)
    assert demand.top_25_percent_selection_share == pytest.approx(
        0.28523552660321505)
    assert demand.top_50_percent_selection_share == pytest.approx(
        0.541835663369968)
    assert demand.expert_demand_skew_gate == "SKEW_PRESENT"


def test_uniform_control_and_mixtral_analytical_regression(profile, workload) -> None:
    metrics = evaluate_moe_decode(workload)
    assert metrics.expert_selection_probabilities_per_layer == (0.25,) * 8
    assert metrics.routing_semantics_status == "STRUCTURAL_NEUTRAL_BASELINE"
    assert metrics.total_parameters == 46_702_792_704
    assert metrics.active_parameters_per_token == 12_879_925_248
    assert metrics.active_expert_weight_bytes_per_decode_step == 21 * 2**30
    assert profile.selection_probability[0] != (0.25,) * 8


def test_dense_workload_regression() -> None:
    workload = load_workload_spec(DENSE, project_root=ROOT).decode
    metrics = evaluate_llm_decode(workload)
    assert metrics.weight_footprint_bytes == 16_000_000_000
    assert metrics.kv_footprint_bytes == 17_179_869_184
    assert metrics.flops_per_token == 83_728_793_600
