"""Targeted structural, traffic, and M3D capacity tests for Mixtral."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import ValidationError

from om3dthermal.experiment import (
    load_moe_workload_spec,
    load_workload_spec,
)
from om3dthermal.power import (
    calculate_memory_power,
    calculate_physical_access_latency,
    calculate_physical_capacity_layout,
    load_case_config,
    resolve_case_geometry,
)
from om3dthermal.power.feol_route import calculate_feol_route
from om3dthermal.power.m3d_subarray import calculate_m3d_subarray
from om3dthermal.workload import (
    M3DMoECapacityError,
    build_m3d_moe_capacity_layout,
    build_moe_resident_objects,
    evaluate_llm_decode,
    evaluate_moe_decode,
)


ROOT = Path(__file__).parents[1]
MIXTRAL = (
    ROOT / "configs" / "workload"
    / "mixtral_8x7b_v01_decode_b1_s32768.yaml"
)
DENSE = ROOT / "configs" / "workload" / "llama31_8b_decode_b1_s131072.yaml"
M3D_CASE = ROOT / "configs" / "cases" / "orthogonal_m3d_igzo.yaml"
GIB = 2**30
MIB = 2**20


@pytest.fixture(scope="module")
def workload():
    return load_moe_workload_spec(MIXTRAL, project_root=ROOT).decode


@pytest.fixture(scope="module")
def physical_layout():
    case = load_case_config(M3D_CASE)
    geometry = resolve_case_geometry(case)
    power = calculate_memory_power(case, project_root=ROOT, geometry=geometry)
    assert geometry.m3d is not None
    topology = calculate_m3d_subarray(
        case.architecture.m3d_subarray, geometry.m3d)
    feol = calculate_feol_route(case.architecture.feol_route, topology)
    latency = calculate_physical_access_latency(
        case.architecture.physical_access_latency,
        feol_route=feol,
        miv_length_per_layer_um=power.diagnostics["miv_length_per_layer_um"],
        miv_delay_per_layer_ns=power.diagnostics["miv_delay_per_layer_ns"],
        miv_status=power.diagnostics["miv_latency_status"],
        miv_parameter_status=power.diagnostics[
            "miv_resistance_parameter_status"],
        miv_provenance=power.diagnostics["miv_resistance_provenance"],
    )
    return calculate_physical_capacity_layout(
        topology,
        latency,
        slab_count=geometry.memory_region_count,
        expected_total_bits=power.diagnostics["total_stored_bits"],
    )


def test_official_config_values_and_context_bound(workload) -> None:
    assert workload.model_id == "mistralai/Mixtral-8x7B-v0.1"
    assert (
        workload.num_hidden_layers,
        workload.hidden_size,
        workload.intermediate_size,
    ) == (32, 4096, 14336)
    assert (
        workload.num_attention_heads,
        workload.num_key_value_heads,
        workload.head_dim,
    ) == (32, 8, 128)
    assert (workload.num_local_experts, workload.num_experts_per_tok) == (8, 2)
    assert workload.vocab_size == 32000
    assert workload.tie_word_embeddings is False
    assert workload.dtype == "BF16"
    assert workload.context_length == workload.max_position_embeddings == 32768
    with pytest.raises(ValidationError, match="max_position_embeddings"):
        type(workload).model_validate(
            {**workload.model_dump(), "context_length": 32769})


def test_structural_parameter_and_bf16_closure(workload) -> None:
    metrics = evaluate_moe_decode(workload)
    assert metrics.expert_count == 32 * 8 == 256
    assert metrics.parameters_per_expert == 3 * 4096 * 14336 == 176_160_768
    assert metrics.expert_parameters_total == 45_097_156_608
    assert metrics.attention_parameters_total == 1_342_177_280
    assert metrics.router_parameters_total == 1_048_576
    assert metrics.rmsnorm_parameters_total == 266_240
    assert metrics.token_embedding_parameters == 131_072_000
    assert metrics.lm_head_parameters == 131_072_000
    assert metrics.nonexpert_parameters_total == 1_605_636_096
    assert metrics.total_parameters == 46_702_792_704
    assert metrics.active_expert_parameters_per_token == 11_274_289_152
    assert metrics.active_parameters_per_token == 12_879_925_248
    assert metrics.active_parameters_per_token < metrics.total_parameters
    assert metrics.active_to_total_parameter_ratio == pytest.approx(
        12_879_925_248 / 46_702_792_704)
    assert metrics.expert_footprint_bytes == 352_321_536 == 336 * MIB
    assert metrics.all_expert_footprint_bytes == 90_194_313_216 == 84 * GIB
    assert metrics.nonexpert_footprint_bytes == 3_211_272_192
    assert metrics.total_weight_footprint_bytes == 93_405_585_408


def test_all_experts_resident_but_only_top_two_active(workload) -> None:
    metrics = evaluate_moe_decode(workload)
    assert metrics.total_weight_footprint_bytes == (
        metrics.all_expert_footprint_bytes + metrics.nonexpert_footprint_bytes)
    assert metrics.active_expert_weight_bytes_per_decode_step == (
        32 * 2 * metrics.expert_footprint_bytes)
    assert metrics.active_expert_weight_bytes_per_decode_step == 22_548_578_304
    assert metrics.active_nonexpert_weight_bytes_per_decode_step == 3_211_272_192
    assert metrics.active_weight_bytes_per_decode_step == 25_759_850_496
    assert metrics.active_expert_flops_per_token == (
        2 * metrics.active_expert_parameters_per_token)
    assert metrics.active_expert_parameters_per_token == (
        metrics.expert_parameters_total // 4)
    assert 256 * metrics.uniform_expected_read_bytes_per_expert_per_decode_step == (
        metrics.active_expert_weight_bytes_per_decode_step)


def test_batch_tile_reuse_and_kv_closure(workload) -> None:
    one = evaluate_moe_decode(workload)
    eight = evaluate_moe_decode(workload.model_copy(update={"batch_size": 8}))
    assert eight.active_weight_bytes_per_decode_step == (
        one.active_weight_bytes_per_decode_step)
    assert one.total_weight_read_bytes_per_token == (
        8 * eight.total_weight_read_bytes_per_token)
    assert one.kv_bytes_per_token_per_request == 131_072
    assert one.kv_write_bytes_per_token_per_request == 131_072
    assert one.kv_bytes_per_request == 4 * GIB
    assert eight.kv_footprint_bytes == 8 * one.kv_footprint_bytes


def test_uniform_routing_is_only_trace_free_structural_closure(workload) -> None:
    metrics = evaluate_moe_decode(workload)
    assert metrics.expert_selection_probabilities_per_layer == (0.25,) * 8
    assert sum(metrics.expert_selection_probabilities_per_layer) == 2
    assert metrics.routing_semantics_status == "STRUCTURAL_NEUTRAL_BASELINE"
    assert metrics.routing_trace_status == "NOT_REAL_ROUTING_TRACE"
    assert metrics.expert_popularity_status == "NOT_WORKLOAD_POPULARITY"
    assert metrics.expert_demand_skew_status == (
        "EXPERT_DEMAND_SKEW_UNAVAILABLE_WITHOUT_TRACE")
    assert workload.real_expert_popularity_available == "NO"


def test_expert_objects_are_complete_and_deterministic(workload) -> None:
    objects = build_moe_resident_objects(workload)
    experts = [obj for obj in objects if obj.object_id.startswith("expert.")]
    assert len(experts) == 256
    assert experts[0].object_id == "expert.layer.00.expert.00"
    assert experts[-1].object_id == "expert.layer.31.expert.07"
    assert len({obj.object_id for obj in objects}) == len(objects)
    assert objects == build_moe_resident_objects(workload)
    assert all(obj.object_type == "WEIGHT" for obj in experts)


def test_weights_only_page_rounded_capacity(workload, physical_layout) -> None:
    weights_only = workload.model_copy(update={"context_length": 0})
    result = build_m3d_moe_capacity_layout(weights_only, physical_layout)
    assert result.weight_logical_bytes == 93_405_585_408
    assert result.kv_logical_bytes == 0
    assert result.page_layout.page_count == 44_540
    assert result.page_rounded_allocated_bytes / GIB == 86.9921875
    assert result.occupancy_fraction == pytest.approx(44_540 / 219_520)


@pytest.mark.parametrize(
    ("requests", "expected_pages", "expected_allocated_gib"),
    [(1, 46_588, 90.9921875),
     (8, 60_924, 118.9921875),
     (16, 77_308, 150.9921875)],
)
def test_n1_n8_n16_m3d_capacity_closure(
    workload, physical_layout, requests, expected_pages, expected_allocated_gib
) -> None:
    resolved = workload.model_copy(update={"batch_size": requests})
    result = build_m3d_moe_capacity_layout(resolved, physical_layout)
    metrics = evaluate_moe_decode(resolved)
    assert physical_layout.total_capacity_gib == 428.75
    assert physical_layout.slot_capacity_bytes == 2 * MIB
    assert physical_layout.physical_slot_count == 219_520
    assert result.expert_object_count == 256
    assert result.weight_logical_bytes == 93_405_585_408
    assert result.kv_logical_bytes == requests * 4 * GIB
    assert result.runtime_logical_bytes == 0
    assert result.total_logical_bytes == metrics.required_capacity_bytes
    assert result.page_layout.page_count == expected_pages
    assert result.page_rounded_allocated_bytes / GIB == expected_allocated_gib
    assert result.occupancy_fraction == pytest.approx(
        expected_pages / 219_520)
    assert result.capacity_status == "M3D_ONLY_PAGE_ALLOCATED_CAPACITY_PASS"
    assert result.residency_semantics == "ALL_EXPERTS_STORED_TOP_K_EXPERTS_ACCESSED"


def test_oversize_page_rounded_workload_fails_without_spill(
    workload, physical_layout
) -> None:
    too_small = replace(
        physical_layout,
        physical_slot_count=1,
        total_capacity_bytes=physical_layout.slot_capacity_bytes,
    )
    with pytest.raises(M3DMoECapacityError, match="M3D_ONLY_CAPACITY_FAIL"):
        build_m3d_moe_capacity_layout(workload, too_small)


def test_dense_analytical_outputs_remain_exact() -> None:
    dense = load_workload_spec(DENSE, project_root=ROOT).decode
    metrics = evaluate_llm_decode(dense)
    assert metrics.weight_footprint_bytes == 16_000_000_000
    assert metrics.weight_active_per_step_bytes == 16_000_000_000
    assert metrics.kv_footprint_bytes == 17_179_869_184
    assert metrics.kv_read_bytes_per_token == 17_179_869_184
    assert metrics.kv_write_bytes_per_token == 131_072
    assert metrics.flops_per_token == 83_728_793_600
