"""Published Mixtral demand through pages, placement, and existing E2E."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from om3dthermal.experiment import load_moe_workload_spec
from om3dthermal.placement import (
    evaluate_published_moe_placement_e2e,
    place_pages_on_slots,
    select_physical_slots,
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
    build_published_moe_page_demand,
    evaluate_moe_decode,
    expert_only_page_demand_view,
    load_fiddler_published_profile,
)


ROOT = Path(__file__).parents[1]
CASE = ROOT / "configs/cases/orthogonal_m3d_igzo.yaml"
WORKLOAD = ROOT / "configs/workload/mixtral_8x7b_v01_decode_b1_s32768.yaml"
PROFILE = (
    ROOT / "configs/workload/profiles"
    / "mixtral_8x7b_fiddler_iclr2025_sharegpt.csv")
BW = 3.92e13
COMPUTE = 1.0e14


@pytest.fixture(scope="module")
def canonical():
    case = load_case_config(CASE)
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
    layout = calculate_physical_capacity_layout(
        topology,
        latency,
        slab_count=geometry.memory_region_count,
        expected_total_bits=power.diagnostics["total_stored_bits"],
    )
    workload = load_moe_workload_spec(WORKLOAD, project_root=ROOT).decode
    profile = load_fiddler_published_profile(
        PROFILE, PROFILE.with_suffix(".metadata.json"))
    return layout, workload, profile


@pytest.mark.parametrize("requests", (1, 8, 16))
def test_expert_pages_and_all_traffic_close(canonical, requests) -> None:
    layout, base, profile = canonical
    workload = base.model_copy(update={"batch_size": requests})
    metrics = evaluate_moe_decode(workload)
    demand = build_published_moe_page_demand(profile, workload, layout)
    assert demand.page_layout.page_size_bytes == 2 * 2**20
    assert demand.expert_page_count == 256 * 168
    assert demand.total_expert_read_bytes_per_decode_step == 21 * 2**30
    assert demand.total_shared_weight_read_bytes_per_decode_step == (
        metrics.active_nonexpert_weight_bytes_per_decode_step)
    assert demand.total_kv_read_bytes_per_decode_step == (
        requests * metrics.kv_read_bytes_per_token_per_request)
    assert demand.kv_write_bytes_per_decode_step == (
        requests * metrics.kv_write_bytes_per_token_per_request)
    assert abs(demand.expert_traffic_closure_error_bytes) <= 1e-6
    assert abs(demand.shared_weight_traffic_closure_error_bytes) <= 1e-6
    assert abs(demand.kv_traffic_closure_error_bytes) <= 1e-6
    assert abs(demand.total_traffic_closure_error_bytes) <= 1e-6
    assert demand.allocated_page_bytes <= layout.total_capacity_bytes


def test_expert_internal_pages_have_no_synthetic_hotness(canonical) -> None:
    layout, workload, profile = canonical
    demand = build_published_moe_page_demand(profile, workload, layout)
    pages = tuple(
        item for item in demand.page_demands
        if item.parent_object_id == "expert.layer.09.expert.05")
    assert len(pages) == 168
    assert len({item.read_demand_bytes_per_decode_step for item in pages}) == 1


def test_random_seed_is_deterministic(canonical) -> None:
    layout, workload, profile = canonical
    demand = build_published_moe_page_demand(profile, workload, layout)
    first = place_pages_on_slots(
        demand, layout, slot_policy="RANDOM", random_seed=7)
    repeated = place_pages_on_slots(
        demand, layout, slot_policy="RANDOM", random_seed=7)
    assert first == repeated


def test_p1_prefix_and_p2_same_slots_with_sorted_pairing(canonical) -> None:
    layout, workload, profile = canonical
    demand = build_published_moe_page_demand(profile, workload, layout)
    p1 = place_pages_on_slots(
        demand, layout, slot_policy="FASTEST", page_ordering="CANONICAL")
    p2 = place_pages_on_slots(
        demand, layout, slot_policy="FASTEST",
        page_ordering="DEMAND_DESCENDING")
    prefix = select_physical_slots(layout, demand.page_count, policy="FASTEST")
    expected_slots = {
        (item.slab_id, item.cluster_id, item.layer_id)
        for item in prefix.selected_slots}
    p1_slots = {
        (item.slab_id, item.cluster_id, item.layer_id)
        for item in p1.assignments}
    p2_slots = {
        (item.slab_id, item.cluster_id, item.layer_id)
        for item in p2.assignments}
    assert p1_slots == p2_slots == expected_slots
    demands = tuple(
        item.read_demand_bytes_per_decode_step for item in p2.assignments)
    latencies = tuple(
        item.physical_access_latency_ns for item in p2.assignments)
    assert demands == tuple(sorted(demands, reverse=True))
    assert latencies == tuple(sorted(latencies))


@pytest.fixture(scope="module")
def evaluated(canonical):
    layout, workload, profile = canonical
    return evaluate_published_moe_placement_e2e(
        profile,
        workload,
        layout,
        matched_payload_bandwidth_bits_per_second=BW,
        effective_compute_flops_per_second=COMPUTE,
        random_seeds=(0, 1, 2, 3),
    )


def test_physical_ordering_and_uniform_control(evaluated) -> None:
    result = evaluated
    all_read = result.all_read_physical
    assert (
        all_read.popularity_aware_fast_region.weighted_average_latency_ns
        <= all_read.fast_region_only.weighted_average_latency_ns
        <= all_read.random.weighted_average_latency_ns)
    assert result.expert_only_physical.popularity_ordering_gain > 0.0
    assert result.uniform_expert_only_physical.popularity_ordering_gain == (
        pytest.approx(0.0, abs=1e-14))
    assert result.uniform_all_read_physical.popularity_ordering_gain >= 0.0
    assert all_read.occupied_slot_set_closure == (
        "P1_P2_IDENTICAL_FASTEST_CAPACITY_PREFIX")


def test_e2e_reuses_existing_service_equation(evaluated) -> None:
    result = evaluated
    p0 = result.random_timing
    p1 = result.fast_region_timing
    p2 = result.popularity_aware_timing
    assert p2.total_step_time_ms <= p1.total_step_time_ms <= p0.total_step_time_ms
    assert p2.aggregate_tokens_per_s >= p1.aggregate_tokens_per_s >= (
        p0.aggregate_tokens_per_s)
    for timing in (p0, p1, p2):
        assert timing.memory_stage_step_time_ms == pytest.approx(
            timing.memory_bandwidth_step_time_ms
            + timing.memory_access_latency_step_time_ms)
        assert timing.gpu_resource_step_time_ms == pytest.approx(max(
            timing.memory_stage_step_time_ms,
            timing.compute_step_time_ms,
        ))
    assert result.physical_latency_exposure_model == (
        "SERIAL_ONE_LATENCY_EXPOSURE_PER_2MIB_READ_PAGE_EQUIVALENT")
    assert result.verdict_scope == "CONDITIONAL_ON_CURRENT_MEMORY_SERVICE_MODEL"


def test_expert_only_view_is_exactly_21_gib(canonical) -> None:
    layout, workload, profile = canonical
    demand = build_published_moe_page_demand(profile, workload, layout)
    view = expert_only_page_demand_view(demand)
    assert view.page_count == demand.expert_page_count
    assert math.fsum(
        item.read_demand_bytes_per_decode_step for item in view.page_demands
    ) == 21 * 2**30
