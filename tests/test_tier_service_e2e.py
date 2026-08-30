"""Targeted Stratum-style tier-service placement E2E tests."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from om3dthermal.experiment import load_experiment_spec, load_workload_spec
from om3dthermal.placement import (
    compare_fast_region_placements,
    compare_placement_serving_performance,
    evaluate_tier_service_placement,
    sweep_local_service_fraction,
)
from om3dthermal.power import (
    calculate_memory_power,
    calculate_physical_access_latency,
    calculate_physical_capacity_layout,
    derive_architecture_bandwidth,
    load_case_config,
    resolve_case_geometry,
)
from om3dthermal.power.feol_route import calculate_feol_route
from om3dthermal.power.m3d_subarray import calculate_m3d_subarray
from om3dthermal.workload import build_m3d_workload_page_demand
import om3dthermal.placement.tier_service_e2e as tier_module


ROOT = Path(__file__).parents[1]
CASE = ROOT / "configs/cases/orthogonal_m3d_igzo.yaml"
WORKLOAD = ROOT / "configs/workload/llama31_8b_decode_b1_s131072.yaml"
EXPERIMENT = (
    ROOT / "configs/experiment/m3d_igzo_llama31_8b_decode_conditional_v0.yaml")


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
        miv_length_per_layer_um=power.diagnostics[
            "miv_length_per_layer_um"],
        miv_delay_per_layer_ns=power.diagnostics["miv_delay_per_layer_ns"],
        miv_status=power.diagnostics["miv_latency_status"],
        miv_parameter_status=power.diagnostics[
            "miv_resistance_parameter_status"],
        miv_provenance=power.diagnostics["miv_resistance_provenance"],
    )
    layout = calculate_physical_capacity_layout(
        topology, latency,
        slab_count=geometry.memory_region_count,
        expected_total_bits=power.diagnostics["total_stored_bits"],
    )
    bandwidth = derive_architecture_bandwidth(
        case.architecture.memory_service, layout, topology,
        feol_io_channels=case.architecture.feol_route.io_channels)
    workload = load_workload_spec(WORKLOAD, project_root=ROOT).decode
    scenario = load_experiment_spec(EXPERIMENT, project_root=ROOT).scenario
    return case, geometry, power, topology, feol, latency, layout, bandwidth, workload, scenario


def _evaluate(canonical, requests: int, fraction: float):
    *_, layout, bandwidth, base, scenario = canonical
    workload = base.model_copy(update={"batch_size": requests})
    demand = build_m3d_workload_page_demand(workload, layout)
    placement = compare_fast_region_placements(
        demand, layout, random_seeds=(0, 1, 2, 3))
    return evaluate_tier_service_placement(
        workload, demand, layout, bandwidth, placement.fast_pack,
        matched_external_bandwidth_bits_per_second=(
            scenario.matched_payload_bandwidth_bits_per_second),
        effective_compute_flops_per_second=(
            scenario.effective_compute_flops_per_second),
        local_service_fraction=fraction,
    ), workload, demand, placement


def test_no_tier_is_global_physical_worst_case(canonical):
    result, _, _, _ = _evaluate(canonical, 1, 0.5)
    layout = canonical[6]
    expected = max(
        slot.physical_access_latency_ns for slot in layout.slot_classes)
    assert result.no_tier.policy == "NO_TIER_WORST_CASE"
    assert result.no_tier.physical_service_latency_ns == pytest.approx(expected)
    assert expected == pytest.approx(18.008616609416016)


@pytest.mark.parametrize(
    ("requests", "expected"),
    ((1, 10.072213443776443),
     (8, 10.427075358422606),
     (16, 11.276452163080243)),
)
def test_fast_pack_latency_reuses_existing_placement(canonical, requests, expected):
    result, _, _, placement = _evaluate(canonical, requests, 0.5)
    assert result.tier_aware_fast_pack.policy == "TIER_AWARE_FAST_PACK"
    assert result.tier_aware_fast_pack.physical_service_latency_ns == (
        placement.fast_pack.weighted_average_access_latency_ns
    ) == pytest.approx(expected)


def test_service_rate_formula_and_internal_bandwidth_closure(canonical):
    result, _, _, _ = _evaluate(canonical, 1, 1.0)
    no_tier = result.no_tier
    fast = result.tier_aware_fast_pack
    assert fast.service_rate_speedup == pytest.approx(
        no_tier.physical_service_latency_ns
        / fast.physical_service_latency_ns)
    assert fast.internal_bandwidth_effective_bytes_per_s == pytest.approx(
        no_tier.internal_bandwidth_no_tier_bytes_per_s
        * fast.service_rate_speedup)


def test_zero_local_fraction_is_fixed_interface_negative_control(canonical):
    result, _, _, _ = _evaluate(canonical, 8, 0.0)
    assert result.end_to_end_speedup == pytest.approx(1.0, abs=1e-14)
    assert result.no_tier.aggregate_tokens_per_s == pytest.approx(
        result.tier_aware_fast_pack.aggregate_tokens_per_s)
    assert result.no_tier.local_memory_time_ms == 0.0
    assert result.tier_aware_fast_pack.local_memory_time_ms == 0.0


def test_full_local_fraction_exposes_physical_service_rate(canonical):
    result, _, _, _ = _evaluate(canonical, 1, 1.0)
    assert result.tier_aware_fast_pack.aggregate_tokens_per_s >= (
        result.no_tier.aggregate_tokens_per_s)
    assert result.end_to_end_speedup == pytest.approx(
        result.tier_aware_fast_pack.service_rate_speedup)
    assert result.no_tier.bottleneck == result.tier_aware_fast_pack.bottleneck == (
        "MEMORY")


@pytest.mark.parametrize("requests", (1, 8, 16))
def test_e2e_speedup_is_monotonic_in_local_fraction(canonical, requests):
    *_, layout, bandwidth, base, scenario = canonical
    workload = base.model_copy(update={"batch_size": requests})
    demand = build_m3d_workload_page_demand(workload, layout)
    placement = compare_fast_region_placements(
        demand, layout, random_seeds=(0, 1, 2, 3))
    sweep = sweep_local_service_fraction(
        workload, demand, layout, bandwidth, placement.fast_pack,
        matched_external_bandwidth_bits_per_second=(
            scenario.matched_payload_bandwidth_bits_per_second),
        effective_compute_flops_per_second=(
            scenario.effective_compute_flops_per_second),
    )
    gains = tuple(item.end_to_end_speedup for item in sweep)
    assert gains[0] == pytest.approx(1.0, abs=1e-14)
    assert all(left <= right for left, right in zip(gains, gains[1:]))


def test_service_rate_potential_decreases_with_occupancy(canonical):
    points = tuple(_evaluate(canonical, requests, 1.0)[0]
                   for requests in (1, 8, 16))
    speedups = tuple(point.tier_aware_fast_pack.service_rate_speedup
                     for point in points)
    occupancies = tuple(point.tier_aware_fast_pack.occupancy_fraction
                        for point in points)
    assert occupancies[0] < occupancies[1] < occupancies[2]
    assert speedups[0] > speedups[1] > speedups[2]


def test_compute_bottleneck_truncates_placement_gain(canonical):
    *_, layout, bandwidth, base, scenario = canonical
    workload = base.model_copy(update={"batch_size": 1})
    demand = build_m3d_workload_page_demand(workload, layout)
    placement = compare_fast_region_placements(
        demand, layout, random_seeds=(0, 1))
    result = evaluate_tier_service_placement(
        workload, demand, layout, bandwidth, placement.fast_pack,
        matched_external_bandwidth_bits_per_second=(
            scenario.matched_payload_bandwidth_bits_per_second),
        effective_compute_flops_per_second=1.0e12,
        local_service_fraction=1.0,
    )
    assert result.no_tier.bottleneck == "COMPUTE"
    assert result.tier_aware_fast_pack.bottleneck == "COMPUTE"
    assert result.end_to_end_speedup == pytest.approx(1.0, abs=1e-14)


def test_existing_external_streaming_negative_control_remains_small(canonical):
    *_, layout, _, base, scenario = canonical
    workload = base.model_copy(update={"batch_size": 1})
    demand = build_m3d_workload_page_demand(workload, layout)
    placement = compare_fast_region_placements(
        demand, layout, random_seeds=(0, 1, 2, 3))
    existing = compare_placement_serving_performance(
        workload, demand, placement, layout,
        matched_payload_bandwidth_bits_per_second=(
            scenario.matched_payload_bandwidth_bits_per_second),
        effective_compute_flops_per_second=(
            scenario.effective_compute_flops_per_second),
    )
    assert existing.tokens_per_s_gain_vs_conventional < 0.01
    assert existing.tokens_per_s_gain_vs_conventional > 0.0


def test_tier_model_does_not_mutate_physics_placement_or_thermal(canonical):
    case, _, power, topology, feol, latency, *_ = canonical
    before = (
        case.model_dump(), power.as_dict(), topology.as_dict(),
        feol.as_dict(), latency.as_dict())
    _evaluate(canonical, 16, 1.0)
    after = (
        case.model_dump(), power.as_dict(), topology.as_dict(),
        feol.as_dict(), latency.as_dict())
    assert after == before
    source = inspect.getsource(tier_module)
    assert "om3dthermal.thermal" not in source
    assert case.architecture.physical_access_latency.mat_latency_ns == 10.0
    assert case.architecture.vertical.miv_resistance_ohm_per_um == 10.0
    assert case.architecture.feol_route.wire.resistance_ohm_per_um == 2.0
