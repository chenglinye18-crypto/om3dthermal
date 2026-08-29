"""Placement-dependent physical latency through existing serving timing."""

from __future__ import annotations

from pathlib import Path

import pytest

from om3dthermal.experiment.config import load_workload_spec
from om3dthermal.placement import (
    compare_fast_region_placements,
    compare_placement_serving_performance,
    evaluate_fast_region_occupancy_sweep,
    evaluate_placement_serving_timing,
    propagate_occupancy_sweep_to_serving,
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
from om3dthermal.workload import build_m3d_workload_page_demand


ROOT = Path(__file__).parents[1]
CASE = ROOT / "configs" / "cases" / "orthogonal_m3d_igzo.yaml"
WORKLOAD = ROOT / "configs" / "workload" / "llama31_8b_decode_b1_s131072.yaml"
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
    workload = load_workload_spec(WORKLOAD, project_root=ROOT).decode
    return layout, workload


def _timing(canonical, requests: int, latency_ns: float, strategy: str):
    layout, base = canonical
    workload = base.model_copy(update={"batch_size": requests})
    demand = build_m3d_workload_page_demand(workload, layout)
    return evaluate_placement_serving_timing(
        workload,
        demand,
        layout,
        strategy=strategy,
        physical_access_latency_avg_ns=latency_ns,
        physical_access_latency_max_ns=latency_ns,
        matched_payload_bandwidth_bits_per_second=BW,
        effective_compute_flops_per_second=COMPUTE,
    ), demand


def test_units_and_memory_stage_closure(canonical) -> None:
    result, demand = _timing(canonical, 8, 10.0, "UNIT_TEST")
    expected_page_equivalents = (
        demand.total_read_bytes_per_decode_step
        / demand.page_layout.page_size_bytes)
    expected_access_ms = expected_page_equivalents * 10.0 * 1e-6
    expected_bandwidth_ms = (
        (demand.total_read_bytes_per_decode_step
         + demand.kv_write_bytes_per_decode_step)
        * 8.0 / BW * 1e3)
    assert result.read_page_equivalents_per_decode_step == pytest.approx(
        expected_page_equivalents)
    assert result.memory_access_latency_step_time_ms == pytest.approx(
        expected_access_ms)
    assert result.memory_bandwidth_step_time_ms == pytest.approx(
        expected_bandwidth_ms)
    assert result.memory_stage_step_time_ms == pytest.approx(
        expected_access_ms + expected_bandwidth_ms)


def test_placement_latency_propagates_and_equal_latency_closes(canonical) -> None:
    fast, _ = _timing(canonical, 8, 10.0, "FAST")
    slow, _ = _timing(canonical, 8, 13.0, "SLOW")
    equal, _ = _timing(canonical, 8, 10.0, "EQUAL")
    assert fast.memory_stage_step_time_ms < slow.memory_stage_step_time_ms
    assert fast.total_step_time_ms < slow.total_step_time_ms
    assert fast.aggregate_tokens_per_s > slow.aggregate_tokens_per_s
    assert fast.total_step_time_ms == equal.total_step_time_ms
    assert fast.aggregate_tokens_per_s == equal.aggregate_tokens_per_s


@pytest.mark.parametrize("requests", (1, 8, 16))
def test_canonical_fast_conventional_random_e2e(canonical, requests) -> None:
    layout, base = canonical
    workload = base.model_copy(update={"batch_size": requests})
    demand = build_m3d_workload_page_demand(workload, layout)
    placement = compare_fast_region_placements(
        demand, layout, random_seeds=(0, 1, 2, 3))
    result = compare_placement_serving_performance(
        workload,
        demand,
        placement,
        layout,
        matched_payload_bandwidth_bits_per_second=BW,
        effective_compute_flops_per_second=COMPUTE,
    )
    assert result.fast_pack.total_step_time_ms < (
        result.conventional.total_step_time_ms)
    assert result.fast_pack.aggregate_tokens_per_s > (
        result.conventional.aggregate_tokens_per_s)
    assert result.end_to_end_latency_gain_vs_conventional > 0.0


def test_full_occupancy_physical_and_e2e_gain_closes_to_zero(canonical) -> None:
    layout, base = canonical
    workload = base.model_copy(update={"batch_size": 8})
    demand = build_m3d_workload_page_demand(workload, layout)
    sweep = evaluate_fast_region_occupancy_sweep(
        layout, (1.0,), random_seeds=(0, 1, 2))
    closure = propagate_occupancy_sweep_to_serving(
        workload,
        demand,
        layout,
        sweep,
        matched_payload_bandwidth_bits_per_second=BW,
        effective_compute_flops_per_second=COMPUTE,
    )[0]
    conventional = select_physical_slots(
        layout, layout.physical_slot_count,
        policy="CONVENTIONAL_LATENCY_OBLIVIOUS")
    assert closure.fast_physical_latency_ns == pytest.approx(
        conventional.mean_slot_latency_ns)
    assert closure.physical_latency_gain == pytest.approx(0.0, abs=1e-14)
    assert closure.end_to_end_latency_gain == pytest.approx(0.0, abs=1e-14)
    assert closure.tokens_per_s_gain == pytest.approx(0.0, abs=1e-14)
