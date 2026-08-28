"""Targeted M3D-only fast-region placement tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from om3dthermal.experiment.config import load_workload_spec
from om3dthermal.placement import (
    FastRegionCapacityError,
    compare_fast_region_placements,
    evaluate_fast_region_occupancy_sweep,
    place_pages_on_slots,
    select_physical_slots,
)
from om3dthermal.power import (
    calculate_memory_power,
    calculate_physical_access_latency,
    calculate_physical_capacity_layout,
    iter_physical_slots,
    load_case_config,
    resolve_case_geometry,
)
from om3dthermal.power.feol_route import calculate_feol_route
from om3dthermal.power.m3d_subarray import calculate_m3d_subarray
from om3dthermal.workload import build_m3d_workload_page_demand


ROOT = Path(__file__).parents[1]
CASE = ROOT / "configs" / "cases" / "orthogonal_m3d_igzo.yaml"
WORKLOAD = ROOT / "configs" / "workload" / "llama31_8b_decode_b1_s131072.yaml"


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
    capacity = calculate_physical_capacity_layout(
        topology,
        latency,
        slab_count=geometry.memory_region_count,
        expected_total_bits=power.diagnostics["total_stored_bits"],
    )
    workload = load_workload_spec(WORKLOAD, project_root=ROOT).decode
    return capacity, workload


def test_fast_pack_is_global_capacity_prefix_without_duplicates(canonical) -> None:
    layout, _ = canonical
    count = 12_345
    result = select_physical_slots(layout, count, policy="FASTEST")
    all_latencies = sorted(
        slot.physical_access_latency_ns for slot in iter_physical_slots(layout))
    identities = {(slot.slab_id, slot.cluster_id, slot.layer_id)
                  for slot in result.selected_slots}
    assert result.selected_slot_count == count
    assert len(identities) == count
    assert [slot.physical_access_latency_ns
            for slot in result.selected_slots] == all_latencies[:count]


def test_random_is_seeded_distinct_and_without_replacement(canonical) -> None:
    layout, _ = canonical
    first = select_physical_slots(layout, 1000, policy="RANDOM", random_seed=7)
    same = select_physical_slots(layout, 1000, policy="RANDOM", random_seed=7)
    different = select_physical_slots(
        layout, 1000, policy="RANDOM", random_seed=8)
    assert first.selected_slots == same.selected_slots
    assert first.selected_slots != different.selected_slots
    identities = {(slot.slab_id, slot.cluster_id, slot.layer_id)
                  for slot in first.selected_slots}
    assert len(identities) == 1000


def test_full_occupancy_uniform_latency_and_zero_gain(canonical) -> None:
    layout, _ = canonical
    point = evaluate_fast_region_occupancy_sweep(
        layout, (1.0,), random_seeds=(0, 1, 2))[0]
    sequential = select_physical_slots(
        layout, layout.physical_slot_count, policy="SEQUENTIAL")
    random_full = select_physical_slots(
        layout, layout.physical_slot_count, policy="RANDOM", random_seed=0)
    assert point.selected_slot_count == layout.physical_slot_count
    assert point.fast_pack_average_slot_latency_ns == pytest.approx(
        point.random_mean_average_slot_latency_ns)
    assert point.fast_pack_average_slot_latency_ns == pytest.approx(
        sequential.mean_slot_latency_ns)
    assert point.fast_pack_average_slot_latency_ns == pytest.approx(
        random_full.mean_slot_latency_ns)
    assert point.slot_selection_gain_vs_random == pytest.approx(0.0, abs=1e-14)


@pytest.mark.parametrize("requests", (1, 8, 16))
def test_canonical_workloads_fit_and_fast_pack_beats_random(
        canonical, requests) -> None:
    layout, workload = canonical
    demand = build_m3d_workload_page_demand(
        workload.model_copy(update={"batch_size": requests}), layout)
    comparison = compare_fast_region_placements(
        demand, layout, random_seeds=(0, 1, 2, 3))
    assert comparison.page_count == demand.page_count
    assert comparison.fast_pack.weighted_average_access_latency_ns <= (
        comparison.random.mean_average_access_latency_ns)
    assert comparison.fast_pack.total_read_demand_bytes_per_decode_step == (
        pytest.approx(demand.total_read_bytes_per_decode_step))
    assert comparison.page_ordering_gain >= 0.0


def test_oversize_workload_fails_loudly(canonical) -> None:
    layout, _ = canonical
    with pytest.raises(
            FastRegionCapacityError, match="M3D_FAST_REGION_CAPACITY_FAIL"):
        select_physical_slots(
            layout, layout.physical_slot_count + 1, policy="FASTEST")


def test_compact_multiplicity_closes_to_expanded_slots(canonical) -> None:
    layout, _ = canonical
    assert sum(slot.multiplicity for slot in layout.slot_classes) == (
        layout.physical_slot_count)
    assert len(tuple(iter_physical_slots(layout))) == layout.physical_slot_count
    assert layout.physical_slot_count == 219_520


def test_fast_capacity_cutoff_closure(canonical) -> None:
    layout, _ = canonical
    fractions = tuple(
        cutoff.capacity_fraction for cutoff in layout.capacity_latency_cutoffs)
    sweep = evaluate_fast_region_occupancy_sweep(
        layout, fractions, random_seeds=(0, 1))
    assert tuple(point.fast_pack_max_occupied_latency_ns for point in sweep) == (
        pytest.approx(tuple(
            cutoff.latency_cutoff_ns
            for cutoff in layout.capacity_latency_cutoffs)))


def test_occupancy_reduces_placement_freedom_and_gain(canonical) -> None:
    layout, _ = canonical
    sweep = evaluate_fast_region_occupancy_sweep(
        layout, random_seeds=(0, 1, 2))
    gains = tuple(point.slot_selection_gain_vs_random for point in sweep)
    assert all(left >= right for left, right in zip(gains, gains[1:]))
    assert gains[0] > 0.20
    assert gains[-1] == pytest.approx(0.0, abs=1e-14)


def test_same_slot_set_exposes_only_page_ordering_gain(canonical) -> None:
    layout, workload = canonical
    demand = build_m3d_workload_page_demand(
        workload.model_copy(update={"batch_size": 1}), layout)
    aware = place_pages_on_slots(
        demand, layout, slot_policy="FASTEST",
        page_ordering="DEMAND_DESCENDING")
    canonical_order = place_pages_on_slots(
        demand, layout, slot_policy="FASTEST", page_ordering="CANONICAL")
    aware_slots = {(x.slab_id, x.cluster_id, x.layer_id)
                   for x in aware.assignments}
    canonical_slots = {(x.slab_id, x.cluster_id, x.layer_id)
                       for x in canonical_order.assignments}
    assert aware_slots == canonical_slots
    assert aware.weighted_average_access_latency_ns <= (
        canonical_order.weighted_average_access_latency_ns)
