"""Targeted tests for compact physical slot/capacity semantics."""

from dataclasses import replace
from itertools import islice
from pathlib import Path

import pytest

from om3dthermal.architecture import resolve_packing_from_legacy_power_result
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


ROOT = Path(__file__).parents[1]
CASE = ROOT / "configs" / "cases" / "orthogonal_m3d_igzo.yaml"


@pytest.fixture(scope="module")
def canonical_layout():
    case = load_case_config(CASE)
    geometry = resolve_case_geometry(case)
    power = calculate_memory_power(
        case, project_root=ROOT, geometry=geometry)
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
        topology,
        latency,
        slab_count=geometry.memory_region_count,
        expected_total_bits=power.diagnostics["total_stored_bits"],
    )
    return case, geometry, power, topology, feol, latency, layout


def test_subarray_and_slot_capacity_closure(canonical_layout):
    *_, layout = canonical_layout
    assert layout.subarray_rows == 512
    assert layout.subarray_cols == 512
    assert layout.subarray_capacity_bytes == 32 * 2**10
    assert layout.subarrays_per_cluster == 64
    assert layout.slot_capacity_bytes == 2 * 2**20
    assert all(slot.capacity_bytes == 2 * 2**20
               for slot in layout.slot_classes)
    assert all(slot.capacity_mib == 2.0 for slot in layout.slot_classes)


def test_architecture_capacity_closure(canonical_layout):
    case, geometry, power, *_, layout = canonical_layout
    assert layout.slab_count == 98
    assert layout.clusters_per_slab == 280
    assert layout.layers_per_cluster == 8
    assert layout.slot_class_count == 2240
    assert layout.physical_slot_count == 280 * 8 * 98 == 219520
    assert layout.capacity_per_layer_per_slab_bytes == 560 * 2**20
    assert layout.capacity_per_slab_bytes == int(4.375 * 2**30)
    assert layout.total_capacity_bytes == 460366807040
    assert layout.total_capacity_gib == 428.75
    assert layout.total_capacity_bytes * 8 == power.diagnostics[
        "total_stored_bits"]
    packing = resolve_packing_from_legacy_power_result(case, geometry, power)
    assert layout.total_capacity_bytes == packing.system_capacity_bytes


def test_slot_classes_bind_existing_latency_without_recalculation(
        canonical_layout):
    *_, latency, layout = canonical_layout
    latency_by_location = {
        (location.cluster_id, location.layer_id): location
        for location in latency.locations
    }
    assert len(layout.slot_classes) == len(latency_by_location)
    for slot in layout.slot_classes:
        source = latency_by_location[(slot.cluster_id, slot.layer_id)]
        assert slot.physical_access_latency_ns == source.total_latency_ns
        assert slot.feol_route_length_um == source.feol_route_length_um
        assert slot.miv_length_um == source.miv_length_um
        assert slot.multiplicity == 98


def test_slab_symmetry_and_lazy_expansion(canonical_layout):
    *_, layout = canonical_layout
    assert layout.slab_symmetry is True
    assert layout.host_capacity_included is False
    assert layout.workload_weighted is False
    first_class_slots = tuple(islice(iter_physical_slots(layout), 98))
    assert tuple(slot.slab_id for slot in first_class_slots) == tuple(range(98))
    assert len({(slot.cluster_id, slot.layer_id)
                for slot in first_class_slots}) == 1
    assert all(slot.capacity_bytes == layout.slot_capacity_bytes
               for slot in first_class_slots)


def test_cumulative_capacity_curve_closes(canonical_layout):
    *_, layout = canonical_layout
    expected_gib = (42.875, 107.1875, 214.375,
                    321.5625, 385.875, 428.75)
    expected_cutoffs = (
        10.109013444422171,
        10.500345597231911,
        11.997266370238673,
        14.500023374300042,
        17.446238134993205,
        18.008616609416016,
    )
    assert tuple(point.cumulative_capacity_gib
                 for point in layout.capacity_latency_cutoffs) == expected_gib
    assert tuple(point.latency_cutoff_ns
                 for point in layout.capacity_latency_cutoffs) == (
                     pytest.approx(expected_cutoffs))
    assert layout.capacity_latency_cutoffs[-1].cumulative_capacity_bytes == (
        layout.total_capacity_bytes)


def test_capacity_changes_with_dynamic_geometry(canonical_layout):
    *_, topology, _, latency, _ = canonical_layout
    rows = 1024
    cols = 384
    subarrays_x = 4
    subarrays_y = 4
    subarrays_per_cluster = subarrays_x * subarrays_y
    bits_per_subarray = rows * cols
    modified = replace(
        topology,
        Nrow=rows,
        Ncol=cols,
        cluster_subarrays_x=subarrays_x,
        cluster_subarrays_y=subarrays_y,
        subarrays_per_cluster=subarrays_per_cluster,
        bits_per_subarray=bits_per_subarray,
        subarrays_per_layer=(
            topology.clusters_per_layer * subarrays_per_cluster),
        bits_per_layer=(
            topology.clusters_per_layer
            * subarrays_per_cluster
            * bits_per_subarray),
    )
    layout = calculate_physical_capacity_layout(
        modified, latency, slab_count=98)
    assert layout.subarray_capacity_bytes == 48 * 2**10
    assert layout.slot_capacity_bytes == 768 * 2**10
    assert layout.total_capacity_gib == pytest.approx(160.78125)


def test_capacity_layout_has_no_workload_probability_dependence(
        canonical_layout):
    case, geometry, baseline_power, *_, baseline_layout = canonical_layout
    workload = case.workload.model_copy(update={
        "layer_access_probability": (1.0, 0.0, 0.0, 0.0,
                                     0.0, 0.0, 0.0, 0.0),
    })
    changed = calculate_memory_power(
        case.model_copy(update={"workload": workload}),
        project_root=ROOT,
        geometry=geometry,
    )
    assert changed.diagnostics["total_capacity_bytes"] == (
        baseline_layout.total_capacity_bytes)
    assert changed.diagnostics["slot_classes"] == (
        baseline_power.diagnostics["slot_classes"])
    assert changed.diagnostics["capacity_latency_cutoffs"] == (
        baseline_power.diagnostics["capacity_latency_cutoffs"])


def test_capacity_layout_has_no_cluster_access_assumption_dependence(
        canonical_layout):
    case, _, power, topology, feol, _, baseline = canonical_layout
    changed_spec = case.architecture.feol_route.model_copy(
        update={"access_assumption": "SYNTHETIC_NON_WORKLOAD_ASSUMPTION"})
    changed_feol = calculate_feol_route(changed_spec, topology)
    changed_latency = calculate_physical_access_latency(
        case.architecture.physical_access_latency,
        feol_route=changed_feol,
        miv_length_per_layer_um=power.diagnostics[
            "miv_length_per_layer_um"],
        miv_delay_per_layer_ns=power.diagnostics["miv_delay_per_layer_ns"],
        miv_status=power.diagnostics["miv_latency_status"],
        miv_parameter_status=power.diagnostics[
            "miv_resistance_parameter_status"],
        miv_provenance=power.diagnostics["miv_resistance_provenance"],
    )
    changed = calculate_physical_capacity_layout(
        topology, changed_latency, slab_count=98,
        expected_total_bits=power.diagnostics["total_stored_bits"])
    assert changed_feol.feol_route_length_per_cluster_um == (
        feol.feol_route_length_per_cluster_um)
    assert changed.slot_classes == baseline.slot_classes
    assert changed.capacity_latency_cutoffs == baseline.capacity_latency_cutoffs


def test_latency_energy_and_capacity_regressions(canonical_layout):
    _, _, power, _, feol, latency, layout = canonical_layout
    assert latency.min_total_latency_ns == pytest.approx(10.050912300102683)
    assert latency.max_total_latency_ns == pytest.approx(18.008616609416016)
    assert power.E_vertical_pj_bit == pytest.approx(0.002445862111816407)
    assert power.E_feol_route_pj_bit == pytest.approx(0.16705631334524151)
    assert feol.feol_route_min_length_um == pytest.approx(314.0405233765899)
    assert feol.feol_route_max_length_um == pytest.approx(4905.139070860771)
    assert layout.total_capacity_gib == 428.75
