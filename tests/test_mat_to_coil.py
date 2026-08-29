"""Targeted tests for the read-only hierarchical MAT-to-coil diagnostic."""

from pathlib import Path

import pytest

from om3dthermal.power import (
    audit_dream_latency_decomposition,
    calculate_hierarchical_mat_to_coil,
    calculate_memory_power,
    calculate_normalized_single_path_delay,
    calculate_wire_rc_delay,
    load_case_config,
    resolve_case_geometry,
)
from om3dthermal.power.feol_route import calculate_feol_route
from om3dthermal.power.m3d_subarray import calculate_m3d_subarray


ROOT = Path(__file__).parents[1]
CASE = ROOT / "configs" / "cases" / "orthogonal_m3d_igzo.yaml"


@pytest.fixture(scope="module")
def canonical():
    case = load_case_config(CASE)
    geometry = resolve_case_geometry(case)
    assert geometry.m3d is not None
    topology = calculate_m3d_subarray(
        case.architecture.m3d_subarray, geometry.m3d)
    feol = calculate_feol_route(case.architecture.feol_route, topology)
    audit = calculate_hierarchical_mat_to_coil(feol)
    return case, geometry, topology, feol, audit


def test_cluster_port_and_fan_in_closure(canonical):
    _, _, _, feol, audit = canonical
    assert audit.cluster_count == 280
    assert audit.port_count == 50
    assert audit.active_port_count == 35
    assert len(audit.paths) == 280
    assert len(audit.ports) == 50
    assert sum(audit.port_fan_in_counts) == 280
    assert audit.port_fan_in_distribution == {0: 15, 8: 35}
    assert (
        audit.fan_in_min, audit.fan_in_median,
        audit.fan_in_mean, audit.fan_in_max) == (0, 8.0, 5.6, 8)
    assert (
        audit.active_fan_in_min, audit.active_fan_in_median,
        audit.active_fan_in_mean, audit.active_fan_in_max) == (
            8, 8.0, 8.0, 8)
    assert tuple(path.assigned_port for path in audit.paths) == (
        feol.feol_route_nearest_port_index)
    assert len({path.cluster_id for path in audit.paths}) == 280


def test_collectors_are_deterministic_geometry_medians(canonical):
    _, _, _, feol, first = canonical
    second = calculate_hierarchical_mat_to_coil(feol)
    assert first.paths == second.paths
    assert first.collector_provenance == "GEOMETRY_DERIVED"
    for port in first.ports:
        assigned = [path for path in first.paths
                    if path.assigned_port == port.port_id]
        if not assigned:
            assert port.collector_x_um is None
            assert port.collector_y_um is None
            continue
        assert port.collector_x_um == pytest.approx(
            assigned[0].collector_x_um)
        assert port.collector_y_um == pytest.approx(
            assigned[0].collector_y_um)
        assert port.collector_x_um == pytest.approx(
            sorted(path.cluster_x_um for path in assigned)[3:5][0])
        middle_y = sorted(path.cluster_y_um for path in assigned)[3:5]
        assert port.collector_y_um == pytest.approx(sum(middle_y) / 2.0)


def test_branch_trunk_lengths_and_no_cluster_serialization(canonical):
    *_, audit = canonical
    assert all(path.branch_length_um > 0.0 for path in audit.paths)
    assert all(path.trunk_length_um > 0.0 for path in audit.paths)
    for path in audit.paths:
        assert path.total_physical_path_um == pytest.approx(
            path.branch_length_um + path.trunk_length_um)
        assert path.total_lateral_delay_ns == pytest.approx(
            path.branch_wire_delay_ns + path.shared_trunk_wire_delay_ns)
        assert path.port_selection_delay_ns == 0.0
    # One access traverses one branch and one trunk. Fan-in is not used as a
    # latency multiplier and no other cluster's branch is serialized into it.
    assert all(path.structural_aggregation_delay_ns == pytest.approx(
        path.branch_wire_delay_ns + path.shared_trunk_wire_delay_ns)
        for path in audit.paths)
    assert audit.serialization_included is False
    assert audit.contention_included is False


def test_rc_unit_closure():
    result = calculate_wire_rc_delay(
        length_um=1_000.0,
        resistance_ohm_per_um=2.0,
        capacitance_fF_per_um=0.20,
        driver_resistance_ohm=100.0,
        endpoint_load_pF=0.006,
    )
    assert result.wire_resistance_ohm == pytest.approx(2_000.0)
    assert result.wire_capacitance_pF == pytest.approx(0.20)
    assert result.driver_cap_time_constant_ps == pytest.approx(20.6)
    assert result.wire_load_time_constant_ps == pytest.approx(12.0)
    assert result.distributed_wire_time_constant_ps == pytest.approx(200.0)
    assert result.time_constant_ps == pytest.approx(232.6)
    assert result.delay_ns == pytest.approx(
        1.6094379124341003 * 232.6e-3)


def test_direct_regression_and_hierarchical_lower_bound(canonical):
    *_, feol, audit = canonical
    direct = audit.legacy_direct_latency
    assert direct.min_ns == pytest.approx(0.04888440215277369)
    assert direct.median_ns == pytest.approx(2.340850719107647)
    assert direct.p90_ns == pytest.approx(7.436958959498224)
    assert direct.max_ns == pytest.approx(7.998330746824804)
    assert direct.mean_ns == pytest.approx(2.9899599707123805)
    hierarchical = audit.hierarchical_latency
    assert hierarchical.min_ns == pytest.approx(2.1854557458199926)
    assert hierarchical.median_ns == pytest.approx(2.9057538467419777)
    assert hierarchical.p90_ns == pytest.approx(3.9788701865006777)
    assert hierarchical.max_ns == pytest.approx(4.128970063191266)
    assert hierarchical.mean_ns == pytest.approx(3.0301946692195023)
    assert audit.hierarchical_model_status == (
        "HIERARCHICAL_TOPOLOGY_LOWER_BOUND")
    assert audit.port_count_architecture_status == "ARCHITECTURE_FEATURE"
    assert audit.port_connectivity_status == (
        "CONNECTIVITY_ASSUMPTION_NOT_VALIDATED")
    assert "COLLECTOR_LOGIC_DELAY_ZERO_LOWER_BOUND" in (
        audit.segment_driver_assumption)
    assert audit.port_selection_status == (
        "NOT_CALIBRATED_ZERO_NS_LOWER_BOUND")
    assert feol.feol_delay_per_cluster_ns is not None


def test_aggregation_load_sensitivity_is_monotonic(canonical):
    *_, audit = canonical
    rows = audit.load_sensitivity
    assert tuple(row.aggregation_load_multiplier for row in rows) == (
        0.0, 1.0, 2.0, 4.0)
    medians = tuple(row.latency.median_ns for row in rows)
    maxima = tuple(row.latency.max_ns for row in rows)
    assert all(right > left for left, right in zip(medians, medians[1:]))
    assert all(right > left for left, right in zip(maxima, maxima[1:]))


def test_dream_lateral_and_11mm_normalized_regressions(canonical):
    *_, feol, _ = canonical
    dream = audit_dream_latency_decomposition(ROOT)
    normalized = calculate_normalized_single_path_delay(feol)
    assert dream.tcl_lateral_bus_ns == pytest.approx(35.46576092306267)
    assert normalized.length_um == pytest.approx(11_000.0)
    assert normalized.delay_ns == pytest.approx(39.5158852888295)


def test_diagnostic_does_not_mutate_canonical_or_frozen_inputs(canonical):
    case, geometry, topology, feol, audit = canonical
    case_before = case.model_dump()
    topology_before = topology.as_dict()
    feol_before = feol.as_dict()
    power_before = calculate_memory_power(
        case, project_root=ROOT, geometry=geometry)

    calculate_hierarchical_mat_to_coil(feol, aggregation_load_multiplier=4.0)
    power_after = calculate_memory_power(
        case, project_root=ROOT, geometry=geometry)

    assert case.model_dump() == case_before
    assert topology.as_dict() == topology_before
    assert feol.as_dict() == feol_before
    assert audit.canonical_feol_mutated is False
    assert case.architecture.physical_access_latency.mat_latency_ns == 10.0
    assert case.architecture.vertical.miv_resistance_ohm_per_um == 10.0
    assert case.architecture.memory_service.coil.data_rate_gbps_per_link == 8.0
    assert case.architecture.memory_service.coil.links_per_die == 50
    assert case.architecture.m3d_subarray.access.accessed_clusters_per_access == 4
    assert power_after.E_vertical_pj_bit == power_before.E_vertical_pj_bit
    assert power_after.E_feol_route_pj_bit == power_before.E_feol_route_pj_bit
    assert power_after.diagnostics["miv_delay_per_layer_ns"] == (
        power_before.diagnostics["miv_delay_per_layer_ns"])


def test_taxonomy_and_current_path_audit_are_explicit(canonical):
    *_, audit = canonical
    current = audit.current_feol_data_path_audit
    for field in (
            "explicit_mux", "regional_collector", "shared_global_bus",
            "arbitration", "serialization", "repeater_or_buffer",
            "port_fan_in_loading"):
        assert current[field] == "NOT_MODELED"
    assert current["port_connectivity_validation"] == (
        "CONNECTIVITY_ASSUMPTION_NOT_VALIDATED")
    assert audit.taxonomy["L2_CLUSTER_TO_REGIONAL_AGGREGATION"]["status"] == (
        "EXPLICITLY_MODELED")
    assert audit.taxonomy["L4_EDGE_PORT_FAN_IN_SELECTION"]["status"] == (
        "LUMPED")
