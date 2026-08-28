"""Targeted tests for the unweighted physical access-latency chain."""

from pathlib import Path

import pytest

from om3dthermal.power import (
    calculate_memory_power,
    calculate_physical_access_latency,
    load_case_config,
    resolve_case_geometry,
)
from om3dthermal.power.feol_route import calculate_feol_route
from om3dthermal.power.m3d_subarray import calculate_m3d_subarray
from om3dthermal.power.miv import (
    build_miv_topology,
    calculate_miv_propagation_latency,
)


ROOT = Path(__file__).parents[1]
CASE = ROOT / "configs" / "cases" / "orthogonal_m3d_igzo.yaml"


@pytest.fixture(scope="module")
def canonical_chain():
    case = load_case_config(CASE)
    geometry = resolve_case_geometry(case)
    power = calculate_memory_power(
        case, project_root=ROOT, geometry=geometry)
    assert geometry.m3d is not None
    topology = calculate_m3d_subarray(
        case.architecture.m3d_subarray, geometry.m3d)
    feol = calculate_feol_route(case.architecture.feol_route, topology)
    spec = case.architecture.physical_access_latency
    assert spec is not None
    physical = calculate_physical_access_latency(
        spec,
        feol_route=feol,
        miv_length_per_layer_um=power.diagnostics[
            "miv_length_per_layer_um"],
        miv_delay_per_layer_ns=power.diagnostics["miv_delay_per_layer_ns"],
        miv_status=power.diagnostics["miv_latency_status"],
        miv_parameter_status=power.diagnostics[
            "miv_resistance_parameter_status"],
        miv_provenance=power.diagnostics["miv_resistance_provenance"],
    )
    return case, geometry, power, topology, feol, physical


def test_canonical_feol_nominal_closure(canonical_chain):
    case, _, power, _, feol, _ = canonical_chain
    assert case.architecture.feol_route.wire.resistance_ohm_per_um == 2.0
    assert "not a measured FEOL value" in (
        case.architecture.feol_route.wire.resistance_provenance.source_note)
    assert feol.feol_min_delay_ns == pytest.approx(0.04888440215277369)
    assert feol.feol_median_delay_ns == pytest.approx(2.340850719107647)
    assert feol.feol_p90_delay_ns == pytest.approx(7.436958959498224)
    assert feol.feol_max_delay_ns == pytest.approx(7.998330746824804)
    assert power.diagnostics["feol_resistance_ohm_per_um"] == 2.0


def test_location_count_and_summary_closure(canonical_chain):
    *_, physical = canonical_chain
    assert physical.number_of_clusters == 280
    assert physical.number_of_layers == 8
    assert physical.number_of_locations == 2240
    assert len(physical.locations) == 2240
    assert len(physical.latency_map_ns) == 280
    assert all(len(row) == 8 for row in physical.latency_map_ns)
    assert physical.min_total_latency_ns == pytest.approx(
        10.050912300102683)
    assert physical.max_total_latency_ns == pytest.approx(
        18.008616609416016)
    assert physical.min_location == (26, 1)
    assert physical.max_location == (277, 8)


def test_every_location_has_exact_additive_closure(canonical_chain):
    *_, physical = canonical_chain
    for location in physical.locations:
        assert location.total_latency_ns == pytest.approx(
            location.mat_latency_ns
            + location.miv_latency_ns
            + location.feol_latency_ns
            + location.interface_latency_ns,
            rel=0.0,
            abs=1e-14,
        )


def test_layer_and_cluster_monotonicity(canonical_chain):
    *_, physical = canonical_chain
    for row in physical.latency_map_ns:
        assert all(right > left for left, right in zip(row, row[1:]))

    first_layer = sorted(
        (location.feol_route_length_um,
         location.feol_latency_ns,
         location.total_latency_ns)
        for location in physical.locations
        if location.layer_id == 1)
    for left, right in zip(first_layer, first_layer[1:]):
        if right[0] > left[0]:
            assert right[1] > left[1]
            assert right[2] > left[2]


def test_map_is_independent_of_layer_probability(canonical_chain):
    case, geometry, baseline, *_ = canonical_chain
    workload = case.workload.model_copy(update={
        "layer_access_probability": (1.0, 0.0, 0.0, 0.0,
                                     0.0, 0.0, 0.0, 0.0),
    })
    changed_case = case.model_copy(update={"workload": workload})
    changed = calculate_memory_power(
        changed_case, project_root=ROOT, geometry=geometry)
    assert changed.diagnostics["latency_map_ns"] == (
        baseline.diagnostics["latency_map_ns"])


def test_map_is_independent_of_cluster_access_assumption(canonical_chain):
    case, _, power, topology, feol, baseline = canonical_chain
    changed_spec = case.architecture.feol_route.model_copy(
        update={"access_assumption": "SYNTHETIC_NON_WORKLOAD_ASSUMPTION"})
    changed_feol = calculate_feol_route(changed_spec, topology)
    spec = case.architecture.physical_access_latency
    changed = calculate_physical_access_latency(
        spec,
        feol_route=changed_feol,
        miv_length_per_layer_um=power.diagnostics[
            "miv_length_per_layer_um"],
        miv_delay_per_layer_ns=power.diagnostics["miv_delay_per_layer_ns"],
        miv_status=power.diagnostics["miv_latency_status"],
        miv_parameter_status=power.diagnostics[
            "miv_resistance_parameter_status"],
        miv_provenance=power.diagnostics["miv_resistance_provenance"],
    )
    assert changed.latency_map_ns == baseline.latency_map_ns
    assert changed_feol.feol_route_length_per_cluster_um == (
        feol.feol_route_length_per_cluster_um)


def test_interface_placeholder_is_explicit_and_included(canonical_chain):
    case, _, _, _, _, physical = canonical_chain
    spec = case.architecture.physical_access_latency
    assert spec.interface_latency_ns == 0.0
    assert case.architecture.interface.energy_pj_per_bit == 0.5
    assert physical.interface_classification == "MODELING_PLACEHOLDER"
    assert physical.interface_status == "NOT_YET_CALIBRATED"
    assert physical.interface_included_in_total is True
    assert "not a claim" in physical.interface_note
    assert all(location.interface_latency_ns == 0.0
               for location in physical.locations)
    assert all(location.mat_latency_ns == 10.0
               for location in physical.locations)


def test_component_status_and_semantics(canonical_chain):
    *_, physical = canonical_chain
    assert physical.mat_classification == "MODELING_CHOICE_PLACEHOLDER"
    assert physical.mat_status == "NOT_CAPABILITY_VALIDATED"
    assert physical.miv_status == "RESOLVED"
    assert physical.miv_parameter_status == "CONDITIONAL_MODELING_CHOICE"
    assert physical.feol_status == "CONDITIONAL_MODELING_CHOICE"
    assert physical.physical_latency_semantics == (
        "ONE_WAY_PHYSICAL_MEMORY_ACCESS_PATH")
    assert physical.workload_weighted is False
    assert physical.serialization_included is False
    assert "serialization" in physical.excluded_components
    assert "contention" in physical.excluded_components


def test_energy_geometry_and_miv_regressions(canonical_chain):
    _, _, power, _, feol, _ = canonical_chain
    assert power.E_feol_route_pj_bit == pytest.approx(
        0.16705631334524151)
    assert power.E_vertical_pj_bit == pytest.approx(0.002445862111816407)
    assert feol.feol_route_min_length_um == pytest.approx(314.0405233765899)
    assert feol.feol_route_max_length_um == pytest.approx(4905.139070860771)
    assert power.diagnostics["miv_resistance_ohm_per_um"] == 10.0
    assert power.diagnostics["miv_delay_per_layer_ns"][0] == pytest.approx(
        0.0020278979499085505)
    assert power.diagnostics["miv_delay_per_layer_ns"][-1] == pytest.approx(
        0.010285862591212216)


@pytest.mark.parametrize(
    ("layers", "expected_far", "expected_spread"),
    [
        (64, 18.126486374668968, 8.075574074566285),
        (256, 19.22981672271369, 9.178904422611007),
        (512, 22.385031576534068, 12.334119276431386),
        (1024, 34.46954490621168, 24.418632606109),
    ],
)
def test_miv_layer_geometry_probe(
        canonical_chain, layers, expected_far, expected_spread):
    case, _, _, _, feol, _ = canonical_chain
    topology = build_miv_topology(
        m3d_layers=layers,
        layer_pitch_um=0.288,
        data_width_before_vertical=256,
        vertical_serialization_factor="unresolved",
        row_miv_count=15,
        col_miv_count=18,
    )
    miv = calculate_miv_propagation_latency(
        topology,
        vertical_capacitance_pF_per_um=0.022,
        fixed_load_pF=0.006,
        miv_load_resistance_ohm=100.0,
        miv_resistance_ohm_per_um=10.0,
        parameter_status="CONDITIONAL_MODELING_CHOICE",
        provenance="MODELING_CHOICE",
    )
    physical = calculate_physical_access_latency(
        case.architecture.physical_access_latency,
        feol_route=feol,
        miv_length_per_layer_um=miv.miv_length_per_layer_um,
        miv_delay_per_layer_ns=miv.miv_delay_per_layer_ns,
        miv_status="RESOLVED",
        miv_parameter_status=miv.parameter_status,
        miv_provenance=miv.provenance,
    )
    assert physical.number_of_layers == layers
    assert physical.min_total_latency_ns == pytest.approx(
        10.050912300102683)
    assert physical.max_total_latency_ns == pytest.approx(expected_far)
    assert physical.latency_spread_ns == pytest.approx(expected_spread)
    assert case.geometry.m3d_stack.bitcell_layers == 8
