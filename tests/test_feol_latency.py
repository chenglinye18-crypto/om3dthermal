"""Targeted tests for pure position-dependent FEOL propagation latency."""

from pathlib import Path

import pytest

from om3dthermal.power import (
    calculate_memory_power,
    load_case_config,
    resolve_case_geometry,
)
from om3dthermal.power.feol_route import calculate_feol_route
from om3dthermal.power.m3d_subarray import calculate_m3d_subarray


ROOT = Path(__file__).parents[1]
CASE = ROOT / "configs" / "cases" / "orthogonal_m3d_igzo.yaml"


@pytest.fixture(scope="module")
def canonical_inputs():
    case = load_case_config(CASE)
    geometry = resolve_case_geometry(case)
    assert geometry.m3d is not None
    topology = calculate_m3d_subarray(
        case.architecture.m3d_subarray, geometry.m3d)
    spec = case.architecture.feol_route
    assert spec is not None
    return case, geometry, topology, spec


@pytest.fixture(scope="module")
def canonical_route(canonical_inputs):
    _, _, topology, spec = canonical_inputs
    return calculate_feol_route(spec, topology)


def test_existing_geometry_distribution_is_unchanged(canonical_route):
    route = canonical_route
    assert route.feol_route_cluster_count == 280
    assert route.feol_io_channel_count == 50
    assert route.feol_io_channel_pitch_um == pytest.approx(440.0)
    assert route.feol_route_min_length_um == pytest.approx(314.0405233765899)
    assert route.feol_route_median_length_um == pytest.approx(
        2609.589797118681)
    assert route.feol_route_p90_length_um == pytest.approx(
        4727.037662662098)
    assert route.feol_route_max_length_um == pytest.approx(
        4905.139070860771)
    for index, (center, port_index, length) in enumerate(zip(
            route.feol_route_cluster_centers_um,
            route.feol_route_nearest_port_index,
            route.feol_route_length_per_cluster_um,
            strict=True)):
        port = route.feol_io_channel_coordinates_um[port_index]
        assert length == pytest.approx(
            abs(center[0] - port[0]) + abs(center[1] - port[1]))
        assert length == pytest.approx(
            route.feol_route_lateral_component_per_cluster_um[index]
            + route.feol_route_perpendicular_component_per_cluster_um[index])


def test_unit_and_component_closure(canonical_route):
    route = canonical_route
    index = 0
    length = route.feol_route_length_per_cluster_um[index]
    capacitance_pF = 0.20 * length * 1e-3
    resistance_ohm = 2.0 * length
    driver_ps = 100.0 * (capacitance_pF + 0.006)
    wire_load_ps = resistance_ohm * 0.006
    distributed_ps = 0.5 * resistance_ohm * capacitance_pF
    tau_ps = driver_ps + wire_load_ps + distributed_ps

    assert route.feol_capacitance_pF_per_um == pytest.approx(0.20e-3)
    assert route.feol_wire_capacitance_per_cluster_pF[index] == pytest.approx(
        capacitance_pF)
    assert route.feol_wire_resistance_per_cluster_ohm[index] == pytest.approx(
        resistance_ohm)
    assert route.feol_driver_cap_time_constant_component_per_cluster_ps[
        index] == pytest.approx(driver_ps)
    assert route.feol_wire_load_time_constant_component_per_cluster_ps[
        index] == pytest.approx(wire_load_ps)
    assert route.feol_distributed_wire_time_constant_component_per_cluster_ps[
        index] == pytest.approx(distributed_ps)
    assert route.feol_time_constant_per_cluster_ps[index] == pytest.approx(
        tau_ps)
    assert route.feol_delay_per_cluster_ns[index] == pytest.approx(
        1.6094379124341003 * tau_ps * 1e-3)
    assert route.feol_latency_unit_conversion == (
        "FF_PER_UM_TO_PF_PER_UM_BY_1E-3__"
        "OHM_TIMES_PF_EQUALS_PS__PS_TO_NS_BY_1E-3")


def test_longer_routes_have_larger_r_c_and_delay(canonical_route):
    route = canonical_route
    ordered = sorted(zip(
        route.feol_route_length_per_cluster_um,
        route.feol_wire_resistance_per_cluster_ohm,
        route.feol_wire_capacitance_per_cluster_pF,
        route.feol_delay_per_cluster_ns,
        strict=True))
    for left, right in zip(ordered, ordered[1:]):
        if right[0] > left[0]:
            assert right[1] > left[1]
            assert right[2] > left[2]
            assert right[3] > left[3]


def test_distributed_component_scales_with_length_squared(canonical_route):
    route = canonical_route
    lengths = route.feol_route_length_per_cluster_um
    components = (
        route.feol_distributed_wire_time_constant_component_per_cluster_ps)
    short_index = lengths.index(min(lengths))
    long_index = lengths.index(max(lengths))
    length_ratio = lengths[long_index] / lengths[short_index]
    assert components[long_index] / components[short_index] == pytest.approx(
        length_ratio ** 2)
    length = lengths[short_index]
    at_length = 0.5 * 2.0 * length * 0.20e-3 * length
    at_double_length = 0.5 * 2.0 * (2 * length) * 0.20e-3 * (2 * length)
    assert at_double_length == pytest.approx(4.0 * at_length)


def test_summary_and_pure_physical_semantics_close(canonical_route):
    route = canonical_route
    delays = route.feol_delay_per_cluster_ns
    assert route.feol_min_delay_ns == min(delays)
    assert route.feol_max_delay_ns == max(delays)
    assert route.feol_delay_spread_ns == pytest.approx(max(delays) - min(delays))
    assert route.feol_far_near_ratio == pytest.approx(max(delays) / min(delays))
    assert route.feol_uniform_average_delay_ns == pytest.approx(
        sum(delays) / len(delays))
    assert route.feol_latency_model_name == (
        "FIRST_ORDER_DISTRIBUTED_RC_ELMORE")
    assert route.feol_latency_status == "CONDITIONAL_MODELING_CHOICE"
    assert route.feol_latency_provenance["resistance"][
        "classification"] == "MODELING_CHOICE"
    assert route.feol_latency_provenance["driver_resistance"][
        "status"] == "CONDITIONAL_MODELING_CHOICE"
    assert route.feol_latency_provenance["load_capacitance"][
        "classification"] == "MODELING_CHOICE_PLACEHOLDER"
    assert route.feol_latency_workload_weighted is False
    assert route.feol_latency_serialization_included is False


def test_resistance_and_load_sensitivities_are_monotonic(canonical_inputs):
    _, _, topology, spec = canonical_inputs
    resistance_delays = []
    for resistance in (0.1, 1.0, 2.0):
        wire = spec.wire.model_copy(
            update={"resistance_ohm_per_um": resistance})
        resistance_delays.append(
            calculate_feol_route(
                spec.model_copy(update={"wire": wire}), topology
            ).feol_max_delay_ns)
    assert resistance_delays[0] < resistance_delays[1] < resistance_delays[2]

    load_delays = []
    for load in (0.003, 0.006, 0.012):
        wire = spec.wire.model_copy(update={"fixed_load_pF": load})
        load_delays.append(
            calculate_feol_route(
                spec.model_copy(update={"wire": wire}), topology
            ).feol_max_delay_ns)
    assert load_delays[0] < load_delays[1] < load_delays[2]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("resistance_ohm_per_um", -0.1, "resistance per length"),
        ("fixed_driver_resistance_ohm", -100.0, "driver resistance"),
        ("capacitance_fF_per_um", -0.2, "capacitance per length"),
        ("fixed_load_pF", -0.006, "load capacitance"),
    ],
)
def test_invalid_latency_inputs_fail_loudly(
        canonical_inputs, field, value, message):
    _, _, topology, spec = canonical_inputs
    wire = spec.wire.model_copy(update={field: value})
    with pytest.raises(ValueError, match=message):
        calculate_feol_route(spec.model_copy(update={"wire": wire}), topology)


def test_energy_miv_topology_capacitance_and_serialization_regression(
        canonical_inputs, canonical_route):
    case, geometry, topology, spec = canonical_inputs
    topology_before = topology.as_dict()
    baseline_energy = canonical_route.feol_route_energy_pj_per_bit
    changed_wire = spec.wire.model_copy(
        update={"resistance_ohm_per_um": 2.0})
    changed = calculate_feol_route(
        spec.model_copy(update={"wire": changed_wire}), topology)
    assert changed.feol_route_energy_pj_per_bit == baseline_energy
    assert baseline_energy == pytest.approx(0.16705631334524151)
    assert changed.feol_route_length_per_cluster_um == (
        canonical_route.feol_route_length_per_cluster_um)
    assert changed.feol_wire_capacitance_per_cluster_pF == (
        canonical_route.feol_wire_capacitance_per_cluster_pF)
    assert topology.as_dict() == topology_before
    assert canonical_route.feol_serialization_applied is False

    power = calculate_memory_power(case, project_root=ROOT, geometry=geometry)
    assert power.E_feol_route_pj_bit == pytest.approx(baseline_energy)
    assert power.E_vertical_pj_bit == pytest.approx(0.002445862111816407)
    assert power.diagnostics["miv_resistance_ohm_per_um"] == 10.0
    assert power.diagnostics["miv_latency_status"] == "RESOLVED"
