"""Targeted tests for pure length-dependent MIV propagation latency."""

import math
from pathlib import Path

import pytest

from om3dthermal.power import (
    calculate_memory_power,
    load_case_config,
    resolve_case_geometry,
)
from om3dthermal.power.miv import (
    build_miv_topology,
    calculate_length_scaled_miv_energy,
    calculate_miv_propagation_latency,
)


ROOT = Path(__file__).parents[1]
CASE = ROOT / "configs" / "cases" / "orthogonal_m3d_igzo.yaml"
K80 = -math.log(0.2)


def _topology(*, layers: int = 8, serialization=4):
    return build_miv_topology(
        m3d_layers=layers,
        layer_pitch_um=0.288,
        data_width_before_vertical=256,
        vertical_serialization_factor=serialization,
        row_miv_count=15,
        col_miv_count=18,
    )


def _latency(*, layers: int = 8, resistance_per_um: float = 1.0,
             driver_resistance: float = 100.0, fixed_load: float = 0.006,
             serialization="unresolved"):
    return calculate_miv_propagation_latency(
        _topology(layers=layers, serialization=serialization),
        vertical_capacitance_pF_per_um=0.022,
        fixed_load_pF=fixed_load,
        miv_load_resistance_ohm=driver_resistance,
        miv_resistance_ohm_per_um=resistance_per_um,
    )


@pytest.mark.parametrize("resistance", [0.0, -1.0])
def test_zero_or_negative_driver_resistance_is_rejected(resistance):
    with pytest.raises(ValueError, match="driver resistance"):
        _latency(driver_resistance=resistance)


def test_negative_resistance_per_length_is_rejected():
    with pytest.raises(ValueError, match="resistance per unit length"):
        _latency(resistance_per_um=-0.1)


@pytest.mark.parametrize(
    ("slope", "fixed_load"),
    [(0.0, 0.006), (-0.022, 0.006), (0.022, 0.0), (0.022, -0.006)],
)
def test_invalid_capacitance_is_rejected(slope, fixed_load):
    with pytest.raises(ValueError, match="capacitance"):
        calculate_miv_propagation_latency(
            _topology(serialization="unresolved"),
            vertical_capacitance_pF_per_um=slope,
            fixed_load_pF=fixed_load,
            miv_load_resistance_ohm=100.0,
            miv_resistance_ohm_per_um=1.0,
        )


def test_component_and_unit_closure():
    result = _latency(layers=1, resistance_per_um=2.0)
    length = 0.288
    cwire = 0.022 * length
    cload = 0.006
    rwire = 2.0 * length
    driver_ps = 100.0 * (cwire + cload)
    wire_load_ps = rwire * cload
    distributed_ps = 0.5 * rwire * cwire
    tau_ps = driver_ps + wire_load_ps + distributed_ps

    assert result.miv_wire_resistance_per_layer_ohm[0] == pytest.approx(rwire)
    assert result.miv_effective_capacitance_per_layer_pF[0] == pytest.approx(
        cwire + cload)
    assert result.driver_cap_time_constant_component_per_layer_ps[0] == (
        pytest.approx(driver_ps))
    assert result.wire_load_time_constant_component_per_layer_ps[0] == (
        pytest.approx(wire_load_ps))
    assert result.distributed_wire_time_constant_component_per_layer_ps[0] == (
        pytest.approx(distributed_ps))
    assert result.miv_time_constant_per_layer_ps[0] == pytest.approx(tau_ps)
    assert result.miv_delay_per_layer_ns[0] == pytest.approx(
        K80 * tau_ps * 1e-3)
    assert result.miv_delay_per_layer_ns[0] == pytest.approx(
        result.driver_cap_delay_component_ns[0]
        + result.wire_load_delay_component_ns[0]
        + result.distributed_wire_delay_component_ns[0])
    assert result.unit_conversion == (
        "OHM_TIMES_PF_EQUALS_PS_CONVERTED_TO_NS_BY_1E-3")
    assert result.model_name == "FIRST_ORDER_DISTRIBUTED_RC_ELMORE"
    assert result.serialization_included is False


def test_positive_parameters_are_strictly_monotonic():
    result = _latency(resistance_per_um=1.0)
    for values in (
        result.miv_length_per_layer_um,
        result.miv_effective_capacitance_per_layer_pF,
        result.miv_wire_resistance_per_layer_ohm,
        result.miv_delay_per_layer_ns,
    ):
        assert all(right > left for left, right in zip(values, values[1:]))


def test_distributed_component_has_exact_quadratic_length_scaling():
    result = _latency(
        layers=1024,
        resistance_per_um=1.0,
        driver_resistance=1e-12,
        fixed_load=1e-12,
    )
    distributed = (
        result.distributed_wire_time_constant_component_per_layer_ps)
    assert distributed[1023] == pytest.approx(4.0 * distributed[511])
    assert result.miv_delay_per_layer_ns[1023] == pytest.approx(
        4.0 * result.miv_delay_per_layer_ns[511], rel=1e-9)


def test_single_layer_min_max_average_closure():
    result = _latency(layers=1)
    only = result.miv_delay_per_layer_ns[0]
    assert result.miv_min_delay_ns == only
    assert result.miv_max_delay_ns == only
    assert result.miv_uniform_average_delay_ns == only
    assert result.miv_delay_spread_ns == 0.0
    assert result.miv_far_near_ratio == 1.0


def test_multi_layer_summary_is_unweighted_by_access_probability():
    topology = build_miv_topology(
        m3d_layers=3,
        layer_pitch_um=0.288,
        data_width_before_vertical=256,
        vertical_serialization_factor="unresolved",
        row_miv_count=15,
        col_miv_count=18,
        layer_access_probability=(1.0, 0.0, 0.0),
    )
    result = calculate_miv_propagation_latency(
        topology,
        vertical_capacitance_pF_per_um=0.022,
        fixed_load_pF=0.006,
        miv_load_resistance_ohm=100.0,
        miv_resistance_ohm_per_um=1.0,
    )
    delays = result.miv_delay_per_layer_ns
    assert result.miv_min_delay_ns == min(delays)
    assert result.miv_max_delay_ns == max(delays)
    assert result.miv_uniform_average_delay_ns == pytest.approx(
        sum(delays) / len(delays))
    assert result.miv_delay_spread_ns == pytest.approx(max(delays) - min(delays))
    assert result.miv_far_near_ratio == pytest.approx(max(delays) / min(delays))
    assert result.miv_uniform_average_delay_ns != delays[0]


def test_latency_does_not_change_energy_topology_or_serialization():
    topology = _topology(serialization=4)
    topology_before = topology.as_dict()
    energy_kwargs = dict(
        vertical_capacitance_pF_per_um=0.022,
        fixed_load_pF=0.006,
        row_voltage_product_V2=1.21,
        col_voltage_product_V2=1.21,
        data_voltage_product_V2=0.44,
        data_pumps=32,
        data_transition_factor=0.5,
        control_address_reuse=64,
        atom_size_bits=256,
    )
    energy_before = calculate_length_scaled_miv_energy(
        topology, **energy_kwargs)
    result = calculate_miv_propagation_latency(
        topology,
        vertical_capacitance_pF_per_um=0.022,
        fixed_load_pF=0.006,
        miv_load_resistance_ohm=100.0,
        miv_resistance_ohm_per_um=1.0,
    )
    energy_after = calculate_length_scaled_miv_energy(topology, **energy_kwargs)
    assert energy_after == energy_before
    assert topology.as_dict() == topology_before
    assert topology.vertical_serialization_factor == 4
    assert topology.active_data_miv_count == 64
    assert result.serialization_included is False


def test_canonical_case_resolves_nominal_resistance_and_latency():
    case = load_case_config(CASE)
    vertical = case.architecture.vertical
    assert vertical.miv_resistance_ohm_per_um == 10.0
    assert vertical.miv_resistance_provenance is not None
    assert vertical.miv_resistance_provenance.classification == (
        "MODELING_CHOICE")
    assert vertical.miv_resistance_provenance.status == (
        "CONDITIONAL_MODELING_CHOICE")
    assert "not a measured MIV value" in (
        vertical.miv_resistance_provenance.note)
    assert "not derived from an explicitly modeled MIV cross section" in (
        vertical.miv_resistance_provenance.note)
    geometry = resolve_case_geometry(case)
    result = calculate_memory_power(case, project_root=ROOT, geometry=geometry)
    metadata = result.diagnostics
    assert result.E_vertical_pj_bit == pytest.approx(0.002445862111816407)
    assert metadata["miv_latency_model"] == (
        "FIRST_ORDER_DISTRIBUTED_RC_ELMORE")
    assert metadata["miv_latency_status"] == "RESOLVED"
    assert metadata["miv_resistance_ohm_per_um"] == 10.0
    assert metadata["miv_resistance_provenance"] == "MODELING_CHOICE"
    assert metadata["miv_resistance_parameter_status"] == (
        "CONDITIONAL_MODELING_CHOICE")
    assert metadata["miv_fixed_driver_resistance_ohm"] == 100.0
    assert metadata["miv_fixed_driver_resistance_provenance"] == (
        "DREAMRAM_REFERENCE_PLACEHOLDER")
    assert metadata["miv_latency_serialization_included"] is False
    assert metadata["miv_delay_per_layer_ns"][0] == pytest.approx(
        0.00202789794991)
    assert metadata["miv_delay_per_layer_ns"][-1] == pytest.approx(
        0.0102858625912)
    assert metadata["miv_delay_spread_ns"] == pytest.approx(
        0.0082579646413)
    assert metadata["miv_effective_capacitance_per_layer_pF"] == pytest.approx(
        tuple(0.006 + 0.022 * 0.288 * index for index in range(1, 9)))


def test_nominal_1024_layer_scaling_closure():
    result = _latency(layers=1024, resistance_per_um=10.0)
    assert result.miv_length_per_layer_um[-1] == pytest.approx(294.912)
    assert result.miv_max_delay_ns == pytest.approx(16.4712141594)


def test_resistance_per_length_sensitivity_is_monotonic():
    far_delays = [
        _latency(layers=1024, resistance_per_um=value).miv_max_delay_ns
        for value in (1.0, 10.0, 20.0)
    ]
    assert far_delays[0] < far_delays[1] < far_delays[2]
