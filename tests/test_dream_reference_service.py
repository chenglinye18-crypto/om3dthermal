"""DreamRAM/reference-baseline memory-service audit tests.

Covers provenance labeling, latency and service-cycle closure, unit
conversion for internal and interface bandwidth, effective-bandwidth
min-semantics, bottleneck identity, M3D-parameter isolation, and
regression guards that the existing M3D coil derivation, internal
bandwidth, and physical latency remain unchanged.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from om3dthermal.power import (
    audit_dream_reference_service,
    calculate_memory_power,
    calculate_physical_access_latency,
    calculate_physical_capacity_layout,
    classify_bottleneck,
    derive_architecture_bandwidth,
    load_case_config,
    resolve_case_geometry,
)
from om3dthermal.power import dream_reference_service as dream_module
from om3dthermal.power.feol_route import calculate_feol_route
from om3dthermal.power.m3d_subarray import calculate_m3d_subarray


ROOT = Path(__file__).parents[1]
CASE = ROOT / "configs/cases/orthogonal_m3d_igzo.yaml"

ALLOWED_PROVENANCE = {
    "PAPER_REPORTED",
    "DERIVED_FROM_PAPER",
    "EXISTING_REPO_REFERENCE",
    "MODELING_CHOICE",
    "UNAVAILABLE",
}


@pytest.fixture(scope="module")
def dream():
    return audit_dream_reference_service(ROOT)


@pytest.fixture(scope="module")
def m3d():
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
    closure = derive_architecture_bandwidth(
        case.architecture.memory_service,
        layout,
        topology,
        feol_io_channels=case.architecture.feol_route.io_channels,
    )
    return case, latency, layout, closure


def test_provenance_labels_are_explicit_and_allowed(dream) -> None:
    assert set(dream.provenance) >= {
        "organization",
        "timing",
        "internal_stage_rates",
        "aggregation_rule",
        "interface_bandwidth",
    }
    assert set(dream.provenance.values()) <= ALLOWED_PROVENANCE
    assert dream.latency.timing_classification in ALLOWED_PROVENANCE
    assert dream.aggregation_rule_classification in ALLOWED_PROVENANCE
    assert all(stage.classification in ALLOWED_PROVENANCE
               for stage in dream.internal_stages)


def test_latency_decomposition_closure(dream) -> None:
    latency = dream.latency
    assert latency.first_access_latency_ns == pytest.approx(
        latency.trp_ns + latency.trcd_ns + latency.tcl_ns)
    assert latency.trp_ns > 0.0
    assert latency.trcd_ns > 0.0
    assert latency.tcl_ns > 0.0
    assert "TRP" in latency.first_access_definition
    assert "TRCD" in latency.first_access_definition
    assert "TCL" in latency.first_access_definition


def test_service_cycle_closure(dream) -> None:
    latency = dream.latency
    assert latency.repeated_service_cycle_ns == pytest.approx(
        latency.bank_clks_per_atom * latency.core_tck_ns)
    assert latency.dq_atom_window_ns > 0.0
    assert latency.repeated_service_cycle_ns > 0.0
    # The DQ atom window reflects the interface-side serialization and is
    # never slower than the core column cycle in this baseline.
    assert latency.dq_atom_window_ns == pytest.approx(
        0.5 * latency.core_tck_ns, rel=1e-6)


def test_internal_bandwidth_unit_conversion_and_stage_closure(dream) -> None:
    org = dream.organization
    core_clock_hz = 1.0 / (dream.latency.core_tck_ns * 1e-9)
    pseudochannels = org["channels"] * org["pseudochannels"]
    for stage in dream.internal_stages:
        assert stage.aggregate_bits_per_core_clock_per_pseudochannel == (
            pytest.approx(
                stage.parallel_units_per_pseudochannel
                * stage.payload_bits_per_core_clock_per_unit))
        assert stage.aggregate_bits_per_s == pytest.approx(
            pseudochannels
            * stage.aggregate_bits_per_core_clock_per_pseudochannel
            * core_clock_hz)
    # The bus hierarchy narrows monotonically from array to TSV.
    ordered = [item.aggregate_bits_per_s for item in dream.internal_stages]
    assert all(left >= right for left, right in zip(ordered, ordered[1:]))
    assert dream.internal_bandwidth_bits_per_s == min(ordered)
    assert dream.internal_bandwidth_bytes_per_s == (
        dream.internal_bandwidth_bits_per_s / 8.0)
    assert dream.internal_binding_stages == ("gbus", "tsv")


def test_interface_bandwidth_unit_conversion(dream) -> None:
    reconstructed_bits_per_s = (
        dream.interface_num_links
        * dream.interface_rate_gbps_per_link
        * 1e9
        / dream.interface_payload_ecc_factor)
    assert dream.interface_bandwidth_bits_per_s == pytest.approx(
        reconstructed_bits_per_s)
    assert dream.interface_bandwidth_bytes_per_s == (
        dream.interface_bandwidth_bits_per_s / 8.0)
    # 16 channels x 2 pseudochannels x 34 DQ = 1088 links from the
    # reference model's own dq_count derivation.
    org = dream.organization
    assert dream.interface_num_links == (
        org["channels"] * org["pseudochannels"] * 34)


def test_effective_bandwidth_is_min_of_stages(dream) -> None:
    assert dream.effective_bandwidth_bytes_per_s == min(
        dream.internal_bandwidth_bytes_per_s,
        dream.interface_bandwidth_bytes_per_s)


def test_bottleneck_identity(dream) -> None:
    assert dream.ratio_internal_over_interface == pytest.approx(1.0)
    assert dream.bottleneck == "BALANCED"
    assert classify_bottleneck(2.0) == "EXTERNAL_INTERFACE"
    assert classify_bottleneck(0.5) == "INTERNAL_MEMORY"
    assert classify_bottleneck(1.0) == "BALANCED"
    with pytest.raises(ValueError):
        classify_bottleneck(0.0)


def test_gates_all_pass_for_pinned_baseline(dream) -> None:
    assert dream.latency_gate == "PASS"
    assert dream.internal_bandwidth_gate == "PASS"
    assert dream.interface_bandwidth_gate == "PASS"


def test_no_m3d_parameter_leakage_into_dream_audit() -> None:
    tree = ast.parse(inspect.getsource(dream_module))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    offenders = [name for name in imported if "m3d" in name.lower()]
    assert not offenders, f"M3D modules imported by Dream audit: {offenders}"
    signature = inspect.signature(
        dream_module.audit_dream_reference_service)
    for parameter in signature.parameters.values():
        assert "m3d" not in parameter.name.lower()


def test_no_hardcoded_m3d_comparison_numbers_in_dream_audit() -> None:
    source = inspect.getsource(dream_module)
    for token in ("4.9e12", "4900000000000", "39.2", "12.1", "15.3",
                  "15.4", "coil", "slab"):
        assert token not in source.lower() or token in {
            "slab"}, f"unexpected M3D token {token!r} in Dream audit"
    assert "coil" not in source.lower()


def test_m3d_coil_derivation_unchanged(m3d) -> None:
    _, _, layout, closure = m3d
    expected_bits = 98 * 50 * 8.0 * 1e9
    assert closure.num_m3d_dies == layout.slab_count == 98
    assert closure.coil_links_per_die == 50
    assert closure.coil_data_rate_gbps_per_link == 8.0
    assert closure.coil_bandwidth_bits_per_s == expected_bits == 39.2e12
    assert closure.coil_bandwidth_bytes_per_s == expected_bits / 8


def test_m3d_internal_bandwidth_regression_unchanged(m3d) -> None:
    _, _, _, closure = m3d
    assert closure.total_parallel_service_units == 98 * 50
    assert closure.clusters_per_service == 4
    assert closure.subarrays_per_service == 256
    assert closure.delivered_bits_per_service == 256
    assert closure.read_payload_bytes_per_service == 32
    assert closure.internal_bandwidth_fast_bytes_per_s >= (
        closure.internal_bandwidth_average_bytes_per_s)
    assert closure.internal_bandwidth_average_bytes_per_s >= (
        closure.internal_bandwidth_slow_bytes_per_s) > 0


def test_m3d_physical_latency_regression_unchanged(m3d) -> None:
    case, latency, _, _ = m3d
    mat_ns = case.architecture.physical_access_latency.mat_latency_ns
    assert latency.min_total_latency_ns >= mat_ns
    assert latency.max_total_latency_ns > latency.min_total_latency_ns
    # Spatial spread remains within the documented first-order regime.
    assert latency.max_total_latency_ns < 2.0 * mat_ns
    assert latency.latency_spread_ns == pytest.approx(
        latency.max_total_latency_ns - latency.min_total_latency_ns)
