"""Latency decomposition audit tests.

Verifies decomposition closure and provenance for both architectures,
canonical-parameter immutability, sensitivity determinism, and
regression guards that the bandwidth model and placement-relevant
latency map remain unchanged by the audit.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from om3dthermal.power import (
    audit_dream_latency_decomposition,
    audit_dream_reference_service,
    build_m3d_latency_decomposition,
    build_risk_ranking,
    build_unified_taxonomy,
    calculate_memory_power,
    calculate_physical_access_latency,
    calculate_physical_capacity_layout,
    classify_gates,
    derive_architecture_bandwidth,
    load_case_config,
    resolve_case_geometry,
    run_feol_resistance_sensitivity,
)
from om3dthermal.power.feol_route import calculate_feol_route
from om3dthermal.power.m3d_subarray import calculate_m3d_subarray


ROOT = Path(__file__).parents[1]
CASE = ROOT / "configs/cases/orthogonal_m3d_igzo.yaml"


def _build_m3d_pipeline():
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
    return case, topology, power, latency


@pytest.fixture(scope="module")
def dream():
    return audit_dream_latency_decomposition(ROOT)


@pytest.fixture(scope="module")
def m3d_pipeline():
    return _build_m3d_pipeline()


@pytest.fixture(scope="module")
def m3d(m3d_pipeline):
    case, _, _, latency = m3d_pipeline
    return build_m3d_latency_decomposition(case, latency)


def test_dream_first_access_closure(dream) -> None:
    assert dream.first_access_ns == pytest.approx(
        dream.trp_ns + dream.trcd_ns + dream.tcl_ns)
    assert dream.trcd_ns == pytest.approx(
        dream.trcd_signal_ns + dream.trcd_sensing_ns)
    assert dream.tcl_ns == pytest.approx(
        dream.tcl_lateral_bus_ns + dream.tcl_tsv_vertical_ns
        + dream.tcl_dq_window_ns)
    assert dream.trcd_signal_ns == pytest.approx(
        dream.inputs["reference_trcd_signal_ns"])
    assert dream.tcl_reference_calibration_ns == pytest.approx(38.4)
    assert dream.tcl_reference_core_tck_ns == pytest.approx(2.0)


def test_dream_trp_aliasing_is_explicit(dream) -> None:
    assert dream.trp_aliased_to_trcd is True
    assert dream.trp_ns == pytest.approx(dream.trcd_ns)
    assert "tRP ~= tRCD" in dream.formulas["tRP"]


def test_dream_access_case_and_row_state_semantics(dream) -> None:
    assert dream.access_case == "ROW_CONFLICT_WORST_CASE_MODEL_DEFAULT"
    assert dream.row_hit_ns == pytest.approx(dream.tcl_ns)
    assert dream.row_miss_ns == pytest.approx(dream.trcd_ns + dream.tcl_ns)
    assert dream.row_conflict_ns == pytest.approx(dream.first_access_ns)
    assert dream.row_hit_ns < dream.row_miss_ns < dream.row_conflict_ns
    assert dream.row_state_source == (
        "DERIVED_FROM_REFERENCE_COMPONENT_SEMANTICS")


def test_dream_decomposition_provenance(dream) -> None:
    assert dream.classification == "DERIVED_FROM_PAPER"
    assert dream.tcl_physical_scope.startswith(
        "LATERAL_BGBUS_GBUS_BASE_TRANSPORT")
    # tCL is dominated by the lateral bus term calibrated from the
    # standard HBM3 tCL reference, not by array physics.
    assert dream.tcl_lateral_bus_ns > dream.tcl_tsv_vertical_ns
    assert dream.tcl_lateral_bus_ns > dream.tcl_dq_window_ns


def test_unified_taxonomy_structure() -> None:
    taxonomy = build_unified_taxonomy()
    assert len(taxonomy) == 8
    by_stage = {row.stage: row for row in taxonomy}
    assert by_stage["E_VERTICAL_INTERCONNECT"].comparable is True
    assert by_stage["F_LATERAL_GLOBAL_ROUTING"].comparable is True
    assert by_stage["A_PRECHARGE_RESET"].m3d_status == "NOT_MODELED"
    assert by_stage["G_MEMORY_SIDE_INTERFACE_STARTUP"].m3d_status == (
        "NOT_MODELED")
    assert by_stage["C_SENSING_MAT_READ"].m3d_status == (
        "INCLUDED_INSIDE_ANOTHER_TERM")


def test_m3d_near_far_closure(m3d, m3d_pipeline) -> None:
    _, _, _, latency = m3d_pipeline
    assert m3d.near_total_ns == pytest.approx(latency.min_total_latency_ns)
    assert m3d.far_total_ns == pytest.approx(latency.max_total_latency_ns)
    assert m3d.near_total_ns == pytest.approx(
        m3d.mat_latency_ns + m3d.miv_min_ns + m3d.feol_min_ns
        + m3d.interface_latency_ns)
    assert m3d.far_total_ns == pytest.approx(
        m3d.mat_latency_ns + m3d.miv_max_ns + m3d.feol_max_ns
        + m3d.interface_latency_ns)
    shares = (
        m3d.far_mat_share + m3d.far_feol_share + m3d.miv_share_of_far_total)
    assert shares == pytest.approx(1.0)


def test_m3d_mat_is_dominant_lumped_placeholder(m3d) -> None:
    assert m3d.mat_latency_ns == pytest.approx(10.0)
    assert "LUMPED_PLACEHOLDER" in m3d.mat_scope
    assert m3d.near_mat_share > 0.9
    assert m3d.far_mat_share > m3d.far_feol_share
    assert m3d.precharge_status.startswith("NOT_MODELED")
    assert m3d.sensing_status.startswith("NOT_EXPLICITLY_MODELED")
    assert m3d.interface_latency_ns == 0.0
    assert "POSITION_INDEPENDENT" in m3d.interface_status_note


def test_miv_contribution_is_negligible(m3d) -> None:
    assert m3d.miv_max_ns < 0.02
    assert m3d.miv_share_of_far_total < 0.001


def test_feol_sensitivity_deterministic_and_ordering_robust(
        m3d_pipeline) -> None:
    case, topology, power, _ = m3d_pipeline
    miv_delays = power.diagnostics["miv_delay_per_layer_ns"]
    first = run_feol_resistance_sensitivity(case, topology, miv_delays)
    second = run_feol_resistance_sensitivity(case, topology, miv_delays)
    assert first == second
    assert [row.resistance_ohm_per_um for row in first] == [1.0, 2.0, 4.0]
    assert all(row.argmax_cluster_unchanged for row in first)
    totals_max = [row.total_max_ns for row in first]
    assert totals_max == sorted(totals_max)
    # Canonical R' = 2 ohm/um row reproduces the canonical latency range.
    canonical = first[1]
    _, _, _, latency = _build_m3d_pipeline()
    assert canonical.total_min_ns == pytest.approx(
        latency.min_total_latency_ns)
    assert canonical.total_max_ns == pytest.approx(
        latency.max_total_latency_ns)


def test_canonical_parameters_not_mutated(m3d_pipeline) -> None:
    case, topology, power, latency_before = m3d_pipeline
    spec = case.architecture.physical_access_latency
    wire = case.architecture.feol_route.wire
    vertical = case.architecture.vertical
    assert spec.mat_latency_ns == pytest.approx(10.0)
    assert spec.mat_status == "NOT_CAPABILITY_VALIDATED"
    assert spec.interface_latency_ns == 0.0
    assert wire.resistance_ohm_per_um == pytest.approx(2.0)
    assert vertical.miv_resistance_ohm_per_um == pytest.approx(10.0)
    # Running the sensitivity must not mutate the parsed configuration.
    run_feol_resistance_sensitivity(
        case, topology, power.diagnostics["miv_delay_per_layer_ns"])
    assert case.architecture.feol_route.wire.resistance_ohm_per_um == (
        pytest.approx(2.0))
    _, _, _, latency_after = _build_m3d_pipeline()
    assert latency_after == latency_before


def test_gates_and_risk_ranking(dream, m3d) -> None:
    gates = classify_gates(dream, m3d)
    assert gates.dream_latency_decomposition_gate == "PASS"
    assert gates.m3d_latency_decomposition_gate == "PARTIAL"
    assert gates.latency_semantic_match_gate == "PARTIALLY_MATCHED"
    assert gates.m3d_absolute_latency_confidence == "LOW"
    assert gates.m3d_spatial_latency_ranking_confidence == "HIGH"
    assert "CURRENT_M3D_ABSOLUTE_LATENCY_NOT_YET_VALIDATED" in (
        gates.reasons["absolute_confidence"])
    risks = build_risk_ranking(dream, m3d)
    assert [item.rank for item in risks] == [1, 2, 3, 4, 5]
    assert risks[0].impact == "HIGH"
    assert "tMAT" in risks[0].item
    assert risks[-1].impact == "LOW"


def test_bandwidth_model_unchanged(m3d_pipeline) -> None:
    case, topology, _, latency = m3d_pipeline
    power = calculate_memory_power(
        case, project_root=ROOT,
        geometry=resolve_case_geometry(case))
    layout = calculate_physical_capacity_layout(
        topology,
        latency,
        slab_count=resolve_case_geometry(case).memory_region_count,
        expected_total_bits=power.diagnostics["total_stored_bits"],
    )
    closure = derive_architecture_bandwidth(
        case.architecture.memory_service,
        layout,
        topology,
        feol_io_channels=case.architecture.feol_route.io_channels,
    )
    assert closure.coil_bandwidth_bits_per_s == 98 * 50 * 8.0 * 1e9
    assert closure.total_parallel_service_units == 98 * 50
    dream_service = audit_dream_reference_service(ROOT)
    assert dream_service.bottleneck == "BALANCED"
    assert dream_service.ratio_internal_over_interface == pytest.approx(1.0)
