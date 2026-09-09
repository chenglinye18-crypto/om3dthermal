"""Targeted tests for the config-driven Memory Power v0 framework."""

from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from om3dthermal.architecture_comparison import (
    _resolve_case_power_operating_point_kwargs,
)
from om3dthermal.power import (
    calculate_memory_power,
    load_case_config,
    map_system_power_to_thermal,
    resolve_case_geometry,
    resolve_system_power,
)
from om3dthermal.power.config import (
    RowPolicy,
)
from om3dthermal.power.si_packing import pack_si_primitive


ROOT = Path(__file__).parents[1]
CASE_CONFIGS = ROOT / "configs" / "cases"


def _with_row_utilization(config, value: float):
    workload = config.workload.model_copy(
        update={"row_policy": RowPolicy(
            activated_row_data_utilization=value)})
    return config.model_copy(update={"workload": workload})


def test_active_cases_parse_and_resolve_system_power():
    hbm_case = load_case_config(
        CASE_CONFIGS / "conventional_hbm_2x1.yaml")
    hbm_geometry = resolve_case_geometry(hbm_case)
    hbm_system = resolve_system_power(
        hbm_case, project_root=ROOT, geometry=hbm_geometry,
        **_resolve_case_power_operating_point_kwargs(hbm_case, ROOT))
    assert hbm_system.gpu_power_W == 522.512
    assert hbm_system.memory_result is not None
    hbm = hbm_system.memory_result
    assert hbm.diagnostics["activated_row_data_utilization"] == 0.10
    assert hbm.diagnostics["effective_rd_per_act"] == 6.4
    assert hbm.P_refresh_W == pytest.approx(0.9614665609424703)
    assert hbm.diagnostics["total_stored_bits"] == 1159641169920  # rev v2: 135 GiB
    assert hbm.E_base_route_pj_bit == pytest.approx(
        0.15710373866310487, abs=0.0)
    assert hbm.E_vertical_pj_bit > 0.0
    assert hbm.diagnostics["dies_stacked"] == 8
    assert hbm.diagnostics["physical_stack_count"] == 4
    assert hbm.diagnostics["dram_dies_per_stack"] == 12
    # Rev v2: 12.2x11.8 mm HBM3E-class die packs 180 banks (was 144).
    assert hbm.diagnostics["packed_banks_per_die"] == 180
    assert hbm.E_memory_internal_pj_bit == pytest.approx(
        1.2392882489481523, abs=0.0)
    assert hbm.E_vertical_pj_bit == pytest.approx(
        0.5490276175199488, abs=0.0)
    assert hbm.E_interface_pj_bit == pytest.approx(
        0.05008039486879381, abs=0.0)
    assert hbm.E_access_total_pj_bit == pytest.approx(
        (0.978 + 3.013) / 2, abs=0.0)
    assert hbm.diagnostics["hbm_read_energy_status"] == (
        "FROZEN_ROW_STATE_ARITHMETIC_MEAN")
    assert hbm.diagnostics["geometry_feasible"] is True

    m3d_case = load_case_config(CASE_CONFIGS / "orthogonal_m3d_igzo.yaml")
    m3d_geometry = resolve_case_geometry(m3d_case)
    m3d = calculate_memory_power(m3d_case, read_bandwidth_gbps=m3d_case.workload.read_bandwidth_gbps, project_root=ROOT, geometry=m3d_geometry)
    assert m3d.E_access_total_pj_bit == 0.8552605756733209
    assert m3d.E_vertical_pj_bit == 0.002445862111816407
    assert m3d.E_feol_route_pj_bit == 0.16705631334524151
    assert m3d.E_interface_pj_bit == 0.5
    assert m3d.P_refresh_W == pytest.approx(
        318 * 0.0003484694872064)
    assert m3d.diagnostics["geometry_source_config"] == (
        "canonical_case:orthogonal_m3d_igzo")


def test_active_case_system_mapping_uses_resolved_power():
    for name in (
        "conventional_hbm_2x1.yaml",
        "orthogonal_si.yaml",
        "orthogonal_m3d_igzo.yaml",
    ):
        case = load_case_config(CASE_CONFIGS / name)
        geometry = resolve_case_geometry(case)
        system = resolve_system_power(
            case, project_root=ROOT, geometry=geometry,
            **_resolve_case_power_operating_point_kwargs(case, ROOT))
        mapping = map_system_power_to_thermal(case, system)
        assert mapping.unresolved is False
        assert mapping.total_mapped_power_W == pytest.approx(
            522.512 + system.resolved_total_memory_power_W)


def test_orthogonal_si_uses_matched_row_workload_and_refresh():
    case = load_case_config(CASE_CONFIGS / "orthogonal_si.yaml")
    geometry = resolve_case_geometry(case)
    result = calculate_memory_power(case, read_bandwidth_gbps=case.workload.read_bandwidth_gbps, project_root=ROOT, geometry=geometry)
    assert result.diagnostics["activated_row_data_utilization"] == 0.10
    assert result.diagnostics["effective_rd_per_act"] == 6.4
    assert result.E_vertical_pj_bit == 0.0
    assert result.E_base_route_pj_bit == 0.0
    assert result.E_interface_pj_bit == 0.5
    assert result.P_refresh_W > 0.11395159240799647
    assert result.E_access_total_pj_bit == pytest.approx(
        1.3676557831180527, abs=0.0)


def test_orthogonal_si_capacity_is_integer_geometry_packing():
    case = load_case_config(CASE_CONFIGS / "orthogonal_si.yaml")
    geometry = resolve_case_geometry(case)
    result = calculate_memory_power(case, read_bandwidth_gbps=case.workload.read_bandwidth_gbps, project_root=ROOT, geometry=geometry)
    d = result.diagnostics
    assert d["primitive_type"] == (
        "DREAMRAM_BANK_TILE_WITH_DECODERS_SWD_BLSA")
    assert d["primitive_bits"] == 134217728
    assert isinstance(d["packed_nx"], int)
    assert isinstance(d["packed_ny"], int)
    assert d["primitives_per_slab"] == d["packed_nx"] * d["packed_ny"]
    assert d["bits_per_slab"] == (
        d["primitives_per_slab"] * d["primitive_bits"])
    assert d["total_system_bits"] == d["bits_per_slab"] * 98
    assert d["gib_per_slab"] == pytest.approx(
        d["bits_per_slab"] / 8 / (2 ** 30))
    assert d["total_system_gib"] == pytest.approx(
        d["total_system_bits"] / 8 / (2 ** 30))
    assert d["gross_density_Mb_mm2"] == pytest.approx(
        d["bits_per_slab"] / 1e6 / (22.0 * 5.5))
    assert d["packed_width_um"] <= geometry.configured_x_mm * 1e3
    assert d["packed_height_um"] <= geometry.configured_y_mm * 1e3
    assert d["geometry_feasible"] is True
    assert d["dreamram_reference_geometry_feasible"] is False
    assert d["dreamram_reference_geometry_role"] == "REFERENCE_ONLY"
    assert d["refresh_events_per_full_memory_cycle"] * d[
        "refresh_bits_per_event"] == d["total_system_bits"]


def test_orthogonal_si_slab_dimensions_drive_capacity_and_rotation():
    case = load_case_config(CASE_CONFIGS / "orthogonal_si.yaml")
    baseline_geometry = resolve_case_geometry(case)
    baseline = calculate_memory_power(case, read_bandwidth_gbps=case.workload.read_bandwidth_gbps, project_root=ROOT, geometry=baseline_geometry)

    smaller_orthogonal = case.geometry.orthogonal.model_copy(update={
        "slab_plane_y_mm": 11.0})
    smaller_case = case.model_copy(update={
        "geometry": case.geometry.model_copy(update={
            "orthogonal": smaller_orthogonal})})
    smaller_geometry = resolve_case_geometry(smaller_case)
    smaller = calculate_memory_power(smaller_case, read_bandwidth_gbps=smaller_case.workload.read_bandwidth_gbps, project_root=ROOT, geometry=smaller_geometry)
    assert smaller.diagnostics["bits_per_slab"] < (
        baseline.diagnostics["bits_per_slab"])

    rotated_orthogonal = case.geometry.orthogonal.model_copy(update={
        "slab_plane_y_mm": 5.5, "slab_height_z_mm": 22.0})
    rotated_case = case.model_copy(update={
        "geometry": case.geometry.model_copy(update={
            "orthogonal": rotated_orthogonal})})
    rotated = calculate_memory_power(rotated_case, read_bandwidth_gbps=rotated_case.workload.read_bandwidth_gbps, project_root=ROOT,
        geometry=resolve_case_geometry(rotated_case))
    assert rotated.diagnostics["rotated_90_deg"] is True
    assert rotated.diagnostics["bits_per_slab"] == (
        baseline.diagnostics["bits_per_slab"])


def test_orthogonal_si_refresh_scales_with_packed_capacity():
    case = load_case_config(CASE_CONFIGS / "orthogonal_si.yaml")
    baseline = calculate_memory_power(case, read_bandwidth_gbps=case.workload.read_bandwidth_gbps, project_root=ROOT, geometry=resolve_case_geometry(case))
    half_orthogonal = case.geometry.orthogonal.model_copy(update={
        "slab_count": 49})
    half_case = case.model_copy(update={
        "geometry": case.geometry.model_copy(update={
            "orthogonal": half_orthogonal})})
    half = calculate_memory_power(half_case, read_bandwidth_gbps=half_case.workload.read_bandwidth_gbps, project_root=ROOT, geometry=resolve_case_geometry(half_case))
    assert half.diagnostics["total_system_bits"] == (
        baseline.diagnostics["total_system_bits"] // 2)
    assert half.P_refresh_W == pytest.approx(0.5 * baseline.P_refresh_W)
    assert half.E_access_total_pj_bit == baseline.E_access_total_pj_bit


def test_si_packing_rejects_fractional_or_nonfitting_primitives():
    packed = pack_si_primitive(
        slab_width_um=10.0, slab_height_um=10.0, slab_count=2,
        primitive_type="toy", primitive_width_um=3.0,
        primitive_height_um=4.0, primitive_bits=128)
    assert (packed.packed_nx, packed.packed_ny) == (3, 2)
    assert packed.primitives_per_slab == 6
    with pytest.raises(ValueError, match="no complete"):
        pack_si_primitive(
            slab_width_um=1.0, slab_height_um=1.0, slab_count=1,
            primitive_type="toy", primitive_width_um=2.0,
            primitive_height_um=3.0, primitive_bits=1)


def test_conventional_full_row_same_boundary_remains_stable():
    case = load_case_config(
        CASE_CONFIGS / "conventional_hbm_2x1.yaml")
    full = _with_row_utilization(case, 1.0)
    geometry = resolve_case_geometry(full)
    result = calculate_memory_power(full, read_bandwidth_gbps=full.workload.read_bandwidth_gbps, project_root=ROOT, geometry=geometry)
    assert result.diagnostics["effective_rd_per_act"] == 64.0
    assert result.E_access_total_pj_bit == pytest.approx(1.9955)
    # Refresh is deliberately enabled in the active case; the old split
    # logic-removed power input predated refresh accounting.
    # Rev v2: refresh scales with capacity 116.0 -> 145.0 GB.
    assert result.P_refresh_W == pytest.approx(0.9614665609424703)


def test_conventional_12hi_preserves_frozen_nominal_total():
    case = load_case_config(
        CASE_CONFIGS / "conventional_hbm_2x1.yaml")
    geometry_12hi = resolve_case_geometry(case)
    geometry_8hi = replace(geometry_12hi, memory_dies_per_region=8)
    result_8hi = calculate_memory_power(case, read_bandwidth_gbps=case.workload.read_bandwidth_gbps, project_root=ROOT, geometry=geometry_8hi)
    result_12hi = calculate_memory_power(case, read_bandwidth_gbps=case.workload.read_bandwidth_gbps, project_root=ROOT, geometry=geometry_12hi)

    assert geometry_12hi.memory_dies_per_region == 12
    assert result_12hi.diagnostics["total_stored_bits"] == 1159641169920  # rev v2
    assert result_12hi.P_refresh_W == pytest.approx(0.9614665609424703)  # rev v2
    assert result_12hi.E_access_total_pj_bit == pytest.approx(1.9955)
    assert result_8hi.E_access_total_pj_bit == pytest.approx(1.9955)
    assert result_12hi.E_base_route_pj_bit > 0.0
    diagnostics = result_12hi.diagnostics
    assert diagnostics["electrical_reference_stack_die_count"] == 8
    assert diagnostics["electrical_resolved_stack_die_count"] == 12
    assert diagnostics["reference_average_tsv_layers_crossed"] == 4.0
    assert diagnostics["resolved_average_tsv_layers_crossed"] == 6.0
    assert diagnostics["reference_average_tsv_length_um"] == 120.0
    assert diagnostics["resolved_average_tsv_length_um"] == 180.0
    assert diagnostics["tsv_capacitance_per_crossed_layer_pF"] == 0.78
    assert diagnostics["reference_average_tsv_capacitance_pF"] == 3.12
    assert diagnostics["resolved_average_tsv_capacitance_pF"] == 4.68
    assert diagnostics["tsv_data_serialization_factor"] == 4
    assert diagnostics["active_data_tsv_count_per_command"] == 68
    reference_vertical = sum(diagnostics[
        "vertical_components_reference_pJ_per_bit"].values())
    resolved_vertical = sum(diagnostics[
        "vertical_components_resolved_pJ_per_bit"].values())
    assert resolved_vertical == pytest.approx(1.5 * reference_vertical)


def test_canonical_m3d_has_single_geometry_and_operation_sources():
    path = CASE_CONFIGS / "orthogonal_m3d_igzo.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert "geometry_source" not in raw["architecture"]
    assert "memory_region" not in raw["geometry"]
    assert "total_pitch_um" not in raw["geometry"]["m3d_stack"]
    assert "layers" not in raw["architecture"]
    assert "stored_bits" not in raw["workload"]
    assert "operations" not in raw["geometry"]
    assert "power_models" not in raw["geometry"]
    assert "bitcell_layers" not in raw["memory"]
    assert "bitcell_layer_pitch_nm" not in raw["memory"]
    assert set(raw["memory"]["cell_model"]["operations"]) == {
        "read_0_pj_per_bit", "read_1_pj_per_bit",
        "write_00_pj_per_bit", "write_01_pj_per_bit",
        "write_10_pj_per_bit", "write_11_pj_per_bit",
        "refresh_0_pj_per_bit", "refresh_1_pj_per_bit",
    }


def test_canonical_geometry_drives_capacity_and_miv_without_second_yaml():
    case = load_case_config(CASE_CONFIGS / "orthogonal_m3d_igzo.yaml")
    geometry = resolve_case_geometry(case)
    baseline = calculate_memory_power(case, read_bandwidth_gbps=case.workload.read_bandwidth_gbps, project_root=ROOT, geometry=geometry)
    assert baseline.diagnostics["clusters_per_layer"] == 280
    assert baseline.diagnostics["subarrays_per_layer"] == 17920
    assert baseline.diagnostics["bits_per_layer"] == 4697620480
    assert baseline.diagnostics["total_stored_bits"] == 318 * 37580963840  # rev v3
    assert baseline.diagnostics["placed_width_um"] == pytest.approx(
        21794.548876360117)
    assert baseline.diagnostics["placed_height_um"] == pytest.approx(
        4999.693110067114)

    raw = case.model_dump(mode="json")
    raw["geometry"]["m3d_stack"]["bitcell_layers"] = 16
    raw["geometry"]["m3d_stack"]["si_substrate_um"] = 90.242
    doubled_case = type(case).model_validate(raw)
    doubled_geometry = resolve_case_geometry(doubled_case)
    doubled = calculate_memory_power(doubled_case, read_bandwidth_gbps=doubled_case.workload.read_bandwidth_gbps, project_root=ROOT, geometry=doubled_geometry)
    assert doubled.diagnostics["total_stored_bits"] == (
        2 * baseline.diagnostics["total_stored_bits"])
    assert doubled.diagnostics["miv_average_length_um"] != (
        baseline.diagnostics["miv_average_length_um"])
    assert doubled_case.memory.cell_model == case.memory.cell_model

    raw = case.model_dump(mode="json")
    raw["geometry"]["m3d_stack"]["bitcell_layer_pitch_nm"] = 300.0
    raw["geometry"]["m3d_stack"]["si_substrate_um"] = 92.45
    wider_pitch_case = type(case).model_validate(raw)
    wider_geometry = resolve_case_geometry(wider_pitch_case)
    wider = calculate_memory_power(wider_pitch_case, read_bandwidth_gbps=wider_pitch_case.workload.read_bandwidth_gbps, project_root=ROOT, geometry=wider_geometry)
    assert wider.diagnostics["miv_average_length_um"] != (
        baseline.diagnostics["miv_average_length_um"])
    assert wider.E_vertical_pj_bit != baseline.E_vertical_pj_bit
