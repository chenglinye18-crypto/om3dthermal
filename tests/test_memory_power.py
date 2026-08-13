"""Targeted tests for the config-driven Memory Power v0 framework."""

from pathlib import Path

import pytest
import yaml

from om3dthermal.power import calculate_memory_power, load_power_config
from om3dthermal.power.backends import DreamRAMBackend, OperationTableCellModel
from om3dthermal.power.cell_model import (
    MissingCellReplacementError,
    ONE_T_ONE_C_SPECIFIC,
    REUSABLE_STRUCTURE,
)
from om3dthermal.power.config import MemoryPowerConfig, RowPolicy, TransportInput
from om3dthermal.power.geometry import evaluate_geometry_fit
from om3dthermal.power.miv import build_miv_topology


ROOT = Path(__file__).parents[1]
POWER_CONFIGS = ROOT / "configs" / "power"


def _with_rd_per_act(config, value: int):
    workload = config.workload.model_copy(
        update={"row_policy": RowPolicy(rd_per_act=value)})
    return config.model_copy(update={"workload": workload})


def _with_component_replacement(
        config, *, required=("bl-act",), replacements=None):
    raw = config.model_dump()
    raw["memory"]["cell_model"] = {
        "type": "component_replacement",
        "replacement": {
            "mapping_status": "validated",
            "components": list(required),
            "component_energy_pj_per_bit": replacements or {},
        },
    }
    return MemoryPowerConfig.model_validate(raw)


def _without_miv(config):
    architecture = config.architecture.model_copy(update={
        "vertical": TransportInput(type="none", source="none")})
    return config.model_copy(update={"architecture": architecture})


@pytest.fixture(scope="module")
def conventional():
    return load_power_config(POWER_CONFIGS / "hbm3_si.yaml")


def test_all_four_configs_parse():
    for name in (
        "hbm3_si.yaml",
        "hbm3_si_logic_remove.yaml",
        "orthogonal_si.yaml",
        "orthogonal_m3d_igzo.yaml",
    ):
        load_power_config(POWER_CONFIGS / name)


def test_igzo_cell_geometry_parses_and_closes():
    config = load_power_config(POWER_CONFIGS / "orthogonal_m3d_igzo.yaml")
    geometry = config.memory.cell_model.geometry
    assert geometry is not None
    assert geometry.cell_area_um2 == pytest.approx(0.023)
    assert geometry.pitch_x_um == pytest.approx(0.15166)
    assert geometry.pitch_y_um == pytest.approx(0.15166)
    assert geometry.aspect_ratio == pytest.approx(1.0)
    assert geometry.pitch_x_um * geometry.pitch_y_um == pytest.approx(
        geometry.cell_area_um2, rel=1e-4)
    assert geometry.pitch_x_um / geometry.pitch_y_um == pytest.approx(
        geometry.aspect_ratio, rel=1e-9)
    assert geometry.provenance.model_dump() == {
        "cell_area_um2": "PAPER_REPORTED",
        "pitch_x_um": "DERIVED_FROM_REFERENCE",
        "pitch_y_um": "DERIVED_FROM_REFERENCE",
        "aspect_ratio": "MODELING_CHOICE",
    }


@pytest.mark.parametrize("mutation, message", [
    ({"pitch_x_um": 0.0}, "greater than 0"),
    ({"pitch_x_um": 0.2}, "cell geometry area does not close"),
    ({"aspect_ratio": 2.0}, "cell geometry aspect ratio does not close"),
])
def test_invalid_igzo_cell_geometry_fails_loudly(mutation, message):
    config = load_power_config(POWER_CONFIGS / "orthogonal_m3d_igzo.yaml")
    raw = config.model_dump()
    raw["memory"]["cell_model"]["geometry"].update(mutation)
    with pytest.raises(ValueError, match=message):
        MemoryPowerConfig.model_validate(raw)


def test_igzo_geometry_does_not_change_operation_energy_or_hold():
    config = load_power_config(POWER_CONFIGS / "orthogonal_m3d_igzo.yaml")
    cell_model = config.memory.cell_model
    energy = cell_model.operations
    assert energy is not None
    assert (
        energy.read_0_pj_per_bit, energy.read_1_pj_per_bit,
    ) == pytest.approx((0.00060, 0.36800))
    assert (
        energy.write_00_pj_per_bit, energy.write_01_pj_per_bit,
        energy.write_10_pj_per_bit, energy.write_11_pj_per_bit,
    ) == pytest.approx((0.00030, 0.00037, 0.00058, 0.00024))
    assert (
        energy.refresh_0_pj_per_bit, energy.refresh_1_pj_per_bit,
    ) == pytest.approx((0.00090, 0.37000))
    assert cell_model.background is not None
    assert cell_model.background.type == "per_row"
    assert cell_model.background.value_w == pytest.approx(4.26e-15)


def test_igzo_table_i_operation_values_and_provenance_are_frozen():
    config = load_power_config(POWER_CONFIGS / "orthogonal_m3d_igzo.yaml")
    cell_model = config.memory.cell_model
    operations = cell_model.operations
    provenance = cell_model.operation_energy_provenance
    replacement = cell_model.replacement
    assert cell_model.source == "IEDM2026_HaotongZhu_V5"
    assert operations is not None
    assert operations.model_dump() == pytest.approx({
        "read_0_pj_per_bit": 0.00060,
        "read_1_pj_per_bit": 0.36800,
        "write_00_pj_per_bit": 0.00030,
        "write_01_pj_per_bit": 0.00037,
        "write_10_pj_per_bit": 0.00058,
        "write_11_pj_per_bit": 0.00024,
        "refresh_0_pj_per_bit": 0.00090,
        "refresh_1_pj_per_bit": 0.37000,
    })
    assert provenance is not None
    assert provenance.source == "IEDM2026_HaotongZhu_V5"
    assert provenance.accounting_level == (
        "SPICE_EXTRACTED_MAT_LOCAL_OPERATION_ENERGY")
    assert provenance.sensing_included is True
    assert provenance.distributed_rc_included is True
    assert replacement is not None
    assert replacement.mapping_status == "validated"
    assert replacement.energy_source == "operation_table"
    assert replacement.components == ("bl-act", "bl-pre")
    assert replacement.component_energy_pj_per_bit == {}


def test_igzo_pitch_maps_to_independent_dreamram_tech():
    config = load_power_config(POWER_CONFIGS / "orthogonal_m3d_igzo.yaml")
    result = DreamRAMBackend(ROOT).calculate(config)
    metadata = result.metadata
    assert metadata["cell_pitch_mapping"] == "pitch_x_to_bl__pitch_y_to_wl"
    assert metadata["pitch_bl_um"] == pytest.approx(0.15166)
    assert metadata["pitch_wl_um"] == pytest.approx(0.15166)
    assert metadata["pitch_ldl_um"] == pytest.approx(
        metadata["pitch_wl_um"] * metadata["pitch_ldl_to_wl_min"])
    assert metadata["pitch_mdl_um"] == pytest.approx(
        metadata["pitch_bl_um"] * metadata["pitch_mdl_to_bl_min"])


def test_igzo_pitch_changes_mat_bank_and_reusable_energy(conventional):
    native = DreamRAMBackend(ROOT).calculate(conventional)
    igzo = DreamRAMBackend(ROOT).calculate(
        load_power_config(POWER_CONFIGS / "orthogonal_m3d_igzo.yaml"))
    for dimension in ("mat_x_um", "mat_y_um", "bank_x_um", "bank_y_um"):
        assert igzo.metadata[dimension] != pytest.approx(native.metadata[dimension])
    changed = {
        name for name in ("lwl", "ldl", "mdl")
        if igzo.native_internal_components[name]
        != pytest.approx(native.native_internal_components[name])
    }
    assert changed


def test_existing_geometry_sources_are_resolved_for_all_power_configs():
    expected = {
        "hbm3_si.yaml": (10.8, 10.8, "hbm_dram_die"),
        "hbm3_si_logic_remove.yaml": (10.8, 10.8, "hbm_dram_die"),
        "orthogonal_si.yaml": (22.0, 5.5, "orthogonal_memory_slab"),
        "orthogonal_m3d_igzo.yaml": (
            22.0, 5.5, "orthogonal_m3d_slab"),
    }
    for name, (configured_x, configured_y, region) in expected.items():
        backend = DreamRAMBackend(ROOT).calculate(
            load_power_config(POWER_CONFIGS / name))
        assert backend.metadata["configured_x_mm"] == pytest.approx(configured_x)
        assert backend.metadata["configured_y_mm"] == pytest.approx(configured_y)
        assert backend.metadata["memory_region"] == region
        assert backend.metadata["x_utilization"] == pytest.approx(
            backend.metadata["required_x_mm"] / configured_x)
        assert backend.metadata["y_utilization"] == pytest.approx(
            backend.metadata["required_y_mm"] / configured_y)
        assert backend.metadata["geometry_feasible"] == (
            backend.metadata["required_x_mm"] <= configured_x
            and backend.metadata["required_y_mm"] <= configured_y)


@pytest.mark.parametrize(
    "configured_x, configured_y, expected",
    [(11.0, 12.0, True), (9.0, 12.0, False), (11.0, 9.0, False)],
)
def test_geometry_fit_checks_each_axis(configured_x, configured_y, expected):
    fit = evaluate_geometry_fit(
        configured_x_mm=configured_x,
        configured_y_mm=configured_y,
        required_x_mm=10.0,
        required_y_mm=10.0,
    )
    assert fit.geometry_feasible is expected


def test_dreamram_hbm3_full_row_regression(conventional):
    result = calculate_memory_power(conventional, project_root=ROOT)
    assert result.E_access_total_pj_bit == pytest.approx(0.9782367131)
    assert result.P_access_W == pytest.approx(
        39200 * result.E_access_total_pj_bit * 1e-3)
    assert result.P_logic_background_W is None
    assert result.P_total_W is None
    assert result.diagnostics["rd_per_act"] == 64
    assert result.diagnostics["atoms_per_page"] == 64


def test_dreamram_hbm3_closed_row_regression(conventional):
    result = calculate_memory_power(
        _with_rd_per_act(conventional, 1), project_root=ROOT)
    assert result.E_access_total_pj_bit == pytest.approx(3.0133613062)


def test_dreamram_decomposition_closes(conventional):
    result = calculate_memory_power(conventional, project_root=ROOT)
    reconstructed = (
        result.E_memory_internal_pj_bit + result.E_vertical_pj_bit
        + result.E_base_route_pj_bit + result.E_interface_pj_bit)
    assert reconstructed == pytest.approx(result.E_access_total_pj_bit)


def test_internal_component_partition_closes(conventional):
    result = calculate_memory_power(conventional, project_root=ROOT)
    components = result.diagnostics["native_components_pj_bit"]
    assert set(components) == ONE_T_ONE_C_SPECIFIC | REUSABLE_STRUCTURE
    specific = sum(components[name] for name in ONE_T_ONE_C_SPECIFIC)
    reusable = sum(components[name] for name in REUSABLE_STRUCTURE)
    assert specific + reusable == pytest.approx(
        result.E_memory_internal_pj_bit, abs=1e-15)


def test_rd_per_act_cannot_exceed_atoms_per_page(conventional):
    with pytest.raises(ValueError, match="exceeds atoms_per_page=64"):
        calculate_memory_power(
            _with_rd_per_act(conventional, 65), project_root=ROOT)


def test_orthogonal_si_keeps_same_dreamram_internal_energy(conventional):
    hbm = calculate_memory_power(conventional, project_root=ROOT)
    orthogonal = calculate_memory_power(
        load_power_config(POWER_CONFIGS / "orthogonal_si.yaml"),
        project_root=ROOT)
    assert orthogonal.E_memory_internal_pj_bit == pytest.approx(
        hbm.E_memory_internal_pj_bit, abs=0.0)
    assert orthogonal.E_vertical_pj_bit == 0.0
    assert orthogonal.E_base_route_pj_bit == 0.0
    assert orthogonal.E_interface_pj_bit == pytest.approx(0.5)
    assert orthogonal.E_memory_internal_pj_bit == pytest.approx(0.6288729797)
    assert orthogonal.E_access_total_pj_bit == pytest.approx(1.1288729797)


def test_igzo_then_si_has_no_shared_tech_contamination(conventional):
    before = calculate_memory_power(conventional, project_root=ROOT)
    DreamRAMBackend(ROOT).calculate(
        load_power_config(POWER_CONFIGS / "orthogonal_m3d_igzo.yaml"))
    after = calculate_memory_power(conventional, project_root=ROOT)
    assert after.E_access_total_pj_bit == pytest.approx(
        before.E_access_total_pj_bit, abs=0.0)
    assert after.diagnostics["pitch_bl_um"] == before.diagnostics["pitch_bl_um"]
    assert after.diagnostics["pitch_wl_um"] == before.diagnostics["pitch_wl_um"]


def test_logic_removed_only_drops_dreamram_base_route(conventional):
    baseline = calculate_memory_power(conventional, project_root=ROOT)
    removed = calculate_memory_power(
        load_power_config(POWER_CONFIGS / "hbm3_si_logic_remove.yaml"),
        project_root=ROOT)
    assert removed.E_memory_internal_pj_bit == pytest.approx(
        baseline.E_memory_internal_pj_bit, abs=0.0)
    assert removed.E_vertical_pj_bit == pytest.approx(
        baseline.E_vertical_pj_bit, abs=0.0)
    assert removed.E_interface_pj_bit == pytest.approx(
        baseline.E_interface_pj_bit, abs=0.0)
    assert removed.E_base_route_pj_bit == 0.0
    assert removed.P_logic_background_W == 0.0
    assert removed.P_total_W == pytest.approx(removed.P_access_W)


def test_synthetic_replacement_has_no_double_count(conventional):
    native = calculate_memory_power(conventional, project_root=ROOT)
    old_component = native.diagnostics["native_components_pj_bit"]["bl-act"]
    replacement = 0.125
    modified = calculate_memory_power(
        _with_component_replacement(
            conventional, replacements={"bl-act": replacement}),
        project_root=ROOT)
    expected = native.E_memory_internal_pj_bit - old_component + replacement
    assert modified.E_memory_internal_pj_bit == pytest.approx(
        expected, abs=1e-15)
    assert "bl-act" not in modified.diagnostics["native_components_pj_bit"]
    assert modified.diagnostics["replacement_components_pj_bit"] == {
        "bl-act": replacement}


def test_modified_internal_is_architecture_independent(conventional):
    modified = _with_component_replacement(
        conventional, replacements={"bl-act": 0.125})
    orthogonal_architecture = load_power_config(
        POWER_CONFIGS / "orthogonal_si.yaml").architecture
    orthogonal = modified.model_copy(
        update={"architecture": orthogonal_architecture})
    hbm_result = calculate_memory_power(modified, project_root=ROOT)
    orthogonal_result = calculate_memory_power(orthogonal, project_root=ROOT)
    assert orthogonal_result.E_memory_internal_pj_bit == pytest.approx(
        hbm_result.E_memory_internal_pj_bit, abs=0.0)
    assert hbm_result.E_vertical_pj_bit != orthogonal_result.E_vertical_pj_bit
    assert hbm_result.E_base_route_pj_bit != orthogonal_result.E_base_route_pj_bit
    assert hbm_result.E_interface_pj_bit != orthogonal_result.E_interface_pj_bit


def test_required_replacement_missing_fails_loudly(conventional):
    config = _with_component_replacement(
        conventional,
        required=("bl-act", "bl-pre"),
        replacements={"bl-act": 0.125})
    with pytest.raises(
            MissingCellReplacementError,
            match="required replacement components are unresolved: bl-pre"):
        calculate_memory_power(config, project_root=ROOT)


def test_operation_table_is_device_energy_not_complete_memory_energy():
    config = load_power_config(POWER_CONFIGS / "orthogonal_m3d_igzo.yaml")
    device = OperationTableCellModel().calculate(config)
    assert device.weighted_read(p0=0.5, p1=0.5) == pytest.approx(0.1843)
    assert device.weighted_write(
        p00=0.25, p01=0.25, p10=0.25, p11=0.25,
    ) == pytest.approx((0.00030 + 0.00037 + 0.00058 + 0.00024) / 4)
    assert device.background_value_W == pytest.approx(4.26e-15)


def test_igzo_operation_primitive_replaces_native_block_without_double_count():
    config = load_power_config(POWER_CONFIGS / "orthogonal_m3d_igzo.yaml")
    backend = DreamRAMBackend(ROOT).calculate(config)
    result = calculate_memory_power(_without_miv(config), project_root=ROOT)
    operation = 0.5 * 0.00060 + 0.5 * 0.36800
    reusable = sum(
        backend.native_internal_components[name]
        for name in REUSABLE_STRUCTURE)
    assert result.E_memory_internal_pj_bit == pytest.approx(
        reusable + operation, abs=1e-15)
    assert set(result.diagnostics["native_components_pj_bit"]) == (
        REUSABLE_STRUCTURE)
    assert result.diagnostics["replacement_components_pj_bit"] == {
        "mat_local_operation": pytest.approx(operation)}
    assert not ({"bl-act", "bl-pre"}
                & set(result.diagnostics["native_components_pj_bit"]))
    assert "current_sense_energy" not in result.diagnostics
    assert "CSA_energy" not in result.diagnostics
    assert "RBL_energy" not in result.diagnostics


def test_m3d_topology_uses_geometry_layers_and_uniform_access():
    config = load_power_config(POWER_CONFIGS / "orthogonal_m3d_igzo.yaml")
    metadata = DreamRAMBackend(ROOT).calculate(config).metadata
    assert config.architecture.layers is None
    assert metadata["m3d_layers"] == 8
    assert metadata["layer_pitch_um"] == pytest.approx(0.288)
    assert metadata["m3d_layers_source"] == (
        "geometry_source.m3d_beol.bitcell_layers")
    assert metadata["layer_pitch_source"] == (
        "geometry_source.m3d_beol.bitcell_layer_pitch_nm")
    assert metadata["layer_access_assumption"] == "uniform"
    assert metadata["miv_length_per_layer_um"] == pytest.approx(
        tuple(0.288 * index for index in range(1, 9)))
    assert metadata["miv_average_length_um"] == pytest.approx(1.296)
    assert metadata["m3d_layers_independent_of_dies_stacked"] is True
    assert metadata["dies_stacked"] == 8


def test_miv_length_changes_with_layers_and_pitch_not_dies_stacked():
    common = {
        "data_width_before_vertical": 17,
        "vertical_serialization_factor": "unresolved",
        "row_miv_count": 24,
        "col_miv_count": 19,
    }
    four_layers = build_miv_topology(
        m3d_layers=4, layer_pitch_um=0.288, **common)
    eight_layers = build_miv_topology(
        m3d_layers=8, layer_pitch_um=0.288, **common)
    wider_pitch = build_miv_topology(
        m3d_layers=8, layer_pitch_um=0.4, **common)
    assert four_layers.miv_average_length_um == pytest.approx(0.72)
    assert eight_layers.miv_average_length_um == pytest.approx(1.296)
    assert wider_pitch.miv_average_length_um == pytest.approx(1.8)
    # No dies_stacked argument exists: physical die count cannot enter MIV
    # length, and 4-layer average is not DreamRAM dies_stacked/2 * pitch.
    assert four_layers.miv_average_length_um != pytest.approx(4 * 0.288)


@pytest.mark.parametrize(
    "layers,pitch_nm,expected_average_um",
    [(4, 288.0, 0.72), (8, 400.0, 1.8)],
)
def test_miv_adapter_tracks_existing_geometry_config(
        tmp_path, layers, pitch_nm, expected_average_um):
    source_path = ROOT / "configs" / "orthogonal_m3d_edram_v0.yaml"
    raw_geometry = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    stack_um = layers * pitch_nm * 1e-3
    raw_geometry["m3d_beol"].update({
        "bitcell_layers": layers,
        "bitcell_layer_pitch_nm": pitch_nm,
        "bitcell_stack_um": stack_um,
        "total_um": stack_um + raw_geometry["m3d_beol"]["interconnect_um"],
    })
    raw_geometry["m3d_memory"]["layers"] = layers
    raw_geometry["slab"]["si_substrate_um"] = (
        raw_geometry["slab"]["total_pitch_um"]
        - raw_geometry["slab"]["feol_um"]
        - stack_um
        - raw_geometry["m3d_beol"]["interconnect_um"]
        - raw_geometry["orthogonal"]["daa_um"])
    geometry_path = tmp_path / "m3d_geometry.yaml"
    geometry_path.write_text(
        yaml.safe_dump(raw_geometry, sort_keys=False), encoding="utf-8")

    config = load_power_config(POWER_CONFIGS / "orthogonal_m3d_igzo.yaml")
    geometry_source = config.architecture.geometry_source.model_copy(
        update={"config": geometry_path})
    architecture = config.architecture.model_copy(
        update={"geometry_source": geometry_source})
    modified_config = config.model_copy(update={"architecture": architecture})
    metadata = DreamRAMBackend(ROOT).calculate(modified_config).metadata
    baseline = DreamRAMBackend(ROOT).calculate(config).metadata
    assert metadata["m3d_layers"] == layers
    assert metadata["layer_pitch_um"] == pytest.approx(pitch_nm * 1e-3)
    assert metadata["miv_average_length_um"] == pytest.approx(
        expected_average_um)
    assert metadata["dies_stacked"] == 8
    expected_segments = (layers + 1) / 2
    assert metadata["miv_average_segments"] == pytest.approx(expected_segments)
    if layers == 8:
        # Physical pitch changes, but pF/segment energy does not.
        assert metadata["miv_access_energy_pJ_per_bit"] == pytest.approx(
            baseline["miv_access_energy_pJ_per_bit"], abs=0.0)
    else:
        assert metadata["miv_access_energy_pJ_per_bit"] == pytest.approx(
            baseline["miv_access_energy_pJ_per_bit"]
            * expected_segments / baseline["miv_average_segments"])


def test_miv_count_uses_dreamram_tsv_equivalent_serialization():
    config = load_power_config(POWER_CONFIGS / "orthogonal_m3d_igzo.yaml")
    metadata = DreamRAMBackend(ROOT).calculate(config).metadata
    assert metadata["data_width_before_vertical"] == 272
    assert metadata["vertical_serialization_factor"] == 4
    assert metadata["vertical_serialization_status"] == "resolved"
    assert metadata["miv_serialization_factor"] == 4
    assert metadata["miv_serialization_source"] == (
        "DREAMRAM_TSV_EQUIVALENT")
    assert metadata["active_data_miv_count"] == 68
    assert metadata["row_miv_count"] == 23
    assert metadata["col_miv_count"] == 17
    resolved = build_miv_topology(
        m3d_layers=8,
        layer_pitch_um=0.288,
        data_width_before_vertical=metadata["data_width_before_vertical"],
        vertical_serialization_factor=5,
        row_miv_count=metadata["row_miv_count"],
        col_miv_count=metadata["col_miv_count"],
    )
    assert resolved.active_data_miv_count == 55
    assert resolved.active_data_miv_count == (
        metadata["data_width_before_vertical"] + 5 - 1) // 5


def test_m3d_path_excludes_hbm_vertical_base_dq_and_tsv_area():
    config = load_power_config(POWER_CONFIGS / "orthogonal_m3d_igzo.yaml")
    metadata = DreamRAMBackend(ROOT).calculate(config).metadata
    assert metadata["vertical_interconnect_type"] == "MIV"
    assert metadata["miv_connection_model"] == (
        "per_layer_local_selection_to_shared_vertical")
    assert metadata["direct_bitline_to_feol"] is False
    assert metadata["miv_components"] == (
        "row-miv", "col-miv", "data-miv")
    assert metadata["tsv_energy_included"] is False
    assert metadata["base_route_included"] is False
    assert metadata["dq_included"] is False
    assert metadata["miv_dedicated_koz_area_modeled"] is False
    assert metadata["miv_planar_footprint_basis"] == (
        "bankdie_without_tsv_koz_bands")
    assert set(metadata["excluded_hbm_components"]) == {
        "row-tsv", "col-tsv", "tsv",
        "row-base", "col-base", "base",
        "row-dq", "col-dq", "dq",
    }


def test_tsv_equivalent_miv_energy_resolves_and_access_closes():
    config = load_power_config(POWER_CONFIGS / "orthogonal_m3d_igzo.yaml")
    metadata = DreamRAMBackend(ROOT).calculate(config).metadata
    assert metadata["miv_electrical_model"] == "TSV_EQUIVALENT_BASELINE"
    assert metadata["miv_modeling_class"] == "MODELING_CHOICE"
    assert metadata["miv_capacitance_status"] == "resolved"
    assert metadata["miv_energy_status"] == "resolved"
    assert metadata["miv_capacitance_per_segment_pF"] == pytest.approx(0.78)
    assert metadata["miv_capacitance_source"] == (
        "DREAMRAM_TSV_EQUIVALENT")
    assert metadata["miv_capacitance_physical_interpretation"] == (
        "effective_capacitance_per_vertical_segment")
    assert metadata["miv_segments_per_layer"] == tuple(range(1, 9))
    assert metadata["miv_average_segments"] == pytest.approx(4.5)
    assert metadata["row_miv_energy_pJ"] == pytest.approx(48.84165)
    assert metadata["col_miv_energy_pJ"] == pytest.approx(36.10035)
    assert metadata["data_miv_energy_pJ"] == pytest.approx(26.2548)
    assert metadata["miv_total_energy_pJ"] == pytest.approx(111.1968)
    assert metadata["row_miv_voltage_source"] == (
        "DREAMRAM_TSV_EQUIVALENT")
    assert metadata["col_miv_voltage_source"] == (
        "DREAMRAM_TSV_EQUIVALENT")
    assert metadata["data_miv_voltage_source"] == (
        "DREAMRAM_TSV_EQUIVALENT")

    result = calculate_memory_power(config, project_root=ROOT)
    mat_local = result.diagnostics["replacement_components_pj_bit"][
        "mat_local_operation"]
    reusable = sum(result.diagnostics["native_components_pj_bit"].values())
    assert mat_local == pytest.approx(0.1843)
    assert reusable == pytest.approx(1.6002246445189394)
    assert result.E_vertical_pj_bit == pytest.approx(0.24804639129638675)
    assert result.E_base_route_pj_bit == 0.0
    assert result.E_interface_pj_bit == pytest.approx(0.5)
    assert result.E_access_total_pj_bit == pytest.approx(
        mat_local + reusable + result.E_vertical_pj_bit
        + result.E_interface_pj_bit)
