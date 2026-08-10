"""Targeted tests for the config-driven Memory Power v0 framework."""

from pathlib import Path

import pytest

from om3dthermal.power import calculate_memory_power, load_power_config
from om3dthermal.power.backends import OperationTableCellModel
from om3dthermal.power.cell_model import (
    MissingCellReplacementError,
    ONE_T_ONE_C_SPECIFIC,
    REUSABLE_STRUCTURE,
)
from om3dthermal.power.config import MemoryPowerConfig, RowPolicy


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


def test_igzo_nominal_fails_at_unvalidated_replacement_boundary():
    config = load_power_config(POWER_CONFIGS / "orthogonal_m3d_igzo.yaml")
    with pytest.raises(
            MissingCellReplacementError,
            match="IGZO cell energy exists but has not been mapped"):
        calculate_memory_power(config, project_root=ROOT)
