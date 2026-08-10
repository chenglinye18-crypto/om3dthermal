"""Targeted tests for the config-driven Memory Power v0 framework."""

from pathlib import Path

import pytest

from om3dthermal.power import calculate_memory_power, load_power_config
from om3dthermal.power.config import RowPolicy


ROOT = Path(__file__).parents[1]
POWER_CONFIGS = ROOT / "configs" / "power"


def _with_rd_per_act(config, value: int):
    workload = config.workload.model_copy(
        update={"row_policy": RowPolicy(rd_per_act=value)})
    return config.model_copy(update={"workload": workload})


def _resolved_igzo_config():
    config = load_power_config(POWER_CONFIGS / "orthogonal_m3d_igzo.yaml")
    vertical = config.architecture.vertical.model_copy(
        update={"energy_pj_per_bit": 0.125})
    architecture = config.architecture.model_copy(update={"vertical": vertical})
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


def test_operation_table_read_weighting_and_architecture_transport():
    config = _resolved_igzo_config()
    result = calculate_memory_power(config, project_root=ROOT)
    expected_internal = 0.5 * 0.00060 + 0.5 * 0.36800
    assert result.E_memory_internal_pj_bit == pytest.approx(expected_internal)
    assert result.E_vertical_pj_bit == pytest.approx(0.125)
    assert result.E_base_route_pj_bit == 0.0
    assert result.E_interface_pj_bit == pytest.approx(0.5)
    assert result.E_access_total_pj_bit == pytest.approx(
        expected_internal + 0.125 + 0.5)


def test_missing_miv_energy_fails_loudly():
    config = load_power_config(POWER_CONFIGS / "orthogonal_m3d_igzo.yaml")
    with pytest.raises(ValueError, match="energy_pj_per_bit is unresolved"):
        calculate_memory_power(config, project_root=ROOT)


def test_refresh_and_per_row_background_use_explicit_activity_only():
    config = _resolved_igzo_config()
    raw = config.model_dump()
    raw["memory"]["retention_s"] = 2.0
    raw["workload"].update({
        "stored_bits": 1000.0,
        "active_rows": 10,
        "refresh_data": {"p0": 0.25, "p1": 0.75},
    })
    raw["power"]["refresh"]["enabled"] = True
    raw["power"]["background"]["enabled"] = True
    from om3dthermal.power.config import MemoryPowerConfig
    resolved = MemoryPowerConfig.model_validate(raw)
    result = calculate_memory_power(resolved, project_root=ROOT)
    expected_refresh = (
        1000.0 * (0.25 * 0.00090 + 0.75 * 0.37000) * 1e-12 / 2.0)
    assert result.P_refresh_W == pytest.approx(expected_refresh)
    assert result.P_memory_background_W == pytest.approx(10 * 4.26e-15)
