"""Targeted 2T0C operation-energy power-accounting tests."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from om3dthermal.config import (
    M3DOperationEnergyPowerConfig,
    load_orthogonal_m3d_template,
)
from om3dthermal.thermal import (
    UnresolvedM3DActivityError,
    calculate_operation_energy_power,
    femtojoules_to_joules,
    resolve_m3d_memory_power,
)


CONFIG = Path(__file__).parents[1] / "configs" / "orthogonal_m3d_edram_v0.yaml"


@pytest.fixture(scope="module")
def template():
    return load_orthogonal_m3d_template(CONFIG)


def _resolved_toy_model(template) -> M3DOperationEnergyPowerConfig:
    raw = template.power_models.operation_energy.model_dump()
    raw["activity"] = {
        "read_bit_rate_per_s": 1000.0,
        "write_bit_rate_per_s": 2000.0,
        "read_state_probability": {"p0": 0.25, "p1": 0.75},
        "write_transition_probability": {
            "p00": 0.10, "p01": 0.20, "p10": 0.30, "p11": 0.40,
        },
        "refresh_period_s": 2.0,
        "refresh_state_probability": {"p0": 0.60, "p1": 0.40},
        "active_rows": 10,
    }
    return M3DOperationEnergyPowerConfig.model_validate(raw)


def test_all_paper_reported_2t0c_energy_values_parse(template):
    model = template.power_models.operation_energy
    energy = model.operation_energy_fJ_per_bit
    assert energy.read_0 == pytest.approx(0.60)
    assert energy.read_1 == pytest.approx(368.0)
    assert energy.write_0_to_0 == pytest.approx(0.30)
    assert energy.write_0_to_1 == pytest.approx(0.37)
    assert energy.write_1_to_0 == pytest.approx(0.58)
    assert energy.write_1_to_1 == pytest.approx(0.24)
    assert energy.refresh_0 == pytest.approx(0.90)
    assert energy.refresh_1 == pytest.approx(370.0)
    assert model.hold_power_W_per_row == pytest.approx(4.26e-15)
    assert model.energy_provenance == "PAPER_REPORTED"


def test_femtojoule_to_joule_conversion():
    assert femtojoules_to_joules(1.0) == pytest.approx(1e-15)
    assert femtojoules_to_joules(368.0) == pytest.approx(368e-15)


def test_toy_activity_matches_manual_dynamic_power(template):
    model = _resolved_toy_model(template)
    result = calculate_operation_energy_power(model, total_memory_bits=1000.0)
    expected_read = 1000.0 * (0.25 * 0.60 + 0.75 * 368.0) * 1e-15
    expected_write = 2000.0 * (
        0.10 * 0.30 + 0.20 * 0.37 + 0.30 * 0.58 + 0.40 * 0.24
    ) * 1e-15
    expected_refresh = (1000.0 / 2.0) * (
        0.60 * 0.90 + 0.40 * 370.0) * 1e-15
    assert result.read_W == pytest.approx(expected_read)
    assert result.write_W == pytest.approx(expected_write)
    assert result.refresh_W == pytest.approx(expected_refresh)
    assert result.dynamic_W == pytest.approx(
        expected_read + expected_write + expected_refresh)
    assert result.memory_total_W == pytest.approx(
        expected_read + expected_write + expected_refresh + result.hold_W)


def test_hold_power_accounting(template):
    model = _resolved_toy_model(template)
    result = calculate_operation_energy_power(model, total_memory_bits=1000.0)
    assert result.hold_W == pytest.approx(10 * 4.26e-15)


@pytest.mark.parametrize("probability_block", [
    ("read_state_probability", {"p0": 0.2, "p1": 0.7}),
    ("refresh_state_probability", {"p0": 0.8, "p1": 0.3}),
    ("write_transition_probability", {
        "p00": 0.1, "p01": 0.2, "p10": 0.3, "p11": 0.5,
    }),
])
def test_probability_sum_not_one_fails_clearly(template, probability_block):
    raw = _resolved_toy_model(template).model_dump()
    field, invalid = probability_block
    raw["activity"][field] = invalid
    with pytest.raises(ValidationError, match="must (equal|sum to) 1"):
        M3DOperationEnergyPowerConfig.model_validate(raw)


def test_nominal_operation_mode_and_legacy_iso_total_are_independent(template):
    nominal = resolve_m3d_memory_power(template)
    assert nominal.mode == "operation_energy"
    assert nominal.memory_total_W == pytest.approx(14.4256)
    assert nominal.per_bitcell_layer_W == pytest.approx(1.8032)
    assert nominal.target_region == "m3d_bitcell_stack"
    iso = resolve_m3d_memory_power(template, mode="iso_total")
    assert iso.mode == "iso_total"
    assert iso.memory_total_W == pytest.approx(156.8)
    assert iso.per_bitcell_layer_W == pytest.approx(19.6)
    assert iso.target_region == "m3d_bitcell_stack"
    with pytest.raises(UnresolvedM3DActivityError) as caught:
        calculate_operation_energy_power(
            template.power_models.operation_energy,
            total_memory_bits=1.0)
    assert "read_bit_rate_per_s" in str(caught.value)
    assert "refresh_period_s" in str(caught.value)
    assert "active_rows" in str(caught.value)


def test_operation_power_mapping_targets_only_bitcell_stack(template):
    distribution = template.power_models.operation_energy.distribution
    assert distribution.target_region == "m3d_bitcell_stack"
    assert distribution.direct_power_regions == ("m3d_bitcell_stack",)
