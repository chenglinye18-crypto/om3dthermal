"""Tests for the per-material-pair interface resistance registry."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from om3dthermal.config import (
    InterfaceResistanceConfig,
    ThermalConductanceConfig,
)
from om3dthermal.thermal.interfaces import InterfaceResistanceRegistry


# ---------------------------------------------------------------------------
# Unit parser
# ---------------------------------------------------------------------------

def test_areal_resistance_accepts_plain_m2K_per_W():
    from om3dthermal.units import parse_areal_thermal_resistance
    assert parse_areal_thermal_resistance(0) == 0.0
    assert parse_areal_thermal_resistance(1e-8) == 1e-8
    assert parse_areal_thermal_resistance(2e-6) == 2e-6


def test_areal_resistance_accepts_unit_strings():
    from om3dthermal.units import parse_areal_thermal_resistance
    assert parse_areal_thermal_resistance("0 m^2*K/W") == 0.0
    assert parse_areal_thermal_resistance("1e-8 m^2*K/W") == 1e-8
    assert parse_areal_thermal_resistance("2 mm^2*K/W") == pytest.approx(
        2e-6)


def test_areal_resistance_rejects_negative_nan_inf():
    from om3dthermal.units import parse_areal_thermal_resistance
    with pytest.raises(ValueError):
        parse_areal_thermal_resistance(-1.0)
    with pytest.raises(ValueError):
        parse_areal_thermal_resistance(float("nan"))
    with pytest.raises(ValueError):
        parse_areal_thermal_resistance(float("inf"))
    with pytest.raises(ValueError):
        parse_areal_thermal_resistance("-1 m^2*K/W")


def test_areal_resistance_rejects_wrong_dimensions():
    from om3dthermal.units import parse_areal_thermal_resistance
    with pytest.raises(ValueError):
        parse_areal_thermal_resistance("1 m")            # length
    with pytest.raises(ValueError):
        parse_areal_thermal_resistance("1 W/(m*K)")      # conductivity
    with pytest.raises(ValueError):
        parse_areal_thermal_resistance("1 s")            # time
    with pytest.raises(ValueError):
        parse_areal_thermal_resistance("1 K/W")          # resistance


def test_areal_resistance_rejects_non_numeric_input():
    from om3dthermal.units import parse_areal_thermal_resistance
    with pytest.raises(TypeError):
        parse_areal_thermal_resistance(True)
    with pytest.raises(TypeError):
        parse_areal_thermal_resistance(object())


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def test_interface_resistance_config_round_trip():
    rule = InterfaceResistanceConfig.model_validate({
        "materials": ["Silicon", "Oxide"],
        "areal_resistance": "1e-8 m^2*K/W",
        "metadata": {"source": "test"},
    })
    assert rule.materials == ("Silicon", "Oxide")
    assert rule.areal_resistance == pytest.approx(1e-8)
    assert rule.metadata == {"source": "test"}


def test_interface_resistance_config_rejects_wrong_pair_length():
    with pytest.raises(ValidationError):
        InterfaceResistanceConfig.model_validate({
            "materials": ["Silicon"],
            "areal_resistance": 0,
        })
    with pytest.raises(ValidationError):
        InterfaceResistanceConfig.model_validate({
            "materials": ["A", "B", "C"],
            "areal_resistance": 0,
        })


def test_thermal_conductance_config_defaults():
    cfg = ThermalConductanceConfig.model_validate({})
    assert cfg.rotation_policy == "axis_aligned_only"
    assert cfg.default_interface_areal_resistance == 0.0
    assert cfg.interfaces == []


# ---------------------------------------------------------------------------
# Registry behaviour
# ---------------------------------------------------------------------------

def test_registry_default_is_used_when_no_rule_matches():
    reg = InterfaceResistanceRegistry(default_areal_resistance=0.0, rules=[])
    q = reg.lookup("Silicon", "Oxide")
    assert q.value == 0.0
    assert q.used_default is True
    assert q.rule_index == -1


def test_registry_unordered_pair_lookup():
    rule = InterfaceResistanceConfig.model_validate({
        "materials": ["Silicon", "Oxide"],
        "areal_resistance": "1e-8 m^2*K/W",
    })
    reg = InterfaceResistanceRegistry(default_areal_resistance=0.0,
                                       rules=[rule])
    q1 = reg.lookup("Silicon", "Oxide")
    q2 = reg.lookup("Oxide", "Silicon")
    assert q1.used_default is False
    assert q2.used_default is False
    assert q1.value == pytest.approx(1e-8)
    assert q2.value == pytest.approx(1e-8)
    assert q1.rule_index == 0
    assert q2.rule_index == 0


def test_registry_rejects_duplicate_unordered_pair():
    rule1 = InterfaceResistanceConfig.model_validate({
        "materials": ["Silicon", "Oxide"],
        "areal_resistance": 0,
    })
    rule2 = InterfaceResistanceConfig.model_validate({
        "materials": ["Oxide", "Silicon"],
        "areal_resistance": 0,
    })
    with pytest.raises(ValueError, match="duplicate interface rule"):
        InterfaceResistanceRegistry(default_areal_resistance=0.0,
                                     rules=[rule1, rule2])


def test_registry_allows_same_material_pair():
    rule = InterfaceResistanceConfig.model_validate({
        "materials": ["Silicon", "Silicon"],
        "areal_resistance": 0,
    })
    reg = InterfaceResistanceRegistry(default_areal_resistance=0.0, rules=[rule])
    q = reg.lookup("Silicon", "Silicon")
    assert q.used_default is False
    assert q.rule_index == 0


def test_registry_rule_count_and_default():
    rule = InterfaceResistanceConfig.model_validate({
        "materials": ["A", "B"],
        "areal_resistance": "1e-9 m^2*K/W",
    })
    reg = InterfaceResistanceRegistry(default_areal_resistance=2e-9,
                                       rules=[rule])
    assert reg.rule_count == 1
    assert reg.default == pytest.approx(2e-9)


def test_registry_rejects_negative_default():
    with pytest.raises(ValueError):
        InterfaceResistanceRegistry(default_areal_resistance=-1.0, rules=[])


def test_registry_rejects_negative_rule_value():
    # The schema validator catches this before the registry is even
    # constructed, so the right place to assert is the model
    # validator.
    with pytest.raises(ValidationError):
        InterfaceResistanceConfig.model_validate({
            "materials": ["A", "B"],
            "areal_resistance": -1.0,
        })
