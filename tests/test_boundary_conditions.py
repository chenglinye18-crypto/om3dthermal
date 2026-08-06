"""Unit tests for boundary condition parsing, schema, and matching."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from om3dthermal.config import (
    BoundaryConditionConfig,
    BoundarySelector,
    ThermalBoundaryConditionsConfig,
)
from om3dthermal.thermal.boundary import select_boundary_rule
from om3dthermal.discretization.models import BoundaryFace
from om3dthermal.discretization.models import ThermalCell


# ---------------------------------------------------------------------------
# Unit parsers
# ---------------------------------------------------------------------------

def test_parse_power_accepts_plain_W():
    from om3dthermal.units import parse_power
    assert parse_power(414) == 414.0
    assert parse_power(0) == 0.0
    assert parse_power("414 W") == pytest.approx(414.0)
    assert parse_power("40 W") == pytest.approx(40.0)
    assert parse_power("1 kW") == pytest.approx(1000.0)


def test_parse_power_rejects_negative_nan_inf():
    from om3dthermal.units import parse_power
    with pytest.raises(ValueError):
        parse_power(-1.0)
    with pytest.raises(ValueError):
        parse_power(float("nan"))
    with pytest.raises(ValueError):
        parse_power(float("inf"))
    with pytest.raises(ValueError):
        parse_power("-1 W")


def test_parse_power_rejects_wrong_dimensions():
    from om3dthermal.units import parse_power
    with pytest.raises(ValueError):
        parse_power("1 m")
    with pytest.raises(ValueError):
        parse_power("1 K")
    with pytest.raises(ValueError):
        parse_power("1 V")


def test_parse_htc_accepts_positive_values():
    from om3dthermal.units import parse_heat_transfer_coefficient
    assert parse_heat_transfer_coefficient(30000) == 30000.0
    assert parse_heat_transfer_coefficient("30000 W/m^2/K") == pytest.approx(30000.0)
    assert parse_heat_transfer_coefficient("200 W/m^2/K") == pytest.approx(200.0)


def test_parse_htc_rejects_non_positive():
    from om3dthermal.units import parse_heat_transfer_coefficient
    with pytest.raises(ValueError):
        parse_heat_transfer_coefficient(0)
    with pytest.raises(ValueError):
        parse_heat_transfer_coefficient(-1.0)
    with pytest.raises(ValueError):
        parse_heat_transfer_coefficient("0 W/m^2/K")
    with pytest.raises(ValueError):
        parse_heat_transfer_coefficient(float("nan"))


def test_parse_htc_rejects_wrong_dimensions():
    from om3dthermal.units import parse_heat_transfer_coefficient
    with pytest.raises(ValueError):
        parse_heat_transfer_coefficient("1 m")
    with pytest.raises(ValueError):
        parse_heat_transfer_coefficient("1 W")


def test_parse_temperature_accepts_K_and_bare_numbers():
    from om3dthermal.units import parse_temperature
    assert parse_temperature(293.15) == pytest.approx(293.15)
    assert parse_temperature(0) == 0.0
    assert parse_temperature("293.15 K") == pytest.approx(293.15)
    assert parse_temperature("0 K") == 0.0


def test_parse_temperature_rejects_negative_nan_inf():
    from om3dthermal.units import parse_temperature
    with pytest.raises(ValueError):
        parse_temperature(-1.0)
    with pytest.raises(ValueError):
        parse_temperature(float("nan"))
    with pytest.raises(ValueError):
        parse_temperature(float("inf"))


def test_parse_temperature_rejects_wrong_dimensions():
    from om3dthermal.units import parse_temperature
    with pytest.raises(ValueError):
        parse_temperature("1 m")
    with pytest.raises(ValueError):
        parse_temperature("1 W")


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def test_adiabatic_rule_rejects_h_ambient_surface():
    for field_name, value in (
        ("heat_transfer_coefficient", 10.0),
        ("ambient_temperature", 293.15),
        ("surface_temperature", 293.15),
    ):
        with pytest.raises(ValidationError):
            BoundaryConditionConfig.model_validate({
                "name": "ad",
                "kind": "adiabatic",
                "selector": {},
                field_name: value,
            })


def test_convection_rule_requires_h_and_ambient():
    base = {
        "name": "conv",
        "kind": "convection",
        "selector": {"axis": "z", "side": "plus"},
        "heat_transfer_coefficient": 100.0,
        "ambient_temperature": 293.15,
    }
    BoundaryConditionConfig.model_validate(base)
    bad = dict(base)
    del bad["heat_transfer_coefficient"]
    with pytest.raises(ValidationError):
        BoundaryConditionConfig.model_validate(bad)
    bad = dict(base)
    del bad["ambient_temperature"]
    with pytest.raises(ValidationError):
        BoundaryConditionConfig.model_validate(bad)
    bad = dict(base)
    bad["surface_temperature"] = 300.0
    with pytest.raises(ValidationError):
        BoundaryConditionConfig.model_validate(bad)


def test_fixed_temperature_rule_requires_surface_temp():
    base = {
        "name": "ft",
        "kind": "fixed_temperature",
        "selector": {},
        "surface_temperature": 293.15,
    }
    BoundaryConditionConfig.model_validate(base)
    bad = dict(base)
    del bad["surface_temperature"]
    with pytest.raises(ValidationError):
        BoundaryConditionConfig.model_validate(bad)
    bad = dict(base)
    bad["heat_transfer_coefficient"] = 10.0
    with pytest.raises(ValidationError):
        BoundaryConditionConfig.model_validate(bad)


def test_thermal_boundary_conditions_default_is_adiabatic():
    cfg = ThermalBoundaryConditionsConfig.model_validate({})
    assert cfg.default == "adiabatic"
    assert cfg.rules == []


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def _face(cell_id: int = 0, axis: str = "z", side: str = "plus",
          classification: str = "scene_outer_boundary",
          material: str = "Lid") -> BoundaryFace:
    return BoundaryFace(
        id=0, cell_id=cell_id, axis=axis, side=side, coordinate=0.0,
        area=1.0e-4, normal=(0.0, 0.0, 1.0),
        component="top", material=material, classification=classification,
    )


def _cell(cell_id: int = 0, component: str = "top",
          material: str = "Lid", parent_box_name: str = "top.lid",
          tags: dict | None = None) -> ThermalCell:
    return ThermalCell(
        id=cell_id, ix=0, iy=0, iz=0,
        x0=0.0, x1=1.0, y0=0.0, y1=1.0, z0=0.0, z1=1.0,
        material=material,
        parent_box_id="b", parent_box_name=parent_box_name,
        component=component, source_path="t", tags=tags or {},
    )


def test_select_boundary_rule_no_match_returns_none():
    face = _face()
    cell = _cell()
    cfg = ThermalBoundaryConditionsConfig.model_validate({})
    assert select_boundary_rule(face, cell, cfg.rules) is None


def test_select_boundary_rule_chooses_highest_priority():
    face = _face()
    cell = _cell()
    rules = [
        BoundaryConditionConfig.model_validate({
            "name": "low", "kind": "convection",
            "selector": {"axis": "z", "side": "plus", "priority": 1},
            "heat_transfer_coefficient": 10.0,
            "ambient_temperature": 300.0,
        }),
        BoundaryConditionConfig.model_validate({
            "name": "high", "kind": "fixed_temperature",
            "selector": {"axis": "z", "side": "plus", "priority": 100},
            "surface_temperature": 350.0,
        }),
    ]
    cfg = ThermalBoundaryConditionsConfig.model_validate({"rules": rules})
    match = select_boundary_rule(face, cell, cfg.rules)
    assert match is not None
    idx, rule = match
    assert rule.name == "high"
    assert idx == 1


def test_select_boundary_rule_tie_raises():
    face = _face()
    cell = _cell()
    rules = [
        BoundaryConditionConfig.model_validate({
            "name": "a", "kind": "convection",
            "selector": {"axis": "z", "side": "plus", "priority": 10},
            "heat_transfer_coefficient": 10.0,
            "ambient_temperature": 300.0,
        }),
        BoundaryConditionConfig.model_validate({
            "name": "b", "kind": "convection",
            "selector": {"axis": "z", "side": "plus", "priority": 10},
            "heat_transfer_coefficient": 20.0,
            "ambient_temperature": 300.0,
        }),
    ]
    cfg = ThermalBoundaryConditionsConfig.model_validate({"rules": rules})
    with pytest.raises(ValueError, match="multiple rules with the same priority"):
        select_boundary_rule(face, cell, cfg.rules)


def test_select_boundary_rule_tag_match():
    face = _face(material="DRAM_BEOL")
    cell = _cell(material="DRAM_BEOL", tags={"role": "dram_beol"})
    rules = [
        BoundaryConditionConfig.model_validate({
            "name": "beol", "kind": "convection",
            "selector": {"tags": {"role": "dram_beol"}, "priority": 1},
            "heat_transfer_coefficient": 10.0,
            "ambient_temperature": 300.0,
        }),
    ]
    cfg = ThermalBoundaryConditionsConfig.model_validate({"rules": rules})
    match = select_boundary_rule(face, cell, cfg.rules)
    assert match is not None
    assert match[1].name == "beol"
