"""Tests for the power source mapping."""
from __future__ import annotations

import pytest

from om3dthermal.config import (
    PowerSelector,
    PowerSourceConfig,
    ThermalPowerSourcesConfig,
)
from om3dthermal.thermal.power import map_power_sources
from om3dthermal.discretization.models import ThermalCell


def _cell(*, id: int, component: str = "gpu", material: str = "FEOL",
          parent_box_name: str = "x", volume: float = 1.0e-9,
          tags: dict | None = None) -> ThermalCell:
    return ThermalCell(
        id=id, ix=0, iy=0, iz=0,
        x0=0.0, x1=1.0, y0=0.0, y1=1.0, z0=0.0, z1=volume,
        material=material,
        parent_box_id="b", parent_box_name=parent_box_name,
        component=component, source_path="t", tags=tags or {},
    )


def test_uniform_volume_splits_total_in_proportion_to_volumes():
    cells = [_cell(id=0, volume=2.0e-9), _cell(id=1, volume=3.0e-9)]
    src = PowerSourceConfig.model_validate({
        "name": "p", "total_power": 50.0, "selector": {},
        "distribution": "uniform_volume",
    })
    pv = map_power_sources(cells, ThermalPowerSourcesConfig.model_validate(
        {"sources": [src.model_dump()]}))
    # Volumes are 2 and 3, total 5. Power 50 -> 20 + 30.
    assert pv.power_W[0] == pytest.approx(20.0)
    assert pv.power_W[1] == pytest.approx(30.0)
    assert pv.total_power_W == pytest.approx(50.0)
    assert pv.power_by_source == {"p": 50.0}


def test_source_with_no_match_raises():
    cells = [_cell(id=0, component="gpu")]
    src = PowerSourceConfig.model_validate({
        "name": "p", "total_power": 50.0,
        "selector": {"component": "hbm"},
        "distribution": "uniform_volume",
    })
    with pytest.raises(ValueError, match="selected no cells"):
        map_power_sources(cells, ThermalPowerSourcesConfig.model_validate(
            {"sources": [src.model_dump()]}))


def test_two_sources_additive_on_overlapping_cells():
    cells = [_cell(id=0, component="gpu"), _cell(id=1, component="hbm")]
    src_a = PowerSourceConfig.model_validate({
        "name": "a", "total_power": 10.0, "selector": {"component": "gpu"},
        "distribution": "uniform_volume",
    })
    src_b = PowerSourceConfig.model_validate({
        "name": "b", "total_power": 20.0, "selector": {"component": "hbm"},
        "distribution": "uniform_volume",
    })
    pv = map_power_sources(cells, ThermalPowerSourcesConfig.model_validate(
        {"sources": [src_a.model_dump(), src_b.model_dump()]}))
    assert pv.total_power_W == pytest.approx(30.0)
    assert pv.power_by_source == {"a": 10.0, "b": 20.0}


def test_tag_matcher_selects_only_cells_with_matching_tag():
    cells = [
        _cell(id=0, tags={"role": "dram_beol"}),
        _cell(id=1, tags={"role": "feol"}),
        _cell(id=2, tags={"role": "dram_beol"}),
    ]
    src = PowerSourceConfig.model_validate({
        "name": "beol", "total_power": 100.0,
        "selector": {"tags": {"role": "dram_beol"}},
        "distribution": "uniform_volume",
    })
    pv = map_power_sources(cells, ThermalPowerSourcesConfig.model_validate(
        {"sources": [src.model_dump()]}))
    # Only cells 0 and 2 selected; volumes are equal so 50/50.
    assert pv.power_W[0] == pytest.approx(50.0)
    assert pv.power_W[1] == 0.0
    assert pv.power_W[2] == pytest.approx(50.0)
    assert pv.selected_cell_count_by_source == {"beol": 2}


def test_layer_selector_matches_parent_box_name():
    cells = [
        _cell(id=0, parent_box_name="gpu.feol"),
        _cell(id=1, parent_box_name="gpu.beol_mxy"),
    ]
    src = PowerSourceConfig.model_validate({
        "name": "feol", "total_power": 4.0,
        "selector": {"layer": "gpu.feol"},
        "distribution": "uniform_volume",
    })
    pv = map_power_sources(cells, ThermalPowerSourcesConfig.model_validate(
        {"sources": [src.model_dump()]}))
    assert pv.power_W[0] == pytest.approx(4.0)
    assert pv.power_W[1] == 0.0


def test_conservation_with_non_equal_volumes():
    cells = [
        _cell(id=0, volume=1.0e-9),
        _cell(id=1, volume=2.0e-9),
        _cell(id=2, volume=3.0e-9),
    ]
    src = PowerSourceConfig.model_validate({
        "name": "p", "total_power": 12.0, "selector": {},
        "distribution": "uniform_volume",
    })
    pv = map_power_sources(cells, ThermalPowerSourcesConfig.model_validate(
        {"sources": [src.model_dump()]}))
    # Volumes 1,2,3 -> total 6. Powers 2, 4, 6.
    assert pv.power_W[0] == pytest.approx(2.0)
    assert pv.power_W[1] == pytest.approx(4.0)
    assert pv.power_W[2] == pytest.approx(6.0)
    assert abs(pv.total_power_W - 12.0) < 1e-9


def test_no_cells_raises():
    with pytest.raises(ValueError, match="no cells"):
        map_power_sources([], ThermalPowerSourcesConfig.model_validate({}))
