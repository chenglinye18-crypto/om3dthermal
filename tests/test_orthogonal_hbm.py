"""Geometry, power, and steady-state checks for the MOSAIC baseline."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from om3dthermal.cli import build_scene, solve_steady
from om3dthermal.config import load_config
from om3dthermal.geometry.horizontal_columns import _boxes_overlap_3d
from om3dthermal.geometry.orthogonal_hbm import ORTHOGONAL_DIE_ROTATION
from om3dthermal.thermal import is_signed_axis_permutation


CONFIG = Path(__file__).parents[1] / "configs" / "exp_orth_mosaic98_g414_m156p8_uniform.yaml"


@pytest.fixture(scope="module")
def cfg():
    return load_config(CONFIG)


@pytest.fixture(scope="module")
def scene(cfg):
    return build_scene(cfg)


def _die_components(scene):
    return sorted({
        box.tags["component"] for box in scene.boxes
        if str(box.tags.get("component", "")).startswith("orthogonal_hbm:die_")
    })


def _cube_boxes(scene):
    return [
        box for box in scene.boxes
        if str(box.tags.get("component", "")).startswith("orthogonal_hbm:")
        and box.tags.get("component") != "orthogonal_hbm:adhesive"
    ]


def test_config_parses_and_preserves_paper_parameters(cfg):
    orthogonal = cfg.orthogonal_hbm
    die = orthogonal.memory_die
    cube = cfg.footprints[orthogonal.cube_footprint]
    assert (cube.size_x, cube.size_y, orthogonal.cube_height) == pytest.approx(
        (30e-3, 22e-3, 5.5e-3))
    assert die.count == 98
    assert (die.width, die.height, die.thickness) == pytest.approx(
        (22e-3, 5.5e-3, 300e-6))
    assert [layer.thickness for layer in die.layers] == pytest.approx(
        [293e-6, 5e-6, 2e-6])
    assert orthogonal.adhesive.material == "Adhesive"
    assert orthogonal.adhesive.thickness == pytest.approx(3e-6)
    assert cfg.materials["Adhesive"].k_local == pytest.approx((0.2, 0.2, 0.2))
    assert "UNRESOLVED" not in cfg.metadata["provenance"]


def test_generates_exactly_98_vertical_dies_with_expected_pitch(scene):
    components = _die_components(scene)
    assert len(components) == 98
    bounds = []
    for component in components:
        boxes = scene.filter(component=component)
        assert len(boxes) == 3
        x0 = min(box.x0 for box in boxes)
        x1 = max(box.x1 for box in boxes)
        assert x1 - x0 == pytest.approx(300e-6)
        bounds.append((x0, x1))
    bounds.sort()
    assert bounds[-1][1] - bounds[0][0] == pytest.approx(29.4e-3)
    assert [bounds[i + 1][0] - bounds[i][0] for i in range(97)] == pytest.approx(
        [300e-6] * 97)
    assert [bounds[i + 1][0] - bounds[i][1] for i in range(97)] == pytest.approx(
        [0.0] * 97, abs=1e-12)


def test_cube_bounds_material_layers_and_orientation(cfg, scene):
    cube_boxes = _cube_boxes(scene)
    x0, x1 = min(b.x0 for b in cube_boxes), max(b.x1 for b in cube_boxes)
    y0, y1 = min(b.y0 for b in cube_boxes), max(b.y1 for b in cube_boxes)
    z0, z1 = min(b.z0 for b in cube_boxes), max(b.z1 for b in cube_boxes)
    assert (x1 - x0, y1 - y0, z1 - z0) == pytest.approx(
        (30e-3, 22e-3, 5.5e-3))
    assert len([b for b in cube_boxes if b.material == "MOSAIC_Si"]) == 98
    assert len([b for b in cube_boxes if b.material == "MOSAIC_BEOL"]) == 98
    assert len([b for b in cube_boxes if b.material == "MOSAIC_DAA"]) == 98
    die_boxes = [b for b in cube_boxes if "die_index" in b.tags]
    assert all(b.rotation == ORTHOGONAL_DIE_ROTATION for b in die_boxes)
    assert is_signed_axis_permutation(ORTHOGONAL_DIE_ROTATION)
    hotspot_x, hotspot_y = 13.75e-3, 0.1465e-3
    assert x0 <= hotspot_x <= x1
    assert y0 <= hotspot_y <= y1


def test_mold_fills_only_remaining_cube_volume_without_overlap(cfg, scene):
    cube_boxes = _cube_boxes(scene)
    mold = [box for box in cube_boxes if box.material == "Mold"]
    assert len(mold) == 2
    mold_volume = sum((b.x1-b.x0)*(b.y1-b.y0)*(b.z1-b.z0) for b in mold)
    assert mold_volume == pytest.approx(22e-3 * 0.6e-3 * 5.5e-3)
    total_volume = sum((b.x1-b.x0)*(b.y1-b.y0)*(b.z1-b.z0) for b in cube_boxes)
    assert total_volume == pytest.approx(22e-3 * 30e-3 * 5.5e-3)
    for index, first in enumerate(scene.boxes):
        for second in scene.boxes[index + 1:]:
            assert not _boxes_overlap_3d(first, second)


def test_adhesive_fully_covers_gpu_to_cube_interface(cfg, scene):
    gpu = scene.filter(component="gpu")
    adhesive = scene.filter(component="orthogonal_hbm:adhesive")
    cube_boxes = _cube_boxes(scene)
    assert len(adhesive) == 1
    layer = adhesive[0]
    assert layer.material == "Adhesive"
    assert layer.z1 - layer.z0 == pytest.approx(3e-6)
    assert (layer.x0, layer.x1, layer.y0, layer.y1) == pytest.approx(
        (-15e-3, 15e-3, -11e-3, 11e-3))
    assert layer.z0 == pytest.approx(max(box.z1 for box in gpu))
    assert layer.z1 == pytest.approx(min(box.z0 for box in cube_boxes))
    assert (layer.x1 - layer.x0) * (layer.y1 - layer.y0) == pytest.approx(
        660e-6)


def test_uniform_power_sources_total_414_W_gpu_and_156_8_W_memory(cfg):
    sources = cfg.thermal_power_sources.sources
    gpu = sum(s.total_power for s in sources if s.name.startswith("gpu"))
    memory = [s for s in sources if s.name.startswith("hbm_die_")]
    assert gpu == pytest.approx(414.0)
    assert len(memory) == 98
    assert all(s.total_power == pytest.approx(1.6) for s in memory)
    assert sum(s.total_power for s in memory) == pytest.approx(156.8)
    assert all(s.selector.tags == {"role": "active_beol"} for s in memory)


@pytest.fixture(scope="module")
def steady_result(tmp_path_factory):
    output = tmp_path_factory.mktemp("orthogonal_hbm_steady")
    summary = solve_steady(
        CONFIG, output, method="pcg", rtol=1e-6, max_iterations=3000)
    hottest = None
    with (output / "temperature_cells.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            temperature = float(row["temperature_K"])
            if hottest is None or temperature > hottest[0]:
                hottest = (temperature, row)
    return summary, hottest


def test_steady_state_converges_is_finite_and_balanced(steady_result):
    summary, hottest = steady_result
    values = np.array([
        summary["min_temperature_K"], summary["max_temperature_K"],
        summary["final_relative_residual"], summary["relative_power_imbalance"],
    ])
    assert np.all(np.isfinite(values))
    assert summary["converged"] is True
    assert summary["final_relative_residual"] < 1e-6
    assert summary["relative_power_imbalance"] < 1e-6
    assert summary["gpu_power_W"] == pytest.approx(414.0)
    assert summary["hbm_power_W"] == pytest.approx(156.8)
    assert summary["total_input_power_W"] == pytest.approx(570.8)
    assert hottest is not None
