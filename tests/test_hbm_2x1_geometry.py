"""Canonical HBM grouping, inset and layer geometry."""
from pathlib import Path

import pytest

from om3dthermal.bandwidth_thermal_sweep import ARCHITECTURES, _base_case
from om3dthermal.cli import build_scene
from om3dthermal.geometry.horizontal_columns import _boxes_overlap_3d


@pytest.fixture(scope="module")
def case():
    canonical, config, _ = _base_case(Path(__file__).parents[1], ARCHITECTURES[0])
    return canonical, build_scene(config)


def test_groups_follow_configured_footprint(case):
    canonical, scene = case
    layout = canonical.geometry.layout
    width, height = layout["visible_group_footprint_mm"]
    assert layout["visible_group_count"] == 2
    for name, (cx, cy) in zip(("hbm_left", "hbm_right"), layout["group_centers_mm"]):
        boxes = scene.filter(component=f"memory_column:{name}")
        assert min(b.x0 for b in boxes) == pytest.approx((cx - width / 2) * 1e-3)
        assert max(b.x1 for b in boxes) == pytest.approx((cx + width / 2) * 1e-3)
        assert min(b.y0 for b in boxes) == pytest.approx((cy - height / 2) * 1e-3)
        assert max(b.y1 for b in boxes) == pytest.approx((cy + height / 2) * 1e-3)


def test_dram_layers_cross_center_without_internal_seam(case):
    canonical, scene = case
    for name in ("hbm_left", "hbm_right"):
        boxes = scene.filter(component=f"memory_column:{name}")
        for role in ("dram_si", "dram_beol", "hybrid_bonding"):
            layers = [box for box in boxes if box.tags.get("role") == role]
            assert len(layers) == canonical.geometry.layout["dram_dies_per_stack"]
            assert all(box.y0 < 0 < box.y1 for box in layers)
            assert all(box.y1 - box.y0 == pytest.approx(
                canonical.geometry.memory_region.height_mm * 1e-3) for box in layers)


def test_no_geometry_overlap(case):
    _, scene = case
    for index, first in enumerate(scene.boxes):
        assert all(not _boxes_overlap_3d(first, second) for second in scene.boxes[index + 1:])
