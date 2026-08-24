"""Targeted geometry/power checks for the y-merged 2x1 HBM layout."""

from pathlib import Path

import pytest

from om3dthermal.cli import build_scene
from om3dthermal.config import load_config
from om3dthermal.geometry.horizontal_columns import _boxes_overlap_3d


ROOT = Path(__file__).parents[1]
CONFIG_414 = ROOT / "configs" / "legacy" / "exp_conv_2x1_g414_m160.yaml"
CONFIG_300 = ROOT / "configs" / "legacy" / "exp_conv_2x1_g300_m160.yaml"
GROUPS = ("hbm_left", "hbm_right")


@pytest.fixture(scope="module", params=[CONFIG_414, CONFIG_300])
def case(request):
    cfg = load_config(request.param)
    return cfg, build_scene(cfg)


def _diff(first, second, path=""):
    if type(first) is not type(second):
        return [(path, first, second)]
    if isinstance(first, dict):
        out = []
        for key in sorted(set(first) | set(second)):
            child = f"{path}.{key}" if path else key
            out.extend(_diff(first.get(key), second.get(key), child))
        return out
    if isinstance(first, (list, tuple)):
        out = []
        for index, (left, right) in enumerate(zip(first, second)):
            out.extend(_diff(left, right, f"{path}[{index}]"))
        if len(first) != len(second):
            out.append((f"{path}.length", len(first), len(second)))
        return out
    return [] if first == second else [(path, first, second)]


def test_two_groups_have_11x22_parent_footprints(case):
    cfg, scene = case
    expected_x = {"hbm_left": (-15e-3, -4e-3),
                  "hbm_right": (4e-3, 15e-3)}
    assert cfg.metadata["layout"]["hbm_arrangement"] == "2x1_hbm"
    for name in GROUPS:
        boxes = scene.filter(component=f"memory_column:{name}")
        assert min(box.x0 for box in boxes) == pytest.approx(expected_x[name][0])
        assert max(box.x1 for box in boxes) == pytest.approx(expected_x[name][1])
        assert min(box.y0 for box in boxes) == pytest.approx(-11e-3)
        assert max(box.y1 for box in boxes) == pytest.approx(11e-3)
        assert max(box.z1 for box in boxes) - min(box.z0 for box in boxes) == pytest.approx(775e-6)


def test_dram_layers_cross_y_zero_without_internal_mold_seam(case):
    _, scene = case
    for name in GROUPS:
        component = f"memory_column:{name}"
        for role in ("dram_si", "dram_beol", "hybrid_bonding"):
            central = [box for box in scene.filter(component=component)
                       if box.tags.get("role") == role]
            assert len(central) == 12
            assert all(box.y0 == pytest.approx(-10.9e-3) for box in central)
            assert all(box.y1 == pytest.approx(10.9e-3) for box in central)
            assert all(box.y0 < 0 < box.y1 for box in central)
        seam_mold = [box for box in scene.filter(component=component)
                     if box.material == "Mold" and box.x1 - box.x0 > 10e-3
                     and box.y0 < 0 < box.y1]
        assert seam_mold == []


def test_no_geometry_overlap(case):
    _, scene = case
    for index, first in enumerate(scene.boxes):
        for second in scene.boxes[index + 1:]:
            assert not _boxes_overlap_3d(first, second)


def test_hbm_power_is_two_times_80_W(case):
    cfg, _ = case
    hbm = cfg.thermal_power_sources.sources[1:]
    group_totals = {
        group: sum(source.total_power for source in hbm
                   if source.metadata.get("stack") == group)
        for group in GROUPS
    }
    assert group_totals == pytest.approx({"hbm_left": 80.0, "hbm_right": 80.0})
    assert sum(source.total_power for source in hbm
               if source.metadata.get("component_class") == "logic") == pytest.approx(32.0)
    assert sum(source.total_power for source in hbm
               if source.metadata.get("component_class") == "dram") == pytest.approx(128.0)
    expected_gpu = 414.0 if cfg.name == "exp_conv_2x1_g414_m160" else 300.0
    assert cfg.thermal_power_sources.sources[0].total_power == pytest.approx(expected_gpu)
    assert sum(source.total_power for source in cfg.thermal_power_sources.sources) == pytest.approx(
        expected_gpu + 160.0)


def test_two_power_variants_differ_only_in_name_and_gpu_power():
    first = load_config(CONFIG_414).model_dump()
    second = load_config(CONFIG_300).model_dump()
    assert _diff(first, second) == [
        ("metadata.case_id", "exp_conv_2x1_g414_m160", "exp_conv_2x1_g300_m160"),
        ("name", "exp_conv_2x1_g414_m160", "exp_conv_2x1_g300_m160"),
        ("thermal_power_sources.sources[0].total_power", 414.0, 300.0),
    ]
