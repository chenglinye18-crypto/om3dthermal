import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from om3dthermal.cli import build
from om3dthermal.config import SimulationConfig, load_config
from om3dthermal.geometry.horizontal_columns import HorizontalColumnsBuilder

CONFIG = Path(__file__).parents[1] / "configs" / "hbm_on_gpu_12hi.yaml"


def test_geometry_z_order_continuity_and_identity_rotation():
    scene = HorizontalColumnsBuilder(load_config(CONFIG)).build()
    for component in {box.tags.get("component") for box in scene.boxes
                      if str(box.tags.get("component", "")).startswith("memory_column:")}:
        boxes = sorted(scene.filter(component=component), key=lambda box: box.z0)
        for lower, upper in zip(boxes, boxes[1:]):
            assert lower.z1 == pytest.approx(upper.z0)
        assert boxes[-1].z1 - boxes[0].z0 == pytest.approx(755e-6)
    assert all(box.rotation == ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
               for box in scene.boxes)


def test_each_hbm_column_has_twelve_dram_boxes():
    scene = HorizontalColumnsBuilder(load_config(CONFIG)).build()
    for name in ("hbm_west", "hbm_east", "hbm_north", "hbm_south"):
        boxes = scene.filter(component=f"memory_column:{name}", material="DRAM_Si")
        assert len(boxes) == 12


def test_foundation_gpu_memory_top_are_derived_in_order():
    scene = HorizontalColumnsBuilder(load_config(CONFIG)).build()
    foundation_top = max(box.z1 for box in scene.filter(component="foundation"))
    gpu = scene.filter(component="gpu")
    memory = scene.filter(component="memory_zone_background")
    top = scene.filter(component="top")
    assert min(box.z0 for box in gpu) == pytest.approx(foundation_top)
    assert memory[0].z0 == pytest.approx(max(box.z1 for box in gpu))
    assert min(box.z0 for box in top) == pytest.approx(memory[0].z1)


def test_summary_and_cli_outputs(tmp_path):
    scene = build(CONFIG, tmp_path)
    expected = {"regions.csv", "geometry_summary.json", "top_view.png",
                "xz_section.png", "yz_section.png"}
    assert expected == {path.name for path in tmp_path.iterdir()}
    summary = json.loads((tmp_path / "geometry_summary.json").read_text(encoding="utf-8"))
    assert summary["total_boxes"] == len(scene.boxes)
    assert summary["boxes_by_material"]["DRAM_Si"] == 48
    assert summary["stack_heights_m"]["hbm_12hi"] == pytest.approx(755e-6)
    assert summary["minimum_dimension_m"] > 0
    assert summary["maximum_dimension_m"] >= summary["minimum_dimension_m"]
    assert {"foundation", "gpu", "memory_zone_background", "top"} <= set(summary["component_bounds_m"])


def test_footprint_outside_package_is_rejected():
    data = load_config(CONFIG).model_dump()
    data["footprints"]["gpu"]["center_x"] = 1.0
    with pytest.raises(ValidationError, match="exceeds package bounds"):
        SimulationConfig.model_validate(data)

def test_missing_match_height_reference_is_rejected():
    data = load_config(CONFIG).model_dump()
    data["horizontal"]["memory_zone"]["columns"][4]["match_height_of"] = "missing"
    with pytest.raises(ValidationError, match="unknown stack reference"):
        SimulationConfig.model_validate(data)


def test_short_stack_without_fill_is_rejected():
    data = load_config(CONFIG).model_dump()
    data["horizontal"]["memory_zone"]["columns"][5]["fill_above"] = None
    with pytest.raises(ValidationError, match="requires fill_above"):
        SimulationConfig.model_validate(data)
