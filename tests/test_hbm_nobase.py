"""Targeted geometry and power checks for the no-base HBM intervention."""
from pathlib import Path

import pytest

from om3dthermal.cli import build_scene
from om3dthermal.config import load_config


ROOT = Path(__file__).parents[1]
CASES = {
    ("2x2", 414): ROOT / "configs" / "exp_conv_2x2_nobase_g414_m128.yaml",
    ("2x2", 300): ROOT / "configs" / "exp_conv_2x2_nobase_g300_m128.yaml",
    ("2x1", 414): ROOT / "configs" / "exp_conv_2x1_nobase_g414_m128.yaml",
    ("2x1", 300): ROOT / "configs" / "exp_conv_2x1_nobase_g300_m128.yaml",
}


@pytest.mark.parametrize("layout,gpu", CASES)
def test_nobase_geometry_removes_logic_layers_only(layout, gpu):
    cfg = load_config(CASES[(layout, gpu)])
    scene = build_scene(cfg)
    hbm_components = sorted({
        box.tags["component"] for box in scene.boxes
        if str(box.tags.get("component", "")).startswith("memory_column:hbm_")
    })
    assert len(hbm_components) == (4 if layout == "2x2" else 2)
    assert cfg.stack_templates["hbm_12hi"].total_thickness == pytest.approx(720e-6)
    for component in hbm_components:
        boxes = scene.filter(component=component)
        assert not [box for box in boxes if box.material == "HBM_Base_BEOL"]
        assert not [box for box in boxes if box.tags.get("role") == "hbm_base"]
        ubump = [box for box in boxes
                 if box.tags.get("role") == "gpu_hbm_interface"]
        assert len(ubump) == 1
        assert ubump[0].z1 - ubump[0].z0 == pytest.approx(40e-6)
        assert len([box for box in boxes if box.tags.get("role") == "dram_beol"]) == 12
        assert len([box for box in boxes if box.tags.get("role") == "dram_si"]) == 12
        fill = [box for box in boxes if box.tags.get("fill_above")]
        assert len(fill) == 1
        assert fill[0].material == "Mold"
        assert fill[0].z1 - fill[0].z0 == pytest.approx(55e-6)


@pytest.mark.parametrize("layout,gpu", CASES)
def test_nobase_power_is_dram_only_and_conserved(layout, gpu):
    cfg = load_config(CASES[(layout, gpu)])
    sources = cfg.thermal_power_sources.sources
    assert sources[0].total_power == pytest.approx(float(gpu))
    hbm = sources[1:]
    assert not [source for source in hbm
                if source.metadata.get("component_class") == "logic"]
    assert all(source.metadata.get("component_class") == "dram"
               for source in hbm)
    assert all(source.selector.layer and "dram_beol" in source.selector.layer
               for source in hbm)
    assert sum(source.total_power for source in hbm) == pytest.approx(128.0)
    assert sum(source.total_power for source in sources) == pytest.approx(
        gpu + 128.0)
    expected_groups = 4 if layout == "2x2" else 2
    stacks = {source.metadata["stack"] for source in hbm}
    assert len(stacks) == expected_groups
    expected_group_power = 32.0 if layout == "2x2" else 64.0
    assert {stack: sum(source.total_power for source in hbm
                       if source.metadata["stack"] == stack)
            for stack in stacks} == pytest.approx(
                {stack: expected_group_power for stack in stacks})


@pytest.mark.parametrize("layout", ["2x2", "2x1"])
def test_nobase_300W_case_changes_only_gpu_power(layout):
    high = load_config(CASES[(layout, 414)]).model_dump()
    low = load_config(CASES[(layout, 300)]).model_dump()
    for data in (high, low):
        data.pop("name")
        data.pop("metadata")
    high["thermal_power_sources"]["sources"][0]["total_power"] = 300.0
    assert high == low
