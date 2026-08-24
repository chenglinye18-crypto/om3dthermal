"""Targeted accounting and placement tests for the Son23 HBM power model."""
from pathlib import Path

import pytest

from om3dthermal.cli import build_scene
from om3dthermal.config import load_config
from om3dthermal.discretization.models import ThermalCell
from om3dthermal.thermal.power import build_power_breakdown, map_power_sources


ROOT = Path(__file__).parents[1]
UNIFORM = ROOT / "configs" / "legacy" / "exp_conv_2x2_g414_m160_legacy_uniform.yaml"
SON23 = ROOT / "configs" / "legacy" / "exp_conv_2x2_g414_m160.yaml"
STACKS = (
    "hbm_left_top", "hbm_left_bottom",
    "hbm_right_top", "hbm_right_bottom",
)


@pytest.fixture(scope="module")
def cases():
    return load_config(UNIFORM), load_config(SON23)


def _scene_signature(config):
    scene = build_scene(config)
    return sorted(
        (box.name, box.material, box.x0, box.x1, box.y0, box.y1,
         box.z0, box.z1, box.rotation, repr(sorted(box.tags.items())))
        for box in scene.boxes)


def _carrier_cells(config):
    scene = build_scene(config)
    boxes = [
        box for box in scene.boxes
        if box.material in {"FEOL", "HBM_Base_BEOL", "DRAM_BEOL"}
    ]
    return [
        ThermalCell(
            id=index, ix=index, iy=0, iz=0,
            x0=box.x0, x1=box.x1,
            y0=box.y0, y1=box.y1,
            z0=box.z0, z1=box.z1,
            material=box.material,
            parent_box_id=f"box-{index}",
            parent_box_name=box.name,
            component=box.tags.get("component"),
            source_path=box.source_path,
            tags=dict(box.tags),
        )
        for index, box in enumerate(boxes)
    ]


def test_son23_config_parse_and_total_power(cases):
    _, son23 = cases
    sources = son23.thermal_power_sources.sources
    assert son23.name == "exp_conv_2x2_g414_m160"
    assert len(sources) == 1 + 4 * (2 + 12 * 2)
    assert sum(source.total_power for source in sources) == pytest.approx(574.0)


def test_son23_is_geometry_material_bc_mesh_solver_identical(cases):
    uniform, son23 = cases
    assert _scene_signature(uniform) == _scene_signature(son23)
    left = uniform.model_dump()
    right = son23.model_dump()
    for resolved in (left, right):
        resolved.pop("name")
        resolved.pop("metadata")
        resolved.pop("thermal_power_sources")
    assert left == right


def test_son23_component_values_and_vertical_carriers(cases):
    _, son23 = cases
    hbm_sources = son23.thermal_power_sources.sources[1:]
    assert all(source.selector.material != "Silicon" for source in hbm_sources)
    for stack in STACKS:
        stack_sources = [
            source for source in hbm_sources
            if source.metadata["stack"] == stack]
        logic = [source for source in stack_sources
                 if source.metadata["component_class"] == "logic"]
        dram = [source for source in stack_sources
                if source.metadata["component_class"] == "dram"]
        assert [source.total_power for source in logic] == pytest.approx(
            [16.0 / 3.0, 8.0 / 3.0])
        assert all(source.selector.material == "HBM_Base_BEOL"
                   for source in logic)
        assert len(dram) == 24
        for die_index in range(1, 13):
            die_sources = [source for source in dram
                           if source.metadata["dram_die_index"] == die_index]
            assert [source.total_power for source in die_sources] == pytest.approx(
                [2.0, 2.0 / 3.0])
            assert all(source.selector.layer.endswith(
                "top_dram_beol" if die_index == 12
                else f"dram_beol_{die_index:02d}") for source in die_sources)
        assert sum(source.total_power for source in stack_sources) == pytest.approx(40.0)


def test_actual_carriers_map_once_and_diagnostics_close(cases):
    _, son23 = cases
    power = map_power_sources(
        _carrier_cells(son23), son23.thermal_power_sources)
    breakdown = build_power_breakdown(power, son23.thermal_power_sources)
    assert power.total_power_W == pytest.approx(574.0)
    assert breakdown["power_model"] == "son23split"
    for stack in breakdown["per_stack"].values():
        assert stack == pytest.approx({
            "logic_phy_W": 16.0 / 3.0,
            "logic_tsv_W": 8.0 / 3.0,
            "logic_total_W": 8.0,
            "dram_bank_W_per_die": 2.0,
            "dram_tsv_W_per_die": 2.0 / 3.0,
            "dram_total_W_per_die": 8.0 / 3.0,
            "dram_total_W": 32.0,
            "stack_total_W": 40.0,
        })
    assert breakdown["whole_package"] == pytest.approx({
        "hbm_logic_total_W": 32.0,
        "hbm_dram_total_W": 128.0,
        "hbm_total_W": 160.0,
        "gpu_total_W": 414.0,
        "package_total_W": 574.0,
    })
    assert breakdown["accounting"]["missing_or_duplicated_power_W"] == pytest.approx(0.0)
