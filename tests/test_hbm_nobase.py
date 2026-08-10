"""Targeted geometry and power checks for the no-base HBM intervention."""
from pathlib import Path

import pytest

from om3dthermal.cli import build_scene
from om3dthermal.config import load_config
from om3dthermal.discretization import (
    build_adjacency,
    build_global_grid,
    generate_cells,
)
from om3dthermal.thermal import build_conductance_table


ROOT = Path(__file__).parents[1]
CASES = {
    ("2x2", 414): ROOT / "configs" / "exp_conv_2x2_nobase_g414_m128.yaml",
    ("2x2", 300): ROOT / "configs" / "exp_conv_2x2_nobase_g300_m128.yaml",
    ("2x1", 414): ROOT / "configs" / "exp_conv_2x1_nobase_g414_m128.yaml",
    ("2x1", 300): ROOT / "configs" / "exp_conv_2x1_nobase_g300_m128.yaml",
}
BASE_PRESENT = {
    "2x2": ROOT / "configs" / "exp_conv_2x2_g414_m160.yaml",
    "2x1": ROOT / "configs" / "exp_conv_2x1_g414_m160.yaml",
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
        assert not [box for box in boxes if box.tags.get("fill_above")]

        hbm_top = max(box.z1 for box in boxes)
        tim = [box for box in scene.boxes
               if box.tags.get("component") == "top" and box.material == "TIM"]
        assert len(tim) == 1
        assert hbm_top == pytest.approx(tim[0].z0)


@pytest.mark.parametrize("layout", ["2x2", "2x1"])
def test_nobase_top_moves_down_and_directly_contacts_tim(layout):
    corrected_cfg = load_config(CASES[(layout, 414)])
    corrected = build_scene(corrected_cfg)
    base = build_scene(load_config(BASE_PRESENT[layout]))

    corrected_tim = next(box for box in corrected.boxes
                         if box.tags.get("component") == "top"
                         and box.material == "TIM")
    base_tim = next(box for box in base.boxes
                    if box.tags.get("component") == "top"
                    and box.material == "TIM")
    assert corrected_tim.z0 == pytest.approx(base_tim.z0 - 55e-6)
    assert corrected_tim.z1 - corrected_tim.z0 == pytest.approx(200e-6)
    assert max(box.z1 for box in corrected.boxes) == pytest.approx(
        max(box.z1 for box in base.boxes) - 55e-6)

    top_contact_area = 0.0
    for box in corrected.boxes:
        if not str(box.tags.get("component", "")).startswith(
                "memory_column:hbm_"):
            continue
        if abs(box.z1 - corrected_tim.z0) > 1e-12:
            continue
        overlap_x = max(0.0, min(box.x1, corrected_tim.x1)
                        - max(box.x0, corrected_tim.x0))
        overlap_y = max(0.0, min(box.y1, corrected_tim.y1)
                        - max(box.y0, corrected_tim.y0))
        top_contact_area += overlap_x * overlap_y
    assert top_contact_area == pytest.approx(484e-6)


@pytest.mark.parametrize("layout", ["2x2", "2x1"])
def test_nobase_hbm_top_has_zero_Rpp_operator_edges_to_tim(layout):
    cfg = load_config(CASES[(layout, 414)])
    scene = build_scene(cfg)
    grid = build_global_grid(scene.boxes, cfg.discretization.max_cell_size)
    cells = generate_cells(scene.boxes, grid)
    edges = build_adjacency(cells, grid)
    table = build_conductance_table(
        cells, edges, cfg.materials, cfg.thermal_conductance)
    cell_by_id = {cell.id: cell for cell in cells}
    interface_indices = []
    for index, edge in enumerate(edges):
        a = cell_by_id[edge.cell_a]
        b = cell_by_id[edge.cell_b]
        pair = (a, b)
        if edge.axis != "z":
            continue
        if any(cell.material == "TIM" for cell in pair) and any(
                str(cell.component).startswith("memory_column:hbm_")
                for cell in pair):
            interface_indices.append(index)
    assert interface_indices
    assert sum(edges[i].face_area for i in interface_indices) == pytest.approx(
        484e-6)
    assert all(table.interface_areal_resistance_m2K_W[i] == 0.0
               for i in interface_indices)
    assert sum(table.conductance_W_K[i] for i in interface_indices) > 0.0


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
