"""Targeted geometry/material/power checks for formal M3D-v1 cases."""

from pathlib import Path

import pytest

from om3dthermal.cli import build_scene
from om3dthermal.config import load_config
from om3dthermal.geometry.orthogonal_hbm import ORTHOGONAL_DIE_ROTATION


ROOT = Path(__file__).parents[1]
CASES = {
    414: ROOT / "configs" / "exp_orth_m3d8_g414_bw39p2_all1read.yaml",
    300: ROOT / "configs" / "exp_orth_m3d8_g300_bw39p2_all1read.yaml",
}


@pytest.mark.parametrize("gpu", [414, 300])
def test_formal_config_parses_with_closed_slab_and_materials(gpu):
    cfg = load_config(CASES[gpu])
    die = cfg.orthogonal_hbm.memory_die
    assert die.count == 98
    assert die.thickness == pytest.approx(300e-6)
    assert [layer.role for layer in die.layers] == [
        "si_substrate", "feol", "m3d_bitcell_stack",
        "beol_interconnect", "daa"]
    assert die.layers[2].thickness == pytest.approx(2.304e-6)
    expected_k = {
        "M3D_Si": 140.0,
        "M3D_FEOL": 7.9,
        "M3D_Bitcell": 0.85,
        "M3D_BEOL": 0.85,
        "M3D_DAA": 0.2,
    }
    for material, k in expected_k.items():
        assert cfg.materials[material].k_local == pytest.approx((k, k, k))


@pytest.mark.parametrize("gpu", [414, 300])
def test_generates_98_slabs_with_one_homogenized_bitcell_stack(gpu):
    cfg = load_config(CASES[gpu])
    scene = build_scene(cfg)
    components = sorted({
        box.tags["component"] for box in scene.boxes
        if str(box.tags.get("component", "")).startswith(
            "orthogonal_hbm:die_")
    })
    assert len(components) == 98
    for component in components:
        boxes = scene.filter(component=component)
        assert len(boxes) == 5
        assert sum(box.x1 - box.x0 for box in boxes) == pytest.approx(300e-6)
        bitcells = [box for box in boxes
                    if box.tags.get("role") == "m3d_bitcell_stack"]
        assert len(bitcells) == 1
        assert bitcells[0].x1 - bitcells[0].x0 == pytest.approx(2.304e-6)
        assert all(box.rotation == ORTHOGONAL_DIE_ROTATION for box in boxes)
        assert len([box for box in boxes if box.tags.get("role") == "daa"]) == 1


@pytest.mark.parametrize("gpu", [414, 300])
def test_operation_energy_power_maps_only_to_homogenized_bitcell_stack(gpu):
    cfg = load_config(CASES[gpu])
    scene = build_scene(cfg)
    sources = cfg.thermal_power_sources.sources
    gpu_sources = [source for source in sources
                   if source.metadata.get("component_class") == "gpu"]
    memory = [source for source in sources
              if source.metadata.get("power_model") == "m3d_operation_energy"]
    assert len(gpu_sources) == 1
    assert gpu_sources[0].total_power == pytest.approx(float(gpu))
    assert len(memory) == 98
    assert sum(source.total_power for source in memory) == pytest.approx(14.4256)
    assert all(source.total_power == pytest.approx(14.4256 / 98)
               for source in memory)
    for source in memory:
        assert source.selector.tags == {"role": "m3d_bitcell_stack"}
        selected = [box for box in scene.boxes
                    if box.tags.get("component") == source.selector.component
                    and box.tags.get("role") == "m3d_bitcell_stack"]
        assert len(selected) == 1
    assert sum(source.total_power for source in sources) == pytest.approx(
        gpu + 14.4256)


def test_414_and_300_cases_differ_only_in_gpu_identity_fields():
    high = load_config(CASES[414]).model_dump()
    low = load_config(CASES[300]).model_dump()
    high["name"] = low["name"]
    high["metadata"]["case_id"] = low["metadata"]["case_id"]
    high["thermal_power_sources"]["sources"][0]["total_power"] = 300.0
    assert high == low


@pytest.mark.parametrize("gpu", [414, 300])
def test_summary_bookkeeping_is_present_and_power_is_derived(gpu):
    cfg = load_config(CASES[gpu])
    book = cfg.metadata["architecture_bookkeeping"]
    assert book["cube_footprint_mm"] == [30, 22]
    assert book["slab_count"] == 98
    assert book["bitcell_layers"] == 8
    assert book["bitcell_thermal_geometry"] == "homogenized_8_layer_stack"
    assert book["capacity_density_Mb_mm2"] == pytest.approx(240)
    assert book["capacity_density_MB_mm2"] == pytest.approx(30)
    assert book["upper_bound_capacity_GB"] == pytest.approx(355.74)
    assert book["matched_delivered_bandwidth_Tb_s"] == pytest.approx(39.2)
    assert book["all1_read_energy_pJ_per_bit"] == pytest.approx(0.368)
    assert book["array_read_power_W"] == pytest.approx(14.4256)
    assert "delivered_bandwidth_bit_per_s" in book["memory_power_derivation"]
