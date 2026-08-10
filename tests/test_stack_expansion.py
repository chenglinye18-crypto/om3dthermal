from pathlib import Path

import pytest
from pydantic import ValidationError

from om3dthermal.config import RepeatBlock, StackTemplate, load_config

CONFIG = Path(__file__).parents[1] / "configs" / "exp_conv_2x2_g414_m160.yaml"


def test_hbm_total_thickness_is_775_um():
    stack = load_config(CONFIG).stack_templates["hbm_12hi"]
    assert stack.total_thickness == pytest.approx(775e-6)


def test_hbm_has_twelve_dram_si_layers():
    expanded = load_config(CONFIG).stack_templates["hbm_12hi"].expand()
    dram = [layer for layer in expanded if layer.tags.get("role") == "dram_si"]
    assert len(dram) == 12
    assert sum(layer.name.startswith("dram_si_") for layer in dram) == 11
    assert sum(layer.name == "top_dram_si" for layer in dram) == 1


def test_hbm_has_twelve_dram_beol_layers():
    expanded = load_config(CONFIG).stack_templates["hbm_12hi"].expand()
    beol = [layer for layer in expanded if layer.tags.get("role") == "dram_beol"]
    assert len(beol) == 12


def test_hbm_has_twelve_hybrid_bonding_layers():
    expanded = load_config(CONFIG).stack_templates["hbm_12hi"].expand()
    hb = [layer for layer in expanded if layer.tags.get("role") == "hybrid_bonding"]
    assert len(hb) == 12


def test_hbm_has_one_base_si_and_one_base_beol():
    expanded = load_config(CONFIG).stack_templates["hbm_12hi"].expand()
    base = [layer for layer in expanded if layer.tags.get("role") == "hbm_base"]
    assert len(base) == 2
    by_name = {layer.name: layer for layer in base}
    assert by_name["hbm_base_si"].thickness == pytest.approx(50e-6)
    assert by_name["hbm_base_beol"].thickness == pytest.approx(5e-6)


def test_hbm_has_one_gpu_hbm_ubump():
    expanded = load_config(CONFIG).stack_templates["hbm_12hi"].expand()
    ubumps = [layer for layer in expanded if layer.tags.get("role") == "gpu_hbm_interface"]
    assert len(ubumps) == 1
    assert ubumps[0].thickness == pytest.approx(40e-6)


def test_hbm_layer_thickness_breakdown():
    expanded = load_config(CONFIG).stack_templates["hbm_12hi"].expand()
    by_role_thickness = {role: [] for role in (
        "dram_si", "dram_beol", "hybrid_bonding", "hbm_base", "gpu_hbm_interface")}
    for layer in expanded:
        role = layer.tags.get("role")
        if role in by_role_thickness:
            by_role_thickness[role].append(layer.thickness)

    assert sorted(by_role_thickness["dram_si"]) == pytest.approx(
        sorted([41e-6] * 11 + [169e-6]))
    assert sorted(by_role_thickness["dram_beol"]) == pytest.approx(sorted([3e-6] * 12))
    assert sorted(by_role_thickness["hybrid_bonding"]) == pytest.approx(
        sorted([2e-6] * 12))
    assert sorted(by_role_thickness["hbm_base"]) == pytest.approx(sorted([5e-6, 50e-6]))
    assert by_role_thickness["gpu_hbm_interface"] == pytest.approx([40e-6])


def test_hbm_layer_order_matches_fig_3b():
    """Per Fig. 3(b): base_si -> HB_01, then each die as HB -> BEOL -> Si,
    then top_die as HB -> BEOL -> Si (with the 169 um Si thickness)."""
    expanded = load_config(CONFIG).stack_templates["hbm_12hi"].expand()
    names = [layer.name for layer in expanded]
    index = {name: i for i, name in enumerate(names)}

    # base_si is immediately followed by hybrid_bonding_01.
    assert names[index["hbm_base_si"] + 1] == "hybrid_bonding_01"

    # For every repeated die N in 1..11: HB_N -> BEOL_N -> Si_N.
    for n in range(1, 12):
        hb, beol, si = f"hybrid_bonding_{n:02d}", f"dram_beol_{n:02d}", f"dram_si_{n:02d}"
        assert names.index(beol) == index[hb] + 1, f"{beol} must follow {hb}"
        assert names.index(si) == index[beol] + 1, f"{si} must follow {beol}"

    # dram_si_NN must connect to the next HB, except dram_si_11 which connects
    # to the top HB.
    for n in range(1, 11):
        next_hb = f"hybrid_bonding_{n + 1:02d}"
        assert names[index[f"dram_si_{n:02d}"] + 1] == next_hb
    assert names[index["dram_si_11"] + 1] == "top_hybrid_bonding"

    # Top die order: top_hybrid_bonding -> top_dram_beol -> top_dram_si.
    assert names[index["top_hybrid_bonding"] + 1] == "top_dram_beol"
    assert names[index["top_dram_beol"] + 1] == "top_dram_si"


def test_repeat_names_are_unique_and_numbered():
    expanded = load_config(CONFIG).stack_templates["hbm_12hi"].expand()
    names = [layer.name for layer in expanded]
    assert len(names) == len(set(names))
    assert "dram_si_01" in names
    assert "dram_si_11" in names
    assert "hybrid_bonding_01" in names
    assert "hybrid_bonding_11" in names
    assert "dram_beol_01" in names
    assert "dram_beol_11" in names


@pytest.mark.parametrize("count", [0, -1, 1.0, 1.5])
def test_repeat_count_must_be_positive_integer(count):
    with pytest.raises(ValidationError):
        RepeatBlock.model_validate({
            "kind": "repeat", "count": count,
            "layers": [{"name": "x", "material": "Si", "thickness": "1 um"}],
        })


def test_layer_thickness_must_be_positive():
    with pytest.raises(ValidationError):
        StackTemplate.model_validate({
            "items": [{"kind": "layer", "name": "bad", "material": "Si", "thickness": "0 um"}]
        })
