from pathlib import Path

import pytest
from pydantic import ValidationError

from om3dthermal.config import RepeatBlock, StackTemplate, load_config

CONFIG = Path(__file__).parents[1] / "configs" / "hbm_on_gpu_12hi.yaml"


def test_hbm_total_thickness_is_755_um():
    stack = load_config(CONFIG).stack_templates["hbm_12hi"]
    assert stack.total_thickness == pytest.approx(755e-6)


def test_hbm_has_twelve_dram_si_layers():
    expanded = load_config(CONFIG).stack_templates["hbm_12hi"].expand()
    dram = [layer for layer in expanded if layer.tags.get("role") == "dram_si"]
    assert len(dram) == 12
    assert sum(layer.name.startswith("dram_si_") for layer in dram) == 11
    assert sum(layer.name == "top_dram_si" for layer in dram) == 1


def test_repeat_names_are_unique_and_numbered():
    expanded = load_config(CONFIG).stack_templates["hbm_12hi"].expand()
    names = [layer.name for layer in expanded]
    assert len(names) == len(set(names))
    assert "dram_si_01" in names
    assert "dram_si_11" in names


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
