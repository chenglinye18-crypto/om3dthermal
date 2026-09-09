
import pytest
from pydantic import ValidationError

from om3dthermal.config import (
    RepeatBlock,
    StackTemplate,
)


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


@pytest.mark.parametrize("count", [1, 4, 8, 12])
def test_stack_repeat_preserves_layers_and_thickness(count):
    stack = StackTemplate.model_validate({"items": [{
        "kind": "repeat", "count": count,
        "layers": [{"name": "die", "material": "Si", "thickness": "2 um"}],
    }]})
    expanded = stack.expand()
    assert len(expanded) == count
    assert len({layer.name for layer in expanded}) == count
    assert stack.total_thickness == pytest.approx(count * 2e-6)
