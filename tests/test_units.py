import pytest

from om3dthermal.units import parse_length


@pytest.mark.parametrize(("text", "expected"), [
    ("65 mm", 0.065),
    ("41 um", 41e-6),
    ("1 meter", 1.0),
    (2.5, 2.5),
])
def test_parse_length_to_si(text, expected):
    assert parse_length(text) == pytest.approx(expected)


def test_reject_non_length_quantity():
    with pytest.raises(ValueError):
        parse_length("3 seconds")


def test_reject_boolean():
    with pytest.raises(TypeError):
        parse_length(True)
