"""Unit handling. All public parsers return SI values."""
from __future__ import annotations

import math
from typing import Any

import pint

ureg = pint.UnitRegistry()


def parse_length(value: Any) -> float:
    """Parse a length and return metres; bare numeric values are already SI."""
    if isinstance(value, bool):
        raise TypeError("boolean is not a length")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            quantity = ureg.Quantity(value)
            return float(quantity.to("m").magnitude)
        except (pint.PintError, ValueError) as exc:
            raise ValueError(f"invalid length {value!r}") from exc
    raise TypeError(f"length must be a number or unit string, got {type(value).__name__}")


def format_length(value_m: float) -> str:
    """Human-readable length for summaries."""
    magnitude = abs(value_m)
    if magnitude and magnitude < 1e-3:
        return f"{value_m * 1e6:.6g} um"
    return f"{value_m * 1e3:.6g} mm"


def parse_areal_thermal_resistance(value: Any) -> float:
    """Parse an areal thermal resistance and return SI m^2*K/W.

    Bare numeric values are interpreted as SI m^2*K/W. Strings are
    routed through Pint and must reduce to a unit dimensionally
    equivalent to ``m^2*K/W`` (i.e. ``[length]^2 * [temperature] /
    [power]``).

    The function rejects:

    - negative, NaN or infinite values (must be finite and
      non-negative);
    - non-``m^2*K/W`` units (e.g. ``W/(m*K)`` conductivity, ``m``
      length, ``s`` time).
    """
    if isinstance(value, bool):
        raise TypeError("boolean is not an areal thermal resistance")
    if isinstance(value, (int, float)):
        result = float(value)
    elif isinstance(value, str):
        try:
            quantity = ureg.Quantity(value)
        except (pint.PintError, ValueError) as exc:
            raise ValueError(
                f"invalid areal thermal resistance {value!r}") from exc
        try:
            converted = quantity.to("m^2 * K / W")
        except pint.DimensionalityError as exc:
            raise ValueError(
                f"areal thermal resistance must have units of m^2*K/W, "
                f"got {value!r}") from exc
        result = float(converted.magnitude)
    else:
        raise TypeError(
            "areal thermal resistance must be a number or unit string, "
            f"got {type(value).__name__}")
    if math.isnan(result) or math.isinf(result):
        raise ValueError(
            f"areal thermal resistance must be finite, got {value!r}")
    if result < 0:
        raise ValueError(
            f"areal thermal resistance must be non-negative, got {value!r}")
    return result
