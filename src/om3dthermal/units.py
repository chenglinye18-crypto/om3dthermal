"""Unit handling. All public parsers return SI values."""

from __future__ import annotations

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
