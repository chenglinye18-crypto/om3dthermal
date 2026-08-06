"""Global cut-plane generation and grid voxel construction.

The discretiser never quantises coordinates. Instead it builds three
strictly increasing 1-D cut arrays per axis by unioning:

1. every box boundary (x0/x1/y0/y1/z0/z1) — these are exact, no
   floating-point error is introduced;
2. for each box interior, a uniform subdivision produced by
   :func:`subdivide_interval`, so that the resulting cell along that
   axis never exceeds ``max_cell_size``.

The cut arrays are deduplicated with a length tolerance so a real
material boundary that happens to coincide (up to ``_LENGTH_TOL``)
with a subdivision plane is not duplicated.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..geometry.horizontal_columns import _LENGTH_TOL


def subdivide_interval(a: float, b: float, max_size: float) -> list[float]:
    """Return the strictly increasing cut list that subdivides ``[a, b]``
    into intervals of length at most ``max_size``.

    The first cut is exactly ``a`` and the last cut is exactly ``b``; all
    interior cuts are uniform. The function refuses to operate on
    degenerate or inverted intervals and on non-positive ``max_size``.

    The number of segments is ``ceil(length / max_size)``, but if the
    ratio is essentially an integer (within 1e-9 of round(ratio)) we
    snap to the rounded value so that unit-parsing noise from Pint
    (``100 um`` may serialise back to ``9.999999999999999e-05``) does
    not introduce an extra ghost cell.
    """
    if not (b > a + _LENGTH_TOL):
        raise ValueError(f"subdivide_interval: degenerate interval [{a}, {b}]")
    if max_size <= 0:
        raise ValueError(f"subdivide_interval: max_size must be > 0, got {max_size}")
    length = b - a
    ratio = length / max_size
    rounded = round(ratio)
    if rounded >= 1 and abs(ratio - rounded) < 1e-9:
        n = rounded
    else:
        n = max(1, math.ceil(ratio))
    if n == 1:
        return [a, b]
    step = length / n
    return [a + i * step for i in range(n + 1)]


def _merge_tolerance(values: list[float], tol: float = _LENGTH_TOL) -> list[float]:
    """Deduplicate a list of real numbers using a length tolerance so
    that two values within ``tol`` of each other are treated as one.

    The returned list is strictly increasing and preserves the smallest
    representative of each merged group.
    """
    if not values:
        return []
    sorted_values = sorted(values)
    out = [sorted_values[0]]
    for value in sorted_values[1:]:
        if value - out[-1] > tol:
            out.append(value)
    return out


@dataclass(frozen=True)
class GlobalGrid:
    """The three strictly-increasing per-axis cut arrays and the derived
    integer-index bounds.

    Cell ``(ix, iy, iz)`` spans ``[x_cuts[ix], x_cuts[ix+1]]`` in x, etc.
    """

    x_cuts: tuple[float, ...]
    y_cuts: tuple[float, ...]
    z_cuts: tuple[float, ...]

    @property
    def nx(self) -> int:
        return len(self.x_cuts) - 1

    @property
    def ny(self) -> int:
        return len(self.y_cuts) - 1

    @property
    def nz(self) -> int:
        return len(self.z_cuts) - 1

    def x_range(self, ix: int) -> tuple[float, float]:
        return self.x_cuts[ix], self.x_cuts[ix + 1]

    def y_range(self, iy: int) -> tuple[float, float]:
        return self.y_cuts[iy], self.y_cuts[iy + 1]

    def z_range(self, iz: int) -> tuple[float, float]:
        return self.z_cuts[iz], self.z_cuts[iz + 1]

    def x0(self) -> float:
        return self.x_cuts[0]

    def x1(self) -> float:
        return self.x_cuts[-1]

    def y0(self) -> float:
        return self.y_cuts[0]

    def y1(self) -> float:
        return self.y_cuts[-1]

    def z0(self) -> float:
        return self.z_cuts[0]

    def z1(self) -> float:
        return self.z_cuts[-1]


def build_global_grid(boxes, max_cell_size) -> GlobalGrid:
    """Union every box boundary with per-box uniform subdivisions and
    return the resulting strictly-increasing cut arrays.

    ``boxes`` is any iterable of objects with attributes ``x0``, ``x1``,
    ``y0``, ``y1``, ``z0``, ``z1`` (e.g. ``AxisAlignedBox``). ``max_cell_size``
    must be an object with ``x``, ``y``, ``z`` SI-metre values.
    """
    x_cuts: list[float] = []
    y_cuts: list[float] = []
    z_cuts: list[float] = []
    for box in boxes:
        x_cuts.append(box.x0); x_cuts.append(box.x1)
        y_cuts.append(box.y0); y_cuts.append(box.y1)
        z_cuts.append(box.z0); z_cuts.append(box.z1)
        x_cuts.extend(subdivide_interval(box.x0, box.x1, max_cell_size.x)[1:-1])
        y_cuts.extend(subdivide_interval(box.y0, box.y1, max_cell_size.y)[1:-1])
        z_cuts.extend(subdivide_interval(box.z0, box.z1, max_cell_size.z)[1:-1])
    return GlobalGrid(
        x_cuts=tuple(_merge_tolerance(x_cuts)),
        y_cuts=tuple(_merge_tolerance(y_cuts)),
        z_cuts=tuple(_merge_tolerance(z_cuts)),
    )
