"""Power source mapping.

A ``PowerSource`` from the YAML selects a set of ``ThermalCell``
nodes; this stage distributes the source's ``total_power`` across
the selected cells in proportion to their volumes (the only
supported distribution is ``uniform_volume``). Multiple sources
covering the same cell are additive — the per-cell power ends up
as the sum of the per-source contributions.

Conservation is a hard invariant: the sum of the per-source
``Σ P_i`` must equal the source's configured total within
floating-point tolerance, and the sum across sources must equal
the sum of configured totals. The function refuses to operate on
empty selections or zero-volume selections.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..config import PowerSourceConfig, ThermalPowerSourcesConfig
from ..discretization.models import ThermalCell


def _cell_matches(cell: ThermalCell, source: PowerSourceConfig) -> bool:
    sel = source.selector
    if sel.component is not None and cell.component != sel.component:
        return False
    if sel.material is not None and cell.material != sel.material:
        return False
    if sel.layer is not None and cell.parent_box_name != sel.layer:
        return False
    if sel.tags:
        for k, v in sel.tags.items():
            if cell.tags.get(k) != v:
                return False
    return True


@dataclass
class PowerSourceResult:
    """Per-source diagnostics.

    ``selected_cell_count`` and ``selected_volume`` are
    bookkeeping for the summary; the actual per-cell power is
    accumulated into :class:`PowerVector.power_W` (caller-side).
    """

    name: str
    total_power_W: float
    selected_cell_count: int
    selected_volume_m3: float
    distributed_power_W: float


@dataclass
class PowerVector:
    """The right-hand-side power vector plus per-source diagnostics."""

    power_W: np.ndarray                     # float64, length N_cells
    source_count: int
    total_power_W: float
    power_by_source: dict[str, float]
    selected_cell_count_by_source: dict[str, int]
    selected_volume_by_source: dict[str, float]

    @property
    def cell_count(self) -> int:
        return int(self.power_W.shape[0])


def map_power_sources(
    cells: Sequence[ThermalCell],
    config: ThermalPowerSourcesConfig,
) -> PowerVector:
    """Map every configured source onto the cells and return the
    per-cell power vector.

    A source that matches zero cells or whose selected cells sum to
    zero volume is a hard error: it would silently drop the
    configured power. The total ``power_W`` sum across all sources
    must equal the sum of configured totals within ``1e-9``
    relative tolerance.
    """
    if not cells:
        raise ValueError("map_power_sources: no cells to distribute power on")
    power = np.zeros(len(cells), dtype=np.float64)
    power_by_source: dict[str, float] = {}
    count_by_source: dict[str, int] = {}
    volume_by_source: dict[str, float] = {}
    for source in config.sources:
        if source.distribution != "uniform_volume":
            raise NotImplementedError(
                f"only 'uniform_volume' power distribution is implemented "
                f"in this stage, got {source.distribution!r} for source "
                f"{source.name!r}")
        selected = [
            (idx, cell) for idx, cell in enumerate(cells)
            if _cell_matches(cell, source)
        ]
        if not selected:
            raise ValueError(
                f"power source {source.name!r} selected no cells "
                f"(selector: {source.selector.model_dump()})")
        total_volume = sum(cell.volume for _, cell in selected)
        if total_volume <= 0:
            raise ValueError(
                f"power source {source.name!r} selected cells with zero "
                f"total volume; uniform_volume cannot distribute")
        distributed = 0.0
        for idx, cell in selected:
            share = source.total_power * cell.volume / total_volume
            power[idx] += share
            distributed += share
        # Floating-point conservation check.
        if not abs(distributed - source.total_power) <= max(
                1e-9 * abs(source.total_power), 1e-18):
            raise ValueError(
                f"power source {source.name!r} failed conservation: "
                f"configured {source.total_power} W but distributed "
                f"{distributed} W")
        power_by_source[source.name] = distributed
        count_by_source[source.name] = len(selected)
        volume_by_source[source.name] = total_volume
    return PowerVector(
        power_W=power,
        source_count=len(config.sources),
        total_power_W=float(power.sum()),
        power_by_source=power_by_source,
        selected_cell_count_by_source=count_by_source,
        selected_volume_by_source=volume_by_source,
    )
