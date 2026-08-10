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
import math
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
        if source.name in power_by_source:
            raise ValueError(
                f"duplicate power source name {source.name!r} would make "
                "power diagnostics ambiguous")
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


def build_power_breakdown(
    power: PowerVector,
    config: ThermalPowerSourcesConfig,
) -> dict:
    """Build component-aware accounting from mapped source metadata.

    The function never redistributes power. It only totals the already mapped
    source contributions and raises if the accounting loses or duplicates any
    configured input.
    """
    source_total_W = sum(power.power_by_source.values())
    if not math.isclose(
            source_total_W, power.total_power_W, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(
            "power breakdown does not match mapped total: "
            f"{source_total_W} W != {power.total_power_W} W")

    gpu_total_W = 0.0
    hbm_total_W = 0.0
    model_names: set[str] = set()
    per_stack_raw: dict[str, dict] = {}
    for source in config.sources:
        distributed_W = power.power_by_source[source.name]
        metadata = source.metadata
        component_class = metadata.get("component_class")
        source_model = metadata.get("power_model")
        if source_model:
            model_names.add(str(source_model))
        if component_class == "gpu":
            gpu_total_W += distributed_W
            continue
        stack_name = metadata.get("stack")
        if stack_name is None:
            if source.name.lower().startswith("hbm"):
                hbm_total_W += distributed_W
            continue
        hbm_total_W += distributed_W
        stack = per_stack_raw.setdefault(str(stack_name), {
            "logic_phy_W": 0.0,
            "logic_tsv_W": 0.0,
            "dram_by_die": {},
        })
        functional_component = metadata.get("functional_component")
        if component_class == "logic":
            key = f"logic_{functional_component}_W"
            if key not in {"logic_phy_W", "logic_tsv_W"}:
                raise ValueError(
                    f"unknown Son23 logic component {functional_component!r}")
            stack[key] += distributed_W
        elif (component_class == "dram"
              and str(source_model).startswith("son23")):
            die_index = int(metadata["dram_die_index"])
            die = stack["dram_by_die"].setdefault(
                die_index, {"bank_W": 0.0, "tsv_W": 0.0})
            if functional_component not in {"bank", "tsv"}:
                raise ValueError(
                    f"unknown Son23 DRAM component {functional_component!r}")
            die[f"{functional_component}_W"] += distributed_W

    son23_models = sorted(
        name for name in model_names if name.startswith("son23"))
    son23 = bool(son23_models)
    if not son23:
        return {
            "power_model": "uniform",
            "component_split_available": False,
            "whole_package": {
                "hbm_total_W": hbm_total_W,
                "gpu_total_W": gpu_total_W,
                "package_total_W": source_total_W,
            },
            "accounting": {
                "mapped_total_W": power.total_power_W,
                "source_total_W": source_total_W,
                "missing_or_duplicated_power_W":
                    power.total_power_W - source_total_W,
            },
        }

    per_stack: dict[str, dict] = {}
    hbm_logic_total_W = 0.0
    hbm_dram_total_W = 0.0
    for stack_name, raw in per_stack_raw.items():
        dram_by_die = raw.pop("dram_by_die")
        if sorted(dram_by_die) != list(range(1, 13)):
            raise ValueError(
                f"Son23 stack {stack_name!r} does not account for DRAM dies 1..12")
        bank_values = [dram_by_die[i]["bank_W"] for i in range(1, 13)]
        tsv_values = [dram_by_die[i]["tsv_W"] for i in range(1, 13)]
        if not all(math.isclose(v, bank_values[0], rel_tol=1e-12, abs_tol=1e-12)
                   for v in bank_values):
            raise ValueError(f"Son23 stack {stack_name!r} has unequal bank power")
        if not all(math.isclose(v, tsv_values[0], rel_tol=1e-12, abs_tol=1e-12)
                   for v in tsv_values):
            raise ValueError(f"Son23 stack {stack_name!r} has unequal DRAM TSV power")
        logic_total_W = raw["logic_phy_W"] + raw["logic_tsv_W"]
        dram_total_W_per_die = bank_values[0] + tsv_values[0]
        dram_total_W = sum(bank_values) + sum(tsv_values)
        stack_total_W = logic_total_W + dram_total_W
        per_stack[stack_name] = {
            "logic_phy_W": raw["logic_phy_W"],
            "logic_tsv_W": raw["logic_tsv_W"],
            "logic_total_W": logic_total_W,
            "dram_bank_W_per_die": bank_values[0],
            "dram_tsv_W_per_die": tsv_values[0],
            "dram_total_W_per_die": dram_total_W_per_die,
            "dram_total_W": dram_total_W,
            "stack_total_W": stack_total_W,
        }
        hbm_logic_total_W += logic_total_W
        hbm_dram_total_W += dram_total_W

    accounted_hbm_W = hbm_logic_total_W + hbm_dram_total_W
    if not math.isclose(
            accounted_hbm_W, hbm_total_W, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(
            "Son23 component accounting does not match HBM source total: "
            f"{accounted_hbm_W} W != {hbm_total_W} W")
    return {
        "power_model": son23_models[0],
        "component_split_available": True,
        "per_stack": per_stack,
        "whole_package": {
            "hbm_logic_total_W": hbm_logic_total_W,
            "hbm_dram_total_W": hbm_dram_total_W,
            "hbm_total_W": hbm_total_W,
            "gpu_total_W": gpu_total_W,
            "package_total_W": source_total_W,
        },
        "accounting": {
            "mapped_total_W": power.total_power_W,
            "source_total_W": source_total_W,
            "missing_or_duplicated_power_W": power.total_power_W - source_total_W,
        },
    }
