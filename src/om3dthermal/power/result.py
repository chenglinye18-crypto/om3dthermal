"""Typed results shared by memory-technology power backends."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class EnergyDecomposition:
    """Read-access energy in pJ/bit at architecture boundaries."""

    memory_internal: float
    vertical: float
    base_route: float
    interface: float

    @property
    def total(self) -> float:
        return (
            self.memory_internal + self.vertical
            + self.base_route + self.interface
        )


@dataclass(frozen=True)
class BackendEnergyResult:
    """Technology-level operation energies exposed to the common model."""

    technology: str
    backend: str
    read_default: EnergyDecomposition | None = None
    read_0: float | None = None
    read_1: float | None = None
    write_00: float | None = None
    write_01: float | None = None
    write_10: float | None = None
    write_11: float | None = None
    refresh_0: float | None = None
    refresh_1: float | None = None
    background_type: str | None = None
    background_value_W: float | None = None
    retention_s: float | None = None
    native_internal_components: dict[str, float] = field(default_factory=dict)
    replacement_components: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryPowerResult:
    technology: str
    backend: str
    architecture: str
    E_memory_internal_pj_bit: float
    E_vertical_pj_bit: float
    E_base_route_pj_bit: float
    E_interface_pj_bit: float
    E_access_total_pj_bit: float
    P_read_W: float
    P_write_W: float
    P_access_W: float
    P_refresh_W: float | None
    P_memory_background_W: float | None
    P_logic_background_W: float | None
    P_total_W: float | None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def as_dict(self, *, display_na: bool = False) -> dict[str, Any]:
        data = asdict(self)
        if display_na:
            return {key: ("N/A" if value is None else value)
                    for key, value in data.items()}
        return data
