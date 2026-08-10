"""Memory-technology backend interface and implementations."""

from __future__ import annotations

from typing import Protocol

from ..config import MemoryPowerConfig
from ..result import BackendEnergyResult


class MemoryTechnologyBackend(Protocol):
    def calculate(self, config: MemoryPowerConfig) -> BackendEnergyResult:
        """Resolve technology-level operation energies from configuration."""


from .dreamram import DreamRAMBackend  # noqa: E402
from .operation_table import OperationTableBackend  # noqa: E402

__all__ = [
    "DreamRAMBackend",
    "MemoryTechnologyBackend",
    "OperationTableBackend",
]
