"""Config-provided operation-energy table backend."""

from __future__ import annotations

from ..config import MemoryPowerConfig
from ..result import BackendEnergyResult


class OperationTableBackend:
    def calculate(self, config: MemoryPowerConfig) -> BackendEnergyResult:
        operations = config.memory.operations
        if operations is None:
            raise ValueError("operation_table backend requires memory.operations")
        background = config.memory.background
        return BackendEnergyResult(
            technology=config.memory.technology,
            backend="operation_table",
            read_0=operations.read_0_pj_per_bit,
            read_1=operations.read_1_pj_per_bit,
            write_00=operations.write_00_pj_per_bit,
            write_01=operations.write_01_pj_per_bit,
            write_10=operations.write_10_pj_per_bit,
            write_11=operations.write_11_pj_per_bit,
            refresh_0=operations.refresh_0_pj_per_bit,
            refresh_1=operations.refresh_1_pj_per_bit,
            background_type=None if background is None else background.type,
            background_value_W=None if background is None else background.value_w,
            retention_s=config.memory.retention_s,
        )
