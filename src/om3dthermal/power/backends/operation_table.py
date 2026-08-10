"""Config-provided cell/device operation-energy table."""

from __future__ import annotations

from ..config import MemoryPowerConfig
from ..cell_model import DeviceOperationEnergies


class OperationTableCellModel:
    """Expose device operations without treating them as complete memory energy."""

    def calculate(self, config: MemoryPowerConfig) -> DeviceOperationEnergies:
        cell_model = config.memory.cell_model
        operations = cell_model.operations
        if operations is None:
            raise ValueError("operation_table requires cell_model.operations")
        background = cell_model.background
        return DeviceOperationEnergies(
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
            retention_s=cell_model.retention_s,
        )


# Compatibility name for callers of Memory Power v0. It now represents a
# device-operation table rather than a complete memory-technology backend.
OperationTableBackend = OperationTableCellModel
