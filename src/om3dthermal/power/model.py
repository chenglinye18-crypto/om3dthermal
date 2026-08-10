"""Config-driven architecture and workload resolution for memory power."""

from __future__ import annotations

from pathlib import Path

from .backends import DreamRAMBackend, OperationTableCellModel
from .cell_model import (
    DeviceOperationEnergies,
    MissingCellReplacementError,
    apply_component_replacements,
)
from .config import MemoryPowerConfig, find_project_root, load_power_config
from .result import BackendEnergyResult, EnergyDecomposition, MemoryPowerResult


def _resolve_transport(
        label: str, source: str, constant: float | None,
        dreamram: EnergyDecomposition | None) -> float:
    if source == "none":
        return 0.0
    if source == "constant":
        if constant is None:
            raise ValueError(
                f"architecture {label} uses source=constant but "
                "energy_pj_per_bit is unresolved")
        return constant
    if source == "dreamram":
        if dreamram is None:
            raise ValueError(
                f"architecture {label} requests DreamRAM energy from a "
                "non-DreamRAM backend")
        return float(getattr(dreamram, label))
    raise ValueError(f"unsupported architecture source {source!r}")


def _memory_read_energy(
        backend: BackendEnergyResult,
        config: MemoryPowerConfig,
        ) -> tuple[
            float, EnergyDecomposition, dict[str, float], dict[str, float]]:
    native = backend.read_default
    if native is None:
        raise ValueError("DreamRAM structural backend did not provide read energy")
    cell_model = config.memory.cell_model
    if cell_model.type == "dreamram_native":
        return (
            native.memory_internal, native,
            dict(backend.native_internal_components), {})

    replacement = cell_model.replacement
    if replacement is None:
        raise MissingCellReplacementError("cell replacement definition is absent")
    if replacement.mapping_status != "validated":
        if cell_model.type == "operation_table":
            raise MissingCellReplacementError(
                "IGZO cell energy exists but has not been mapped to a "
                "validated DreamRAM replacement boundary")
        raise MissingCellReplacementError(
            "cell replacement mapping has not been validated")
    resolved = apply_component_replacements(
        backend.native_internal_components,
        required_components=replacement.components,
        replacement_components=replacement.component_energy_pj_per_bit,
    )
    modified = EnergyDecomposition(
        memory_internal=resolved.memory_internal_pj_bit,
        vertical=native.vertical,
        base_route=native.base_route,
        interface=native.interface,
    )
    return (
        resolved.memory_internal_pj_bit, modified,
        resolved.native_components, resolved.replacement_components)


def _refresh_power(
        device: DeviceOperationEnergies | None,
        config: MemoryPowerConfig) -> float:
    if not config.power.refresh.enabled:
        return 0.0
    if device is None:
        raise ValueError("refresh enabled but cell model refresh is unsupported")
    if config.workload.stored_bits is None:
        raise ValueError("refresh enabled but workload.stored_bits is unresolved")
    if device.retention_s is None:
        raise ValueError("refresh enabled but cell_model.retention_s is unresolved")
    probability = config.workload.refresh_data
    if probability is None:
        raise ValueError("refresh enabled but workload.refresh_data is unresolved")
    energy = device.weighted_refresh(p0=probability.p0, p1=probability.p1)
    return config.workload.stored_bits * energy * 1e-12 / device.retention_s


def _background_power(
        device: DeviceOperationEnergies | None,
        config: MemoryPowerConfig) -> float:
    if not config.power.background.enabled:
        return 0.0
    if (device is None or device.background_type is None
            or device.background_value_W is None):
        raise ValueError("background enabled but cell model background is unresolved")
    if device.background_type == "total":
        return device.background_value_W
    if device.background_type == "per_row":
        if config.workload.active_rows is None:
            raise ValueError("per_row background requires workload.active_rows")
        count = config.workload.active_rows
    elif device.background_type == "per_bit":
        if config.workload.stored_bits is None:
            raise ValueError("per_bit background requires workload.stored_bits")
        count = config.workload.stored_bits
    elif device.background_type == "per_die":
        if config.architecture.dies is None:
            raise ValueError("per_die background requires architecture.dies")
        count = config.architecture.dies
    else:
        raise ValueError(f"unsupported background type {device.background_type!r}")
    return count * device.background_value_W


def calculate_memory_power(
        config: MemoryPowerConfig, *, project_root: Path) -> MemoryPowerResult:
    backend = DreamRAMBackend(project_root).calculate(config)
    device = (
        OperationTableCellModel().calculate(config)
        if config.memory.cell_model.type == "operation_table" else None)

    (memory_internal, dreamram_decomposition,
     native_components, replacement_components) = _memory_read_energy(
         backend, config)
    vertical = _resolve_transport(
        "vertical", config.architecture.vertical.source,
        config.architecture.vertical.energy_pj_per_bit,
        dreamram_decomposition)
    base_route = _resolve_transport(
        "base_route", config.architecture.base_route.source,
        config.architecture.base_route.energy_pj_per_bit,
        dreamram_decomposition)
    interface = _resolve_transport(
        "interface", config.architecture.interface.source,
        config.architecture.interface.energy_pj_per_bit,
        dreamram_decomposition)
    read_total = memory_internal + vertical + base_route + interface

    if config.workload.write_bandwidth_gbps > 0:
        raise ValueError(
            "write bandwidth is nonzero but a validated structural write "
            "replacement boundary is unavailable")

    # Gbit/s * pJ/bit = 1e-3 W.
    read_W = config.workload.read_bandwidth_gbps * read_total * 1e-3
    write_W = 0.0
    access_W = read_W + write_W
    refresh_W = _refresh_power(device, config)
    background_W = _background_power(device, config)
    logic_W = config.architecture.logic_background_w
    total_W = None if logic_W is None else (
        access_W + refresh_W + background_W + logic_W)

    return MemoryPowerResult(
        technology=backend.technology,
        backend=backend.backend,
        architecture=config.architecture.name,
        E_memory_internal_pj_bit=memory_internal,
        E_vertical_pj_bit=vertical,
        E_base_route_pj_bit=base_route,
        E_interface_pj_bit=interface,
        E_access_total_pj_bit=read_total,
        P_read_W=read_W,
        P_write_W=write_W,
        P_access_W=access_W,
        P_refresh_W=refresh_W,
        P_memory_background_W=background_W,
        P_logic_background_W=logic_W,
        P_total_W=total_W,
        diagnostics={
            **backend.metadata,
            "cell_model": config.memory.cell_model.type,
            "native_components_pj_bit": native_components,
            "replacement_components_pj_bit": replacement_components,
        },
    )


def run_memory_power(config_path: str | Path) -> MemoryPowerResult:
    path = Path(config_path).resolve()
    return calculate_memory_power(
        load_power_config(path), project_root=find_project_root(path))
