"""Config-driven architecture and workload resolution for memory power."""

from __future__ import annotations

from pathlib import Path

from .backends import DreamRAMBackend, OperationTableBackend
from .config import MemoryPowerConfig, find_project_root, load_power_config
from .result import BackendEnergyResult, EnergyDecomposition, MemoryPowerResult


def _weighted_binary(p0: float, p1: float, e0: float, e1: float) -> float:
    return p0 * e0 + p1 * e1


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
        backend: BackendEnergyResult, config: MemoryPowerConfig) -> tuple[float, EnergyDecomposition | None]:
    if backend.read_default is not None:
        return backend.read_default.memory_internal, backend.read_default
    if backend.read_0 is None or backend.read_1 is None:
        raise ValueError("selected backend does not provide read energy")
    probability = config.workload.read_data
    if probability is None:
        raise ValueError("state-dependent read energy requires workload.read_data")
    return _weighted_binary(
        probability.p0, probability.p1,
        backend.read_0, backend.read_1), None


def _memory_write_energy(
        backend: BackendEnergyResult, config: MemoryPowerConfig) -> float | None:
    energies = (
        backend.write_00, backend.write_01,
        backend.write_10, backend.write_11,
    )
    if any(value is None for value in energies):
        return None
    probability = config.workload.write_transition
    if probability is None:
        if config.workload.write_bandwidth_gbps == 0:
            return 0.0
        raise ValueError(
            "state-dependent write energy requires workload.write_transition")
    return (
        probability.p00 * float(backend.write_00)
        + probability.p01 * float(backend.write_01)
        + probability.p10 * float(backend.write_10)
        + probability.p11 * float(backend.write_11)
    )


def _refresh_power(
        backend: BackendEnergyResult, config: MemoryPowerConfig) -> float:
    if not config.power.refresh.enabled:
        return 0.0
    if backend.refresh_0 is None or backend.refresh_1 is None:
        raise ValueError("refresh enabled but backend refresh energy is unsupported")
    if config.workload.stored_bits is None:
        raise ValueError("refresh enabled but workload.stored_bits is unresolved")
    if backend.retention_s is None:
        raise ValueError("refresh enabled but memory.retention_s is unresolved")
    probability = config.workload.refresh_data
    if probability is None:
        raise ValueError("refresh enabled but workload.refresh_data is unresolved")
    energy = _weighted_binary(
        probability.p0, probability.p1,
        backend.refresh_0, backend.refresh_1)
    return config.workload.stored_bits * energy * 1e-12 / backend.retention_s


def _background_power(
        backend: BackendEnergyResult, config: MemoryPowerConfig) -> float:
    if not config.power.background.enabled:
        return 0.0
    if backend.background_type is None or backend.background_value_W is None:
        raise ValueError("background enabled but backend background is unresolved")
    if backend.background_type == "total":
        return backend.background_value_W
    if backend.background_type == "per_row":
        if config.workload.active_rows is None:
            raise ValueError("per_row background requires workload.active_rows")
        count = config.workload.active_rows
    elif backend.background_type == "per_bit":
        if config.workload.stored_bits is None:
            raise ValueError("per_bit background requires workload.stored_bits")
        count = config.workload.stored_bits
    elif backend.background_type == "per_die":
        if config.architecture.dies is None:
            raise ValueError("per_die background requires architecture.dies")
        count = config.architecture.dies
    else:
        raise ValueError(f"unsupported background type {backend.background_type!r}")
    return count * backend.background_value_W


def calculate_memory_power(
        config: MemoryPowerConfig, *, project_root: Path) -> MemoryPowerResult:
    if config.memory.backend == "dreamram":
        backend = DreamRAMBackend(project_root).calculate(config)
    else:
        backend = OperationTableBackend().calculate(config)

    memory_internal, dreamram_decomposition = _memory_read_energy(
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

    write_memory = _memory_write_energy(backend, config)
    if config.workload.write_bandwidth_gbps > 0 and write_memory is None:
        raise ValueError("write bandwidth is nonzero but backend write is unsupported")
    write_total = 0.0 if write_memory is None else (
        write_memory + vertical + base_route + interface)

    # Gbit/s * pJ/bit = 1e-3 W.
    read_W = config.workload.read_bandwidth_gbps * read_total * 1e-3
    write_W = config.workload.write_bandwidth_gbps * write_total * 1e-3
    access_W = read_W + write_W
    refresh_W = _refresh_power(backend, config)
    background_W = _background_power(backend, config)
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
        diagnostics=backend.metadata,
    )


def run_memory_power(config_path: str | Path) -> MemoryPowerResult:
    path = Path(config_path).resolve()
    return calculate_memory_power(
        load_power_config(path), project_root=find_project_root(path))
