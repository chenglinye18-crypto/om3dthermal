"""Config-driven architecture and workload resolution for memory power."""

from __future__ import annotations

from pathlib import Path

from .backends import DreamRAMBackend, OperationTableCellModel
from .cell_model import (
    DeviceOperationEnergies,
    MissingCellReplacementError,
    apply_component_replacements,
    apply_operation_primitive_replacement,
)
from .config import MemoryPowerConfig, find_project_root, load_power_config
from .geometry import load_m3d_geometry
from .feol_route import calculate_feol_route
from .m3d_subarray import calculate_m3d_subarray
from .result import BackendEnergyResult, EnergyDecomposition, MemoryPowerResult


class UnresolvedMIVEnergyError(ValueError):
    """MIV topology is resolved but electrical energy inputs are not."""

    def __init__(self, diagnostics: dict[str, object]):
        self.diagnostics = diagnostics
        super().__init__(
            "MIV energy is unresolved/N/A: credible MIV capacitance and "
            "vertical serialization are not configured; TSV capacitance is "
            "not used as a substitute")


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
    if source == "miv_topology":
        if dreamram is None:
            raise ValueError("MIV topology requires DreamRAM structural energy")
        return dreamram.vertical
    raise ValueError(f"unsupported architecture source {source!r}")


def _memory_read_energy(
        backend: BackendEnergyResult,
        config: MemoryPowerConfig,
        device: DeviceOperationEnergies | None,
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
    if replacement.energy_source == "operation_table":
        if device is None:
            raise MissingCellReplacementError(
                "operation-table replacement energy is unavailable")
        probability = config.workload.read_data
        if probability is None:
            raise ValueError(
                "operation-table read replacement requires workload.read_data")
        resolved = apply_operation_primitive_replacement(
            backend.native_internal_components,
            required_components=replacement.components,
            operation_energy_pj_per_bit=device.weighted_read(
                p0=probability.p0, p1=probability.p1),
        )
    else:
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
    m3d_subarray = None
    if config.architecture.m3d_subarray is not None:
        m3d_geometry = load_m3d_geometry(
            project_root, config.architecture.geometry_source)
        m3d_subarray = calculate_m3d_subarray(
            config.architecture.m3d_subarray, m3d_geometry)
    backend = DreamRAMBackend(project_root).calculate(
        config, m3d_subarray=m3d_subarray)
    device = (
        OperationTableCellModel().calculate(config)
        if config.memory.cell_model.type == "operation_table" else None)

    if m3d_subarray is None:
        (memory_internal, dreamram_decomposition,
         native_components, replacement_components) = _memory_read_energy(
             backend, config, device)
    else:
        if device is None:
            raise MissingCellReplacementError(
                "M3D embedded-peripheral topology requires an operation-table "
                "cell primitive")
        probability = config.workload.read_data
        if probability is None:
            raise ValueError(
                "M3D operation-table read requires workload.read_data")
        mat_local = device.weighted_read(
            p0=probability.p0, p1=probability.p1)
        memory_internal = (
            mat_local
            + m3d_subarray.global_control_routing_energy_pj_per_bit
            + m3d_subarray.local_read_routing_energy_pj_per_bit
        )
        native_components = {}
        replacement_components = {
            "zhu_mat_local_operation": mat_local,
            "tang_global_control_routing": (
                m3d_subarray.global_control_routing_energy_pj_per_bit),
            "tang_local_read_routing": (
                m3d_subarray.local_read_routing_energy_pj_per_bit),
        }
        if backend.read_default is None:
            raise ValueError("MIV reference backend did not provide energy")
        dreamram_decomposition = EnergyDecomposition(
            memory_internal=memory_internal,
            vertical=backend.read_default.vertical,
            base_route=0.0,
            interface=0.0,
        )
    if config.architecture.vertical.source == "miv_topology":
        if backend.metadata.get("miv_energy_status") != "resolved":
            raise UnresolvedMIVEnergyError(dict(backend.metadata))
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
    feol_route_result = None
    if config.architecture.feol_route is not None:
        if m3d_subarray is None:
            raise ValueError("FEOL route requires resolved M3D topology")
        feol_route_result = calculate_feol_route(
            config.architecture.feol_route, m3d_subarray)
    feol_route = (
        0.0 if feol_route_result is None
        else feol_route_result.feol_route_energy_pj_per_bit)
    read_total = (
        memory_internal + vertical + feol_route + base_route + interface)

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
        E_feol_route_pj_bit=feol_route,
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
            **({} if m3d_subarray is None else m3d_subarray.as_dict()),
            **({} if feol_route_result is None else feol_route_result.as_dict()),
            "cell_model": config.memory.cell_model.type,
            "operation_energy_provenance": (
                None
                if config.memory.cell_model.operation_energy_provenance is None
                else config.memory.cell_model.operation_energy_provenance.model_dump()
            ),
            "native_components_pj_bit": native_components,
            "replacement_components_pj_bit": replacement_components,
            "dreamram_internal_components_excluded": (
                [] if m3d_subarray is None else [
                    "row", "mwl", "lwl", "bl-act", "bl-pre", "col",
                    "csl", "ldl", "mdl", "bgbus+gbus",
                ]
            ),
            "interface_energy_pj_per_bit": interface,
        },
    )


def run_memory_power(config_path: str | Path) -> MemoryPowerResult:
    path = Path(config_path).resolve()
    return calculate_memory_power(
        load_power_config(path), project_root=find_project_root(path))
