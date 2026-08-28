"""Config-driven architecture and workload resolution for memory power."""

from __future__ import annotations

from pathlib import Path
import yaml

from .backends import DreamRAMBackend, OperationTableCellModel
from .cell_model import (
    DeviceOperationEnergies,
    MissingCellReplacementError,
    apply_component_replacements,
    apply_operation_primitive_replacement,
)
from .config import (
    MemoryPowerConfig,
    find_project_root,
    load_case_config,
    load_power_config,
)
from .geometry import (
    ResolvedGeometry,
    resolve_case_geometry,
    resolve_legacy_geometry,
)
from .feol_route import calculate_feol_route
from .m3d_subarray import calculate_m3d_subarray
from .physical_latency import calculate_physical_access_latency
from .refresh import calculate_refresh_power
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


def _scaled_m3d_read_energy(
        config: MemoryPowerConfig, device: DeviceOperationEnergies,
        ) -> tuple[float, dict[str, object]]:
    scaling = config.memory.cell_model.size_scaling
    topology = config.architecture.m3d_subarray
    probability = config.workload.read_data
    if scaling is None or topology is None or probability is None:
        raise ValueError(
            "M3D operation energy requires size_scaling, topology, and read_data")
    resolved_rows = topology.subarray.n_rows
    ratio = resolved_rows / scaling.reference_n_rows
    scaled_read_0 = device.read_0 * ratio
    scaled_read_1 = device.read_1 - device.read_0 + scaled_read_0
    weighted = probability.p0 * scaled_read_0 + probability.p1 * scaled_read_1
    return weighted, {
        "zhu_reference_n_rows": scaling.reference_n_rows,
        "zhu_reference_n_cols": scaling.reference_n_cols,
        "zhu_reference_read_0_pj_per_bit": device.read_0,
        "zhu_reference_read_1_pj_per_bit": device.read_1,
        "zhu_reference_energy_provenance": "PAPER_REPORTED",
        "zhu_size_scaling_model": scaling.model,
        "zhu_size_scaling_provenance": scaling.provenance,
        "zhu_resolved_n_rows": resolved_rows,
        "zhu_nrow_scale_ratio": ratio,
        "zhu_scaled_read_0_pj_per_bit": scaled_read_0,
        "zhu_scaled_read_1_pj_per_bit": scaled_read_1,
        "zhu_scaled_weighted_read_pj_per_bit": weighted,
    }


def calculate_memory_power(
        config: MemoryPowerConfig, *, project_root: Path,
        geometry: ResolvedGeometry | None = None) -> MemoryPowerResult:
    if geometry is None:
        source = config.architecture.geometry_source
        if source is None:
            raise ValueError(
                "canonical case requires its resolved geometry object")
        geometry = resolve_legacy_geometry(project_root, source)
    m3d_subarray = None
    m3d_geometry = geometry.m3d
    if config.architecture.m3d_subarray is not None:
        if m3d_geometry is None:
            raise ValueError("M3D power requires resolved M3D geometry")
        m3d_subarray = calculate_m3d_subarray(
            config.architecture.m3d_subarray, m3d_geometry)
    backend = DreamRAMBackend(project_root).calculate(
        config, m3d_subarray=m3d_subarray, geometry=geometry)
    device = (
        OperationTableCellModel().calculate(config)
        if config.memory.cell_model.type == "operation_table" else None)
    zhu_scaling_diagnostics: dict[str, object] = {}

    if m3d_subarray is None:
        (memory_internal, dreamram_decomposition,
         native_components, replacement_components) = _memory_read_energy(
             backend, config, device)
    else:
        if device is None:
            raise MissingCellReplacementError(
                "M3D embedded-peripheral topology requires an operation-table "
                "cell primitive")
        mat_local, zhu_scaling_diagnostics = _scaled_m3d_read_energy(
            config, device)
        memory_internal = (
            mat_local
            + m3d_subarray.global_control_routing_energy_pj_per_bit
        )
        native_components = {}
        replacement_components = {
            "zhu_scaled_local_operation": mat_local,
            "tang_global_control_routing": (
                m3d_subarray.global_control_routing_energy_pj_per_bit),
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
    physical_latency_result = None
    physical_latency_spec = config.architecture.physical_access_latency
    if physical_latency_spec is not None:
        if feol_route_result is None:
            raise ValueError("physical access latency requires FEOL latency")
        raw_miv_lengths = backend.metadata.get("miv_length_per_layer_um")
        raw_miv_delays = backend.metadata.get("miv_delay_per_layer_ns")
        if (not isinstance(raw_miv_lengths, (tuple, list))
                or not isinstance(raw_miv_delays, (tuple, list))):
            raise ValueError("physical access latency requires MIV latency")
        physical_latency_result = calculate_physical_access_latency(
            physical_latency_spec,
            feol_route=feol_route_result,
            miv_length_per_layer_um=tuple(
                float(value) for value in raw_miv_lengths),
            miv_delay_per_layer_ns=tuple(
                float(value) for value in raw_miv_delays),
            miv_status=str(backend.metadata.get("miv_latency_status")),
            miv_parameter_status=str(backend.metadata.get(
                "miv_resistance_parameter_status")),
            miv_provenance=str(backend.metadata.get(
                "miv_resistance_provenance")),
        )
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
    refresh_result = calculate_refresh_power(
        config,
        backend=backend,
        device=device,
        m3d_subarray=m3d_subarray,
        m3d_layer_count=(
            None if m3d_geometry is None else m3d_geometry.layers),
        memory_region_count=geometry.memory_region_count,
    )
    refresh_W = refresh_result.power_W
    background_W = _background_power(device, config)
    logic_W = config.architecture.logic_background_w
    total_W = None if logic_W is None else (
        access_W + refresh_W + background_W + logic_W)
    electrical_access_comparison: dict[str, object] = {}
    reference_components = backend.metadata.get(
        "electrical_components_reference_pJ_per_bit")
    if isinstance(reference_components, dict):
        reference_decomposition = EnergyDecomposition(**reference_components)
        reference_vertical = _resolve_transport(
            "vertical", config.architecture.vertical.source,
            config.architecture.vertical.energy_pj_per_bit,
            reference_decomposition)
        reference_base = _resolve_transport(
            "base_route", config.architecture.base_route.source,
            config.architecture.base_route.energy_pj_per_bit,
            reference_decomposition)
        reference_interface = _resolve_transport(
            "interface", config.architecture.interface.source,
            config.architecture.interface.energy_pj_per_bit,
            reference_decomposition)
        reference_total = (
            reference_decomposition.memory_internal + reference_vertical
            + reference_base + reference_interface)
        electrical_access_comparison = {
            "architecture_access_reference_pJ_per_bit": {
                "memory_internal": reference_decomposition.memory_internal,
                "vertical": reference_vertical,
                "base_route": reference_base,
                "interface": reference_interface,
                "total": reference_total,
            },
            "architecture_access_resolved_pJ_per_bit": {
                "memory_internal": memory_internal,
                "vertical": vertical,
                "base_route": base_route,
                "interface": interface,
                "total": read_total,
            },
        }

    return MemoryPowerResult(
        technology=backend.technology,
        backend=backend.backend,
        architecture=(
            config.architecture.name
            or getattr(config, "name", "unnamed_architecture")),
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
            **({} if physical_latency_result is None
               else physical_latency_result.as_dict()),
            **({} if m3d_subarray is None else zhu_scaling_diagnostics),
            **refresh_result.diagnostics,
            **electrical_access_comparison,
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
            "interface_energy_status": (
                config.architecture.interface.energy_status),
            "interface_included_components": list(
                config.architecture.interface.included_components),
            "interface_excluded_components": list(
                config.architecture.interface.excluded_components),
            "interface_unconfirmed_components": list(
                config.architecture.interface.unconfirmed_components),
            "interface_source_boundary": (
                config.architecture.interface.source_boundary),
            "P_memory_dynamic_W": access_W + refresh_W + background_W,
        },
    )


def run_memory_power(config_path: str | Path) -> MemoryPowerResult:
    path = Path(config_path).resolve()
    with path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if isinstance(raw, dict) and "geometry" in raw:
        case = load_case_config(path)
        return calculate_memory_power(
            case,
            project_root=find_project_root(path),
            geometry=resolve_case_geometry(case),
        )
    return calculate_memory_power(
        load_power_config(path), project_root=find_project_root(path))
