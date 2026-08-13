"""Technology-specific, time-domain memory refresh-power accounting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .cell_model import DeviceOperationEnergies
from .config import MemoryPowerConfig
from .m3d_subarray import M3DSubarrayResult
from .result import BackendEnergyResult


@dataclass(frozen=True)
class RefreshPowerResult:
    power_W: float
    diagnostics: dict[str, Any] = field(default_factory=dict)


def calculate_refresh_power(
        config: MemoryPowerConfig, *,
        backend: BackendEnergyResult,
        device: DeviceOperationEnergies | None,
        m3d_subarray: M3DSubarrayResult | None,
        m3d_layer_count: int | None,
        memory_region_count: int = 1,
        ) -> RefreshPowerResult:
    """Resolve refresh independently of bandwidth-driven access energy."""
    refresh = config.power.refresh
    if not refresh.enabled:
        return RefreshPowerResult(
            power_W=0.0,
            diagnostics={
                "refresh_enabled": False,
                "refresh_model": None,
                "refresh_power_W": 0.0,
                "refresh_route_boundary": "INTERNAL_MEMORY_ONLY",
            },
        )

    if refresh.model == "operation_table_retention":
        if device is None or m3d_subarray is None or m3d_layer_count is None:
            raise ValueError(
                "operation-table retention refresh requires the IGZO operation "
                "table and resolved M3D capacity")
        probability = config.workload.refresh_data
        if probability is None:
            raise ValueError(
                "operation-table retention refresh requires workload.refresh_data")
        if (refresh.retention_reference_s is None
                or refresh.refresh_safety_factor is None):
            raise ValueError("IGZO refresh retention inputs are unresolved")
        weighted_energy = device.weighted_refresh(
            p0=probability.p0, p1=probability.p1)
        interval_s = (
            refresh.retention_reference_s / refresh.refresh_safety_factor)
        total_bits = (
            m3d_subarray.bits_per_layer * m3d_layer_count
            * memory_region_count)
        full_energy_J = total_bits * weighted_energy * 1e-12
        power_W = full_energy_J / interval_s
        return RefreshPowerResult(
            power_W=power_W,
            diagnostics={
                "refresh_enabled": True,
                "refresh_model": refresh.model,
                "refresh_reference_0_pj_per_bit": device.refresh_0,
                "refresh_reference_1_pj_per_bit": device.refresh_1,
                "refresh_data_p0": probability.p0,
                "refresh_data_p1": probability.p1,
                "refresh_weighted_energy_pj_per_bit": weighted_energy,
                "retention_reference_s": refresh.retention_reference_s,
                "retention_reference_source": (
                    refresh.retention_reference_source),
                "refresh_safety_factor": refresh.refresh_safety_factor,
                "resolved_refresh_interval_s": interval_s,
                "bits_per_layer": m3d_subarray.bits_per_layer,
                "memory_layer_count": m3d_layer_count,
                "physical_slab_count": memory_region_count,
                "total_stored_bits": total_bits,
                "full_memory_refresh_energy_J": full_energy_J,
                "refresh_power_W": power_W,
                "refresh_operation_provenance": "PAPER_REPORTED",
                "retention_provenance": (
                    refresh.retention_reference_provenance),
                "refresh_interval_provenance": (
                    refresh.refresh_interval_provenance),
                "zhu_refresh_size_scaling": "NOT_MODELED",
                "refresh_route_boundary": "INTERNAL_MEMORY_ONLY",
            },
        )

    if refresh.model == "dreamram_internal_refresh":
        if refresh.refresh_window_s is None:
            raise ValueError("DreamRAM refresh window is unresolved")
        metadata = backend.metadata
        event_energy_pJ = float(
            metadata["refresh_internal_event_energy_pJ"])
        event_count = int(
            metadata["refresh_events_per_full_memory_cycle"])
        full_energy_pJ = event_energy_pJ * event_count
        power_W = full_energy_pJ * 1e-12 / refresh.refresh_window_s
        return RefreshPowerResult(
            power_W=power_W,
            diagnostics={
                "refresh_enabled": True,
                "refresh_model": refresh.model,
                "refresh_window_s": refresh.refresh_window_s,
                "refresh_window_provenance": (
                    refresh.refresh_window_provenance),
                "dreamram_refresh_included_components": metadata[
                    "dreamram_refresh_included_components"],
                "dreamram_refresh_included_component_energy_pJ": metadata[
                    "dreamram_refresh_included_component_energy_pJ"],
                "dreamram_refresh_excluded_components": metadata[
                    "dreamram_refresh_excluded_components"],
                "refresh_internal_event_energy_pJ": event_energy_pJ,
                "refresh_event_scope": metadata["refresh_event_scope"],
                "refresh_events_per_full_memory_cycle": event_count,
                "refresh_bits_per_event": metadata["refresh_bits_per_event"],
                "total_stored_bits": metadata["dreamram_total_stored_bits"],
                "dreamram_refresh_organization": metadata[
                    "dreamram_refresh_organization"],
                "full_memory_refresh_energy_pJ": full_energy_pJ,
                "refresh_power_W": power_W,
                "refresh_component_equations_provenance": (
                    "DERIVED_FROM_REFERENCE"),
                "refresh_route_boundary": "INTERNAL_MEMORY_ONLY",
            },
        )

    raise ValueError(f"unsupported refresh model {refresh.model!r}")
