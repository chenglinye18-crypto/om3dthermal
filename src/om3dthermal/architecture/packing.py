"""Packing adapter over the existing validated analytical backend diagnostics."""

from __future__ import annotations

from om3dthermal.power.config import CanonicalCaseConfig
from om3dthermal.power.geometry import ResolvedGeometry
from om3dthermal.power.result import MemoryPowerResult

from .models import ResolvedPacking


PACKING_SOURCE_STATUS = "ANALYTICAL_PACKING_DIAGNOSTICS_BIT_CLOSURE"


def resolve_packing_from_legacy_power_result(
    case: CanonicalCaseConfig,
    geometry: ResolvedGeometry,
    memory: MemoryPowerResult,
) -> ResolvedPacking:
    """Expose exact packing facts without exposing power to capacity consumers.

    The existing backend remains the numerical source during migration.  This
    adapter performs the same architecture-specific interpretation and exact
    bit-closure gate previously embedded in ``resolve_architecture_capacity``.
    """

    diagnostics = memory.diagnostics
    if geometry.memory_region == "hbm_dram_die":
        instances = geometry.memory_region_count
        total_bits = int(diagnostics["total_stored_bits"])
        bits_per_instance = int(diagnostics["bits_per_stack"])
        bits_per_plane = int(diagnostics["bits_per_die"])
        plane_area = geometry.configured_x_mm * geometry.configured_y_mm
        layout = case.geometry.layout
        footprint_area = (
            int(layout["visible_group_count"])
            * float(layout["visible_group_footprint_mm"][0])
            * float(layout["visible_group_footprint_mm"][1])
        )
    elif geometry.memory_region == "orthogonal_memory_slab":
        instances = geometry.memory_region_count
        bits_per_instance = int(diagnostics["bits_per_slab"])
        total_bits = int(diagnostics["total_stored_bits"])
        bits_per_plane = bits_per_instance
        plane_area = geometry.configured_x_mm * geometry.configured_y_mm
        orthogonal = case.geometry.orthogonal
        assert orthogonal is not None
        footprint_area = (
            orthogonal.cube_length_x_mm * orthogonal.slab_plane_y_mm)
    else:
        instances = geometry.memory_region_count
        layers = int(diagnostics["memory_layer_count"])
        bits_per_plane = int(diagnostics["bits_per_layer"])
        bits_per_instance = bits_per_plane * layers
        total_bits = int(diagnostics["total_stored_bits"])
        plane_area = geometry.configured_x_mm * geometry.configured_y_mm
        orthogonal = case.geometry.orthogonal
        assert orthogonal is not None
        footprint_area = (
            orthogonal.cube_length_x_mm * orthogonal.slab_plane_y_mm)

    if total_bits != bits_per_instance * instances:
        raise RuntimeError("system capacity does not close over physical instances")

    return ResolvedPacking(
        architecture_id=case.name,
        instance_count=instances,
        bits_per_instance=bits_per_instance,
        total_bits=total_bits,
        bits_per_plane=bits_per_plane,
        memory_plane_area_mm2=plane_area,
        architecture_footprint_area_mm2=footprint_area,
        source_status=PACKING_SOURCE_STATUS,
    )
