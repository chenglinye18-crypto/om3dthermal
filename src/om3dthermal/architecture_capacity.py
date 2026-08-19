"""Canonical architecture capacity resolved from existing packing diagnostics.

This module contains no thermal construction or solver path.  Exact integer
``total_bits`` from the existing analytical memory resolution is the capacity
source of truth; byte and GiB values are explicit derived display quantities.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .power.config import CanonicalCaseConfig
from .power.geometry import ResolvedGeometry
from .power.system import ResolvedSystemPower


CAPACITY_SOURCE_STATUS = "ANALYTICAL_PACKING_DIAGNOSTICS_BIT_CLOSURE"


@dataclass(frozen=True)
class ResolvedArchitectureCapacity:
    """Exact canonical capacity plus geometry-derived density metrics."""

    architecture: str
    instance_count: int
    bits_per_instance: int
    total_bits: int
    capacity_per_instance_bytes: float
    system_capacity_bytes: float
    capacity_per_instance_GiB: float
    system_capacity_GiB: float
    bits_per_plane: int
    memory_plane_area_mm2: float
    memory_plane_density_Mb_mm2: float
    architecture_footprint_area_mm2: float
    architecture_footprint_density_Gb_mm2: float
    source_status: str

    def as_dict(self) -> dict[str, int | float | str]:
        """Return a compatibility mapping for existing comparison consumers."""
        return asdict(self)


def resolve_architecture_capacity(
    case: CanonicalCaseConfig,
    geometry: ResolvedGeometry,
    system: ResolvedSystemPower,
) -> ResolvedArchitectureCapacity:
    """Resolve canonical system capacity without invoking thermal machinery.

    Packing diagnostics are produced by the existing analytical memory
    resolution path.  This function centralizes their architecture-specific
    interpretation and verifies exact bit closure before deriving bytes/GiB.
    """
    result = system.memory_result
    if result is None:
        raise ValueError(
            "capacity resolution requires validated analytical memory power")
    diagnostics = result.diagnostics

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
            * float(layout["visible_group_footprint_mm"][1]))
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

    capacity_per_instance_bytes = bits_per_instance / 8
    system_capacity_bytes = total_bits / 8
    return ResolvedArchitectureCapacity(
        architecture=case.name,
        instance_count=instances,
        bits_per_instance=bits_per_instance,
        total_bits=total_bits,
        capacity_per_instance_bytes=capacity_per_instance_bytes,
        system_capacity_bytes=system_capacity_bytes,
        capacity_per_instance_GiB=capacity_per_instance_bytes / 2**30,
        system_capacity_GiB=system_capacity_bytes / 2**30,
        bits_per_plane=bits_per_plane,
        memory_plane_area_mm2=plane_area,
        memory_plane_density_Mb_mm2=bits_per_plane / 1e6 / plane_area,
        architecture_footprint_area_mm2=footprint_area,
        architecture_footprint_density_Gb_mm2=(
            total_bits / 1e9 / footprint_area),
        source_status=CAPACITY_SOURCE_STATUS,
    )
