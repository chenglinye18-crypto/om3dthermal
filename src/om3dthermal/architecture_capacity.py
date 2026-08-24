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
from .architecture import resolve_packing_from_legacy_power_result


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
    packing = resolve_packing_from_legacy_power_result(case, geometry, result)

    capacity_per_instance_bytes = packing.bits_per_instance / 8
    system_capacity_bytes = packing.total_bits / 8
    return ResolvedArchitectureCapacity(
        architecture=case.name,
        instance_count=packing.instance_count,
        bits_per_instance=packing.bits_per_instance,
        total_bits=packing.total_bits,
        capacity_per_instance_bytes=capacity_per_instance_bytes,
        system_capacity_bytes=system_capacity_bytes,
        capacity_per_instance_GiB=capacity_per_instance_bytes / 2**30,
        system_capacity_GiB=system_capacity_bytes / 2**30,
        bits_per_plane=packing.bits_per_plane,
        memory_plane_area_mm2=packing.memory_plane_area_mm2,
        memory_plane_density_Mb_mm2=(
            packing.bits_per_plane / 1e6 / packing.memory_plane_area_mm2),
        architecture_footprint_area_mm2=(
            packing.architecture_footprint_area_mm2),
        architecture_footprint_density_Gb_mm2=(
            packing.total_bits / 1e9
            / packing.architecture_footprint_area_mm2),
        source_status=CAPACITY_SOURCE_STATUS,
    )
