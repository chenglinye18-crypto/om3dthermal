"""Architecture domain objects that do not encode application semantics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from om3dthermal.provenance import ProvenanceRecord


class ArchitectureSpec(BaseModel):
    """A stable architecture identity backed by one validated canonical case.

    During the compatibility migration the canonical case remains the sole
    source of numerical hardware truth.  This descriptor adds identity and
    role without copying physical parameters into a second configuration.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    architecture_id: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    role: Literal["baseline", "proposed", "ablation"]
    canonical_case: Path
    provenance: tuple[ProvenanceRecord, ...] = ()


class ResolvedPacking(BaseModel):
    """Exact architecture capacity/packing facts, independent of power units."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    architecture_id: str
    instance_count: int = Field(gt=0)
    bits_per_instance: int = Field(gt=0)
    total_bits: int = Field(gt=0)
    bits_per_plane: int = Field(gt=0)
    memory_plane_area_mm2: float = Field(gt=0.0)
    architecture_footprint_area_mm2: float = Field(gt=0.0)
    source_status: str

    @property
    def system_capacity_bytes(self) -> float:
        return self.total_bits / 8

    @property
    def system_capacity_GiB(self) -> float:
        return self.system_capacity_bytes / 2**30

    @property
    def architecture(self) -> str:
        """Compatibility identity used by the existing capacity adapter."""
        return self.architecture_id


class ResolvedEnergyPrimitives(BaseModel):
    """Architecture energy facts exposed without workload semantics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    read_access_energy_pj_per_bit: float | None = Field(ge=0.0)
    memory_internal_pj_per_bit: float | None = Field(ge=0.0)
    vertical_pj_per_bit: float | None = Field(ge=0.0)
    feol_route_pj_per_bit: float | None = Field(ge=0.0)
    base_route_pj_per_bit: float | None = Field(ge=0.0)
    interface_pj_per_bit: float | None = Field(ge=0.0)
    source_status: str = Field(min_length=1)


class ResolvedStaticPower(BaseModel):
    """Static/reference power components before workload-rate composition."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    refresh_power_W: float | None = Field(ge=0.0)
    memory_background_power_W: float | None = Field(ge=0.0)
    logic_background_power_W: float | None = Field(ge=0.0)
    fixed_gpu_power_W: float = Field(ge=0.0)
    source_status: str = Field(min_length=1)
    completeness_status: Literal[
        "RESOLVED",
        "UNRESOLVED_LOGIC_BACKGROUND",
    ]


class ResolvedArchitectureFacts(BaseModel):
    """Stable hardware facts consumed by workload-aware evaluation stages."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    architecture_id: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    role: Literal["baseline", "proposed", "ablation"]
    canonical_case: Path
    geometry_type: str = Field(min_length=1)
    packing: ResolvedPacking
    energy_primitives: ResolvedEnergyPrimitives
    static_power: ResolvedStaticPower
    provenance: tuple[ProvenanceRecord, ...] = ()


@dataclass(frozen=True)
class ResolvedArchitecture:
    """Compatibility aggregate around existing validated resolution objects."""

    spec: ArchitectureSpec
    case: Any
    geometry: Any
    system_power: Any
    packing: ResolvedPacking
