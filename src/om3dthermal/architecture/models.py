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


@dataclass(frozen=True)
class ResolvedArchitecture:
    """Compatibility aggregate around existing validated resolution objects."""

    spec: ArchitectureSpec
    case: Any
    geometry: Any
    system_power: Any
    packing: ResolvedPacking
