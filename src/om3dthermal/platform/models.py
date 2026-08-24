"""Platform facts shared across memory architecture comparisons."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from om3dthermal.provenance import ProvenanceRecord


class PlatformSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    platform_id: str = Field(min_length=1)
    fixed_gpu_power_W: float = Field(ge=0.0)
    gpu_power_status: Literal[
        "FIXED_EXISTING_BASELINE_NOT_WORKLOAD_ENERGY_MODEL"
    ]
    package_profile_status: str = Field(min_length=1)
    provenance: tuple[ProvenanceRecord, ...] = ()
