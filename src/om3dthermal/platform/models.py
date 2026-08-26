"""Platform facts shared across memory architecture comparisons."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from om3dthermal.provenance import ProvenanceRecord


class HostOffloadSpec(BaseModel):
    """Two-tier host-memory transport facts; decimal GB/s are explicit."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["RESOLVED", "UNRESOLVED"]
    host_memory_bandwidth_GBps: float | None = Field(default=None, gt=0.0)
    host_device_link_bandwidth_GBps: float | None = Field(
        default=None, gt=0.0)
    host_offload_efficiency: float | None = Field(
        default=None, gt=0.0, le=1.0)
    provenance: tuple[ProvenanceRecord, ...]

    @model_validator(mode="after")
    def _status_closure(self) -> "HostOffloadSpec":
        values = (
            self.host_memory_bandwidth_GBps,
            self.host_device_link_bandwidth_GBps,
            self.host_offload_efficiency,
        )
        if not self.provenance:
            raise ValueError("host offload configuration requires provenance")
        if self.status == "RESOLVED" and any(value is None for value in values):
            raise ValueError("RESOLVED host offload requires all numeric inputs")
        if self.status == "UNRESOLVED" and any(value is not None for value in values):
            raise ValueError("UNRESOLVED host offload must not carry nominal numbers")
        return self

    @property
    def effective_bandwidth_bytes_per_second(self) -> float | None:
        if self.status == "UNRESOLVED":
            return None
        assert self.host_memory_bandwidth_GBps is not None
        assert self.host_device_link_bandwidth_GBps is not None
        assert self.host_offload_efficiency is not None
        return (
            min(
                self.host_memory_bandwidth_GBps,
                self.host_device_link_bandwidth_GBps,
            )
            * self.host_offload_efficiency
            * 1e9
        )


class PlatformSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    platform_id: str = Field(min_length=1)
    fixed_gpu_power_W: float = Field(ge=0.0)
    gpu_power_status: Literal[
        "FIXED_EXISTING_BASELINE_NOT_WORKLOAD_ENERGY_MODEL"
    ]
    package_profile_status: str = Field(min_length=1)
    host_offload: HostOffloadSpec | None = None
    provenance: tuple[ProvenanceRecord, ...] = ()
