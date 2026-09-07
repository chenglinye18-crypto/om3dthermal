"""Platform facts shared across memory architecture comparisons."""

from __future__ import annotations

from typing import Literal
from pathlib import Path

import yaml

from pydantic import BaseModel, ConfigDict, Field, model_validator

from om3dthermal.provenance import ProvenanceRecord

from .gpu_power import AffineGPUComputePowerSpec, AffineGPUDecodePowerSpec


class HostOffloadSpec(BaseModel):
    """Host transport facts plus optional incremental dynamic-only power."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["RESOLVED", "UNRESOLVED"]
    host_memory_bandwidth_GBps: float | None = Field(default=None, gt=0.0)
    host_device_link_bandwidth_GBps: float | None = Field(
        default=None, gt=0.0)
    host_offload_efficiency: float | None = Field(
        default=None, gt=0.0, le=1.0)
    power_model_status: Literal[
        "INCREMENTAL_DYNAMIC_OFFLOAD_POWER", "UNRESOLVED"
    ] = "UNRESOLVED"
    host_static_power_status: Literal["UNRESOLVED"] = "UNRESOLVED"
    e_pcie_dynamic_J_per_bit: float | None = Field(default=None, ge=0.0)
    e_pcie_dynamic_uncertainty_J_per_bit: float | None = Field(
        default=None, ge=0.0)
    e_ddr_dynamic_J_per_bit: float | None = Field(default=None, ge=0.0)
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
        power_values = (
            self.e_pcie_dynamic_J_per_bit,
            self.e_pcie_dynamic_uncertainty_J_per_bit,
            self.e_ddr_dynamic_J_per_bit,
        )
        if self.power_model_status == "INCREMENTAL_DYNAMIC_OFFLOAD_POWER":
            if self.status != "RESOLVED":
                raise ValueError(
                    "resolved host dynamic power requires resolved transport")
            if any(value is None for value in power_values):
                raise ValueError(
                    "INCREMENTAL_DYNAMIC_OFFLOAD_POWER requires PCIe and DDR coefficients")
        elif any(value is not None for value in power_values):
            raise ValueError(
                "UNRESOLVED host power must not carry dynamic coefficients")
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

    @property
    def e_host_offload_dynamic_J_per_bit(self) -> float | None:
        """Derived sum; PCIe and DDR remain independently configured."""
        if self.power_model_status == "UNRESOLVED":
            return None
        assert self.e_pcie_dynamic_J_per_bit is not None
        assert self.e_ddr_dynamic_J_per_bit is not None
        return self.e_pcie_dynamic_J_per_bit + self.e_ddr_dynamic_J_per_bit


class PlatformSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    platform_id: str = Field(min_length=1)
    package_profile_status: str = Field(min_length=1)
    host_offload: HostOffloadSpec | None = None
    gpu_decode_power: "AffineGPUDecodePowerSpec | None" = None
    gpu_compute_power: "AffineGPUComputePowerSpec | None" = None
    provenance: tuple[ProvenanceRecord, ...] = ()

    @model_validator(mode="after")
    def _gpu_static_power_closure(self) -> "PlatformSpec":
        if (self.gpu_decode_power is not None
                and self.gpu_compute_power is not None
                and self.gpu_decode_power.static_power_W
                != self.gpu_compute_power.static_power_W):
            raise ValueError(
                "GPU decode and compute power models must share static power")
        return self


def load_platform_spec_file(path: str | Path) -> PlatformSpec:
    """Load one strict platform document from its canonical YAML source."""
    with Path(path).open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise TypeError("platform config root must be a mapping")
    return PlatformSpec.model_validate(raw)
