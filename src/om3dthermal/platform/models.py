"""Platform facts shared across memory architecture comparisons."""

from __future__ import annotations

import math
from typing import Literal
from pathlib import Path

import yaml

from pydantic import BaseModel, ConfigDict, Field, model_validator

from om3dthermal.provenance import ProvenanceRecord

from .gpu_power import (
    AffineGPUComputePowerSpec,
    AffineGPUDecodePowerSpec,
    ReferenceCalibratedGPUPrefillComputeSpec,
)


class GPUBandwidthServiceSpec(BaseModel):
    """Independent platform choice for achieved GPU-side memory service."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: Literal["SUSTAINED_FRACTION_OF_TRANSFER_CEILING"]
    nominal_utilization: float = Field(gt=0.0, le=1.0)
    utilization_status: Literal[
        "MODELING_CHOICE_NOMINAL_GPU_BANDWIDTH_UTILIZATION"]
    provenance: tuple[ProvenanceRecord, ...]

    @model_validator(mode="after")
    def _provenance_required(self) -> "GPUBandwidthServiceSpec":
        if not self.provenance:
            raise ValueError("GPU bandwidth service spec requires provenance")
        return self


class HostOffloadSpec(BaseModel):
    """Canonical direct host path or explicit legacy transport sensitivity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["RESOLVED", "UNRESOLVED"]
    host_path_id: str | None = None
    path_role: Literal[
        "PRIMARY", "PRIMARY_HOST_OFFLOAD_BASELINE", "SENSITIVITY_ONLY"
    ] | None = None
    direct_effective_bandwidth_bytes_per_second: float | None = Field(
        default=None, gt=0.0)
    effective_bandwidth_status: str | None = None
    memory_bandwidth_upper_bound_bytes_per_second: float | None = Field(
        default=None, gt=0.0)
    link_bandwidth_upper_bound_bytes_per_second: float | None = Field(
        default=None, gt=0.0)
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
    memory_dynamic_J_per_bit: float | None = Field(default=None, ge=0.0)
    link_dynamic_J_per_bit: float | None = Field(default=None, ge=0.0)
    total_dynamic_J_per_bit: float | None = Field(default=None, ge=0.0)
    memory_energy_status: str | None = None
    link_energy_status: str | None = None
    total_energy_status: str | None = None
    provenance: tuple[ProvenanceRecord, ...]

    @model_validator(mode="after")
    def _status_closure(self) -> "HostOffloadSpec":
        legacy_transport = (
            self.host_memory_bandwidth_GBps,
            self.host_device_link_bandwidth_GBps,
            self.host_offload_efficiency,
        )
        direct_transport = (
            self.host_path_id, self.path_role,
            self.direct_effective_bandwidth_bytes_per_second,
            self.effective_bandwidth_status,
        )
        if not self.provenance:
            raise ValueError("host offload configuration requires provenance")
        if self.status == "RESOLVED":
            legacy_complete = all(value is not None for value in legacy_transport)
            direct_complete = all(value is not None for value in direct_transport)
            if legacy_complete == direct_complete:
                raise ValueError(
                    "RESOLVED host offload requires exactly one complete "
                    "legacy or direct transport description")
        elif any(value is not None for value in (
            legacy_transport+direct_transport+(
                self.memory_bandwidth_upper_bound_bytes_per_second,
                self.link_bandwidth_upper_bound_bytes_per_second))):
            raise ValueError("UNRESOLVED host offload must not carry nominal numbers")
        legacy_power = (
            self.e_pcie_dynamic_J_per_bit,
            self.e_pcie_dynamic_uncertainty_J_per_bit,
            self.e_ddr_dynamic_J_per_bit,
        )
        generic_power = (
            self.memory_dynamic_J_per_bit, self.link_dynamic_J_per_bit,
            self.total_dynamic_J_per_bit, self.memory_energy_status,
            self.link_energy_status, self.total_energy_status,
        )
        if self.power_model_status == "INCREMENTAL_DYNAMIC_OFFLOAD_POWER":
            if self.status != "RESOLVED":
                raise ValueError(
                    "resolved host dynamic power requires resolved transport")
            legacy_complete = all(value is not None for value in legacy_power)
            generic_complete = all(value is not None for value in generic_power)
            if legacy_complete == generic_complete:
                raise ValueError(
                    "resolved host power requires exactly one complete legacy "
                    "or generic component description")
            if generic_complete and not math.isclose(
                float(self.total_dynamic_J_per_bit),
                float(self.memory_dynamic_J_per_bit)
                + float(self.link_dynamic_J_per_bit), rel_tol=1e-12):
                raise ValueError("host path total dynamic energy must close")
        elif any(value is not None for value in legacy_power+generic_power):
            raise ValueError(
                "UNRESOLVED host power must not carry dynamic coefficients")
        return self

    @property
    def effective_bandwidth_bytes_per_second(self) -> float | None:
        if self.status == "UNRESOLVED":
            return None
        if self.direct_effective_bandwidth_bytes_per_second is not None:
            return self.direct_effective_bandwidth_bytes_per_second
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
        """Canonical path sum for either generic or legacy components."""
        if self.power_model_status == "UNRESOLVED":
            return None
        if self.total_dynamic_J_per_bit is not None:
            return self.total_dynamic_J_per_bit
        assert self.e_pcie_dynamic_J_per_bit is not None
        assert self.e_ddr_dynamic_J_per_bit is not None
        return self.e_pcie_dynamic_J_per_bit + self.e_ddr_dynamic_J_per_bit

    @property
    def host_memory_dynamic_J_per_bit(self) -> float | None:
        if self.power_model_status == "UNRESOLVED":
            return None
        return (self.memory_dynamic_J_per_bit
                if self.memory_dynamic_J_per_bit is not None
                else self.e_ddr_dynamic_J_per_bit)

    @property
    def host_link_dynamic_J_per_bit(self) -> float | None:
        if self.power_model_status == "UNRESOLVED":
            return None
        return (self.link_dynamic_J_per_bit
                if self.link_dynamic_J_per_bit is not None
                else self.e_pcie_dynamic_J_per_bit)


class PlatformSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    platform_id: str = Field(min_length=1)
    package_profile_status: str = Field(min_length=1)
    host_offload: HostOffloadSpec | None = None
    gpu_bandwidth_service: GPUBandwidthServiceSpec
    gpu_decode_power: "AffineGPUDecodePowerSpec | None" = None
    gpu_compute_power: "AffineGPUComputePowerSpec | None" = None
    gpu_prefill_compute: "ReferenceCalibratedGPUPrefillComputeSpec | None" = None
    provenance: tuple[ProvenanceRecord, ...] = ()

    @model_validator(mode="after")
    def _gpu_static_power_closure(self) -> "PlatformSpec":
        if (self.gpu_decode_power is not None
                and self.gpu_compute_power is not None
                and self.gpu_decode_power.static_power_W
                != self.gpu_compute_power.static_power_W):
            raise ValueError(
                "GPU decode and compute power models must share static power")
        if self.gpu_prefill_compute is not None:
            if self.gpu_compute_power is None:
                raise ValueError(
                    "GPU Prefill calibration requires GPU compute peak data")
            peak_tflops = (
                self.gpu_compute_power.peak_compute_BF16_dense_flops_per_s
                / 1e12)
            if max(
                self.gpu_prefill_compute.large_gemm_effective_tflops,
                self.gpu_prefill_compute.causal_attention_effective_tflops,
            ) > peak_tflops:
                raise ValueError(
                    "GPU Prefill effective throughput cannot exceed vendor peak")
        return self


def load_platform_spec_file(path: str | Path) -> PlatformSpec:
    """Load one strict platform document from its canonical YAML source."""
    with Path(path).open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise TypeError("platform config root must be a mapping")
    return PlatformSpec.model_validate(raw)
