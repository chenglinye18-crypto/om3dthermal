"""Shared GPU/package platform specifications."""

from .gpu_power import (
    AffineGPUComputePowerSpec,
    AffineGPUDecodePowerSpec,
    GPUComputePowerOperatingPoint,
    GPUDecodePowerOperatingPoint,
    resolve_gpu_compute_power,
    resolve_gpu_decode_power,
)
from .host_offload_power import (
    HostOffloadPowerOperatingPoint,
    resolve_host_offload_power,
)
from .models import HostOffloadSpec, PlatformSpec

__all__ = [
    "AffineGPUComputePowerSpec",
    "AffineGPUDecodePowerSpec",
    "GPUComputePowerOperatingPoint",
    "GPUDecodePowerOperatingPoint",
    "HostOffloadSpec",
    "HostOffloadPowerOperatingPoint",
    "PlatformSpec",
    "resolve_gpu_compute_power",
    "resolve_gpu_decode_power",
    "resolve_host_offload_power",
]
