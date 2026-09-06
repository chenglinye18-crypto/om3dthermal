"""Shared GPU/package platform specifications."""

from .gpu_power import (
    AffineGPUDecodePowerSpec,
    GPUDecodePowerOperatingPoint,
    resolve_gpu_decode_power,
)
from .models import HostOffloadSpec, PlatformSpec

__all__ = [
    "AffineGPUDecodePowerSpec",
    "GPUDecodePowerOperatingPoint",
    "HostOffloadSpec",
    "PlatformSpec",
    "resolve_gpu_decode_power",
]
