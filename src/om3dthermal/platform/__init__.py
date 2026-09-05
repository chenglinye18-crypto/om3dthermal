"""Shared GPU/package platform specifications."""

from .gpu_power import AffineGPUDecodePowerSpec
from .models import HostOffloadSpec, PlatformSpec

__all__ = ["AffineGPUDecodePowerSpec", "HostOffloadSpec", "PlatformSpec"]
