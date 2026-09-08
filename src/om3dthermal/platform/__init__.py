"""Shared GPU/package platform specifications."""

from .gpu_power import (
    AffineGPUComputePowerSpec,
    AffineGPUDecodePowerSpec,
    GPUComputePowerOperatingPoint,
    GPUDecodePowerOperatingPoint,
    EffectiveThroughputReferenceRangeTFLOPS,
    ReferenceCalibratedGPUPrefillComputeSpec,
    resolve_gpu_compute_power,
    resolve_gpu_decode_power,
)
from .host_offload_power import (
    HostOffloadPowerOperatingPoint,
    resolve_host_offload_power,
)
from .models import (
    GPUBandwidthServiceSpec,
    HostOffloadSpec,
    PlatformSpec,
    load_platform_spec_file,
)
from .transfer import (
    GPUBandwidthServiceOperatingPoint,
    LocalMemoryGPUTransferOperatingPoint,
    TransferBottleneck,
    resolve_gpu_bandwidth_service,
    resolve_local_memory_gpu_transfer,
)

__all__ = [
    "AffineGPUComputePowerSpec",
    "AffineGPUDecodePowerSpec",
    "GPUComputePowerOperatingPoint",
    "GPUDecodePowerOperatingPoint",
    "EffectiveThroughputReferenceRangeTFLOPS",
    "ReferenceCalibratedGPUPrefillComputeSpec",
    "GPUBandwidthServiceOperatingPoint",
    "GPUBandwidthServiceSpec",
    "HostOffloadSpec",
    "HostOffloadPowerOperatingPoint",
    "PlatformSpec",
    "LocalMemoryGPUTransferOperatingPoint",
    "TransferBottleneck",
    "load_platform_spec_file",
    "resolve_gpu_compute_power",
    "resolve_gpu_bandwidth_service",
    "resolve_gpu_decode_power",
    "resolve_host_offload_power",
    "resolve_local_memory_gpu_transfer",
]
