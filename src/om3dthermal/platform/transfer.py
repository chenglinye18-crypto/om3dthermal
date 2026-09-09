"""Pure local-memory-to-GPU transfer-rate closure.

Only traffic that crosses the local-memory-to-GPU boundary belongs here.
Local NMP activity is deliberately outside this operating point.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from om3dthermal.provenance import ProvenanceRecord


TransferBottleneck = Literal["DEMAND", "MEMORY", "GPU"]
class LocalMemoryGPUTransferOperatingPoint(BaseModel):
    """Resolved rate and deterministic bottleneck at the transfer boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    bandwidth_demand_bytes_per_s: float = Field(ge=0.0)
    memory_capability_bytes_per_s: float = Field(gt=0.0)
    gpu_peak_bandwidth_bytes_per_s: float = Field(gt=0.0)
    bandwidth_actual_bytes_per_s: float = Field(ge=0.0)
    demand_limited: bool
    memory_limited: bool
    gpu_limited: bool
    bottleneck: TransferBottleneck


class GPUBandwidthServiceOperatingPoint(BaseModel):
    """Direct GPU-side service at the resolved transfer ceiling."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    transfer_ceiling_bytes_per_s: float = Field(gt=0.0)
    sustained_bandwidth_bytes_per_s: float = Field(gt=0.0)
    service_status: Literal["DIRECT_TRANSFER_CEILING"]
    provenance: tuple[ProvenanceRecord, ...]


def _finite_rate(name: str, value: float, *, allow_zero: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{name} must be finite")
    if resolved < 0.0 or (not allow_zero and resolved == 0.0):
        qualifier = "non-negative" if allow_zero else "strictly positive"
        raise ValueError(f"{name} must be {qualifier}")
    return resolved


def resolve_local_memory_gpu_transfer(
    *,
    bandwidth_demand_bytes_per_s: float,
    memory_capability_bytes_per_s: float,
    gpu_peak_bandwidth_bytes_per_s: float,
) -> LocalMemoryGPUTransferOperatingPoint:
    """Resolve ``min(demand, memory capability, GPU peak)`` exactly once.

    Ties use the fixed priority ``DEMAND > MEMORY > GPU``. The three
    ``*_limited`` flags identify every value equal to the minimum, while
    ``bottleneck`` provides that deterministic primary identity.
    """

    demand = _finite_rate(
        "bandwidth_demand_bytes_per_s",
        bandwidth_demand_bytes_per_s,
        allow_zero=True,
    )
    memory = _finite_rate(
        "memory_capability_bytes_per_s",
        memory_capability_bytes_per_s,
        allow_zero=False,
    )
    gpu = _finite_rate(
        "gpu_peak_bandwidth_bytes_per_s",
        gpu_peak_bandwidth_bytes_per_s,
        allow_zero=False,
    )
    actual = min(demand, memory, gpu)
    demand_limited = demand == actual
    memory_limited = memory == actual
    gpu_limited = gpu == actual
    bottleneck: TransferBottleneck = (
        "DEMAND" if demand_limited
        else "MEMORY" if memory_limited
        else "GPU"
    )
    return LocalMemoryGPUTransferOperatingPoint(
        bandwidth_demand_bytes_per_s=demand,
        memory_capability_bytes_per_s=memory,
        gpu_peak_bandwidth_bytes_per_s=gpu,
        bandwidth_actual_bytes_per_s=actual,
        demand_limited=demand_limited,
        memory_limited=memory_limited,
        gpu_limited=gpu_limited,
        bottleneck=bottleneck,
    )


def resolve_gpu_bandwidth_service(
    *,
    transfer_ceiling_bytes_per_s: float,
    service_status: Literal["DIRECT_TRANSFER_CEILING"],
    provenance: tuple[ProvenanceRecord, ...],
) -> GPUBandwidthServiceOperatingPoint:
    """Use the transfer ceiling as the shared actual service rate."""

    ceiling = _finite_rate(
        "transfer_ceiling_bytes_per_s",
        transfer_ceiling_bytes_per_s,
        allow_zero=False,
    )
    if not provenance:
        raise ValueError("GPU bandwidth service requires provenance")
    return GPUBandwidthServiceOperatingPoint(
        transfer_ceiling_bytes_per_s=ceiling,
        sustained_bandwidth_bytes_per_s=ceiling,
        service_status=service_status,
        provenance=provenance,
    )
