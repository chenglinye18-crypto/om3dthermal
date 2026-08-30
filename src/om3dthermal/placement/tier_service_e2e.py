"""Stratum-style tier-dependent internal memory-service E2E diagnostic.

This is deliberately separate from ``serving_e2e.py``.  The latter remains an
external-GPU-streaming negative control; this module models only the explicit
case where a configurable fraction of decode traffic is served by a
placement-sensitive local M3D internal path.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Literal

from om3dthermal.power.memory_bandwidth import (
    ArchitectureBandwidthClosure,
    resolve_effective_bandwidth,
)
from om3dthermal.power.physical_capacity import PhysicalCapacityLayout
from om3dthermal.workload.llm_decode import LLMDecodeInput, evaluate_llm_decode
from om3dthermal.workload.m3d_page_demand import M3DWorkloadPageDemand

from .fast_region import PagePlacementResult


TierTimingPolicy = Literal["NO_TIER_WORST_CASE", "TIER_AWARE_FAST_PACK"]
TierBottleneck = Literal["MEMORY", "COMPUTE", "BALANCED"]


@dataclass(frozen=True)
class TierServiceTiming:
    policy: TierTimingPolicy
    requested_requests: int
    occupancy_fraction: float
    physical_service_latency_ns: float
    no_tier_latency_ns: float
    fast_pack_latency_ns: float
    service_rate_speedup: float
    internal_bandwidth_no_tier_bytes_per_s: float
    internal_bandwidth_effective_bytes_per_s: float
    local_service_fraction: float
    total_memory_bytes_per_decode_step: float
    local_memory_bytes_per_decode_step: float
    external_memory_bytes_per_decode_step: float
    local_memory_time_ms: float
    external_memory_time_ms: float
    total_memory_stage_ms: float
    compute_time_ms: float
    total_decode_step_ms: float
    aggregate_tokens_per_s: float
    bottleneck: TierBottleneck
    service_semantics: str
    overlap_semantics: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TierServiceComparison:
    no_tier: TierServiceTiming
    tier_aware_fast_pack: TierServiceTiming
    end_to_end_speedup: float
    throughput_gain: float
    local_service_fraction: float
    model_name: str
    model_status: str

    def as_dict(self) -> dict[str, object]:
        return {
            "no_tier": self.no_tier.as_dict(),
            "tier_aware_fast_pack": self.tier_aware_fast_pack.as_dict(),
            "end_to_end_speedup": self.end_to_end_speedup,
            "throughput_gain": self.throughput_gain,
            "local_service_fraction": self.local_service_fraction,
            "model_name": self.model_name,
            "model_status": self.model_status,
        }


def _positive(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _fraction(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite in [0, 1]")
    return result


def _bottleneck(memory_ms: float, compute_ms: float) -> TierBottleneck:
    if math.isclose(memory_ms, compute_ms, rel_tol=1e-12, abs_tol=0.0):
        return "BALANCED"
    return "MEMORY" if memory_ms > compute_ms else "COMPUTE"


def evaluate_tier_service_placement(
        workload: LLMDecodeInput,
        demand: M3DWorkloadPageDemand,
        physical_layout: PhysicalCapacityLayout,
        bandwidth_closure: ArchitectureBandwidthClosure,
        fast_pack: PagePlacementResult,
        *,
        matched_external_bandwidth_bits_per_second: float,
        effective_compute_flops_per_second: float,
        local_service_fraction: float,
        ) -> TierServiceComparison:
    """Compare global-worst no-tier timing against actual fast-pack timing.

    The local path rate follows ``B_fast = B_no_tier * t_no_tier/t_fast``.
    The complementary traffic remains on the unchanged fixed external path.
    Local and external traffic portions are conservatively summed, after
    which the existing memory/compute roofline maximum is applied.
    """
    if workload.batch_size != demand.requested_requests:
        raise ValueError("workload batch and page demand request count differ")
    if fast_pack.page_count != demand.page_count:
        raise ValueError("fast-pack placement and page demand count differ")
    if not math.isclose(
            fast_pack.total_read_demand_bytes_per_decode_step,
            demand.total_read_bytes_per_decode_step,
            rel_tol=1e-12, abs_tol=1e-6):
        raise ValueError("fast pack changed workload read traffic")
    local_fraction = _fraction(local_service_fraction, "local_service_fraction")
    external_bandwidth = _positive(
        matched_external_bandwidth_bits_per_second,
        "matched_external_bandwidth_bits_per_second") / 8.0
    compute_rate = _positive(
        effective_compute_flops_per_second,
        "effective_compute_flops_per_second")
    latencies = tuple(
        slot.physical_access_latency_ns for slot in physical_layout.slot_classes)
    if not latencies:
        raise ValueError("tier service requires physical slot latencies")
    no_tier_latency = max(latencies)
    fast_latency = _positive(
        fast_pack.weighted_average_access_latency_ns,
        "fast_pack.weighted_average_access_latency_ns")
    no_tier_bandwidth = resolve_effective_bandwidth(
        bandwidth_closure, no_tier_latency).internal_bandwidth_bytes_per_s
    service_rate_speedup = no_tier_latency / fast_latency
    fast_bandwidth = no_tier_bandwidth * service_rate_speedup

    metrics = evaluate_llm_decode(workload)
    total_memory_bytes = (
        demand.total_read_bytes_per_decode_step
        + demand.kv_write_bytes_per_decode_step)
    compute_ms = (
        workload.batch_size * metrics.flops_per_token / compute_rate * 1e3)

    def timing(policy: TierTimingPolicy, latency_ns: float,
               internal_bandwidth: float) -> TierServiceTiming:
        local_bytes = local_fraction * total_memory_bytes
        external_bytes = (1.0 - local_fraction) * total_memory_bytes
        local_ms = local_bytes / internal_bandwidth * 1e3
        external_ms = external_bytes / external_bandwidth * 1e3
        memory_ms = local_ms + external_ms
        total_ms = max(memory_ms, compute_ms)
        return TierServiceTiming(
            policy=policy,
            requested_requests=workload.batch_size,
            occupancy_fraction=fast_pack.occupancy_fraction,
            physical_service_latency_ns=latency_ns,
            no_tier_latency_ns=no_tier_latency,
            fast_pack_latency_ns=fast_latency,
            service_rate_speedup=service_rate_speedup,
            internal_bandwidth_no_tier_bytes_per_s=no_tier_bandwidth,
            internal_bandwidth_effective_bytes_per_s=internal_bandwidth,
            local_service_fraction=local_fraction,
            total_memory_bytes_per_decode_step=total_memory_bytes,
            local_memory_bytes_per_decode_step=local_bytes,
            external_memory_bytes_per_decode_step=external_bytes,
            local_memory_time_ms=local_ms,
            external_memory_time_ms=external_ms,
            total_memory_stage_ms=memory_ms,
            compute_time_ms=compute_ms,
            total_decode_step_ms=total_ms,
            aggregate_tokens_per_s=workload.batch_size / (total_ms * 1e-3),
            bottleneck=_bottleneck(memory_ms, compute_ms),
            service_semantics=(
                "LOCAL_INTERNAL_SERVICE_RATE_SCALES_WITH_PHYSICAL_TIER_"
                "LATENCY__EXTERNAL_FRACTION_REMAINS_FIXED_INTERFACE"),
            overlap_semantics=(
                "LOCAL_AND_EXTERNAL_TRAFFIC_PORTIONS_SUM_THEN_EXISTING_"
                "ROOFLINE_MAX_WITH_COMPUTE"),
        )

    no_tier = timing(
        "NO_TIER_WORST_CASE", no_tier_latency, no_tier_bandwidth)
    fast = timing(
        "TIER_AWARE_FAST_PACK", fast_latency, fast_bandwidth)
    return TierServiceComparison(
        no_tier=no_tier,
        tier_aware_fast_pack=fast,
        end_to_end_speedup=(
            no_tier.total_decode_step_ms / fast.total_decode_step_ms),
        throughput_gain=(
            fast.aggregate_tokens_per_s / no_tier.aggregate_tokens_per_s - 1.0),
        local_service_fraction=local_fraction,
        model_name="STRATUM_STYLE_TIER_DEPENDENT_INTERNAL_SERVICE",
        model_status=(
            "FIRST_ORDER_ABSTRACTION_NOT_STRATUM_MICROARCHITECTURE_"
            "REPRODUCTION"),
    )


def sweep_local_service_fraction(
        workload: LLMDecodeInput,
        demand: M3DWorkloadPageDemand,
        physical_layout: PhysicalCapacityLayout,
        bandwidth_closure: ArchitectureBandwidthClosure,
        fast_pack: PagePlacementResult,
        *,
        matched_external_bandwidth_bits_per_second: float,
        effective_compute_flops_per_second: float,
        fractions: tuple[float, ...] = (0.0, 0.25, 0.50, 0.75, 1.0),
        ) -> tuple[TierServiceComparison, ...]:
    """Deterministically sweep the explicit placement-sensitive fraction."""
    if tuple(sorted(set(fractions))) != fractions:
        raise ValueError("local service fractions must be unique and increasing")
    return tuple(
        evaluate_tier_service_placement(
            workload, demand, physical_layout, bandwidth_closure, fast_pack,
            matched_external_bandwidth_bits_per_second=(
                matched_external_bandwidth_bits_per_second),
            effective_compute_flops_per_second=(
                effective_compute_flops_per_second),
            local_service_fraction=fraction,
        )
        for fraction in fractions)
