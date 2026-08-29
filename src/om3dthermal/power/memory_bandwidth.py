"""Hierarchical M3D internal, contactless-coil, and GPU bandwidth model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import statistics
from typing import Literal

from .config import HierarchicalMemoryServiceInput
from .m3d_subarray import M3DSubarrayResult
from .physical_capacity import PhysicalCapacityLayout


BandwidthBottleneck = Literal["INTERNAL", "COIL_INTERFACE", "GPU_INTERNAL"]


@dataclass(frozen=True)
class InternalBandwidthPrefix:
    capacity_fraction: float
    average_service_cycle_ns: float
    internal_bandwidth_bytes_per_s: float


@dataclass(frozen=True)
class ArchitectureBandwidthClosure:
    num_m3d_dies: int
    die_count_source: str
    coil_links_per_die: int
    coil_data_rate_gbps_per_link: float
    coil_bandwidth_bits_per_s: float
    coil_bandwidth_bytes_per_s: float
    coil_classification: str
    coil_parameter_classification: str
    internal_service_model: str
    internal_service_unit: str
    parallel_service_units_per_slab: int
    parallel_slabs: int
    total_parallel_service_units: int
    clusters_per_service: int
    subarrays_per_service: int
    delivered_bits_per_service: int
    read_payload_bytes_per_service: float
    service_cycle_scale: float
    service_cycle_source: str
    fast_service_cycle_ns: float
    average_service_cycle_ns: float
    slow_service_cycle_ns: float
    internal_bandwidth_fast_bytes_per_s: float
    internal_bandwidth_average_bytes_per_s: float
    internal_bandwidth_slow_bytes_per_s: float
    prefix_bandwidth: tuple[InternalBandwidthPrefix, ...]
    gpu_internal_bandwidth_bytes_per_s: float | None
    gpu_internal_status: str
    internal_classification: str
    access_topology_provenance: str
    payload_status: str
    physical_latency_status: str
    aggregation_semantics: str
    layer_parallelism_semantics: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class EffectiveBandwidth:
    physical_access_latency_ns: float
    service_cycle_ns: float
    internal_parallelism_scale: float
    internal_bandwidth_bytes_per_s: float
    coil_bandwidth_bytes_per_s: float
    gpu_internal_bandwidth_bytes_per_s: float | None
    effective_bandwidth_bytes_per_s: float
    bottleneck: BandwidthBottleneck
    model_name: Literal["HIERARCHICAL_BANDWIDTH_MODEL"]


def derive_architecture_bandwidth(
    spec: HierarchicalMemoryServiceInput,
    physical_layout: PhysicalCapacityLayout,
    topology: M3DSubarrayResult,
    *,
    feol_io_channels: int,
) -> ArchitectureBandwidthClosure:
    """Resolve bandwidth only from architecture and explicit choices."""
    if feol_io_channels <= 0:
        raise ValueError("FEOL IO channel count must be positive")
    if physical_layout.slab_count <= 0:
        raise ValueError("physical layout must contain slabs")
    parallel_per_slab = feol_io_channels
    clusters_per_service = topology.accessed_clusters_per_access
    if parallel_per_slab * clusters_per_service > (
            physical_layout.clusters_per_slab):
        raise ValueError(
            "parallel service groups exceed instantiated clusters per slab")
    if topology.accessed_subarrays_per_access != (
            clusters_per_service * topology.subarrays_per_cluster):
        raise ValueError(
            "canonical service payload requires all subarrays in each "
            "participating cluster")
    delivered_bits = topology.delivered_bits_per_access
    expected_bits = (
        topology.accessed_subarrays_per_access
        * topology.selected_bits_per_subarray)
    if delivered_bits != expected_bits or delivered_bits % 8 != 0:
        raise ValueError("M3D delivered service payload must close to whole bytes")
    payload_bytes = delivered_bits / 8.0
    slabs = physical_layout.slab_count
    total_parallel = slabs * parallel_per_slab
    numerator_bytes = total_parallel * payload_bytes
    latencies = tuple(
        item.physical_access_latency_ns for item in physical_layout.slot_classes)
    if not latencies or any(
            not math.isfinite(value) or value <= 0.0 for value in latencies):
        raise ValueError("physical service latencies must be finite and positive")
    scale = spec.internal.service_cycle_scale
    fast_cycle = scale * min(latencies)
    average_cycle = scale * statistics.fmean(latencies)
    slow_cycle = scale * max(latencies)

    ordered = tuple(sorted(latencies))
    prefixes = []
    for fraction in (0.10, 0.25, 0.50, 0.75, 0.90, 1.00):
        count = math.ceil(fraction * len(ordered))
        cycle = scale * statistics.fmean(ordered[:count])
        prefixes.append(InternalBandwidthPrefix(
            capacity_fraction=fraction,
            average_service_cycle_ns=cycle,
            internal_bandwidth_bytes_per_s=(
                numerator_bytes / (cycle * 1e-9)),
        ))
    coil_bits = (
        slabs
        * spec.coil.links_per_die
        * spec.coil.data_rate_gbps_per_link
        * 1e9)
    return ArchitectureBandwidthClosure(
        num_m3d_dies=slabs,
        die_count_source=spec.die_count_source,
        coil_links_per_die=spec.coil.links_per_die,
        coil_data_rate_gbps_per_link=(
            spec.coil.data_rate_gbps_per_link),
        coil_bandwidth_bits_per_s=coil_bits,
        coil_bandwidth_bytes_per_s=coil_bits / 8.0,
        coil_classification=spec.coil.classification,
        coil_parameter_classification=(
            spec.coil.parameter_classification),
        internal_service_model=spec.internal.model,
        internal_service_unit=spec.internal.service_unit,
        parallel_service_units_per_slab=parallel_per_slab,
        parallel_slabs=slabs,
        total_parallel_service_units=total_parallel,
        clusters_per_service=clusters_per_service,
        subarrays_per_service=topology.accessed_subarrays_per_access,
        delivered_bits_per_service=delivered_bits,
        read_payload_bytes_per_service=payload_bytes,
        service_cycle_scale=scale,
        service_cycle_source=spec.internal.service_cycle_source,
        fast_service_cycle_ns=fast_cycle,
        average_service_cycle_ns=average_cycle,
        slow_service_cycle_ns=slow_cycle,
        internal_bandwidth_fast_bytes_per_s=(
            numerator_bytes / (fast_cycle * 1e-9)),
        internal_bandwidth_average_bytes_per_s=(
            numerator_bytes / (average_cycle * 1e-9)),
        internal_bandwidth_slow_bytes_per_s=(
            numerator_bytes / (slow_cycle * 1e-9)),
        prefix_bandwidth=tuple(prefixes),
        gpu_internal_bandwidth_bytes_per_s=(
            spec.gpu_internal.bandwidth_bytes_per_s),
        gpu_internal_status=spec.gpu_internal.status,
        internal_classification=spec.internal.classification,
        access_topology_provenance=topology.access_provenance,
        payload_status="DERIVED_FROM_M3D_ACCESS_TOPOLOGY",
        physical_latency_status=physical_layout.latency_source_status,
        aggregation_semantics=(
            "FEOL_IO_SERVICE_LANES_ADD_ACROSS_SLABS;CLUSTERS_WITHIN_ONE_"
            "ACCESS_GROUP_AND_LAYERS_SHARING_A_LANE_DO_NOT_ADD"),
        layer_parallelism_semantics=(
            "EIGHT_LAYERS_SHARE_PER_SLAB_FEOL_IO_SERVICE_LANES"),
    )


def resolve_effective_bandwidth(
    closure: ArchitectureBandwidthClosure,
    physical_access_latency_ns: float,
    *,
    internal_parallelism_scale: float = 1.0,
) -> EffectiveBandwidth:
    """Resolve one placement-dependent path bottleneck."""
    latency = _positive_finite(
        physical_access_latency_ns, "physical_access_latency_ns")
    parallelism = _positive_finite(
        internal_parallelism_scale, "internal_parallelism_scale")
    service_cycle = closure.service_cycle_scale * latency
    internal = (
        closure.total_parallel_service_units
        * parallelism
        * closure.read_payload_bytes_per_service
        / (service_cycle * 1e-9))
    candidates = [
        (internal, "INTERNAL"),
        (closure.coil_bandwidth_bytes_per_s, "COIL_INTERFACE"),
    ]
    if closure.gpu_internal_bandwidth_bytes_per_s is not None:
        candidates.append((
            closure.gpu_internal_bandwidth_bytes_per_s, "GPU_INTERNAL"))
    effective, bottleneck = min(candidates, key=lambda item: item[0])
    return EffectiveBandwidth(
        physical_access_latency_ns=latency,
        service_cycle_ns=service_cycle,
        internal_parallelism_scale=parallelism,
        internal_bandwidth_bytes_per_s=internal,
        coil_bandwidth_bytes_per_s=closure.coil_bandwidth_bytes_per_s,
        gpu_internal_bandwidth_bytes_per_s=(
            closure.gpu_internal_bandwidth_bytes_per_s),
        effective_bandwidth_bytes_per_s=effective,
        bottleneck=bottleneck,
        model_name="HIERARCHICAL_BANDWIDTH_MODEL",
    )


def _positive_finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result
