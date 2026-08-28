"""Compact local physical-slot capacity semantics over the latency map."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterator

from .m3d_subarray import M3DSubarrayResult
from .physical_latency import PhysicalAccessLatency


@dataclass(frozen=True)
class PhysicalSlotClass:
    cluster_id: int
    layer_id: int
    capacity_bytes: int
    capacity_mib: float
    physical_access_latency_ns: float
    multiplicity: int
    feol_route_length_um: float
    miv_length_um: float


@dataclass(frozen=True)
class PhysicalSlot:
    slab_id: int
    cluster_id: int
    layer_id: int
    capacity_bytes: int
    physical_access_latency_ns: float


@dataclass(frozen=True)
class CapacityLatencyCutoff:
    capacity_fraction: float
    latency_cutoff_ns: float
    cumulative_capacity_bytes: int
    cumulative_capacity_gib: float
    included_slot_class_count: int


@dataclass(frozen=True)
class PhysicalCapacityLayout:
    slab_count: int
    clusters_per_slab: int
    layers_per_cluster: int
    subarrays_per_cluster: int
    subarray_rows: int
    subarray_cols: int
    subarray_capacity_bytes: int
    slot_capacity_bytes: int
    slot_class_count: int
    physical_slot_count: int
    capacity_per_layer_per_slab_bytes: int
    capacity_per_slab_bytes: int
    total_capacity_bytes: int
    total_capacity_gib: float
    slot_classes: tuple[PhysicalSlotClass, ...]
    capacity_latency_cutoffs: tuple[CapacityLatencyCutoff, ...]
    slab_symmetry: bool
    slab_symmetry_basis: str
    capacity_source_status: str
    latency_source_status: str
    host_capacity_included: bool
    workload_weighted: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def calculate_physical_capacity_layout(
        topology: M3DSubarrayResult, latency: PhysicalAccessLatency, *,
        slab_count: int, expected_total_bits: int | None = None,
        ) -> PhysicalCapacityLayout:
    """Build the local resource side; no data pages or placement variables.

    A future placement layer may impose ``sum_i x_ij*S_i <= C_j`` and
    ``sum_j x_ij = 1``.  This function deliberately defines only ``C_j``.
    """
    if slab_count <= 0:
        raise ValueError("physical slab count must be positive")
    if topology.subarrays_per_cluster != (
            topology.cluster_subarrays_x * topology.cluster_subarrays_y):
        raise ValueError("subarrays-per-cluster topology does not close")
    subarray_bits = topology.Nrow * topology.Ncol
    if subarray_bits != topology.bits_per_subarray:
        raise ValueError("subarray capacity does not close to topology")
    if subarray_bits % 8 != 0:
        raise ValueError("subarray capacity must contain whole bytes")
    if latency.number_of_clusters != topology.clusters_per_layer:
        raise ValueError("latency clusters do not match capacity topology")
    if latency.number_of_layers <= 0:
        raise ValueError("latency map must contain physical layers")

    subarray_capacity_bytes = subarray_bits // 8
    slot_capacity_bytes = (
        subarray_capacity_bytes * topology.subarrays_per_cluster)
    slot_classes = tuple(
        PhysicalSlotClass(
            cluster_id=location.cluster_id,
            layer_id=location.layer_id,
            capacity_bytes=slot_capacity_bytes,
            capacity_mib=slot_capacity_bytes / 2**20,
            physical_access_latency_ns=location.total_latency_ns,
            multiplicity=slab_count,
            feol_route_length_um=location.feol_route_length_um,
            miv_length_um=location.miv_length_um,
        )
        for location in latency.locations
    )
    if len({(slot.cluster_id, slot.layer_id) for slot in slot_classes}) != (
            len(slot_classes)):
        raise ValueError("physical slot-class indices must be unique")
    slot_class_count = len(slot_classes)
    expected_class_count = (
        topology.clusters_per_layer * latency.number_of_layers)
    if slot_class_count != expected_class_count:
        raise ValueError("slot classes do not close over clusters and layers")

    physical_slot_count = slot_class_count * slab_count
    capacity_per_layer_per_slab = (
        topology.clusters_per_layer * slot_capacity_bytes)
    capacity_per_slab = (
        capacity_per_layer_per_slab * latency.number_of_layers)
    total_capacity = capacity_per_slab * slab_count
    topology_bits_per_layer = (
        topology.clusters_per_layer
        * topology.subarrays_per_cluster
        * topology.bits_per_subarray)
    if topology_bits_per_layer != topology.bits_per_layer:
        raise ValueError("per-layer slot capacity does not close to topology")
    if expected_total_bits is not None and total_capacity * 8 != (
            expected_total_bits):
        raise ValueError(
            "physical slot capacity does not close to analytical packing")

    ordered = sorted(
        slot_classes, key=lambda slot: (
            slot.physical_access_latency_ns, slot.cluster_id, slot.layer_id))
    cutoffs: list[CapacityLatencyCutoff] = []
    for fraction in (0.10, 0.25, 0.50, 0.75, 0.90, 1.00):
        included_count = math.ceil(fraction * slot_class_count)
        if included_count <= 0:
            included_count = 1
        cumulative_capacity = (
            included_count * slot_capacity_bytes * slab_count)
        cutoffs.append(CapacityLatencyCutoff(
            capacity_fraction=fraction,
            latency_cutoff_ns=(
                ordered[included_count - 1].physical_access_latency_ns),
            cumulative_capacity_bytes=cumulative_capacity,
            cumulative_capacity_gib=cumulative_capacity / 2**30,
            included_slot_class_count=included_count,
        ))

    return PhysicalCapacityLayout(
        slab_count=slab_count,
        clusters_per_slab=topology.clusters_per_layer,
        layers_per_cluster=latency.number_of_layers,
        subarrays_per_cluster=topology.subarrays_per_cluster,
        subarray_rows=topology.Nrow,
        subarray_cols=topology.Ncol,
        subarray_capacity_bytes=subarray_capacity_bytes,
        slot_capacity_bytes=slot_capacity_bytes,
        slot_class_count=slot_class_count,
        physical_slot_count=physical_slot_count,
        capacity_per_layer_per_slab_bytes=capacity_per_layer_per_slab,
        capacity_per_slab_bytes=capacity_per_slab,
        total_capacity_bytes=total_capacity,
        total_capacity_gib=total_capacity / 2**30,
        slot_classes=slot_classes,
        capacity_latency_cutoffs=tuple(cutoffs),
        slab_symmetry=True,
        slab_symmetry_basis=(
            "ONE_CANONICAL_SLAB_GEOMETRY_AND_LATENCY_MAP_REPLICATED_BY_"
            "MEMORY_REGION_COUNT"),
        capacity_source_status=(
            "DYNAMIC_M3D_TOPOLOGY_CLOSED_TO_ANALYTICAL_PACKING_DIAGNOSTICS"),
        latency_source_status="EXISTING_PHYSICAL_ACCESS_LATENCY_MAP",
        host_capacity_included=False,
        workload_weighted=False,
    )


def iter_physical_slots(
        layout: PhysicalCapacityLayout) -> Iterator[PhysicalSlot]:
    """Expand compact slot classes lazily; the layout remains the sole truth."""
    for slot_class in layout.slot_classes:
        for slab_id in range(layout.slab_count):
            yield PhysicalSlot(
                slab_id=slab_id,
                cluster_id=slot_class.cluster_id,
                layer_id=slot_class.layer_id,
                capacity_bytes=slot_class.capacity_bytes,
                physical_access_latency_ns=(
                    slot_class.physical_access_latency_ns),
            )
