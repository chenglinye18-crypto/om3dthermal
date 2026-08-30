"""Dense decode placement accounting with strictly GPU-mediated domains.

This deliberately does not infer that a physical ``slab`` is a package die.
It uses the existing symmetric slab index as an *independent memory-domain*
modeling choice, and charges every cross-domain exchange to the fixed GPU
interface.  In particular, direct die-to-die traffic is structurally zero.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import math
from typing import Literal

from om3dthermal.power.memory_bandwidth import ArchitectureBandwidthClosure
from om3dthermal.power.physical_capacity import PhysicalCapacityLayout, iter_physical_slots
from om3dthermal.workload.llm_decode import LLMDecodeInput, evaluate_llm_decode
from om3dthermal.workload.m3d_page_demand import M3DWorkloadPageDemand

from .fast_region import place_pages_on_slots

CommunicationModel = Literal["FUSED_DIE_LOCAL", "CONSERVATIVE_GPU_BOUNDARY"]
PlacementPolicy = Literal["NO_TIER_WORST_CASE", "DIE_LOCAL_BALANCED_FAST_PACK"]
OverlapPolicy = Literal["NO_OVERLAP", "PERFECT_RESOURCE_OVERLAP"]


@dataclass(frozen=True)
class IndependentDomainSemantics:
    independent_memory_domain_count: int
    capacity_per_domain_bytes: int
    slots_per_domain: int
    domain_id_source: str
    provenance: str = "ARCHITECTURE_DEFINED_ONE_SLAB_PER_PHYSICAL_DIE"


@dataclass(frozen=True)
class DieLocalTraffic:
    model: CommunicationModel
    local_weight_read_bytes: float
    local_kv_read_bytes: float
    local_kv_write_bytes: float
    gpu_to_die_activation_bytes: float
    die_to_gpu_partial_sum_bytes: float
    die_to_gpu_output_bytes: float
    gpu_to_die_kv_update_bytes: float
    direct_die_to_die_bytes: float
    local_memory_bytes: float
    external_interface_bytes: float
    derived_local_byte_fraction: float
    local_compute_flops: float
    gpu_small_operator_flops: float


@dataclass(frozen=True)
class DieLocalFastPack:
    active_domain_count: int
    domain_page_counts: tuple[int, ...]
    occupancy_fraction_per_domain: float
    weighted_average_access_latency_ns: float
    global_fast_pack_latency_ns: float
    balance_status: str


@dataclass(frozen=True)
class DieLocalTiming:
    policy: PlacementPolicy
    overlap: OverlapPolicy
    physical_latency_ns: float
    internal_bandwidth_bytes_per_s: float
    local_memory_time_ms: float
    external_interface_time_ms: float
    memory_time_ms: float
    gpu_small_compute_time_ms: float
    total_step_ms: float
    aggregate_tokens_per_s: float
    required_local_compute_tflops: float
    bottleneck: str


@dataclass(frozen=True)
class DieLocalComparison:
    active_domain_count: int
    traffic: DieLocalTraffic
    fast_pack: DieLocalFastPack
    no_tier_no_overlap: DieLocalTiming
    fast_no_overlap: DieLocalTiming
    no_tier_perfect_overlap: DieLocalTiming
    fast_perfect_overlap: DieLocalTiming
    placement_speedup: float
    e2e_speedup_no_overlap: float
    e2e_speedup_perfect_overlap: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def independent_domain_semantics(layout: PhysicalCapacityLayout) -> IndependentDomainSemantics:
    """Expose the architecture-defined one-slab-per-physical-die mapping."""
    return IndependentDomainSemantics(
        independent_memory_domain_count=layout.slab_count,
        capacity_per_domain_bytes=layout.capacity_per_slab_bytes,
        slots_per_domain=layout.clusters_per_slab * layout.layers_per_cluster,
        domain_id_source="ARCHITECTURE_DEFINED_PhysicalSlot.slab_id__ONE_PHYSICAL_DIE",
    )


def minimum_active_domains(demand: M3DWorkloadPageDemand,
                           domains: IndependentDomainSemantics) -> int:
    return max(math.ceil(demand.allocated_page_bytes / domains.capacity_per_domain_bytes),
               math.ceil(demand.page_count / domains.slots_per_domain))


def _balanced_fast_pack(demand: M3DWorkloadPageDemand, layout: PhysicalCapacityLayout,
                        active_domains: int) -> DieLocalFastPack:
    domains = independent_domain_semantics(layout)
    minimum = minimum_active_domains(demand, domains)
    if active_domains < minimum:
        raise ValueError("ACTIVE_DOMAIN_COUNT_BELOW_D_MIN_FIT")
    if active_domains > domains.independent_memory_domain_count:
        raise ValueError("ACTIVE_DOMAIN_COUNT_EXCEEDS_D_MAX")
    base, remainder = divmod(demand.page_count, active_domains)
    counts = tuple(base + (index < remainder) for index in range(active_domains))
    if max(counts) - min(counts) > 1:
        raise RuntimeError("domain page balance failed")
    pages = sorted(demand.page_demands, key=lambda x: (-x.read_demand_bytes_per_decode_step, x.page_id))
    # Round-robin sends high-demand pages to every symmetric domain, then each
    # domain gets its own fastest capacity prefix.  No page crosses a domain.
    bins = [[] for _ in range(active_domains)]
    for index, page in enumerate(pages):
        bins[index % active_domains].append(page)
    weighted_sum = 0.0
    total = demand.total_read_bytes_per_decode_step
    for domain, assigned in enumerate(bins):
        slots = _domain_fast_slots(layout, domain)
        for page, slot in zip(sorted(assigned, key=lambda x: (-x.read_demand_bytes_per_decode_step, x.page_id)), slots[:len(assigned)], strict=True):
            weighted_sum += page.read_demand_bytes_per_decode_step * slot.physical_access_latency_ns
    global_fast = place_pages_on_slots(
        demand, layout, slot_policy="FASTEST", page_ordering="DEMAND_DESCENDING")
    return DieLocalFastPack(
        active_domain_count=active_domains,
        domain_page_counts=counts,
        occupancy_fraction_per_domain=max(counts) / domains.slots_per_domain,
        weighted_average_access_latency_ns=weighted_sum / total,
        global_fast_pack_latency_ns=global_fast.weighted_average_access_latency_ns,
        balance_status="DOMAIN_PAGE_BALANCE_MAX_MINUS_MIN_LEQ_1",
    )


@lru_cache(maxsize=None)
def _domain_fast_slots(layout: PhysicalCapacityLayout, domain: int):
    return tuple(sorted((slot for slot in iter_physical_slots(layout) if slot.slab_id == domain),
                        key=lambda x: (x.physical_access_latency_ns, x.cluster_id, x.layer_id)))


def _traffic(workload: LLMDecodeInput, demand: M3DWorkloadPageDemand,
             active_domains: int, model: CommunicationModel) -> DieLocalTraffic:
    bytes_per_value = workload.weight_bits / 8.0
    vector = workload.batch_size * workload.d_model * bytes_per_value
    kv_width = workload.n_heads_kv * workload.d_head
    if model == "FUSED_DIE_LOCAL":
        activation = workload.n_layers * active_domains * vector
        partial = workload.n_layers * active_domains * vector
        kv_update = 0.0  # local K/V projection produces its local shard.
    else:
        # Q/K/V/O/gate/up/down inputs and results pass the GPU boundary.
        activation_dims = 6 * workload.d_model + workload.d_ff
        result_dims = 3 * workload.d_model + 2 * kv_width + 2 * workload.d_ff
        activation = workload.n_layers * active_domains * workload.batch_size * activation_dims * bytes_per_value
        partial = workload.n_layers * active_domains * workload.batch_size * result_dims * bytes_per_value
        kv_update = demand.kv_write_bytes_per_decode_step
    output = vector
    local = (demand.total_weight_read_bytes_per_decode_step + demand.total_kv_read_bytes_per_decode_step + demand.kv_write_bytes_per_decode_step)
    external = activation + partial + output + kv_update
    metrics = evaluate_llm_decode(workload)
    linear = demand.total_weight_read_bytes_per_decode_step  # FP16: 2 FLOP/parameter = bytes numerically.
    attention = workload.batch_size * 4 * workload.n_layers * workload.n_heads_q * workload.context_length * workload.d_head
    local_flops = linear + attention
    total_flops = workload.batch_size * metrics.flops_per_token
    return DieLocalTraffic(model, demand.total_weight_read_bytes_per_decode_step,
        demand.total_kv_read_bytes_per_decode_step, demand.kv_write_bytes_per_decode_step,
        activation, partial, output, kv_update, 0.0, local, external,
        local / (local + external), local_flops, max(0.0, total_flops - local_flops))


def evaluate_die_local_placement(workload: LLMDecodeInput, demand: M3DWorkloadPageDemand,
        layout: PhysicalCapacityLayout, bandwidth: ArchitectureBandwidthClosure, *,
        active_domains: int, communication_model: CommunicationModel,
        gpu_compute_flops_per_s: float) -> DieLocalComparison:
    """Placement-only comparison at fixed sharding and fixed external link."""
    if workload.batch_size != demand.requested_requests:
        raise ValueError("workload/demand batch mismatch")
    if gpu_compute_flops_per_s <= 0:
        raise ValueError("gpu_compute_flops_per_s must be positive")
    pack = _balanced_fast_pack(demand, layout, active_domains)
    traffic = _traffic(workload, demand, active_domains, communication_model)
    no_tier_latency = max(item.physical_access_latency_ns for item in layout.slot_classes)
    # Existing architectural semantics: FEOL service lanes add across active
    # slabs.  The external coil/GPU aggregate is intentionally *not* scaled.
    def internal(latency: float) -> float:
        return (active_domains * bandwidth.parallel_service_units_per_slab
                * bandwidth.read_payload_bytes_per_service
                / (bandwidth.service_cycle_scale * latency * 1e-9))
    external_bw = bandwidth.coil_bandwidth_bytes_per_s
    def timing(policy: PlacementPolicy, latency: float, overlap: OverlapPolicy) -> DieLocalTiming:
        local_ms = traffic.local_memory_bytes / internal(latency) * 1e3
        external_ms = traffic.external_interface_bytes / external_bw * 1e3
        memory_ms = local_ms + external_ms if overlap == "NO_OVERLAP" else max(local_ms, external_ms)
        gpu_ms = traffic.gpu_small_operator_flops / gpu_compute_flops_per_s * 1e3
        total = max(memory_ms, gpu_ms)
        return DieLocalTiming(policy, overlap, latency, internal(latency), local_ms, external_ms,
            memory_ms, gpu_ms, total, workload.batch_size / (total * 1e-3),
            traffic.local_compute_flops / (local_ms * 1e-3) / 1e12,
            "MEMORY" if memory_ms >= gpu_ms else "GPU_SMALL_COMPUTE")
    no_no = timing("NO_TIER_WORST_CASE", no_tier_latency, "NO_OVERLAP")
    fast_no = timing("DIE_LOCAL_BALANCED_FAST_PACK", pack.weighted_average_access_latency_ns, "NO_OVERLAP")
    no_ov = timing("NO_TIER_WORST_CASE", no_tier_latency, "PERFECT_RESOURCE_OVERLAP")
    fast_ov = timing("DIE_LOCAL_BALANCED_FAST_PACK", pack.weighted_average_access_latency_ns, "PERFECT_RESOURCE_OVERLAP")
    return DieLocalComparison(active_domains, traffic, pack, no_no, fast_no, no_ov, fast_ov,
        no_tier_latency / pack.weighted_average_access_latency_ns,
        no_no.total_step_ms / fast_no.total_step_ms, no_ov.total_step_ms / fast_ov.total_step_ms)
