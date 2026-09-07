"""A-final dense decode: path-correct NMP and operator-locality accounting.

The NMP path intentionally bypasses the long edge FEOL and external bulk-data
path.  It does not create a die-to-die network: all remaining cross-domain
activation and partial-result traffic is charged to the fixed GPU interface.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import math
import statistics
from typing import Literal

from om3dthermal.power.memory_bandwidth import ArchitectureBandwidthClosure, resolve_internal_service_bandwidth
from om3dthermal.power.physical_capacity import PhysicalCapacityLayout
from om3dthermal.power.physical_latency import PhysicalAccessLatency
from om3dthermal.workload.llm_decode import LLMDecodeInput, evaluate_llm_decode
from om3dthermal.workload.m3d_page_demand import M3DWorkloadPageDemand

Case = Literal["NON_NMP_GPU", "NMP_NAIVE", "NMP_LOCALITY_AWARE_PLACEMENT"]
NMP_BANK_TO_LOCAL_ROUTE_DELAY_NS = 1.0
NMP_LOCAL_ROUTE_PROVENANCE = "MODELING_CHOICE_FIXED_LOCAL_NMP_ROUTE_DELAY__NOT_PHYSICALLY_EXTRACTED__NOT_OPTIMIZED__NOT_POSITION_DEPENDENT"

from om3dthermal.workload.dense_decode_ledger import (DenseDecodePlacementUnit,
    active_weight_read_bytes, build_dense_decode_placement_units, boundary_bytes_per_die)

@dataclass(frozen=True)
class NMPPlacementMetrics:
    case: Case; physical_die_count: int; mean_operator_die_span: float; median_operator_die_span: float
    max_operator_die_span: int; mean_layer_die_span: float; die_used_bytes: tuple[int, ...]
    local_access_latency_ns: float; long_feol_edge_included: bool; local_route_status: str

@dataclass(frozen=True)
class NMPTraffic:
    local_weight_read_bytes: float; local_kv_read_bytes: float; local_kv_write_bytes: float
    weight_bulk_external_bytes: float; kv_bulk_external_bytes: float
    gpu_to_die_activation_bytes: float; die_to_gpu_partial_bytes: float; die_to_gpu_output_bytes: float
    gpu_to_die_next_stage_bytes: float; direct_die_to_die_bytes: float
    local_memory_bytes: float; external_interface_bytes: float

@dataclass(frozen=True)
class NMPCaseTiming:
    case: Case; nmp_aggregate_tflops: float | None; local_memory_ms: float; external_ms: float
    nmp_compute_ms: float; gpu_small_compute_ms: float; total_step_ms: float; tokens_per_s: float
    nmp_compute_crossover_tflops: float | None; average_tflops_per_die: float | None; bottleneck: str
    raw_internal_bandwidth_bytes_per_s: float | None; external_bandwidth_bytes_per_s: float
    memory_serial_ms: float; total_step_serial_ms: float; memory_pipeline_ms: float; total_step_pipeline_ms: float
    tokens_per_s_serial: float; tokens_per_s_pipeline: float

@dataclass(frozen=True)
class NMPFinalResult:
    placement: NMPPlacementMetrics; traffic: NMPTraffic; timing: NMPCaseTiming
    def as_dict(self) -> dict[str, object]: return asdict(self)

def independent_physical_die_count(layout: PhysicalCapacityLayout) -> int:
    """Architecture-defined invariant: one symmetric slab is one M3D die."""
    return layout.slab_count

def _spans(units: tuple[DenseDecodePlacementUnit, ...], layout: PhysicalCapacityLayout,
           locality: bool) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Capacity-balanced stripe versus deterministic operator grouping spans."""
    dies=layout.slab_count; cap=layout.capacity_per_slab_bytes; pagesize=layout.slot_capacity_bytes
    used=[0]*dies; spans=[]; cursor=0
    for unit in units:
        size=max(unit.weight_bytes+unit.kv_bytes, pagesize)
        needed=max(1, math.ceil(size/cap))
        if locality:
            candidates=sorted(range(dies), key=lambda d:(used[d],d))
            chosen=candidates[:needed]
        else:
            # Normal capacity page striping: page-sized pieces continue across dies.
            pages=math.ceil(size/pagesize); chosen=[]
            for _ in range(pages):
                d=cursor % dies; cursor += 1
                if d not in chosen: chosen.append(d)
        share=math.ceil(size/len(chosen))
        for d in chosen: used[d]+=share
        spans.append(tuple(chosen))
    # The true resident allocation remains protected by existing page capacity;
    # this is only an operator-affinity ownership map and cannot exceed it.
    if max(used) > cap:
        raise ValueError("operator ownership exceeded per-die capacity")
    return tuple(spans), tuple(used)

def _placement(case: Case, units: tuple[DenseDecodePlacementUnit, ...], layout: PhysicalCapacityLayout,
               physical: PhysicalAccessLatency) -> NMPPlacementMetrics:
    if case == "NON_NMP_GPU":
        spans, used = _spans(units, layout, False)
        access=physical.uniform_average_total_latency_ns; edge=True
    else:
        spans, used = _spans(units, layout, case == "NMP_LOCALITY_AWARE_PLACEMENT")
        # Existing MIV layer delays, but no long FEOL / edge interface.
        access=statistics.fmean(x.mat_latency_ns+x.miv_latency_ns+NMP_BANK_TO_LOCAL_ROUTE_DELAY_NS for x in physical.locations)
        edge=False
    values=tuple(len(x) for x in spans)
    layers=[]
    for layer in range(max(x.layer_id for x in units)+1):
        layers.append(len(set(d for u,s in zip(units,spans) if u.layer_id==layer for d in s)))
    return NMPPlacementMetrics(case, layout.slab_count, statistics.fmean(values), statistics.median(values), max(values),
        statistics.fmean(layers), used, access, edge,
        NMP_LOCAL_ROUTE_PROVENANCE if not edge else "LONG_FEOL_EDGE_ROUTE_INCLUDED")

def build_locality_aware_unit_ownership(workload: LLMDecodeInput, layout: PhysicalCapacityLayout):
    """Return deterministic operator units and their physical-die ownership."""
    units=build_dense_decode_placement_units(workload)
    spans,_=_spans(units,layout,True)
    return units,spans

def _traffic(case: Case, demand: M3DWorkloadPageDemand, units: tuple[DenseDecodePlacementUnit, ...],
             placement: NMPPlacementMetrics, layout: PhysicalCapacityLayout) -> NMPTraffic:
    weight=active_weight_read_bytes(units); kvread=sum(u.kv_bytes for u in units); kvwrite=demand.kv_write_bytes_per_decode_step
    if case == "NON_NMP_GPU":
        return NMPTraffic(0,0,0,weight,kvread+kvwrite,0,0,0,0,0,0,weight+kvread+kvwrite)
    spans,_=_spans(units,layout,case=="NMP_LOCALITY_AWARE_PLACEMENT")
    score=sum(u.score_bytes for u in units)
    prob=sum(u.probability_bytes for u in units)
    activation=sum(u.activation_input_bytes*len(o) for u,o in zip(units,spans)
                   if u.shard_mode=="ROW_PARALLEL")
    boundary=sum(boundary_bytes_per_die(units,spans,layout.slab_count))
    output=sum(u.partial_output_bytes for u in units if u.shard_mode=="ROW_PARALLEL")
    partial=boundary-score-prob-activation-output
    return NMPTraffic(weight,kvread,kvwrite,0,0,activation+prob,partial,score,output,0,
        weight+kvread+kvwrite,boundary)

def evaluate_nmp_locality_case(workload: LLMDecodeInput, demand: M3DWorkloadPageDemand,
        layout: PhysicalCapacityLayout, physical: PhysicalAccessLatency, bandwidth: ArchitectureBandwidthClosure, *,
        case: Case, gpu_compute_flops_per_s: float,
        external_bandwidth_cap_bytes_per_s: float | None = None) -> NMPFinalResult:
    units=build_dense_decode_placement_units(workload); placement=_placement(case,units,layout,physical); traffic=_traffic(case,demand,units,placement,layout)
    from pathlib import Path
    from om3dthermal.platform import load_platform_spec_file, resolve_gpu_bandwidth_service, resolve_local_memory_gpu_transfer
    platform=load_platform_spec_file(Path(__file__).resolve().parents[3]/"configs/platform/gpu_package_h200_reference.yaml")
    spec=platform.gpu_bandwidth_service
    gpu_bw=resolve_gpu_bandwidth_service(
        transfer_ceiling_bytes_per_s=platform.gpu_decode_power.peak_memory_bandwidth_bytes_per_s,
        gpu_bandwidth_utilization=spec.nominal_utilization,
        utilization_status=spec.utilization_status,utilization_provenance=spec.provenance).sustained_bandwidth_bytes_per_s
    demand_bw=bandwidth.coil_bandwidth_bytes_per_s if external_bandwidth_cap_bytes_per_s is None else external_bandwidth_cap_bytes_per_s
    external_bw=resolve_local_memory_gpu_transfer(bandwidth_demand_bytes_per_s=demand_bw,
        memory_capability_bytes_per_s=bandwidth.coil_bandwidth_bytes_per_s,
        gpu_peak_bandwidth_bytes_per_s=gpu_bw).bandwidth_actual_bytes_per_s
    if case == "NON_NMP_GPU":
        local_bw=resolve_internal_service_bandwidth(bandwidth,placement.local_access_latency_ns)
        local_ms=(traffic.weight_bulk_external_bytes+traffic.kv_bulk_external_bytes)/local_bw*1e3
        external_ms=traffic.external_interface_bytes/external_bw*1e3
        gpu_ms=workload.batch_size*evaluate_llm_decode(workload).flops_per_token/gpu_compute_flops_per_s*1e3
        memory_serial=local_ms+external_ms; total_serial=memory_serial+gpu_ms
        memory_pipeline=max(local_ms,external_ms); total_pipeline=max(memory_pipeline,gpu_ms)
        timing=NMPCaseTiming(case,None,local_ms,external_ms,0,gpu_ms,total_pipeline,workload.batch_size/(total_pipeline*1e-3),None,None,
            "EXTERNAL" if external_ms >= local_ms and external_ms >= gpu_ms else ("GPU" if gpu_ms >= local_ms else "INTERNAL"),
            local_bw,external_bw,memory_serial,total_serial,memory_pipeline,total_pipeline,
            workload.batch_size/(total_serial*1e-3),workload.batch_size/(total_pipeline*1e-3))
        return NMPFinalResult(placement,traffic,timing)
    from om3dthermal.power.nmp_die_activity import canonical_nmp_hardware, evaluate_nmp_die_activity
    hw=canonical_nmp_hardware(layout.slab_count)
    nmp_aggregate_tflops=hw.aggregate_peak_flops/1e12
    spans,_=_spans(units,layout,case=="NMP_LOCALITY_AWARE_PLACEMENT")
    activity=evaluate_nmp_die_activity(workload,demand,layout,bandwidth,
        local_access_latency_ns=placement.local_access_latency_ns,
        bandwidth_demand_bytes_per_s=demand_bw,ownership=spans)
    local_ms=sum(s["memory_ms"] for s in activity.stages if "memory_ms" in s)
    nmp_ms=sum(s["compute_ms"] for s in activity.stages if "compute_ms" in s)
    total=activity.decode_step_interval_ms
    tps=workload.batch_size/(total*1e-3)
    timing=NMPCaseTiming(case,nmp_aggregate_tflops,local_ms,activity.boundary_time_ms,
        nmp_ms,activity.softmax_time_ms,total,tps,None,hw.peak_flops_per_die/1e12,
        "SERIAL_DEPENDENT_STAGES",activity.aggregate_local_bandwidth_bytes_per_s,
        external_bw,local_ms+activity.boundary_time_ms,total,
        local_ms+activity.boundary_time_ms,total,tps,tps)
    return NMPFinalResult(placement,traffic,timing)
