"""Deterministic die-level performance balancing under fixed locality."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import math, statistics
from om3dthermal.power.physical_capacity import PhysicalCapacityLayout
from om3dthermal.workload.llm_decode import LLMDecodeInput
from om3dthermal.workload.m3d_page_demand import M3DWorkloadPageDemand
from om3dthermal.workload.dense_decode_ledger import DenseDecodePlacementUnit, build_dense_decode_placement_units, boundary_bytes_per_die

@dataclass(frozen=True)
class NMPPlacementUnitLoad:
    unit: DenseDecodePlacementUnit; resident_bytes: float
    weight_read_bytes: float; kv_read_bytes: float; kv_write_bytes: float
    local_memory_traffic_bytes: float; nmp_flops: float; minimum_die_span: int

@dataclass(frozen=True)
class NMPPerformanceBalancedPlacement:
    unit_loads: tuple[NMPPlacementUnitLoad,...]; ownership: tuple[tuple[int,...],...]
    resident_used_bytes_per_die: tuple[float,...]; traffic_bytes_per_die: tuple[float,...]
    flops_per_die: tuple[float,...]; service_time_ms_per_die: tuple[float,...]
    operator_die_spans: tuple[int,...]; max_capacity_utilization: float
    mean_capacity_utilization: float; capacity_violations: int
    algorithm: str; locality_constraint: str
    def as_dict(self): return asdict(self)

def derive_unit_loads(workload:LLMDecodeInput,demand:M3DWorkloadPageDemand,layout:PhysicalCapacityLayout)->tuple[NMPPlacementUnitLoad,...]:
    units=build_dense_decode_placement_units(workload)
    if not math.isclose(sum(u.weight_bytes for u in units),demand.weight_footprint_bytes):
        raise ValueError("workload weight ledger and demand disagree")
    if not math.isclose(sum(u.kv_bytes for u in units),demand.kv_footprint_bytes):
        raise ValueError("workload KV ledger and demand disagree")
    raw_resident=sum(u.weight_bytes+u.kv_bytes for u in units)
    runtime=float(demand.runtime_footprint_bytes)
    loads=[]
    for u in units:
        weight_res=u.weight_bytes
        kv_res=u.kv_bytes
        raw=u.weight_bytes+u.kv_bytes
        resident=weight_res+kv_res+(runtime*raw/raw_resident if raw_resident else 0.0)
        weight_read=u.weight_bytes
        kv_read=u.kv_bytes; kv_write=u.kv_write_bytes
        traffic=weight_read+kv_read+kv_write
        unit_flops=(workload.batch_size if u.placement_scope=="SHARED_BATCH" else 1)*u.local_flops
        loads.append(NMPPlacementUnitLoad(u,resident,weight_read,kv_read,kv_write,traffic,unit_flops,max(1,math.ceil(resident/layout.capacity_per_slab_bytes))))
    return tuple(loads)

def build_performance_balanced_placement(workload:LLMDecodeInput,demand:M3DWorkloadPageDemand,
        layout:PhysicalCapacityLayout,*,bandwidth_per_die_bytes_per_s:float,compute_per_die_flops_per_s:float)->NMPPerformanceBalancedPlacement:
    loads=derive_unit_loads(workload,demand,layout); n=layout.slab_count; cap=layout.capacity_per_slab_bytes
    resident=[0.0]*n; traffic=[0.0]*n; flops=[0.0]*n; ownership=[None]*len(loads)
    order=sorted(range(len(loads)),key=lambda i:(-max(loads[i].local_memory_traffic_bytes/bandwidth_per_die_bytes_per_s,loads[i].nmp_flops/compute_per_die_flops_per_s),loads[i].unit.unit_id))
    for index in order:
        load=loads[index]; span=load.minimum_die_span
        # Current dense units all have span one.  The general path greedily
        # chooses exactly the minimum number of feasible dies, never more.
        chosen=[]
        for _ in range(span):
            candidates=[]
            for die in range(n):
                if die in chosen: continue
                share=1/span
                if resident[die]+load.resident_bytes*share>cap: continue
                t=traffic.copy(); f=flops.copy(); r=resident.copy()
                t[die]+=load.local_memory_traffic_bytes*share; f[die]+=load.nmp_flops*share; r[die]+=load.resident_bytes*share
                stage=max(max(t[d]/bandwidth_per_die_bytes_per_s,f[d]/compute_per_die_flops_per_s) for d in range(n))
                maxcap=max(r)/cap
                services=[max(t[d]/bandwidth_per_die_bytes_per_s,f[d]/compute_per_die_flops_per_s) for d in range(n)]
                candidates.append(((stage,maxcap,statistics.pvariance(services),die),die))
            if not candidates: raise ValueError("PERFORMANCE_BALANCED_CAPACITY_FAIL")
            chosen.append(min(candidates)[1])
        share=1/span
        for die in chosen:
            resident[die]+=load.resident_bytes*share; traffic[die]+=load.local_memory_traffic_bytes*share; flops[die]+=load.nmp_flops*share
        ownership[index]=tuple(sorted(chosen))
    service=tuple(max(traffic[d]/bandwidth_per_die_bytes_per_s,flops[d]/compute_per_die_flops_per_s)*1e3 for d in range(n))
    return NMPPerformanceBalancedPlacement(loads,tuple(ownership),tuple(resident),tuple(traffic),tuple(flops),service,
        tuple(len(x) for x in ownership),max(resident)/cap,statistics.fmean(resident)/cap,sum(x>cap for x in resident),
        "DETERMINISTIC_LPT_MINIMIZE_PROJECTED_MAX_SERVICE__TIE_MAX_CAPACITY_VARIANCE_DIE_ID",
        "LEXICOGRAPHIC_MINIMUM_DIE_SPAN_THEN_STAGE_TIME")

def build_locality_only_placement(workload:LLMDecodeInput,demand:M3DWorkloadPageDemand,
        layout:PhysicalCapacityLayout,*,bandwidth_per_die_bytes_per_s:float,compute_per_die_flops_per_s:float)->NMPPerformanceBalancedPlacement:
    """Capacity-balanced first-touch baseline without runtime-load awareness."""
    loads=derive_unit_loads(workload,demand,layout); n=layout.slab_count; cap=layout.capacity_per_slab_bytes
    resident=[0.0]*n; traffic=[0.0]*n; flops=[0.0]*n; ownership=[]
    for load in loads:
        span=load.minimum_die_span; feasible=sorted(
            (d for d in range(n) if resident[d]+load.resident_bytes/span<=cap),
            key=lambda d:(resident[d],d))
        if len(feasible)<span: raise ValueError("LOCALITY_ONLY_CAPACITY_FAIL")
        chosen=tuple(sorted(feasible[:span])); ownership.append(chosen)
        for die in chosen:
            resident[die]+=load.resident_bytes/span; traffic[die]+=load.local_memory_traffic_bytes/span; flops[die]+=load.nmp_flops/span
    service=tuple(max(traffic[d]/bandwidth_per_die_bytes_per_s,flops[d]/compute_per_die_flops_per_s)*1e3 for d in range(n))
    return NMPPerformanceBalancedPlacement(loads,tuple(ownership),tuple(resident),tuple(traffic),tuple(flops),service,
        tuple(len(x) for x in ownership),max(resident)/cap,statistics.fmean(resident)/cap,sum(x>cap for x in resident),
        "CAPACITY_BALANCED_FIRST_TOUCH_LOCALITY_ONLY","MINIMUM_DIE_SPAN_ONLY__RUNTIME_LOAD_OBLIVIOUS")

def remaining_external_bytes_for_ownership(loads, ownership)->float:
    die_count=1+max(d for owners in ownership for d in owners)
    return sum(external_bytes_per_die_for_ownership(loads,ownership,die_count))


def external_bytes_per_die_for_ownership(loads, ownership, die_count)->tuple[float,...]:
    return boundary_bytes_per_die(tuple(x.unit for x in loads),ownership,die_count)
