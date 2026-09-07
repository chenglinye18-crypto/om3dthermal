"""Deterministic stage-level sharding under fixed NMP locality constraints."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import math, statistics
from om3dthermal.power.physical_capacity import PhysicalCapacityLayout
from om3dthermal.workload.llm_decode import LLMDecodeInput
from om3dthermal.workload.m3d_page_demand import M3DWorkloadPageDemand
from om3dthermal.workload.dense_decode_ledger import (
    DenseDecodePlacementUnit, active_weight_read_bytes,
    build_dense_decode_placement_units, boundary_bytes_per_die,
    unit_shard_fractions)

@dataclass(frozen=True)
class NMPPlacementUnitLoad:
    unit: DenseDecodePlacementUnit; resident_bytes: float
    weight_read_bytes: float; kv_read_bytes: float; kv_write_bytes: float
    local_memory_traffic_bytes: float; nmp_flops: float; minimum_die_span: int

@dataclass(frozen=True)
class NMPShardAssignment:
    die_id: int; shard_index: int; shard_count: int; shard_fraction: float
    resident_bytes: float; weight_read_bytes: float; kv_read_bytes: float
    kv_write_bytes: float; local_memory_traffic_bytes: float; nmp_flops: float

@dataclass(frozen=True)
class NMPPerformanceBalancedPlacement:
    unit_loads: tuple[NMPPlacementUnitLoad,...]; ownership: tuple[tuple[int,...],...]
    shard_assignments: tuple[tuple[NMPShardAssignment,...],...]
    resident_used_bytes_per_die: tuple[float,...]; traffic_bytes_per_die: tuple[float,...]
    flops_per_die: tuple[float,...]; service_time_ms_per_die: tuple[float,...]
    operator_die_spans: tuple[int,...]; minimum_capacity_die_spans: tuple[int,...]
    max_capacity_utilization: float; mean_capacity_utilization: float
    capacity_violations: int; mean_exec_die_span: float; median_exec_die_span: float
    max_exec_die_span: int; algorithm: str; locality_constraint: str
    def as_dict(self): return asdict(self)

def derive_unit_loads(workload:LLMDecodeInput,demand:M3DWorkloadPageDemand,layout:PhysicalCapacityLayout)->tuple[NMPPlacementUnitLoad,...]:
    units=build_dense_decode_placement_units(workload)
    if not math.isclose(sum(u.weight_bytes for u in units),demand.weight_footprint_bytes):
        raise ValueError("workload resident-weight ledger and demand disagree")
    if not math.isclose(sum(u.kv_bytes for u in units),demand.kv_footprint_bytes):
        raise ValueError("workload KV ledger and demand disagree")
    if not math.isclose(active_weight_read_bytes(units),demand.total_weight_read_bytes_per_decode_step):
        raise ValueError("workload active-weight ledger and demand disagree")
    raw_resident=sum(u.weight_bytes+u.kv_bytes for u in units)
    runtime=float(demand.runtime_footprint_bytes)
    loads=[]
    for u in units:
        raw=u.weight_bytes+u.kv_bytes
        resident=raw+(runtime*raw/raw_resident if raw_resident else 0.0)
        weight_read=u.active_weight_read_bytes
        kv_read=u.kv_bytes; kv_write=u.kv_write_bytes
        traffic=weight_read+kv_read+kv_write
        unit_flops=(workload.batch_size if u.placement_scope=="SHARED_BATCH" else 1)*u.local_flops
        loads.append(NMPPlacementUnitLoad(u,resident,weight_read,kv_read,kv_write,
            traffic,unit_flops,max(1,math.ceil(resident/layout.capacity_per_slab_bytes))))
    if not math.isclose(sum(x.weight_read_bytes for x in loads),active_weight_read_bytes(units)):
        raise ValueError("active weight traffic ledger failed closure")
    return tuple(loads)

def _counts(total:int,span:int)->tuple[int,...]:
    quotient,remainder=divmod(total,span)
    return tuple(quotient+(index<remainder) for index in range(span))

def shard_fractions(load:NMPPlacementUnitLoad,span:int)->tuple[float,...]:
    """Analytical row/vector partition; an atomic item never crosses a die."""
    return unit_shard_fractions(load.unit,span)

def _stage_time(load:NMPPlacementUnitLoad,span:int,bw:float,compute:float)->float:
    fractions=shard_fractions(load,span)
    return max(max(load.local_memory_traffic_bytes*f/bw,load.nmp_flops*f/compute)
               for f in fractions)

def choose_execution_die_span(load:NMPPlacementUnitLoad,physical_die_count:int,
        bandwidth_per_die_bytes_per_s:float,compute_per_die_flops_per_s:float)->int:
    lower=load.minimum_die_span
    if load.unit.shard_mode == "RESIDENT_ONLY" or (not load.local_memory_traffic_bytes and not load.nmp_flops):
        return lower
    upper=min(physical_die_count,max(lower,load.unit.max_useful_parallelism))
    return min(range(lower,upper+1),key=lambda span:(
        _stage_time(load,span,bandwidth_per_die_bytes_per_s,compute_per_die_flops_per_s),span))

def _assign(load:NMPPlacementUnitLoad,owners:tuple[int,...])->tuple[NMPShardAssignment,...]:
    fractions=shard_fractions(load,len(owners))
    counts=_counts(load.unit.atomic_count,len(owners)) if load.unit.atomic_count else (0,)*len(owners)
    return tuple(NMPShardAssignment(die,index,counts[index],fraction,
        load.resident_bytes*fraction,load.weight_read_bytes*fraction,
        load.kv_read_bytes*fraction,load.kv_write_bytes*fraction,
        load.local_memory_traffic_bytes*fraction,load.nmp_flops*fraction)
        for index,(die,fraction) in enumerate(zip(owners,fractions,strict=True)))

def build_performance_balanced_placement(workload:LLMDecodeInput,demand:M3DWorkloadPageDemand,
        layout:PhysicalCapacityLayout,*,bandwidth_per_die_bytes_per_s:float,compute_per_die_flops_per_s:float)->NMPPerformanceBalancedPlacement:
    loads=derive_unit_loads(workload,demand,layout); n=layout.slab_count; cap=layout.capacity_per_slab_bytes
    resident=[0.0]*n; traffic=[0.0]*n; flops=[0.0]*n
    ownership=[]; assignments=[]; kv_pair_owners={}
    for load in loads:
        span=choose_execution_die_span(load,n,bandwidth_per_die_bytes_per_s,compute_per_die_flops_per_s)
        pair_key=(load.unit.layer_id,load.unit.request_id)
        paired=(load.unit.operator_type=="ATTENTION_AV" and pair_key in kv_pair_owners
                and len(kv_pair_owners[pair_key])==span)
        if paired:
            chosen=kv_pair_owners[pair_key]
        else:
            fractions=shard_fractions(load,span); feasible=[]
            for die in sorted(range(n),key=lambda d:(resident[d],d)):
                fraction=fractions[len(feasible)]
                if resident[die]+load.resident_bytes*fraction<=cap:
                    feasible.append(die)
                    if len(feasible)==span: break
            if len(feasible)<span: raise ValueError("PERFORMANCE_BALANCED_CAPACITY_FAIL")
            chosen=tuple(feasible)
        if load.unit.operator_type=="ATTENTION_QK": kv_pair_owners[pair_key]=chosen
        shards=_assign(load,chosen)
        for shard in shards:
            resident[shard.die_id]+=shard.resident_bytes
            traffic[shard.die_id]+=shard.local_memory_traffic_bytes
            flops[shard.die_id]+=shard.nmp_flops
        ownership.append(chosen); assignments.append(shards)
    service=tuple(max(traffic[d]/bandwidth_per_die_bytes_per_s,
                      flops[d]/compute_per_die_flops_per_s)*1e3 for d in range(n))
    spans=tuple(len(x) for x in ownership)
    active_spans=tuple(span for load,span in zip(loads,spans)
                       if load.unit.shard_mode!="RESIDENT_ONLY")
    return NMPPerformanceBalancedPlacement(loads,tuple(ownership),tuple(assignments),
        tuple(resident),tuple(traffic),tuple(flops),service,spans,
        tuple(x.minimum_die_span for x in loads),max(resident)/cap,
        statistics.fmean(resident)/cap,sum(x>cap for x in resident),
        statistics.fmean(active_spans),statistics.median(active_spans),max(active_spans),
        "DETERMINISTIC_PER_STAGE_SPAN_LATENCY_OPTIMIZER__TIE_FEWER_DIES__CAPACITY_BALANCED_OWNERS",
        "ROW_BLOCKS_AND_WHOLE_TOKEN_KV_HEAD_VECTORS__PAIRED_KV_COLOCATION")

def build_locality_only_placement(workload:LLMDecodeInput,demand:M3DWorkloadPageDemand,
        layout:PhysicalCapacityLayout,*,bandwidth_per_die_bytes_per_s:float,compute_per_die_flops_per_s:float)->NMPPerformanceBalancedPlacement:
    """Minimum-capacity-span reference, retained for placement comparisons."""
    loads=derive_unit_loads(workload,demand,layout); n=layout.slab_count; cap=layout.capacity_per_slab_bytes
    resident=[0.0]*n; traffic=[0.0]*n; flops=[0.0]*n; ownership=[]; assignments=[]
    for load in loads:
        span=load.minimum_die_span; fractions=shard_fractions(load,span); feasible=[]
        for die in sorted(range(n),key=lambda d:(resident[d],d)):
            if resident[die]+load.resident_bytes*fractions[len(feasible)]<=cap:
                feasible.append(die)
                if len(feasible)==span: break
        if len(feasible)<span: raise ValueError("LOCALITY_ONLY_CAPACITY_FAIL")
        chosen=tuple(feasible); shards=_assign(load,chosen)
        ownership.append(chosen); assignments.append(shards)
        for shard in shards:
            resident[shard.die_id]+=shard.resident_bytes; traffic[shard.die_id]+=shard.local_memory_traffic_bytes; flops[shard.die_id]+=shard.nmp_flops
    service=tuple(max(traffic[d]/bandwidth_per_die_bytes_per_s,flops[d]/compute_per_die_flops_per_s)*1e3 for d in range(n))
    spans=tuple(len(x) for x in ownership)
    active_spans=tuple(span for load,span in zip(loads,spans)
                       if load.unit.shard_mode!="RESIDENT_ONLY")
    return NMPPerformanceBalancedPlacement(loads,tuple(ownership),tuple(assignments),tuple(resident),tuple(traffic),tuple(flops),service,spans,
        tuple(x.minimum_die_span for x in loads),max(resident)/cap,statistics.fmean(resident)/cap,sum(x>cap for x in resident),
        statistics.fmean(active_spans),statistics.median(active_spans),max(active_spans),
        "CAPACITY_BALANCED_FIRST_TOUCH_LOCALITY_ONLY","MINIMUM_CAPACITY_DIE_SPAN_ONLY")

def remaining_external_bytes_for_ownership(loads, ownership)->float:
    die_count=1+max(d for owners in ownership for d in owners)
    return sum(external_bytes_per_die_for_ownership(loads,ownership,die_count))

def external_bytes_per_die_for_ownership(loads, ownership, die_count)->tuple[float,...]:
    return boundary_bytes_per_die(tuple(x.unit for x in loads),ownership,die_count)
