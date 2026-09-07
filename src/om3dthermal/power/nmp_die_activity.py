"""Canonical per-die NMP hardware, workload activity, and compute power."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import math, statistics
from om3dthermal.platform import load_platform_spec_file, resolve_gpu_bandwidth_service, resolve_local_memory_gpu_transfer
from pathlib import Path
from om3dthermal.workload.dense_decode_ledger import boundary_bytes_per_die, attention_boundary_by_layer, build_dense_decode_placement_units
from om3dthermal.placement.nmp_load_balance import NMPPlacementUnitLoad, shard_fractions
from om3dthermal.power.memory_bandwidth import ArchitectureBandwidthClosure
from om3dthermal.power.physical_capacity import PhysicalCapacityLayout
from om3dthermal.workload.llm_decode import LLMDecodeInput
from om3dthermal.workload.m3d_page_demand import M3DWorkloadPageDemand

@dataclass(frozen=True)
class NMPHardware:
    precision: str; macs_per_die: int; active_mac_ceiling_per_die: int; clock_hz: float
    flops_per_mac: int; peak_flops_per_die: float; physical_die_count: int
    aggregate_peak_flops: float; mac_energy_pj: float
    mac_count_provenance: str; clock_provenance: str; peak_provenance: str

@dataclass(frozen=True)
class NMPDiePower:
    die_id: int; compute_dynamic_W: float; memory_dynamic_W: float|None
    refresh_W: float|None; total_dynamic_W: float|None; power_status: str
    memory_read_dynamic_W: float|None=None; memory_write_dynamic_W: float|None=None
    mac_dynamic_W: float|None=None; nmp_logic_overhead_factor: float=1.0
    nmp_dynamic_W: float|None=None; residual_external_W: float|None=None
    total_W: float|None=None; thermal_memory_carrier_W: float|None=None
    thermal_nmp_carrier_W: float|None=None; thermal_mapping_status: str="THERMAL_MAPPING_PENDING"

@dataclass(frozen=True)
class NMPDieWorkloadActivity:
    die_id: int; weight_read_bytes: float; kv_read_bytes: float; kv_write_bytes: float
    total_local_memory_bytes: float; nmp_flops: float; arithmetic_intensity_flop_per_byte: float
    memory_service_time_ms: float; compute_service_time_ms: float; active_service_time_ms: float
    memory_utilization: float; compute_utilization: float; bottleneck: str
    compute_energy_j: float; power: NMPDiePower

@dataclass(frozen=True)
class NMPDieActivitySummary:
    hardware: NMPHardware; local_access_latency_ns: float; local_route_delay_ns: float
    local_bandwidth_per_die_bytes_per_s: float; aggregate_local_bandwidth_bytes_per_s: float
    hardware_balance_flop_per_byte: float; activities: tuple[NMPDieWorkloadActivity,...]
    global_nmp_stage_time_ms: float; decode_step_interval_ms: float; straggler_die_id: int
    mean_die_service_time_ms: float; p90_die_service_time_ms: float; max_die_service_time_ms: float
    memory_bound_die_count: int; compute_bound_die_count: int; balanced_die_count: int
    aggregate_compute_energy_j: float; aggregate_compute_dynamic_W: float
    memory_power_status: str
    timing_semantics: str
    stages: tuple[dict, ...]
    attention_layers: dict
    score_bytes: float
    probability_bytes: float
    partial_bytes: float
    attention_boundary_bytes: float
    residual_boundary_bytes: float
    boundary_time_ms: float
    transfer: dict
    softmax_local_bytes: float
    softmax_time_ms: float
    softmax_dynamic_energy_j: float
    gpu_static_energy_j: float
    mean_exec_die_span: float
    median_exec_die_span: float
    max_exec_die_span: int
    realized_effective_local_bandwidth_bytes_per_s: float
    def as_dict(self): return asdict(self)

def canonical_nmp_hardware(physical_die_count:int)->NMPHardware:
    macs=512; clock=1.0e9; flops_per_mac=2; peak=macs*clock*flops_per_mac
    return NMPHardware("FP16",macs,macs,clock,flops_per_mac,peak,physical_die_count,peak*physical_die_count,0.604,
        "ARCHITECTURE_MODELING_CHOICE","REFERENCE_ANCHORED_MODELING_CHOICE__NOT_PHYSICALLY_SYNTHESIZED_FOR_THIS_DESIGN",
        "DERIVED_FROM_MAC_COUNT_CLOCK_AND_PRECISION")

def evaluate_nmp_die_activity(workload:LLMDecodeInput,demand:M3DWorkloadPageDemand,layout:PhysicalCapacityLayout,
        bandwidth:ArchitectureBandwidthClosure,*,local_access_latency_ns:float,bandwidth_demand_bytes_per_s:float,
        ownership:tuple[tuple[int,...],...])->NMPDieActivitySummary:
    hw=canonical_nmp_hardware(layout.slab_count)
    bw_die=(bandwidth.local_service_groups_per_slab*bandwidth.read_payload_bytes_per_service/(bandwidth.service_cycle_scale*local_access_latency_ns*1e-9))
    units=build_dense_decode_placement_units(workload); spans=ownership; n=layout.slab_count
    if len(spans)!=len(units): raise ValueError("unit ownership count mismatch")
    weights=[0.0]*n; kvreads=[0.0]*n; kvwrites=[0.0]*n; flops=[0.0]*n
    stages_by_key={}
    score=probability=partial=0.0
    for u,owners in zip(units,spans,strict=True):
        if not owners or len(set(owners)) != len(owners) or any(d < 0 or d >= n for d in owners):
            raise ValueError("invalid unit ownership")
        key=(u.layer_id,u.operator_type)
        stage_mem,stage_flops=stages_by_key.setdefault(key,([0.0]*n,[0.0]*n))
        score+=u.score_bytes; probability+=u.probability_bytes

        unit_flops=(workload.batch_size if u.placement_scope=="SHARED_BATCH" else 1)*u.local_flops
        load=NMPPlacementUnitLoad(u,u.weight_bytes+u.kv_bytes,u.active_weight_read_bytes,
            u.kv_bytes,u.kv_write_bytes,u.active_weight_read_bytes+u.kv_bytes+u.kv_write_bytes,
            unit_flops,1)
        fractions=shard_fractions(load,len(owners))
        for die,share in zip(owners,fractions,strict=True):
            weights[die]+=u.active_weight_read_bytes*share
            kvreads[die]+=u.kv_bytes*share
            kvwrites[die]+=u.kv_write_bytes*share
            flops[die]+=unit_flops*share
            stage_mem[die]+=(u.active_weight_read_bytes+u.kv_bytes+u.kv_write_bytes)*share
            stage_flops[die]+=unit_flops*share
    platform=load_platform_spec_file(Path(__file__).resolve().parents[3]/"configs/platform/gpu_package_h200_reference.yaml")
    gpu=platform.gpu_decode_power
    service_spec=platform.gpu_bandwidth_service
    gpu_bw=resolve_gpu_bandwidth_service(
        transfer_ceiling_bytes_per_s=gpu.peak_memory_bandwidth_bytes_per_s,
        gpu_bandwidth_utilization=service_spec.nominal_utilization,
        utilization_status=service_spec.utilization_status,
        utilization_provenance=service_spec.provenance).sustained_bandwidth_bytes_per_s
    transfer=resolve_local_memory_gpu_transfer(
        bandwidth_demand_bytes_per_s=bandwidth_demand_bytes_per_s,
        memory_capability_bytes_per_s=bandwidth.coil_bandwidth_bytes_per_s,
        gpu_peak_bandwidth_bytes_per_s=gpu_bw)
    boundary=sum(boundary_bytes_per_die(units,spans,n))
    attention_layers=attention_boundary_by_layer(units,spans)
    partial=sum(row["partial_bytes"] for row in attention_layers.values())
    if boundary and not transfer.bandwidth_actual_bytes_per_s:
        raise ValueError("positive boundary traffic requires positive transfer demand")
    external_boundary_time_ms=boundary/transfer.bandwidth_actual_bytes_per_s*1e3 if boundary else 0
    softmax_ms=(score+probability)/gpu_bw*1e3
    stages=[]
    for (layer,operator),(mem,compute) in stages_by_key.items():
        memory_ms=max(mem)/bw_die*1e3
        compute_ms=max(compute)/hw.peak_flops_per_die*1e3
        stage_ms=max(max(m/bw_die,f/hw.peak_flops_per_die)*1e3 for m,f in zip(mem,compute))
        stage_owners=set(d for u,o in zip(units,spans,strict=True)
                         if u.layer_id==layer and u.operator_type==operator for d in o)
        stages.append(dict(layer=layer,operator=operator,execution_die_span=len(stage_owners),
            memory_ms=memory_ms,compute_ms=compute_ms,stage_ms=stage_ms,time_ms=stage_ms))
        if operator not in ("ATTENTION_QK","ATTENTION_AV"):
            weight_boundary=sum(u.activation_input_bytes*len(o)+u.partial_output_bytes
                for u,o in zip(units,spans) if u.layer_id==layer and u.operator_type==operator)
            if weight_boundary:
                stages.append(dict(layer=layer,operator=operator+"_GPU_BOUNDARY",
                    time_ms=weight_boundary/transfer.bandwidth_actual_bytes_per_s*1e3))
        if operator == "ATTENTION_QK":
            stages.extend((dict(layer=layer,operator="SCORE_TRANSFER",time_ms=attention_layers[layer]["score_bytes"]/transfer.bandwidth_actual_bytes_per_s*1e3),
                dict(layer=layer,operator="GPU_SOFTMAX",time_ms=softmax_ms/workload.n_layers),
                dict(layer=layer,operator="PROBABILITY_TRANSFER",time_ms=attention_layers[layer]["probability_bytes"]/transfer.bandwidth_actual_bytes_per_s*1e3)))
        if operator == "ATTENTION_AV":
            layer_partial=attention_layers[layer]["partial_bytes"]
            stages.append(dict(layer=layer,operator="PARTIAL_TRANSFER",time_ms=layer_partial/transfer.bandwidth_actual_bytes_per_s*1e3))
    for actual,expected in ((sum(weights),sum(u.active_weight_read_bytes for u in units)),
                            (sum(kvreads),sum(u.kv_bytes for u in units)),
                            (sum(kvwrites),demand.kv_write_bytes_per_decode_step)):
        if not math.isclose(actual,expected,rel_tol=1e-12):
            raise ValueError("workload ledger and page demand disagree")
    totals=[weights[i]+kvreads[i]+kvwrites[i] for i in range(n)]
    mem_ms=[x/bw_die*1e3 for x in totals]; comp_ms=[x/hw.peak_flops_per_die*1e3 for x in flops]
    service=[max(mem_ms[i],comp_ms[i]) for i in range(n)]
    stage=sum(s["time_ms"] for s in stages if "memory_ms" in s)
    interval=sum(s["time_ms"] for s in stages)
    ai_balance=hw.peak_flops_per_die/bw_die; rows=[]
    for i in range(n):
        ai=0 if totals[i]==0 else flops[i]/totals[i]; ratio=ai/ai_balance if ai_balance else 0
        label="BALANCED" if math.isclose(ratio,1.0,rel_tol=.02) else ("COMPUTE_BOUND" if ratio>1 else "MEMORY_BOUND")
        energy=flops[i]/2*hw.mac_energy_pj*1e-12; power=energy/(interval*1e-3)
        rows.append(NMPDieWorkloadActivity(i,weights[i],kvreads[i],kvwrites[i],totals[i],flops[i],ai,mem_ms[i],comp_ms[i],service[i],mem_ms[i]/stage,comp_ms[i]/stage,label,energy,
            NMPDiePower(i,power,None,None,None,"COMPUTE_DYNAMIC_RESOLVED__DIE_LEVEL_MEMORY_POWER_DISTRIBUTION_PENDING_B")))
    ordered=sorted(service); p90=ordered[math.ceil(.9*len(ordered))-1]
    exec_spans=tuple(len(o) for u,o in zip(units,spans) if u.shard_mode!="RESIDENT_ONLY")
    memory_stage_seconds=sum(s["stage_ms"] for s in stages
        if "memory_ms" in s and s["memory_ms"]>=s["compute_ms"])*1e-3
    realized_bw=sum(totals)/memory_stage_seconds if memory_stage_seconds else 0.0
    return NMPDieActivitySummary(hw,local_access_latency_ns,1.0,bw_die,bw_die*n,ai_balance,tuple(rows),stage,interval,service.index(max(service)),statistics.fmean(service),p90,max(service),
        sum(r.bottleneck=="MEMORY_BOUND" for r in rows),sum(r.bottleneck=="COMPUTE_BOUND" for r in rows),sum(r.bottleneck=="BALANCED" for r in rows),
        sum(r.compute_energy_j for r in rows),sum(r.power.compute_dynamic_W for r in rows),"DIE_LEVEL_MEMORY_POWER_DISTRIBUTION_PENDING_B",
        "SERIAL_DEPENDENT_STAGES_WITH_GPU_SOFTMAX_BARRIERS",tuple(stages),attention_layers,
        score,probability,partial,score+probability+partial,boundary,external_boundary_time_ms,transfer.model_dump(),
        score+probability,softmax_ms,8*(score+probability)*gpu.e_decode_J_per_bit,
        gpu.static_power_W*interval*1e-3,statistics.fmean(exec_spans),statistics.median(exec_spans),
        max(exec_spans),realized_bw)
