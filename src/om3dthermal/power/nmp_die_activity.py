"""Canonical per-die NMP hardware, workload activity, and compute power."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import math, statistics
from om3dthermal.platform import load_platform_spec_file, resolve_gpu_bandwidth_service, resolve_local_memory_gpu_transfer
from pathlib import Path
from om3dthermal.workload.dense_decode_ledger import (
    attention_boundary_by_layer, build_dense_decode_handoffs,
    build_dense_decode_placement_units, build_dense_decode_small_ops)
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
    handoffs: tuple[dict, ...]
    small_ops: tuple[dict, ...]
    gpu_remaining_local_bytes: float
    gpu_remaining_time_ms: float
    gpu_remaining_dynamic_energy_j: float
    embedding_local_read_bytes: float
    embedding_local_time_ms: float
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
    handoffs=build_dense_decode_handoffs(units,spans,n)
    boundary=sum(x.bytes for x in handoffs)
    attention_layers=attention_boundary_by_layer(units,spans)
    partial=sum(row["partial_bytes"] for row in attention_layers.values())
    if boundary and not transfer.bandwidth_actual_bytes_per_s:
        raise ValueError("positive boundary traffic requires positive transfer demand")
    external_boundary_time_ms=boundary/transfer.bandwidth_actual_bytes_per_s*1e3 if boundary else 0
    softmax_ms=(score+probability)/gpu_bw*1e3
    local_stages={}
    for (layer,operator),(mem,compute) in stages_by_key.items():
        memory_ms=max(mem)/bw_die*1e3
        compute_ms=max(compute)/hw.peak_flops_per_die*1e3
        stage_ms=max(max(m/bw_die,f/hw.peak_flops_per_die)*1e3 for m,f in zip(mem,compute))
        stage_owners=set(d for u,o in zip(units,spans,strict=True)
                         if u.layer_id==layer and u.operator_type==operator for d in o)
        local_stages[layer,operator]=dict(layer=layer,operator=operator,kind="NMP_STAGE",
            execution_die_span=len(stage_owners),memory_ms=memory_ms,compute_ms=compute_ms,
            stage_ms=stage_ms,time_ms=stage_ms)
    small_ops=build_dense_decode_small_ops(workload,units,spans,n)
    def gpu_stage(operator,variant,layer,label):
        rows=[x for x in small_ops if x.operator==operator and x.variant==variant and x.layer_id==layer]
        bytes_=sum(x.gpu_local_total_bytes for x in rows)
        return dict(layer=layer,operator=label,kind="GPU_SMALL_OP",gpu_local_bytes=bytes_,
            time_ms=bytes_/gpu_bw*1e3)
    def handoff_stage(producer,consumer,layer,label):
        rows=[x for x in handoffs if x.producer==producer and x.consumer==consumer and x.layer_id==layer]
        bytes_=sum(x.bytes for x in rows)
        return dict(layer=layer,operator=label,kind="BOUNDARY_HANDOFF",producer=producer,
            consumer=consumer,direction=rows[0].direction,bytes=bytes_,
            reason="; ".join(sorted({x.reason for x in rows})),
            time_ms=bytes_/transfer.bandwidth_actual_bytes_per_s*1e3)
    stages=[local_stages[-1,"TOKEN_EMBED_LOOKUP"],
        handoff_stage("TOKEN_EMBED_LOOKUP","RMSNORM_PRE_ATTENTION",-1,"EMBEDDING_TRANSFER")]
    for layer in range(workload.n_layers):
        stages.append(gpu_stage("RMSNORM","PRE_ATTENTION",layer,"RMSNORM_PRE_ATTENTION"))
        for consumer in ("Q","K","V"):
            stages.append(handoff_stage("RMSNORM_PRE_ATTENTION",consumer,layer,
                f"RMSNORM_TO_{consumer}_TRANSFER"))
        stages.extend((local_stages[layer,"Q"],handoff_stage("Q","ROPE",layer,"Q_TO_ROPE_TRANSFER"),
            local_stages[layer,"K"],handoff_stage("K","ROPE",layer,"K_TO_ROPE_TRANSFER"),
            local_stages[layer,"V"],handoff_stage("V","KV_VECTOR_REPACK",layer,"V_TO_KV_REPACK_TRANSFER"),
            handoff_stage("KV_VECTOR_REPACK","KV_WRITE",layer,"V_KV_REPACK_RETURN_TRANSFER"),
            gpu_stage("ROPE","Q_K",layer,"ROPE"),
            handoff_stage("ROPE_Q","ATTENTION_QK",layer,"ROPE_Q_RETURN_TRANSFER"),
            handoff_stage("ROPE_K","KV_WRITE",layer,"ROPE_K_RETURN_TRANSFER"),
            local_stages[layer,"ATTENTION_QK"],
            handoff_stage("ATTENTION_QK","GPU_SOFTMAX",layer,"SCORE_TRANSFER"),
            dict(layer=layer,operator="GPU_SOFTMAX",kind="GPU_SOFTMAX",
                 gpu_local_bytes=attention_layers[layer]["score_bytes"]+attention_layers[layer]["probability_bytes"],
                 time_ms=(attention_layers[layer]["score_bytes"]+attention_layers[layer]["probability_bytes"])/gpu_bw*1e3),
            handoff_stage("GPU_SOFTMAX","ATTENTION_AV",layer,"PROBABILITY_TRANSFER"),
            local_stages[layer,"ATTENTION_AV"],
            handoff_stage("ATTENTION_AV","AV_REDUCTION",layer,"PARTIAL_TRANSFER"),
            gpu_stage("AV_REDUCTION","FP32_PARTIAL_TO_FP16",layer,"AV_REDUCTION"),
            handoff_stage("AV_REDUCTION","O",layer,"AV_REDUCTION_TO_O_TRANSFER"),
            local_stages[layer,"O"],handoff_stage("O","RESIDUAL_ADD_ATTENTION",layer,"O_TO_RESIDUAL_TRANSFER"),
            gpu_stage("RESIDUAL_ADD","ATTENTION",layer,"RESIDUAL_ADD_ATTENTION"),
            gpu_stage("RMSNORM","PRE_FFN",layer,"RMSNORM_PRE_FFN"),
            handoff_stage("RMSNORM_PRE_FFN","FFN_GATE",layer,"RMSNORM_TO_FFN_GATE_TRANSFER"),
            handoff_stage("RMSNORM_PRE_FFN","FFN_UP",layer,"RMSNORM_TO_FFN_UP_TRANSFER"),
            local_stages[layer,"FFN_GATE"],handoff_stage("FFN_GATE","SWIGLU",layer,"FFN_GATE_TO_SWIGLU_TRANSFER"),
            local_stages[layer,"FFN_UP"],handoff_stage("FFN_UP","SWIGLU",layer,"FFN_UP_TO_SWIGLU_TRANSFER"),
            gpu_stage("SWIGLU","SILU_GATE_TIMES_UP",layer,"SWIGLU"),
            handoff_stage("SWIGLU","FFN_DOWN",layer,"SWIGLU_TO_FFN_DOWN_TRANSFER"),
            local_stages[layer,"FFN_DOWN"],
            handoff_stage("FFN_DOWN","RESIDUAL_ADD_FFN",layer,"FFN_DOWN_TO_RESIDUAL_TRANSFER"),
            gpu_stage("RESIDUAL_ADD","FFN",layer,"RESIDUAL_ADD_FFN")))
    stages.extend((gpu_stage("FINAL_RMSNORM","FINAL",workload.n_layers,"FINAL_RMSNORM"),
        handoff_stage("FINAL_RMSNORM","LM_HEAD",workload.n_layers,"FINAL_RMSNORM_TO_LM_HEAD_TRANSFER"),
        local_stages[workload.n_layers,"LM_HEAD"],
        handoff_stage("LM_HEAD","SAMPLING",workload.n_layers,"LOGITS_TRANSFER"),
        gpu_stage("SAMPLING","GREEDY_ARGMAX",workload.n_layers,"SAMPLING")))
    for actual,expected in ((sum(weights),sum(u.active_weight_read_bytes for u in units)),
                            (sum(kvreads),sum(u.kv_bytes for u in units)),
                            (sum(kvwrites),demand.kv_write_bytes_per_decode_step)):
        if not math.isclose(actual,expected,rel_tol=1e-12):
            raise ValueError("workload ledger and page demand disagree")
    totals=[weights[i]+kvreads[i]+kvwrites[i] for i in range(n)]
    mem_ms=[x/bw_die*1e3 for x in totals]; comp_ms=[x/hw.peak_flops_per_die*1e3 for x in flops]
    service=[max(mem_ms[i],comp_ms[i]) for i in range(n)]
    stage=sum(s["time_ms"] for s in stages if s.get("kind")=="NMP_STAGE")
    interval=sum(s["time_ms"] for s in stages)
    ai_balance=hw.peak_flops_per_die/bw_die; rows=[]
    for i in range(n):
        ai=0 if totals[i]==0 else flops[i]/totals[i]; ratio=ai/ai_balance if ai_balance else 0
        label="BALANCED" if math.isclose(ratio,1.0,rel_tol=.02) else ("COMPUTE_BOUND" if ratio>1 else "MEMORY_BOUND")
        energy=flops[i]/2*hw.mac_energy_pj*1e-12; power=energy/(interval*1e-3)
        rows.append(NMPDieWorkloadActivity(i,weights[i],kvreads[i],kvwrites[i],totals[i],flops[i],ai,mem_ms[i],comp_ms[i],service[i],mem_ms[i]/stage,comp_ms[i]/stage,label,energy,
            NMPDiePower(i,power,None,None,None,"COMPUTE_DYNAMIC_RESOLVED__DIE_LEVEL_MEMORY_POWER_DISTRIBUTION_PENDING_B")))
    ordered=sorted(service); p90=ordered[math.ceil(.9*len(ordered))-1]
    exec_spans=tuple(len(o) for u,o in zip(units,spans)
                     if u.shard_mode not in ("RESIDENT_ONLY","LOCAL_LOOKUP"))
    memory_stage_seconds=sum(s["stage_ms"] for s in stages
        if "memory_ms" in s and s["memory_ms"]>=s["compute_ms"])*1e-3
    realized_bw=sum(totals)/memory_stage_seconds if memory_stage_seconds else 0.0
    gpu_remaining_bytes=sum(x.gpu_local_total_bytes for x in small_ops
                            if x.operator!="TOKEN_EMBED_LOOKUP")
    gpu_remaining_ms=gpu_remaining_bytes/gpu_bw*1e3
    gpu_remaining_j=8*gpu_remaining_bytes*gpu.e_decode_J_per_bit
    embedding_bytes=sum(u.active_weight_read_bytes for u in units
                        if u.operator_type=="TOKEN_EMBED_LOOKUP")
    embedding_ms=local_stages[-1,"TOKEN_EMBED_LOOKUP"]["time_ms"]
    return NMPDieActivitySummary(hw,local_access_latency_ns,1.0,bw_die,bw_die*n,ai_balance,tuple(rows),stage,interval,service.index(max(service)),statistics.fmean(service),p90,max(service),
        sum(r.bottleneck=="MEMORY_BOUND" for r in rows),sum(r.bottleneck=="COMPUTE_BOUND" for r in rows),sum(r.bottleneck=="BALANCED" for r in rows),
        sum(r.compute_energy_j for r in rows),sum(r.power.compute_dynamic_W for r in rows),"DIE_LEVEL_MEMORY_POWER_DISTRIBUTION_PENDING_B",
        "SERIAL_DEPENDENT_STAGES_WITH_GPU_SOFTMAX_BARRIERS",tuple(stages),attention_layers,
        score,probability,partial,score+probability+partial,boundary,external_boundary_time_ms,transfer.model_dump(),
        score+probability,softmax_ms,8*(score+probability)*gpu.e_decode_J_per_bit,
        gpu.static_power_W*interval*1e-3,statistics.fmean(exec_spans),statistics.median(exec_spans),
        max(exec_spans),realized_bw,tuple(asdict(x) for x in handoffs),
        tuple(asdict(x) for x in small_ops),gpu_remaining_bytes,gpu_remaining_ms,
        gpu_remaining_j,embedding_bytes,embedding_ms)
