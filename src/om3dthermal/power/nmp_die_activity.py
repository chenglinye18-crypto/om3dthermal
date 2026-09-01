"""Canonical per-die NMP hardware, workload activity, and compute power."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import math, statistics
from om3dthermal.placement.nmp_locality_e2e import build_locality_aware_unit_ownership
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
    def as_dict(self): return asdict(self)

def canonical_nmp_hardware(physical_die_count:int)->NMPHardware:
    macs=512; clock=1.0e9; flops_per_mac=2; peak=macs*clock*flops_per_mac
    return NMPHardware("FP16",macs,macs,clock,flops_per_mac,peak,physical_die_count,peak*physical_die_count,0.604,
        "ARCHITECTURE_MODELING_CHOICE","REFERENCE_ANCHORED_MODELING_CHOICE__NOT_PHYSICALLY_SYNTHESIZED_FOR_THIS_DESIGN",
        "DERIVED_FROM_MAC_COUNT_CLOCK_AND_PRECISION")

def evaluate_nmp_die_activity(workload:LLMDecodeInput,demand:M3DWorkloadPageDemand,layout:PhysicalCapacityLayout,
        bandwidth:ArchitectureBandwidthClosure,*,local_access_latency_ns:float,external_boundary_time_ms:float)->NMPDieActivitySummary:
    hw=canonical_nmp_hardware(layout.slab_count)
    bw_die=(bandwidth.local_service_groups_per_die*bandwidth.read_payload_bytes_per_service/(bandwidth.service_cycle_scale*local_access_latency_ns*1e-9))
    units,spans=build_locality_aware_unit_ownership(workload,layout); n=layout.slab_count
    weights=[0.0]*n; kvreads=[0.0]*n; kvwrites=[0.0]*n; flops=[0.0]*n
    weight_basis=sum(u.weight_bytes for u in units); kv_basis=sum(u.kv_bytes for u in units)
    for u,owners in zip(units,spans):
        for die in owners:
            share=1/len(owners)
            if u.weight_bytes: weights[die]+=demand.total_weight_read_bytes_per_decode_step*(u.weight_bytes/weight_basis)*share
            if u.kv_bytes:
                kvreads[die]+=demand.total_kv_read_bytes_per_decode_step*(u.kv_bytes/kv_basis)*share
                kvwrites[die]+=demand.kv_write_bytes_per_decode_step*(u.kv_bytes/kv_basis)*share
            flops[die]+=workload.batch_size*u.local_flops*share
    totals=[weights[i]+kvreads[i]+kvwrites[i] for i in range(n)]
    mem_ms=[x/bw_die*1e3 for x in totals]; comp_ms=[x/hw.peak_flops_per_die*1e3 for x in flops]
    service=[max(mem_ms[i],comp_ms[i]) for i in range(n)]; stage=max(service); interval=stage+external_boundary_time_ms
    ai_balance=hw.peak_flops_per_die/bw_die; rows=[]
    for i in range(n):
        ai=0 if totals[i]==0 else flops[i]/totals[i]; ratio=ai/ai_balance if ai_balance else 0
        label="BALANCED" if math.isclose(ratio,1.0,rel_tol=.02) else ("COMPUTE_BOUND" if ratio>1 else "MEMORY_BOUND")
        energy=flops[i]/2*hw.mac_energy_pj*1e-12; power=energy/(interval*1e-3)
        rows.append(NMPDieWorkloadActivity(i,weights[i],kvreads[i],kvwrites[i],totals[i],flops[i],ai,mem_ms[i],comp_ms[i],service[i],mem_ms[i]/stage,comp_ms[i]/stage,label,energy,
            NMPDiePower(i,power,None,None,None,"COMPUTE_DYNAMIC_RESOLVED__DIE_LEVEL_MEMORY_POWER_DISTRIBUTION_PENDING_B")))
    ordered=sorted(service); p90=ordered[math.ceil(.9*len(ordered))-1]
    return NMPDieActivitySummary(hw,local_access_latency_ns,1.0,bw_die,bw_die*n,ai_balance,tuple(rows),stage,interval,service.index(stage),statistics.fmean(service),p90,stage,
        sum(r.bottleneck=="MEMORY_BOUND" for r in rows),sum(r.bottleneck=="COMPUTE_BOUND" for r in rows),sum(r.bottleneck=="BALANCED" for r in rows),
        sum(r.compute_energy_j for r in rows),sum(r.power.compute_dynamic_W for r in rows),"DIE_LEVEL_MEMORY_POWER_DISTRIBUTION_PENDING_B",
        "NON_CANONICAL_DIE_LEVEL_STRAGGLER_DIAGNOSTIC__DOES_NOT_DEFINE_A_CANONICAL_THROUGHPUT")
