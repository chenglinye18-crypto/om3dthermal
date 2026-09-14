"""Formal multi-turn matrix: shared state, physical IOM3D, and Grace overflow."""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import csv
import json
import math
import numpy as np
import yaml

from .decode_policy import CachedWorkload, DecodePolicyModel, ExecutionPolicy, llama31_models
from .mixed_phase_e2e import resolve_conventional_hbm_backend
from .workspace import WorkspaceExecutionConfig, evaluate_prefill_workspace, evaluate_decode_workspace
from om3dthermal.workload import DenseLLMModelSpec, LLMPrefillInput, evaluate_cached_prefix_incremental_prefill, evaluate_llm_decode
from om3dthermal.architecture.feol_floorplan import resolve_feol_floorplan
from om3dthermal.placement.nmp_load_balance import PhysicalResidentPlacement
from om3dthermal.platform import load_platform_spec_file
from om3dthermal.power.feol_energy import FEOLEnergyModel, sum_events


@lru_cache(maxsize=1)
def setup(root):
    root=Path(root)
    config=yaml.safe_load((root/'configs/experiment/formal_iom3d_workload_sweep_v1.yaml').read_text())
    workspace=WorkspaceExecutionConfig(**yaml.safe_load((root/config['workspace_source']).read_text())['workspace'])
    platform=load_platform_spec_file(root/'configs/platform/gpu_package_h200_reference.yaml')
    hbm=resolve_conventional_hbm_backend(root)
    closure=json.loads((root/config['hbm_thermal_source']).read_text())
    hbm_thermal=closure['HBM_Bthermal_TBps']['conventional_hbm_2x1']*1e12
    with (root/config['thermal_limits_source']).open() as stream:
        m3d_thermal=float(next(r for r in csv.DictReader(stream) if r['architecture']=='orthogonal_m3d_igzo')['Bthermal_TBps'])*1e12
    hbm=hbm.model_copy(update={'sustained_bandwidth_bytes_per_s':min(hbm.peak_bandwidth_bytes_per_s,hbm.sustained_bandwidth_bytes_per_s,hbm_thermal),
        'bandwidth_source_status':'CANONICAL_85C_THERMAL_CLOSED_HBM_REUSED_UNCHANGED_MODEL'})
    return config,workspace,platform,hbm,m3d_thermal


def inputs(name,wid,batch,root):
    config,ws,*_=setup(root)
    cw=CachedWorkload(**config['workloads'][wid])
    w=llama31_models()[name].model_copy(update={'batch_size':batch,'context_length':cw.history+cw.prompt+cw.generated})
    spec=DenseLLMModelSpec(model_id=name,model_spec_status='RESOLVED',context_status='CANONICAL_LLAMA31_131072',
        **{k:getattr(w,k) for k in ('n_param','n_layers','n_heads_q','n_heads_kv','d_model','d_ff','vocab_size','weight_bits','kv_bits')})
    return w,cw,spec


@lru_cache(maxsize=27)
def physical_capacity_gate(name,wid,batch,root):
    w,_,_=inputs(name,wid,batch,root);f=resolve_feol_floorplan(root)
    p=PhysicalResidentPlacement.__new__(PhysicalResidentPlacement)
    try:
        p.__init__(w,f,'UNIFORM_STRIPING')
    except ValueError as exc:
        excess=int(np.maximum(p.slot_used-f.layout.slot_capacity_bytes,0).sum())*4
        return 0,str(exc),excess
    free=((f.layout.slot_capacity_bytes-p.slot_used)*4//32)*32
    assert int(p.slot_used.sum())*4==w.n_param*2+2*batch*w.n_layers*w.context_length*w.n_heads_kv*w.d_head*2
    return int(free.sum()),'',0


def capacity(name,wid,batch,system,root):
    config,ws,platform,hbm,_=setup(root)
    w,cw,spec=inputs(name,wid,batch,root)
    f=resolve_feol_floorplan(root)
    weight=w.n_param*2
    kv_token=2*w.n_layers*w.n_heads_kv*w.d_head*2*batch
    pre_ws=evaluate_prefill_workspace(spec,batch_size=batch,context_length=cw.prompt,config=ws).peak_bytes
    dec_ws=evaluate_decode_workspace(spec,batch_size=batch,context_length=w.context_length-1,config=ws,
        proposed_nmp='MAC_NMP' in system,nmp_die_count=f.layout.slab_count if 'MAC_NMP' in system else 0).peak_bytes
    workspace=max(pre_ws,dec_ws)
    persistent=weight+w.context_length*kv_token
    host=system=='HBM_GRACE_C2C'
    cap=int(hbm.capacity_bytes)+config['grace_capacity_bytes'] if host else f.layout.total_capacity_bytes
    # Actual atoms occupy four member-cluster slots. Reserve workspace in
    # free 32-B physical service slots; do not round every object to a whole MAT.
    rounded_workspace=math.ceil(workspace/32)*32
    peak=persistent+rounded_workspace
    status='EVALUATED' if peak<=cap and (not host or workspace<=hbm.capacity_bytes) else ('SYSTEM_CAPACITY_INFEASIBLE' if host else 'CAPACITY_INFEASIBLE')
    slot_excess=0
    if not host and status=='EVALUATED':
        free,physical_error,slot_excess=physical_capacity_gate(name,wid,batch,root)
        if free<rounded_workspace or physical_error:status='CAPACITY_INFEASIBLE'
    else:physical_error=''
    return dict(model=name,workload_id=wid,H=cw.history,P=cw.prompt,G=cw.generated,batch_size=batch,system=system,status=status,
        weight_GB=weight/1e9,initial_KV_GB=cw.history*kv_token/1e9,final_KV_GB=w.context_length*kv_token/1e9,
        workspace_GB=rounded_workspace/1e9,peak_state_GB=peak/1e9,capacity_GB=cap/1e9,
        capacity_utilization=peak/cap,capacity_violation_GB=max(0,peak-cap,slot_excess)/1e9,physical_slot_excess_GB=slot_excess/1e9,
        HBM_capacity_GB=hbm.capacity_bytes/1e9,Grace_capacity_GB=config['grace_capacity_bytes']/1e9,
        HBM_only_fit=peak<=hbm.capacity_bytes,physical_slot_status=physical_error or 'PASS' if status=='EVALUATED' else physical_error or 'TOTAL_CAPACITY_GATE',
        kv_state_fraction=(w.context_length*kv_token)/persistent)


@dataclass
class StateObject:
    name: str
    size: int
    prefill_reads: float
    decode_reads: float
    write_phase: str
    birth: int = -1
    local: int = 0


def grace_residency(w,cw,prefill,hbm_bytes,workspace):
    """Exact static byte-allocation knapsack for divisible persistent extents.

    Benefit is total timed reads+writes per byte. No migration is permitted.
    New extents reserve their eventual location before timing begins. Equal
    benefits prefer write-sensitive extents, then recurring reads.
    """
    step=evaluate_llm_decode(w.model_copy(update={'context_length':cw.history+cw.prompt}))
    dense=int(prefill.active_weight_read_bytes)
    lookup=w.batch_size*w.d_model*2
    # The analytical lookup ledger names B distinct active rows. Prefill's
    # embedding input bytes are GPU-local activations, not extra weight reads.
    objects=[StateObject('dense_weights',dense,1,cw.generated,''),
             StateObject('lookup_weights',lookup,0,cw.generated,''),
             StateObject('inactive_weights',w.n_param*2-dense-lookup,0,0,'')]
    kv=2*w.batch_size*w.n_layers*w.n_heads_kv*w.d_head*2
    objects.extend([StateObject('historical_KV',cw.history*kv,1,cw.generated,''),
                    StateObject('prefill_KV',cw.prompt*kv,0,cw.generated,'prefill')])
    objects.extend(StateObject('decode_KV',kv,0,cw.generated-1-j,'decode',j) for j in range(cw.generated))
    assert sum(o.size for o in objects)==w.n_param*2+w.context_length*kv
    assert dense+lookup==int(step.weight_read_bytes_per_token*w.batch_size)
    remaining=int(hbm_bytes-workspace)
    for o in sorted(objects,key=lambda o:(-(o.prefill_reads+o.decode_reads+bool(o.write_phase)),not bool(o.write_phase),-o.decode_reads,o.birth,o.name)):
        o.local=min(remaining,o.size);remaining-=o.local
    assert all(0<=o.local<=o.size for o in objects)
    return objects


def conventional(name,wid,batch,root,cap):
    config,_,platform,hbm,_=setup(root);w,cw,spec=inputs(name,wid,batch,root)
    inp=LLMPrefillInput(**{k:getattr(w,k) for k in LLMPrefillInput.model_fields if k!='prompt_length'},prompt_length=cw.prompt)
    pref=evaluate_cached_prefix_incremental_prefill(inp,cached_history_tokens=cw.history)
    objects=grace_residency(w,cw,pref,hbm.capacity_bytes,round(cap['workspace_GB']*1e9))
    host=lambda o:o.size-o.local
    host_resident=sum(host(o) for o in objects)
    assert host_resident<=config['grace_capacity_bytes']
    host_spec=platform.host_offload
    host_bw=host_spec.direct_effective_bandwidth_bytes_per_second
    host_e=host_spec.total_dynamic_J_per_bit
    pre_host_read=sum(host(o)*o.prefill_reads for o in objects)
    pre_host_write=sum(host(o) for o in objects if o.write_phase=='prefill')
    pre_hbm_read=pref.total_read_bytes-pre_host_read
    pre_hbm_write=pref.total_write_bytes-pre_host_write
    assert min(pre_hbm_read,pre_hbm_write)>=0
    compute=platform.gpu_prefill_compute
    pre_compute=(pref.linear_flops+pref.lm_head_flops)/(compute.large_gemm_effective_tflops*1e12)+pref.attention_flops/(compute.causal_attention_effective_tflops*1e12)
    pre_s=max(pre_compute,(pre_hbm_read+pre_hbm_write)/hbm.sustained_bandwidth_bytes_per_s,(pre_host_read+pre_host_write)/host_bw)
    times=[];traffic=dict(HBM_read_GB=0.,HBM_write_GB=0.,Grace_weight_read_GB=0.,Grace_KV_read_GB=0.,Grace_KV_write_GB=0.,migration_GB=0.)
    dec_host=dec_hbm_read=dec_hbm_write=dec_bits=critical=0.
    for j,context in enumerate(cw.contexts):
        m=evaluate_llm_decode(w.model_copy(update={'context_length':context}))
        wh=sum(host(o) for o in objects if 'weights' in o.name and o.decode_reads)
        kh=sum(host(o) for o in objects if 'KV' in o.name and (o.birth<j))
        append=sum(host(o) for o in objects if o.write_phase=='decode' and o.birth==j)
        read=m.read_bytes_per_token*batch;write=m.write_bytes_per_token*batch
        local_read=read-wh-kh;local_write=write-append
        assert min(local_read,local_write)>=0
        gpu=max((local_read+local_write)/hbm.sustained_bandwidth_bytes_per_s,m.flops_per_token*batch/platform.gpu_compute_power.peak_compute_BF16_dense_flops_per_s)
        hs=(wh+kh+append)/host_bw;t=max(gpu,hs);times.append(t)
        critical+=t if hs>gpu else 0
        for key,value in zip(('HBM_read_GB','HBM_write_GB','Grace_weight_read_GB','Grace_KV_read_GB','Grace_KV_write_GB'),(local_read,local_write,wh,kh,append)):
            traffic[key]+=value/1e9
        dec_host+=wh+kh+append;dec_hbm_read+=local_read;dec_hbm_write+=local_write;dec_bits+=(read+write)*8
    seconds=sum(times);generated=batch*cw.generated
    gpu=platform.gpu_decode_power;comp=platform.gpu_compute_power
    components=dict(HBM_read_J=dec_hbm_read*8*hbm.read_energy_pJ_per_bit*1e-12,
        Grace_memory_J=dec_host*8*host_spec.memory_dynamic_J_per_bit,C2C_J=dec_host*8*host_spec.link_dynamic_J_per_bit,
        GPU_dynamic_J=dec_bits*gpu.e_decode_J_per_bit,GPU_static_J=seconds*gpu.static_power_W)
    pre_J=pre_hbm_read*8*hbm.read_energy_pJ_per_bit*1e-12+(pre_host_read+pre_host_write)*8*host_e+pref.total_flops*(comp.e_compute_dynamic_J_per_FLOP_min+comp.e_compute_dynamic_J_per_FLOP_max)/2+pre_s*gpu.static_power_W
    total_known=sum(components.values())+pre_J
    traffic.update(total_C2C_GB=(dec_host+pre_host_read+pre_host_write)/1e9,
        C2C_GB_per_generated_token=(dec_host+pre_host_read+pre_host_write)/1e9/generated,
        host_critical_path_fraction=critical/seconds,prefill_HBM_read_GB=pre_hbm_read/1e9,prefill_HBM_write_GB=pre_hbm_write/1e9,
        prefill_Grace_read_GB=pre_host_read/1e9,prefill_Grace_write_GB=pre_host_write/1e9)
    row={**cap,**timing(pre_s,times,generated), 'prefill_host_GB':(pre_host_read+pre_host_write)/1e9,
         'HBM_resident_GB':sum(o.local for o in objects)/1e9,'Grace_resident_GB':host_resident/1e9,
         'residency_policy':'HBM_LOCAL' if host_resident==0 else 'GRACE_C2C_STATE_OFFLOAD',
         'effective_bandwidth_TBps':hbm.sustained_bandwidth_bytes_per_s/1e12}
    energy=dict(energy_status='ENERGY_INCOMPLETE_HBM_WRITE_UNRESOLVED',prefill_known_J=pre_J,decode_known_J=sum(components.values()),
        E2E_known_J=total_known,E2E_known_J_per_token=total_known/generated,
        E2E_J_per_token=None,E2E_tokens_per_J=None,decode_J_per_token=None,decode_tokens_per_J=None,
        unresolved_HBM_write_bytes=pre_hbm_write+dec_hbm_write,host_static_power_status='UNRESOLVED',**components)
    return dict(summary=row,traffic=traffic,energy=energy,host_audit=[vars(o) for o in objects],prefill_ledger=pref.model_dump())


def timing(prefill_s,times,generated):
    decode=sum(times);e2e=prefill_s+decode
    return dict(prefill_s=prefill_s,decode_s=decode,decode_generated_tokens=generated,decode_tok_s=generated/decode,
        first_step_ms=times[0]*1e3,last_step_ms=times[-1]*1e3,mean_TPOT_ms=decode/len(times)*1e3,
        E2E_s=e2e,E2E_generated_tokens=generated,E2E_tok_s=generated/e2e,TTFT=prefill_s+times[0])
