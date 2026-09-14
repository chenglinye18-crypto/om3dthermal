"""Compose frozen/new raw candidates and exact die-grouped event power."""
import argparse
import json
import math
import pickle
from pathlib import Path
import os
import time
import numpy as np
from formal_long_context_support import *
from long_context_spatial_events import project_decode,project_prefill,audit_global
from om3dthermal.serving.decode_policy import DecodePolicyModel
from om3dthermal.placement.nmp_load_balance import PhysicalResidentPlacement
from om3dthermal.architecture.feol_floorplan import resolve_feol_floorplan
from om3dthermal.power.feol_energy import FEOLEnergyModel,SCALARS,LAYERS

PLAN_ROOT=Path(os.environ['LOCALAPPDATA'])/'om3dthermal'/'formal_long_context_plans'


def candidate(m,c,b,path):
    if path=='HBM_GPU':return hbm(m,c,b)
    if c!='LC64K':return existing(m,c,b,path)
    fs=list((OUT/'checkpoints').glob(f'{m}_W64_B{b}_{LEGACY[path]}_{fingerprint()[:16]}.json'))
    if not fs:return None
    r=json.loads(fs[0].read_text())
    if path=='M3D_GPU':
        r=primary.corrected_gpu_result(r,primary.candidate_timing(m,'W64',b,ROOT),ROOT)
    r['summary']['system']=path;r['source_checkpoint']=str(fs[0].relative_to(ROOT));r['reused']=False
    return r


def plan_for(m,c,b,path,r):
    cache=PLAN_ROOT/f'{m}_{c}_B{b}_{path}_{fingerprint()[:16]}.pkl'
    if cache.exists():
        with cache.open('rb') as f:return pickle.load(f)
    w,cw,_=legacy.inputs(m,CONTEXTS[c][0],b,ROOT)
    if path=='M3D_NMP_CPA':
        print('RESTORE_PLACEMENT_ONLY',m,c,b,path,flush=True)
        e=DecodePolicyModel(w,project_root=ROOT,placement_policy='CRITICAL_PATH_AWARE',decode_start_context=cw.history+cw.prompt)
        clean=lambda rows:[{k:v for k,v in a.items() if k!='optimizer_runtime_s'} for a in rows]
        assert clean(e.placement.optimizer_audit)==clean(r['cpa_audit']), 'Frozen CPA trace changed'
        p=e.placement
    else:p=PhysicalResidentPlacement(w,resolve_feol_floorplan(ROOT),'UNIFORM_STRIPING')
    cache.parent.mkdir(parents=True,exist_ok=True)
    with cache.open('wb') as f:pickle.dump(p,f,protocol=5)
    return p


def energy_breakdown(r,pre_events=None):
    e=r['energy'];s=r['summary'];n=s['E2E_generated_tokens']
    if s['system']=='HBM_GPU':
        row={k:e[k] for k in ('GPU_energy_J','local_memory_energy_J','external_memory_energy_J')}
        row.update({k:0. for k in ('NMP_MAC_energy_J','NMP_SRAM_energy_J','NMP_fabric_energy_J','NMP_NoC_energy_J','NMP_reduction_energy_J','NMP_interface_energy_J')})
    else:
        comp={k:sum(e.get(phase+'_'+k,0) for phase in ('prefill','decode')) for k in (
            'gpu_dynamic_J','gpu_static_J','array_read_J','array_write_J','read_peripheral_J','row_select_J','column_select_J','write_driver_J',
            'miv_J','mac_J','sram_read_J','sram_write_J','router_J','pipeline_register_J','reduction_J','interface_J','feol_wire_J','feol_unresolved_J')}
        events={k:r['events'][k]+pre_events[k] for k in SCALARS}
        f=resolve_feol_floorplan(ROOT);wire=f.case.architecture.feol_route.wire
        coeff=wire.activity_factor*wire.capacitance_fF_per_um*1e-15*wire.voltage_V**2
        noc_bits=r['traffic'].get('NoC_GB_per_token',0)*n*1e9*8
        router_coeff=f.energy_config['parameters']['router']['value']*1e-12
        noc_router=noc_bits*router_coeff
        local_wire=sum(events[k] for k in ('sa_to_edge_bit_um','root_to_sa_bit_um'))*coeff
        fabric_wire=sum(events[k] for k in ('sa_to_tile_bit_um','root_to_tile_bit_um'))*coeff
        noc_wire=events['noc_link_bit_um']*coeff;interface_wire=events['root_to_port_bit_um']*coeff
        np.testing.assert_allclose(local_wire+fabric_wire+noc_wire+interface_wire,comp['feol_wire_J'],rtol=1e-12,atol=1e-8)
        row=dict(GPU_energy_J=comp['gpu_dynamic_J']+comp['gpu_static_J'],
            local_memory_energy_J=sum(comp[k] for k in ('array_read_J','array_write_J','read_peripheral_J','row_select_J','column_select_J','write_driver_J','miv_J','feol_unresolved_J'))+local_wire,
            external_memory_energy_J=0.,NMP_MAC_energy_J=comp['mac_J'],NMP_SRAM_energy_J=comp['sram_read_J']+comp['sram_write_J'],
            NMP_fabric_energy_J=comp['router_J']-noc_router+fabric_wire,NMP_NoC_energy_J=noc_wire+noc_router+comp['pipeline_register_J'],
            NMP_reduction_energy_J=comp['reduction_J'],NMP_interface_energy_J=comp['interface_J']+interface_wire)
    total=e['E2E_J']
    np.testing.assert_allclose(sum(row.values()),total,rtol=1e-12,atol=1e-7)
    assert min(row.values())>=0
    return dict(**row,total_E2E_energy_J=total,generated_tokens=n,J_per_token=total/n,tokens_per_J=n/total)


def spatial(m,c,b,path,r):
    key=f'{m}_{c}_B{b}_{path}'
    file=OUT/'spatial'/f'{key}.json'
    if file.exists():
        old=json.loads(file.read_text())
        assert old['physical_fingerprint']==fingerprint()
        assert old['E2E_J']==r['energy']['E2E_J'] and old['E2E_s']==r['summary']['E2E_s']
        return old
    started=time.perf_counter()
    p=plan_for(m,c,b,path,r);w,cw,_=legacy.inputs(m,CONTEXTS[c][0],b,ROOT)
    d=project_decode(p,cw,nmp=path!='M3D_GPU');audit_global(d,r['events'])
    pre=project_prefill(p,cw,r['prefill_ledger'])
    model=FEOLEnergyModel(p.floorplan,setup(ROOT)[2])
    global_pre={k:v.sum(axis=0).tolist() for k,v in pre.items()}
    gp=model.account(global_pre,r['summary']['prefill_s'],phase='prefill',policy='NO_NMP',total_flops=r['prefill_ledger']['total_flops'])
    for k,v in gp['components'].items():np.testing.assert_allclose(v,r['energy']['prefill_'+k],rtol=1e-12,atol=1e-8,err_msg=k)
    die_J=[]
    for die in range(318):
        ev={k:(d[k][die]+pre[k][die]).tolist() for k in d}
        a=model.account(ev,1,phase='decode',policy='NO_NMP')['components']
        die_J.append(sum(v for k,v in a.items() if not k.startswith('gpu_')))
    # Preserve the existing 0.1 W/slab NMP FEOL budget during Decode. GPU-only
    # keeps it inactive, exactly as the selected execution's saved energy.
    background=sum(r['energy'][phase+'_feol_unresolved_J'] for phase in ('prefill','decode'))
    die_J=[x+background/318 for x in die_J]
    total=r['energy']['E2E_J'];GPU=sum(r['energy'][phase+'_'+k] for phase in ('prefill','decode') for k in ('gpu_static_J','gpu_dynamic_J'))
    np.testing.assert_allclose(sum(die_J)+GPU,total,rtol=1e-12,atol=1e-7)
    result=dict(physical_fingerprint=fingerprint(),E2E_s=r['summary']['E2E_s'],E2E_J=total,
        GPU_energy_J=GPU,die_energy_J=die_J,die_power_W=[x/r['summary']['E2E_s'] for x in die_J],
        GPU_power_W=GPU/r['summary']['E2E_s'],projection_s=time.perf_counter()-started,
        event_conservation='PASS',method='EXACT_ATOM_BIRTH_CONTEXT_INTEGRATION__NO_DECODE_LATENCY_REEXECUTION',
        energy_audit=energy_breakdown(r,global_pre))
    save(file,result);print('SPATIAL_COMPLETE',key,result['projection_s'],flush=True)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--existing-only',action='store_true')
    parser.add_argument('--context',choices=CONTEXTS)
    parser.add_argument('--model',choices=MODELS);parser.add_argument('--batch',type=int,choices=(1,8));a=parser.parse_args()
    for m,c,b in points():
        if a.existing_only and c=='LC64K':continue
        if a.context and c!=a.context:continue
        if a.model and m!=a.model:continue
        if a.batch and b!=a.batch:continue
        for path in PATHS:
            r=candidate(m,c,b,path)
            if r is None:continue
            key=f'{m}_{c}_B{b}_{path}'
            save(OUT/'candidates'/f'{key}.json',r)
            if path!='HBM_GPU':spatial(m,c,b,path,r)
    print('AVAILABLE_CANDIDATES_COMPOSED',flush=True)


if __name__=='__main__':main()
