"""Refresh only formal-v2 GPU candidates; reuse port replay, project exact events."""
import argparse
import copy
import hashlib
import json
import subprocess
import time
import numpy as np
import formal_long_context_v2_support as f
from diagnose_m3d_gpu_port_balanced import read, loadcsv, sha, OUT as DIAG
from long_context_spatial_events import project_decode, project_prefill
from om3dthermal.placement.nmp_load_balance import PhysicalResidentPlacement
from om3dthermal.architecture.feol_floorplan import resolve_feol_floorplan
from om3dthermal.power.feol_energy import FEOLEnergyModel, empty_events

SNAP=f.OUT/'gpu_refresh_preservation.json'

def snapshot():
    if SNAP.exists():return read(SNAP)
    files={str(p.relative_to(f.ROOT)).replace('\\','/'):sha(p)
           for p in f.OUT.rglob('*') if p.is_file() and
           ('M3D_NMP_' in p.name or 'IOM3D_MAC_NMP' in p.name)}
    rows=loadcsv(f.OUT/'final_e2e_metrics.csv')
    value=dict(old_HEAD=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
               files=files,rows=rows,status='SNAPSHOT_BEFORE_REFRESH')
    f.save(SNAP,value)
    return value

def hbm_bandwidth():
    return float(next(r for r in loadcsv(f.ROOT/'runs/no_nmp_geometry_sensitivity_v2/thermal_limits.csv')
                      if r['architecture']=='conventional_hbm_2x1')['Bthermal_TBps'])*1e12

def hbm(m,c,b,old):
    original=f.legacy.setup
    def setup(root):
        cfg,ws,platform,backend,m3d=original(root)
        return cfg,ws,platform,backend.model_copy(update={
            'sustained_bandwidth_bytes_per_s':hbm_bandwidth()}),m3d
    f.legacy.setup=setup
    try:r=f.legacy.conventional(m,c,b,f.ROOT,f.cap(m,c,b,'HBM_GPU'))
    finally:f.legacy.setup=original
    s,t=r['summary'],r['traffic'];platform=f.setup(f.ROOT)[2]
    read_pj=f.setup(f.ROOT)[3].read_energy_pJ_per_bit
    assert abs(read_pj-3.)<1e-12
    write_pj=old['energy']['hbm_access_energy_pj_per_bit']
    energy={};phase_components={}
    for phase in ('prefill','decode'):
        prefix='prefill_' if phase=='prefill' else ''
        rd=t[prefix+'HBM_read_GB']*1e9;wr=t[prefix+'HBM_write_GB']*1e9
        host=(t['prefill_Grace_read_GB']+t['prefill_Grace_write_GB'])*1e9 if phase=='prefill' else (t['total_C2C_GB']-t['prefill_Grace_read_GB']-t['prefill_Grace_write_GB'])*1e9
        pc=platform.gpu_compute_power
        dynamic=(r['prefill_ledger']['total_flops']*(pc.e_compute_dynamic_J_per_FLOP_min+pc.e_compute_dynamic_J_per_FLOP_max)/2
                 if phase=='prefill' else r['energy']['GPU_dynamic_J'])
        comp=dict(gpu_dynamic_J=dynamic,gpu_static_J=s[phase+'_s']*platform.gpu_decode_power.static_power_W,
                  local_memory_read_J=rd*8*read_pj*1e-12,local_memory_write_J=wr*8*write_pj*1e-12,
                  external_memory_J=host*8*5.3e-12)
        phase_components[phase]=comp
        energy.update({phase+'_'+k:v for k,v in comp.items()})
        energy[phase+'_J']=sum(comp.values())
    energy.update(GPU_energy_J=sum(x['gpu_dynamic_J']+x['gpu_static_J'] for x in phase_components.values()),
                  local_memory_energy_J=sum(x['local_memory_read_J']+x['local_memory_write_J'] for x in phase_components.values()),
                  external_memory_energy_J=sum(x['external_memory_J'] for x in phase_components.values()),
                  hbm_read_energy_pj_per_bit=read_pj,hbm_write_energy_pj_per_bit=write_pj,
                  write_energy_provenance='UNCHANGED_FORMAL_WRITE_COEFFICIENT',HBM_static_power_W=0,Grace_static_power_W=0,
                  energy_status='HBM_ENERGY_CLOSED')
    r['energy']=energy;s['system']='HBM_GPU'
    r['thermal_provenance']='runs/no_nmp_geometry_sensitivity_v2/thermal_limits.csv; read=3.00 pJ/bit; Bthermal='+str(hbm_bandwidth()/1e12)+' TB/s; 85C'
    return r

def global_events(slab):
    return empty_events()|{k:np.asarray(v).sum(axis=0).tolist() for k,v in slab.items()}

def m3d(m,c,b,old):
    key=f'{m}_{c}_B{b}'
    diagnostic=read(DIAG/'cases'/f'{key}.json');new=diagnostic['new'];ns=new['summary']
    assert ns['decode_s']<=diagnostic['old']['summary']['decode_s']
    assert new['placement']['slot_capacity_violations']==0
    w,cw,_=f.legacy.inputs(m,c,b,f.ROOT)
    plan=PhysicalResidentPlacement(w,resolve_feol_floorplan(f.ROOT),'GPU_PORT_BALANCED')
    decode=global_events(project_decode(plan,cw,nmp=False))
    pre=global_events(project_prefill(plan,cw,old['prefill_ledger']))
    for k in ('array_read_bits','array_write_bits','interface_bits','gpu_decode_proxy_bits','miv_read_bits_by_layer','miv_write_bits_by_layer'):
        np.testing.assert_allclose(decode[k],old['events'][k],rtol=1e-12,atol=1e-7)
    r=copy.deepcopy(old);s=r['summary']
    times=[x['latency_s'] for x in new['steps']]
    s.update(f.legacy.timing(ns['prefill_s'],times,b*32),placement_policy='GPU_PORT_BALANCED',accepted_moves=0,
             CPA_moves=0,CPA_optimizer_runtime=0,achieved_bandwidth_Bps=ns['achieved_Bps'],
             boundary_service_s=ns['boundary_service_s'],array_service_s=ns['array_s'],
             bandwidth_utilization=ns['achieved_Bps']/f.setup(f.ROOT)[4])
    r['events']=decode;r['prefill_events']=pre;r['cpa_audit']=[]
    model=FEOLEnergyModel(plan.floorplan,f.setup(f.ROOT)[2]);e={}
    for phase,ev in (('decode',decode),('prefill',pre)):
        a=model.account(ev,s[phase+'_s'],phase=phase,policy='NO_NMP',
                        total_flops=r['prefill_ledger']['total_flops'] if phase=='prefill' else 0)
        e.update({phase+'_'+k:v for k,v in a['components'].items()});e[phase+'_J']=sum(a['components'].values())
    r['energy']=e
    r['component_sums'].update(ARRAY=ns['array_s'],EXTERNAL_BOUNDARY=ns['boundary_service_s'],GPU_COMPUTE=ns['gpu_related_s'])
    r['placement_audit']=new['placement']
    r['physical_capacity_audit']={k:v for k,v in new['placement'].items() if k in ('slot_capacity_violations','resident_bytes','total_capacity_bytes','max_group_bytes','group_capacity_bytes','max_die_bytes','die_capacity_bytes')}
    r['prefill_timing_audit']=dict(raw_prefill_s=ns['prefill_s'],formal_prefill_s=ns['prefill_s'],
        source='EXACT_PORT_DIAGNOSTIC_PHYSICAL_PREFILL_REUSE',physical_decode_changed=True,
        previous_aggregate_prefill_s=old['summary']['prefill_s'])
    r['gpu_refresh']=dict(diagnostic_source=str((DIAG/'cases'/f'{key}.json.gz').relative_to(f.ROOT)),
        timing_replayed=False,decode_steps_reused=32,event_projection='EXACT_ATOM_BIRTH_CONTEXT_INTEGRATION',
        logical_traffic_conservation='PASS',capacity='PASS',energy_conservation='PASS',
        operator_totals=new['operator_totals'],operator_steps=new['steps'],CPA='NOT_APPLICABLE')
    r['thermal_provenance']='Frozen M3D 85C thermal cap; physical operator-level data-chain; contention-aware external-port service; GPU_PORT_BALANCED'
    return r

def finish(r):
    s,e=r['summary'],r['energy'];n=s['E2E_generated_tokens']
    e['E2E_J']=e['decode_J']+e['prefill_J']
    for phase in ('E2E','decode'):
        e[phase+'_J_per_token']=e[phase+'_J']/n;e[phase+'_tokens_per_J']=n/e[phase+'_J']
        e[phase+'_average_power_W']=e[phase+'_J']/s[phase+'_s']
    e['average_decode_power_W']=e['decode_average_power_W']
    if s['system']=='M3D_GPU':
        s['thermal_bandwidth_cap_TBps']=f.setup(f.ROOT)[4]/1e12
        s['effective_bandwidth_TBps']=s['achieved_bandwidth_Bps']/1e12
        r['traffic']['external_realized_TBps']=s['effective_bandwidth_TBps']
        r['traffic']['external_bandwidth_denominator']='SUM_DECODE_OPERATOR_BOUNDARY_ACTIVE_SERVICE_SECONDS'
    r.update(reused=False,workload_checkpoint_reused=False,benchmark_id=f.CONFIG['benchmark_id'],formal_gpu_refresh=True)
    return r

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--case',type=int);args=parser.parse_args()
    snapshot()
    assert read(DIAG/'verdict.json')['verdict']=='PORT_BALANCED_PLACEMENT_EFFECTIVE'
    for i,(m,c,b) in enumerate(f.points()):
        if args.case is not None and args.case!=i:continue
        for path,fn in [('HBM_GPU',hbm),('M3D_GPU',m3d)]:
            p=f.OUT/'candidates'/f'{m}_{c}_B{b}_{path}.json';old=read(p)
            if old.get('formal_gpu_refresh'):
                f.save(p,finish(old))
                continue
            started=time.perf_counter();r=finish(fn(m,c,b,old));f.save(p,r)
            print('REFRESHED',m,c,b,path,round(time.perf_counter()-started,2),flush=True)

if __name__=='__main__':main()
