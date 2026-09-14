"""Fresh compact-window physical runs; existing runtime, equations and CPA."""
import argparse
import gc
import os
import pickle
import json
import time
import numpy as np
import formal_long_context_v2_support as f
import run_formal_iom3d_workloads as runner
from run_formal_long_context import SavedPlanModel
from long_context_spatial_events import project_decode,audit_global
from om3dthermal.power.feol_energy import FEOLEnergyModel

PLAN_ROOT=f.Path(os.environ.get('OM3DTHERMAL_PLAN_ROOT','F:/om3dthermal_cache/formal_long_context_v2_plans'))


def correct_nmp_prefill(result,m,c,b):
    """Same existing aggregate GPU Prefill model for every M3D executor."""
    candidate=f.primary.candidate_timing(m,c,b,f.ROOT)
    assert result['prefill_ledger']==candidate['ledger']
    s,e=result['summary'],result['energy']
    previous=s['prefill_s'];new=candidate['timing']['prefill_s']
    old_static=e['prefill_gpu_static_J'];new_static=new*f.setup(f.ROOT)[2].gpu_decode_power.static_power_W
    result['prefill_timing_audit']=dict(raw_prefill_s=previous,corrected_prefill_s=new,
        raw_GPU_static_J=old_static,corrected_GPU_static_J=new_static,
        source='EXISTING_CORRECTED_GPU_AGGREGATE_STREAMING_HELPER',physical_decode_changed=False)
    s.update(prefill_s=new,E2E_s=new+s['decode_s'],TTFT=new+s['first_step_ms']/1000)
    s['E2E_tok_s']=s['E2E_generated_tokens']/s['E2E_s']
    e['prefill_gpu_static_J']=new_static;e['prefill_J']+=new_static-old_static
    e['E2E_J']=e['prefill_J']+e['decode_J']
    e['E2E_J_per_token']=e['E2E_J']/s['E2E_generated_tokens']
    e['E2E_tokens_per_J']=s['E2E_generated_tokens']/e['E2E_J']
    return result


def _run(m,c,b,path,workers):
    start=time.perf_counter();key=f'{m}_{c}_B{b}_{path}'
    target=f.OUT/'candidates'/f'{key}.json'
    if target.exists() and json.loads(target.read_text())['summary']['status']=='EVALUATED':
        return key
    cp=f.cap(m,c,b,path)
    if cp['status']!='EVALUATED':
        result=dict(summary=cp,energy={},traffic={},cpa_audit=[],execution_status='NOT_EXECUTED_CAPACITY_GATE')
    elif path=='HBM_GPU':
        result=f.hbm(m,c,b)
    else:
        runner.OUT=f.OUT;runner.inputs=f.legacy.inputs;runner.setup=f.setup
        runner.DecodePolicyModel=SavedPlanModel;SavedPlanModel.workers=workers
        plan_file=PLAN_ROOT/f'{key}_{f.fingerprint()[:16]}.pkl'
        SavedPlanModel.plan_file=plan_file if path!='M3D_GPU' else None
        try:result=runner.run_case((cp,f.fingerprint()))
        finally:
            for e in list(SavedPlanModel.active):e.close()
        if path=='M3D_GPU':
            result=f.primary.corrected_gpu_result(result,f.primary.candidate_timing(m,c,b,f.ROOT),f.ROOT)
        else:
            result=correct_nmp_prefill(result,m,c,b)
            projection_start=time.perf_counter()
            for attempt in range(40):
                try:
                    with plan_file.open('rb') as stream:plan=pickle.load(stream)
                    break
                except PermissionError:
                    if attempt==39:raise
                    time.sleep(.5)
            _,cw,_=f.legacy.inputs(m,c,b,f.ROOT)
            if getattr(plan,'capacity_legalization',{}).get('fallback_triggered'):
                slots=plan.slot_used;layout=plan.floorplan.layout
                result['capacity_legalization']=plan.capacity_legalization
                result['physical_capacity_audit']=dict(max_slot_bytes=int(slots.max()),slot_capacity_bytes=layout.slot_capacity_bytes,
                    slot_capacity_violations=int((slots>layout.slot_capacity_bytes).sum()),
                    max_group_bytes=int(slots.sum(axis=1).max())*4,group_capacity_bytes=layout.slot_capacity_bytes*8*4,
                    max_die_bytes=int(slots.reshape(70,plan.dies,8).sum(axis=(0,2)).max())*4,
                    die_capacity_bytes=layout.total_capacity_bytes//plan.dies,resident_bytes=int(slots.sum())*4,
                    total_capacity_bytes=layout.total_capacity_bytes)
                a=result['physical_capacity_audit']
                assert a['slot_capacity_violations']==0 and a['resident_bytes']<=a['total_capacity_bytes']
                assert a['max_group_bytes']<=a['group_capacity_bytes'] and a['max_die_bytes']<=a['die_capacity_bytes']
                result['summary']['initial_placement_provenance']='UNIFORM_STRIPING_CAPACITY_LEGALIZED'
                if path=='M3D_NMP_UNIFORM':result['summary']['placement_policy']='UNIFORM_STRIPING_CAPACITY_LEGALIZED'
            slab=project_decode(plan,cw,nmp=True);audit_global(slab,result['events'])
            model=FEOLEnergyModel(plan.floorplan,f.setup(f.ROOT)[2])
            duration=result['summary']['decode_s'];energy=result['energy']
            die=[]
            for i in range(plan.floorplan.layout.slab_count):
                ev={k:v[i].tolist() for k,v in slab.items()}
                components=model.account(ev,1,phase='decode',policy='NO_NMP')['components']
                die.append(sum(v for k,v in components.items() if not k.startswith('gpu_')))
            background=energy['decode_feol_unresolved_J']
            die=[x+background/len(die) for x in die]
            gpu=energy['decode_gpu_dynamic_J']+energy['decode_gpu_static_J']
            np.testing.assert_allclose(sum(die)+gpu,energy['decode_J'],rtol=1e-12,atol=1e-7)
            f.save(f.OUT/'spatial_decode'/f'{key}.json',dict(model=m,context=c,batch=b,path=path,
                decode_s=duration,decode_J=energy['decode_J'],GPU_decode_J=gpu,die_decode_J=die,
                GPU_power_W=gpu/duration,die_power_W=[x/duration for x in die],
                phase='DECODE',prefill_energy_included=False,event_conservation='PASS',power_conservation='PASS',
                physical_fingerprint=f.fingerprint(),projection_s=time.perf_counter()-projection_start))
            del plan,slab;gc.collect()
    result['summary']['system']=path
    result.update(reused=False,workload_checkpoint_reused=False,benchmark_id=f.CONFIG['benchmark_id'])
    f.save(target,result)
    f.save(f.OUT/'runtime_rows'/f'{key}.json',dict(model=m,context=c,batch=b,path=path,
        elapsed_s=time.perf_counter()-start,inner_workers=workers,execution_checkpoint_reused=False))
    if path not in ('HBM_GPU','M3D_GPU'):
        cache=PLAN_ROOT/f'{key}_{f.fingerprint()[:16]}.pkl'
        if cache.exists():
            assert cache.resolve().parent==PLAN_ROOT.resolve()
            try:cache.unlink()  # v2-only reconstructible cache; physical checkpoints remain.
            except PermissionError:print('CACHE_RETAINED_FOR_LATER_CLEANUP',cache,flush=True)
    print('V2_COMPLETE',key,time.perf_counter()-start,flush=True)
    return key


def run(m,c,b,path,workers):
    key=f'{m}_{c}_B{b}_{path}'
    target=f.OUT/'candidates'/f'{key}.json'
    locks=PLAN_ROOT/'locks';locks.mkdir(parents=True,exist_ok=True)
    lock=locks/f'{key}.lock'
    while True:
        if target.exists() and json.loads(target.read_text())['summary']['status']=='EVALUATED':return key
        try:
            fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
            os.write(fd,str(os.getpid()).encode());os.close(fd)
            break
        except FileExistsError:time.sleep(1)
    try:return _run(m,c,b,path,workers)
    finally:
        assert lock.resolve().parent==locks.resolve()
        lock.unlink()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',required=True,choices=f.MODELS)
    p.add_argument('--context',required=True,choices=f.CONTEXTS);p.add_argument('--batch',type=int,required=True,choices=f.CONFIG['batches'])
    p.add_argument('--path',required=True,choices=f.PATHS);p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();run(a.model,a.context,a.batch,a.path,a.workers)
