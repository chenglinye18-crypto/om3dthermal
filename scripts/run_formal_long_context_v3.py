"""Isolated v3 H+P+G execution; no writes to v2 or legacy wave results."""
import argparse
import csv
import gc
import gzip
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import yaml

from preflight_formal_long_context_v3 import ROOT, OUT
from om3dthermal.workload.model_registry import load_dense_model_spec
from om3dthermal.serving.decode_policy import CachedWorkload, llama31_models
from om3dthermal.serving.cached_history_wave import evaluate_hbm_policy, select_hbm_best, latency_metrics
from om3dthermal.serving.workload_matrix import setup
from om3dthermal.serving.primary_execution import gpu_memory_closure
from om3dthermal.power.feol_energy import FEOLEnergyModel, empty_events, sum_events, MAXIMA
from formal_parallel_runtime import ParallelDecodeModel
from long_context_spatial_events import project_decode, project_prefill

CONFIG = yaml.safe_load((ROOT/'configs/experiment/formal_long_context_v3.yaml').read_text())
POINTS = [(m,c,b) for m in CONFIG['models'] for c in CONFIG['contexts'] for b in CONFIG['batches']]


def read(path):
    if path.exists():return json.loads(path.read_text(encoding='utf-8'))
    return json.loads(gzip.decompress(path.with_suffix(path.suffix+'.gz').read_bytes()))


def completed(path):
    return path.exists() or path.with_suffix(path.suffix+'.gz').exists()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False,
        default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x.item()), encoding='utf-8')
    temp.replace(path)


def inputs(index):
    m,c,b = POINTS[index]
    spec = load_dense_model_spec(ROOT/CONFIG['models'][m])
    if m == 'Llama-3.1-8B': spec = spec.model_copy(update={'n_param': llama31_models()[m].n_param})
    cw = CachedWorkload(CONFIG['contexts'][c], CONFIG['incremental_prefill_tokens'], CONFIG['generated_tokens'])
    return m,c,b,spec,cw


def hbm(index):
    m,c,b,spec,cw = inputs(index)
    _, ws, platform, backend, _ = setup(ROOT)
    with (ROOT/'runs/no_nmp_geometry_sensitivity_v2/thermal_limits.csv').open() as stream:
        bw = float(next(r for r in csv.DictReader(stream) if r['architecture']=='conventional_hbm_2x1')['Bthermal_TBps'])*1e12
    backend = backend.model_copy(update={'sustained_bandwidth_bytes_per_s': bw})
    policies = [evaluate_hbm_policy(spec,cw,b,ws,platform,backend,policy=p) for p in CONFIG['hbm_policies']]
    policies.append(select_hbm_best(*policies))
    for result in policies:
        result.update(model=m,context=c,batch=b,H=cw.history,P=cw.prompt,G=cw.generated)
        save(OUT/'candidates'/f'{m}_{c}_B{b}_{result["policy"]}.json', result)
    print('HBM COMPLETE', m,c,b,policies[-1]['selected_HBM_policy'], flush=True)


def global_events(slab):
    return empty_events() | {k:np.asarray(v).sum(axis=0).tolist() for k,v in slab.items()}


def physical(index, path):
    m,c,b,_,_=inputs(index)
    lock=OUT/'checkpoints'/f'{m}_{c}_B{b}_{path}.lock'
    target=OUT/'candidates'/f'{m}_{c}_B{b}_{path}.json'
    lock.parent.mkdir(parents=True,exist_ok=True)
    runtime_root=Path('F:/om3dthermal_cache/formal_long_context_v3_shared')
    runtime_root.mkdir(parents=True,exist_ok=True)
    coordinator=runtime_root/'.dispatch.lock'
    waiting=False
    while not completed(target):
        try:
            descriptor=os.open(coordinator,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
        except FileExistsError:
            time.sleep(1);continue
        os.close(descriptor)
        try:
            if lock.exists():
                print('OWNED BY OTHER RUNNER',m,c,b,path,flush=True)
                return
            admitted=b!=32 or len(list(lock.parent.glob('*_B32_*.lock')))<3
            if admitted:
                descriptor=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
                os.close(descriptor)
        finally:coordinator.unlink()
        if not admitted:
            if not waiting:print('WAITING FOR B32 SIMULATION MEMORY SLOT',m,c,path,flush=True)
            waiting=True;time.sleep(2);continue
        try:return _physical(index,path)
        finally:lock.unlink()


def _physical(index, path):
    m,c,b,spec,cw = inputs(index)
    key = f'{m}_{c}_B{b}_{path}'
    target = OUT/'candidates'/f'{key}.json'
    if completed(target): return
    start = time.perf_counter()
    print('START', key, flush=True)
    nmp = path != 'M3D_GPU'
    policy = 'MAC_NMP' if nmp else 'NO_NMP'
    placement = {'M3D_GPU':'GPU_PORT_BALANCED','M3D_NMP_UNIFORM':'UNIFORM_STRIPING','M3D_NMP_CPA':'CRITICAL_PATH_AWARE'}[path]
    os.environ['OM3DTHERMAL_SHARED_ROOT'] = 'F:/om3dthermal_cache/formal_long_context_v3_shared'
    ParallelDecodeModel.workers = 2
    w = spec.decode_input(batch_size=b,context_length=cw.history+cw.prompt+cw.generated)
    engine = ParallelDecodeModel(w,project_root=ROOT,record_energy=True,placement_policy=placement,
        decode_start_context=cw.history+cw.prompt,external_bandwidth_cap=None if nmp else setup(ROOT)[4])
    print('PLACED', key, round(time.perf_counter()-start,1), flush=True)
    pre = engine.prefill(cw)
    if nmp:
        pre['latency_s'] = max(pre['compute_s'], pre['ledger']['total_memory_bytes']/gpu_memory_closure(ROOT)['effective_Bps'])
    exact=False
    try:
        slab = project_decode(engine.placement,cw,nmp=nmp)
    except AssertionError:
        # Projection algebra is only an optimization. Unsupported integer
        # rounding/support patterns use the original physical event recorder.
        exact=True;slab=None
        engine.record_slabs=True;engine.physical.record_slabs=True
        print('EXACT PER-STEP SLAB EVENTS',key,flush=True)
    events = global_events(slab) if slab is not None else None
    pre_events = global_events(project_prefill(engine.placement,cw,pre['ledger']))
    optimizer = getattr(engine.placement,'optimizer_audit',[])
    account = FEOLEnergyModel(engine.floorplan,engine.platform)
    recorded=[]
    steps=[]
    log = OUT/'checkpoints'/f'{key}.jsonl'
    log.parent.mkdir(parents=True,exist_ok=True)
    try:
        with log.open('w',encoding='utf-8') as stream:
            for context in cw.contexts:
                r = engine.step(context,policy)
                if exact:
                    recorded.append(r['energy_events'])
                    if slab is None:slab={k:np.asarray(v).copy() for k,v in r['slab_events'].items()}
                    else:
                        for k,v in r['slab_events'].items():
                            if k in MAXIMA:slab[k]=np.maximum(slab[k],v)
                            else:slab[k]+=np.asarray(v)
                row = {k:r[k] for k in ('context','latency_s','boundary_bytes','local_array_bytes','component_sums')}
                steps.append(row)
                stream.write(json.dumps(row)+'\n');stream.flush()
                if len(steps)%8==0:print('STEPS',key,len(steps),'/32',round(time.perf_counter()-start,1),flush=True)
    finally:
        engine.close()
    duration=sum(r['latency_s'] for r in steps)
    if exact:events=sum_events(recorded)
    assert math.isclose(sum(s['local_array_bytes'] for s in steps)*8,events['array_read_bits']+events['array_write_bits'],rel_tol=1e-12)
    assert math.isclose(sum(s['boundary_bytes'] for s in steps)*8,events['interface_bits'],rel_tol=1e-12)
    die=[]
    for i in range(engine.floorplan.layout.slab_count):
        components=account.account({k:v[i].tolist() for k,v in slab.items() if np.asarray(v).ndim>0},1,phase='decode',policy='NO_NMP')['components']
        die.append(sum(v for k,v in components.items() if not k.startswith('gpu_')))
    dec=account.account(events,duration,phase='decode',policy=policy)
    pref=account.account(pre_events,pre['latency_s'],phase='prefill',policy=policy,total_flops=pre['ledger']['total_flops'])
    energy={**{'decode_'+k:v for k,v in dec['components'].items()},**{'prefill_'+k:v for k,v in pref['components'].items()}}
    total=sum(energy.values());elapsed=pre['latency_s']+duration
    background=dec['components']['feol_unresolved_J']
    die=[v+background/len(die) for v in die]
    gpu=dec['components']['gpu_dynamic_J']+dec['components']['gpu_static_J']
    np.testing.assert_allclose(sum(die)+gpu,dec['total_J'],rtol=1e-12,atol=1e-7)
    requests=[dict(request_id=i,wave=0,queue_delay=0.,admission_delay=0.,prefill_latency=pre['latency_s'],
                   first_decode_step=steps[0]['latency_s'],decode_duration=duration,completion_time=elapsed) for i in range(b)]
    result=dict(model=m,context=c,batch=b,H=cw.history,P=cw.prompt,G=cw.generated,path=path,status='EVALUATED',
        E2E_s=elapsed,decode_s=duration,tokens_per_s=b*cw.generated/elapsed,Decode_tokens_per_s=b*cw.generated/duration,
        E2E_J=total,J_per_token=total/(b*cw.generated),tokens_per_J=b*cw.generated/total,
        energy=energy,requests=requests,steps=steps,prefill=pre,events=events,
        GPU_power_W=gpu/duration,die_power_W=[x/duration for x in die],
        traffic=dict(M3D_boundary_bytes=sum(s['boundary_bytes'] for s in steps),
                     local_memory_bytes=sum(s['local_array_bytes'] for s in steps)),
        placement_policy=placement,optimizer_audit=optimizer,runtime_s=time.perf_counter()-start,
        spatial_accounting='EXACT_PER_STEP_PHYSICAL_EVENTS' if exact else 'EXACT_SUPPORTED_EVENT_PROJECTION',
        **latency_metrics(requests,cw.generated))
    save(target,result)
    print('COMPLETE',key,result['tokens_per_s'],round(result['runtime_s'],1),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--case',type=int);parser.add_argument('--path');parser.add_argument('--hbm-only',action='store_true')
    parser.add_argument('--start',type=int,default=0);parser.add_argument('--end',type=int,default=18)
    args=parser.parse_args()
    with (OUT/'capacity_audit.csv').open() as stream:capacity=list(csv.DictReader(stream))
    if len(capacity)!=18 or any(r['M3D_status']!='PASS' for r in capacity):raise RuntimeError('Capacity preflight must pass all 18 points')
    if args.path:return physical(args.case,args.path)
    for i in range(args.start,args.end): hbm(i)
    if args.hbm_only:return
    for i in range(args.start,args.end):
        for path in CONFIG['paths'][1:]:
            subprocess.run([sys.executable,__file__,'--case',str(i),'--path',path],check=True)


if __name__=='__main__':main()
