"""Isolated 18-case GPU placement diagnostic; never writes formal results."""
import argparse
from collections import Counter, defaultdict
import csv
import gc
import gzip
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import numpy as np
import formal_long_context_v2_support as f
from om3dthermal.serving.decode_policy import DecodePolicyModel
from om3dthermal.power.nmp_die_activity import PhysicalStageModel
from om3dthermal.placement.gpu_port_balanced import port_first_lanes

OUT=f.ROOT/'runs/m3d_gpu_port_balanced_diagnostic'
OLD=f.ROOT/'runs/m3d_gpu_workload_achieved_bw_diagnostic'
FORMAL=f.ROOT/'runs/formal_long_context_v2'


def read(p):
    if not p.exists() and p.suffix=='.json' and p.with_suffix('.json.gz').exists():
        return json.loads(gzip.decompress(p.with_suffix('.json.gz').read_bytes()))
    return json.loads(p.read_text(encoding='utf-8'))
def save(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')
def loadcsv(p):
    with p.open(encoding='utf-8',newline='') as s:return list(csv.DictReader(s))
def writecsv(p,rows):
    with p.open('w',encoding='utf-8',newline='') as s:
        writer=csv.DictWriter(s,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def close(a,b):assert math.isclose(a,b,rel_tol=1e-11,abs_tol=1e-7),(a,b)


class ObservedGPUStage(PhysicalStageModel):
    def evaluate(self,*args,**kwargs):
        kwargs['resources']=True
        return super().evaluate(*args,**kwargs)


def resident_audit(model):
    p=model.placement; audit=p.audit(); _,ports=port_first_lanes(model.floorplan)
    group=p.slot_used.sum(axis=1).reshape(70,p.dies).T*4
    resident=np.zeros((p.dies,len(model.floorplan.ports)),dtype=np.int64)
    for g,port in enumerate(ports):resident[:,port]+=group[:,g]
    audit.pop('group_resident_bytes');audit.pop('group_active_layer_slots')
    audit['per_port_resident_bytes']=resident.tolist()
    active=resident[resident>0]
    audit['resident_port_max_mean_ratio']=float(active.max()/active.mean())
    audit.update(getattr(p,'gpu_port_audit',{}))
    assert audit['slot_capacity_violations']==0
    assert audit['max_group_bytes']<=audit['group_capacity_bytes']
    assert audit['max_die_bytes']<=audit['die_capacity_bytes']
    return audit


def replay(m,c,b,policy,raw_steps):
    w,cw,_=f.legacy.inputs(m,c,b,f.ROOT)
    model=DecodePolicyModel(w,project_root=f.ROOT,placement_policy=policy,record_energy=False,
        external_bandwidth_cap=f.setup(f.ROOT)[4],decode_start_context=cw.history+cw.prompt)
    model.physical=ObservedGPUStage(model.floorplan,model.platform,w,record_events=False)
    placement=resident_audit(model)
    port_totals=np.zeros((model.floorplan.layout.slab_count,len(model.floorplan.ports)))
    op_rows=[]; steps=[]; counts=Counter(); sums=defaultdict(float); operator_totals=defaultdict(lambda:defaultdict(float))
    for index,context in enumerate(cw.contexts):
        r=model.step(context,'NO_NMP',include_stages=True)
        original=raw_steps[index]
        close(r['boundary_bytes'],original['boundary_bytes'])
        assert r['local_array_bytes']==r['boundary_bytes'] and r['noc_bytes']==r['nmp_flops']==0
        assert all(r['component_sums'][k]==0 for k in ('MAC','LOCAL_FABRIC','INTER_REGION_NOC'))
        if policy=='UNIFORM_STRIPING':close(r['latency_s'],original['latency_s'])
        for stage in r['stages']:
            for t in stage.get('external_transfers',[]):
                loads=t['resources']['loads'];active=loads[loads>0]
                if not len(active):continue
                duration=t['external_service_s'];mean=float(active.mean());peak=float(active.max())
                close(loads.sum(),t['total_boundary_bytes'])
                counts[t['limiting_reason']]+=1
                port_totals+=loads
                values=dict(active_ports=len(active),max_port_bytes=peak,mean_active_port_bytes=mean,
                    max_mean_load_ratio=peak/mean,max_port_utilization=t['max_port_utilization'],
                    mean_active_port_utilization=t['mean_active_port_utilization'],boundary_s=duration,
                    cap_only_s=t['global_cap_serialization_s'],port_serialization_s=t['port_cap_serialization_s'],
                    route_startup_s=t['rc_startup_s'])
                for k,v in values.items():sums[k]+=v
                sums[t['limiting_reason']+'_boundary_s']+=duration
                for k in ('boundary_s','cap_only_s','port_serialization_s','route_startup_s'):
                    operator_totals[stage['operator']][k]+=values[k]
                if index==0:
                    layer=stage['layer'];op=stage['operator']
                    baseop='ATTENTION_QK' if op=='KV_APPEND' else op
                    entry=model.placement.get(layer,baseop)
                    atoms=(w.n_heads_kv*b if op=='KV_APPEND' else context*w.n_heads_kv*b
                           if op in ('ATTENTION_QK','ATTENTION_AV') else b if op=='TOKEN_EMBED_LOOKUP' else entry.atom_count)
                    op_rows.append(dict(policy=policy,operator=op,layer=layer,active_atom_count=atoms,
                        atom_bytes=entry.atom_bytes,active_groups=stage['active_groups'],
                        limiting_reason=t['limiting_reason'],**values))
        steps.append(dict(context=context,latency_s=r['latency_s'],boundary_bytes=r['boundary_bytes'],
            external_service_s=r['external_service_s'],array_service_s=r['array_service_s'],
            gpu_related_s=r['component_sums']['GPU_COMPUTE']))
        if index in (0,15,31):print(m,c,b,policy,index+1,'/32',flush=True)
    dec=sum(s['latency_s'] for s in steps)
    totalbytes=sum(s['boundary_bytes'] for s in steps)
    ext=sum(s['external_service_s'] for s in steps)
    transfers=sum(counts.values())
    metrics=dict(transfer_count=transfers,**{k:counts[k] for k in ('PORT_SERIALIZATION','GLOBAL_THERMAL_CAP','ROUTE_STARTUP')},
        **dict(sums),**{k+'_transfer_mean':sums[k]/transfers for k in ('active_ports','max_port_bytes',
            'mean_active_port_bytes','max_mean_load_ratio','max_port_utilization','mean_active_port_utilization')})
    summary=dict(decode_s=dec,energy_status='PERFORMANCE_ONLY; energy model unchanged',
        boundary_bytes=totalbytes,boundary_service_s=ext,achieved_Bps=totalbytes/ext,
        array_s=sum(s['array_service_s'] for s in steps),gpu_related_s=sum(s['gpu_related_s'] for s in steps))
    assert summary['achieved_Bps']<=f.setup(f.ROOT)[4]*(1+1e-12)
    del model;gc.collect()
    return dict(summary=summary,port_metrics=metrics,steps=steps,first_step_operators=op_rows,
        operator_totals=dict(operator_totals),placement=placement,per_port_decode_bytes=port_totals.tolist())


def case(index):
    m,c,b=f.points()[index];key=f'{m}_{c}_B{b}'
    target=OUT/'cases'/f'{key}.json'
    if target.exists() or target.with_suffix('.json.gz').exists():return
    canonical=read(FORMAL/'candidates'/f'{key}_M3D_GPU.json')
    fp=canonical['legacy_fingerprint']
    cp=FORMAL/'checkpoints'/f'{key}_IOM3D_NO_NMP_{fp[:16]}.json'
    raw=read(cp); raw_steps=[json.loads(x) for x in cp.with_suffix('.jsonl').read_text(encoding='utf-8').splitlines()]
    # Old totals are reused; only missing port telemetry requires a GPU-only
    # replay with identical Uniform ownership. This is not a new baseline run.
    old=replay(m,c,b,'UNIFORM_STRIPING',raw_steps)
    new=replay(m,c,b,'GPU_PORT_BALANCED',raw_steps)
    close(old['summary']['decode_s'],raw['summary']['decode_s'])
    p=raw['summary']['prefill_s'];pe=raw['energy']['prefill_J']
    new['summary'].update(prefill_s=p,prefill_J=pe,E2E_s=p+new['summary']['decode_s'])
    new['summary']['tokens_per_s']=b*32/new['summary']['E2E_s']
    new['summary']['tokens_per_J']=None  # energy is outside this placement diagnostic
    close(new['placement']['resident_bytes'],old['placement']['resident_bytes'])
    save(target,dict(model=m,context=c,batch=b,checkpoint_sha256=sha(cp),prefill_status='EXACT_OLD_DIAGNOSTIC_REUSE',old=old,new=new))
    print('COMPLETE',key,new['summary']['tokens_per_s'],flush=True)


def stats(rows,prefix):
    gm=lambda values:math.exp(sum(math.log(v) for v in values)/len(values))
    util=[r[prefix+'_utilization_pct'] for r in rows];drop=[r[prefix+'_throughput_drop_pct'] for r in rows]
    return dict(utilization_min=min(util),utilization_max=max(util),utilization_geomean=gm(util),
        TPS_drop_min=min(drop),TPS_drop_max=max(drop),TPS_drop_geomean=gm(drop) if min(drop)>0 else None,
        drop_from_TPS_ratio_geomean=100*(1-gm([1-v/100 for v in drop])))


def analyze():
    from om3dthermal.architecture.feol_floorplan import resolve_feol_floorplan
    port_Bps=resolve_feol_floorplan(f.ROOT).port_Bps
    references={(r['model'],int(r['cached_history']),int(r['batch'])):r for r in loadcsv(OLD/'results.csv')}
    comparisons=[];ports=[];ops=[];worst=[]
    for m,c,b in f.points():
        result=read(OUT/'cases'/f'{m}_{c}_B{b}.json');h=f.CONTEXTS[c][1];ref=references[m,h,b]
        old=result['old'];new=result['new'];s=new['summary'];peak=float(ref['peak_bandwidth_cap_Bps'])
        close(old['summary']['gpu_related_s'],s['gpu_related_s'])
        for before,after in zip(old['steps'],new['steps']):
            close(before['gpu_related_s'],after['gpu_related_s'])
        close(old['summary']['achieved_Bps'],float(ref['achieved_bandwidth_Bps']))
        r=dict(model=m,context=c,H=h,batch=b,reference_tokens_per_s=float(ref['reference_tokens_per_s']),
            old_tokens_per_s=float(ref['achieved_tokens_per_s']),new_tokens_per_s=s['tokens_per_s'],
            old_throughput_drop_pct=float(ref['throughput_degradation_pct']),
            new_throughput_drop_pct=100*(1-s['tokens_per_s']/float(ref['reference_tokens_per_s'])),
            old_achieved_BW_TBps=old['summary']['achieved_Bps']/1e12,new_achieved_BW_TBps=s['achieved_Bps']/1e12,
            old_utilization_pct=float(ref['bandwidth_utilization_pct']),new_utilization_pct=100*s['achieved_Bps']/peak,
            old_bottleneck=ref['bottleneck'],new_bottleneck=max({'BOUNDARY':s['boundary_service_s'],'ARRAY':s['array_s'],'GPU_RELATED':s['gpu_related_s']},
                key={'BOUNDARY':s['boundary_service_s'],'ARRAY':s['array_s'],'GPU_RELATED':s['gpu_related_s']}.get),
            prefill_s=s['prefill_s'],new_decode_s=s['decode_s'])
        r['utilization_improvement_pp']=r['new_utilization_pct']-r['old_utilization_pct']
        r['TPS_gain_pct']=100*(r['new_tokens_per_s']/r['old_tokens_per_s']-1)
        comparisons.append(r)
        for label in ('old','new'):
            for reason in ('PORT_SERIALIZATION','GLOBAL_THERMAL_CAP','ROUTE_STARTUP'):
                result[label]['port_metrics'].setdefault(reason+'_boundary_s',0.)
            phase_load=np.asarray(result[label]['per_port_decode_bytes'])
            phase_active=phase_load[phase_load>0]
            result[label]['port_metrics'].update(phase_active_external_ports=len(phase_active),
                phase_max_port_bytes=float(phase_active.max()),
                phase_mean_active_port_bytes=float(phase_active.mean()),
                phase_port_max_mean_ratio=float(phase_active.max()/phase_active.mean()),
                phase_max_port_utilization=float(phase_active.max()/port_Bps/result[label]['summary']['decode_s']),
                phase_mean_active_port_utilization=float(phase_active.mean()/port_Bps/result[label]['summary']['decode_s']))
            ports.append(dict(model=m,context=c,batch=b,placement=label,**result[label]['port_metrics']))
            ops.extend(dict(model=m,context=c,batch=b,**row) for row in result[label]['first_step_operators'])
        if (m,c,b)==('Llama-3.1-8B','LC20K',1):
            for op,values in old['operator_totals'].items():
                worst.append(dict(operator=op,old_boundary_s=values['boundary_s'],
                    new_boundary_s=new['operator_totals'][op]['boundary_s'],cap_only_s=values['cap_only_s']))
    assert len(comparisons)==18
    for r in comparisons:
        assert all(math.isfinite(v) for v in r.values() if isinstance(v,(int,float)))
    writecsv(OUT/'comparison.csv',comparisons);writecsv(OUT/'port_comparison.csv',ports)
    writecsv(OUT/'first_step_operator_ports.csv',ops);writecsv(OUT/'worst_case_operator_attribution.csv',worst)
    aggregate={}
    for label in ('old','new'):
        aggregate[label]={'all':stats(comparisons,label)}
        for field in ('batch','model','context'):
            aggregate[label][field]={str(v):stats([r for r in comparisons if r[field]==v],label)
                for v in dict.fromkeys(r[field] for r in comparisons)}
    aggregate['regressions']=[r for r in comparisons if r['TPS_gain_pct']<-1e-9]
    save(OUT/'aggregate.json',aggregate)
    print(json.dumps(aggregate,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--case',type=int);parser.add_argument('--analyze',action='store_true');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.case is not None:
        # Two independent diagnostic workers may resume the same output set.
        # An exclusive per-case lock prevents duplicate physical replays.
        lock=OUT/f'.case_{args.case}.lock'
        while True:
            try:
                descriptor=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
                break
            except FileExistsError:
                time.sleep(1)
        try:return case(args.case)
        finally:
            os.close(descriptor)
            lock.unlink()
    if args.analyze:return analyze()
    protected=OUT/'protected_before.json'
    if not protected.exists():
        paths=[*OLD.rglob('*.csv'),*OLD.rglob('*.json'),*FORMAL.glob('*.csv'),*(FORMAL/'candidates').glob('*.json')]
        save(protected,{str(p.relative_to(f.ROOT)):sha(p) for p in paths})
    for i in range(18):
        subprocess.run([sys.executable,'-u',__file__,'--case',str(i)],check=True)
    analyze()
    assert all(sha(f.ROOT/p)==h for p,h in read(protected).items())
    save(OUT/'preservation.json',dict(status='PASS',protected_files=len(read(protected)),formal_updated=False,
        old_diagnostic_updated=False,HBM_NMP_thermal_runs=0,CPA_optimizer_calls=0,
        new_GPU_cases=18,old_GPU_port_telemetry_replays=18,Prefill_recomputed=False))


if __name__=='__main__':main()
