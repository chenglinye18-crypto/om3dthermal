"""Checkpointed formal H/P/G matrix; no thermal solves or hardware changes."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np

from om3dthermal.serving.workload_matrix import setup, inputs, capacity, conventional, timing
from om3dthermal.serving.decode_policy import DecodePolicyModel
from om3dthermal.power.feol_energy import FEOLEnergyModel,sum_events

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/formal_iom3d_workload_sweep_v1'


def serial(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    raise TypeError(type(x))


def fingerprint():
    config,*_=setup(ROOT)
    paths=[Path(__file__),*sorted((ROOT/'src/om3dthermal').rglob('*.py')),
           *sorted((ROOT/'configs').rglob('*.yaml')),ROOT/config['thermal_limits_source'],ROOT/config['hbm_thermal_source']]
    return hashlib.sha256(b''.join(p.read_bytes() for p in paths)).hexdigest()


def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,default=serial),encoding='utf-8')
    for retry in range(20):
        try:tmp.replace(path);return
        except PermissionError:
            if retry==19:raise
            time.sleep(.5)


def run_case(args):
    cap,fp=args;name,wid,batch,system=(cap[k] for k in ('model','workload_id','batch_size','system'))
    key=f'{name}_{wid}_B{batch}_{system}_{fp[:16]}'
    completed=OUT/'checkpoints'/f'{key}.json'
    if completed.exists():return json.loads(completed.read_text())
    if cap['status']!='EVALUATED':
        result=dict(summary=cap,traffic={},energy={},host_audit=[],cpa_audit=[])
    elif system=='HBM_GRACE_C2C':
        result=conventional(name,wid,batch,ROOT,cap)
    else:
        w,cw,_=inputs(name,wid,batch,ROOT)
        placement='CRITICAL_PATH_AWARE' if system.endswith('_CPA') else 'UNIFORM_STRIPING'
        policy='NO_NMP' if system=='IOM3D_NO_NMP' else 'MAC_NMP'
        config,_,_,_,cap_bw=setup(ROOT)
        started=time.perf_counter()
        print(f'START {key}',flush=True)
        e=DecodePolicyModel(w,project_root=ROOT,placement_policy=placement,record_energy=True,
            decode_start_context=cw.history+cw.prompt,external_bandwidth_cap=cap_bw if policy=='NO_NMP' else None)
        optimizer=getattr(e.placement,'optimizer_audit',[])
        optimized_bytes=moved_bytes=resident_moved_bytes=0.
        for a in optimizer:
            entry=e.placement.get(a['layer'],a['operator'],a.get('request_id'))
            size=entry.atom_count*entry.atom_bytes
            optimized_bytes+=size;moved_bytes+=size*a['moved_atom_fraction']
            resident_moved_bytes+=entry.atom_bytes*a['resident_atoms_moved']
        print(f'PLACED {key} {time.perf_counter()-started:.1f}s',flush=True)
        pre=e.prefill(cw)
        log=OUT/'checkpoints'/f'{key}.jsonl'
        steps=[]
        if log.exists():
            # A terminated process may leave only the final line incomplete.
            for line in log.read_text().splitlines():
                try:steps.append(json.loads(line))
                except json.JSONDecodeError:break
            assert [s['context'] for s in steps]==list(cw.contexts)[:len(steps)]
        log.parent.mkdir(parents=True,exist_ok=True)
        log.write_text(''.join(json.dumps(s)+'\n' for s in steps))
        with log.open('a',encoding='utf-8') as stream:
            for context in list(cw.contexts)[len(steps):]:
                s=e.step(context,policy)
                row={k:s[k] for k in ('context','latency_s','boundary_bytes','external_service_s','local_array_bytes',
                    'array_service_s','noc_bytes','nmp_flops','component_sums','energy_events')}
                steps.append(row);stream.write(json.dumps(row,default=serial)+'\n');stream.flush()
                if len(steps)%32==0:print(f'STEPS {key}: {len(steps)}/{cw.generated}; {time.perf_counter()-started:.1f}s',flush=True)
        generated=batch*cw.generated
        model=FEOLEnergyModel(e.floorplan,e.platform)
        seconds=sum(s['latency_s'] for s in steps);events=sum_events(s['energy_events'] for s in steps)
        dec=model.account(events,seconds,phase='decode',policy=policy)
        pref=model.account(pre['energy_events'],pre['latency_s'],phase='prefill',policy=policy,total_flops=pre['ledger']['total_flops'])
        total=dec['total_J']+pref['total_J']
        boundary=sum(s['boundary_bytes'] for s in steps)
        row={**cap,**timing(pre['latency_s'],[s['latency_s'] for s in steps],generated),
             'effective_bandwidth_TBps':e.floorplan.external_Bps/1e12,'prefill_host_GB':0,
             'optimizer_runtime_s':sum(a['optimizer_runtime_s'] for a in optimizer),
             'accepted_moves':sum(a['accepted_moves'] for a in optimizer),
             'moved_fraction':moved_bytes/optimized_bytes if optimized_bytes else 0,
             'resident_moved_fraction':resident_moved_bytes/optimized_bytes if optimized_bytes else 0}
        traffic=dict(local_array_GB_per_token=sum(s['local_array_bytes'] for s in steps)/generated/1e9,
             GPU_memory_boundary_GB_per_token=boundary/generated/1e9,
             external_realized_TBps=boundary/sum(s['external_service_s'] for s in steps)/1e12,
             NoC_GB_per_token=sum(s['noc_bytes'] for s in steps)/generated/1e9,
             NMP_FLOPs_per_token=sum(s['nmp_flops'] for s in steps)/generated,
             router_bit_traversals_per_token=events['router_bit_traversals']/generated,
             fabric_wire_bit_um_per_token=(events['sa_to_tile_bit_um']+events['root_to_tile_bit_um'])/generated,
             prefill_boundary_GB=pre['ledger']['total_memory_bytes']/1e9)
        energy=dict(energy_status='MODELED_EVENT_ENERGY__WORKLOAD_THERMAL_CLOSURE_PENDING',prefill_J=pref['total_J'],decode_J=dec['total_J'],
            decode_J_per_token=dec['total_J']/generated,decode_tokens_per_J=generated/dec['total_J'],
            E2E_J=total,E2E_J_per_token=total/generated,E2E_tokens_per_J=generated/total,average_decode_power_W=dec['average_power_W'],
            **{'decode_'+k:v for k,v in dec['components'].items()},**{'prefill_'+k:v for k,v in pref['components'].items()})
        result=dict(summary=row,traffic=traffic,energy=energy,host_audit=[],cpa_audit=optimizer,
                    events=events,prefill_ledger=pre['ledger'],component_sums={k:sum(s['component_sums'][k] for s in steps) for k in steps[0]['component_sums']})
    result['fingerprint']=fp
    save(completed,result)
    print(f'DONE {key}: {result["summary"].get("E2E_tok_s",cap["status"])}',flush=True)
    return result


def csv_out(name,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(rows)


def output(results,fp):
    ident=lambda r:{k:r['summary'][k] for k in ('model','workload_id','batch_size','system','status')}
    summaries=[{**r['summary'],**{k:v for k,v in r['energy'].items() if k in ('E2E_J_per_token','E2E_tokens_per_J','energy_status')}} for r in results]
    csv_out('summary.csv',summaries)
    csv_out('traffic.csv',[{**ident(r),**r['traffic']} for r in results])
    csv_out('energy.csv',[{**ident(r),**r['energy']} for r in results])
    csv_out('host_offload_audit.csv',[{**ident(r),**a} for r in results for a in r.get('host_audit',[])])
    csv_out('cpa_audit.csv',[{**ident(r),**{k:v for k,v in a.items() if not isinstance(v,(dict,list))}} for r in results for a in r.get('cpa_audit',[])])
    csv_out('b32_stress.csv',[r for r in summaries if r['batch_size']==32])
    norm=[]
    by={(r['model'],r['workload_id'],r['batch_size'],r['system']):r for r in summaries}
    for r in summaries:
        if r['system']=='HBM_GRACE_C2C':continue
        row={k:r[k] for k in ('model','workload_id','batch_size','system','status')}
        for label,base in [('vs_NO_NMP','IOM3D_NO_NMP'),('vs_UNIFORM','IOM3D_MAC_NMP_UNIFORM'),('vs_HBM','HBM_GRACE_C2C')]:
            b=by.get((r['model'],r['workload_id'],r['batch_size'],base))
            row[label]=r['E2E_tok_s']/b['E2E_tok_s'] if r['status']=='EVALUATED' and b and b['status']=='EVALUATED' else 'N/A_CAPACITY_INFEASIBLE'
        norm.append(row)
    csv_out('normalized.csv',norm)
    config,_,platform,hbm,bw=setup(ROOT)
    save(OUT/'manifest.json',dict(git_HEAD=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_fingerprint=fp,experiment=config,logical_rows=len(results),
        evaluated=sum(r['summary']['status']=='EVALUATED' for r in results),
        HBM_effective_Bps=hbm.sustained_bandwidth_bytes_per_s,HBM_bandwidth_status=hbm.bandwidth_source_status,
        IOM3D_NO_NMP_Bps=bw,MAC_NMP_boundary_cap='UNCHANGED_PHYSICAL_FEOL_CONFIG',
        platform=platform.model_dump(mode='json'),K=1000,timed_boundary='PREFILL(P|H)+G_AGGREGATE_DECODE_STEPS',
        thermal='NOT_RUN',numerator='B*G',host_residency='STATIC_TRAFFIC_MINIMIZING_EXTENTS_NO_MIGRATION',
        host_capacity_bytes=config['grace_capacity_bytes'],HBM_write_energy='UNRESOLVED_ABSOLUTE_TOKENS_PER_J_NOT_REPORTED'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=2);parser.add_argument('--capacity-only',action='store_true');parser.add_argument('--system');parser.add_argument('--batch',type=int)
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    config,*_=setup(ROOT);fp=fingerprint()
    capacities=[]
    for name in config['models']:
        for wid in config['workloads']:
            for batch in config['primary_batches']+config['stress_batches']:
                for system in config['systems']:
                    capacities.append(capacity(name,wid,batch,system,ROOT))
                print('CAPACITY',name,wid,batch,[r['status'] for r in capacities[-4:]],flush=True)
    csv_out('capacity.csv',capacities)
    if args.capacity_only:return
    selected=[c for c in capacities if (not args.system or c['system']==args.system) and (not args.batch or c['batch_size']==args.batch)]
    # B32 physical states can occupy several GB of host RAM; run those serially.
    ordinary=[c for c in selected if c['batch_size']!=32 or c['system']=='HBM_GRACE_C2C' or c['status']!='EVALUATED']
    large=[c for c in selected if c not in ordinary]
    results=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(run_case,[(c,fp) for c in ordinary]):
            results.append(result);output(results,fp)
    for c in large:
        # Fresh process releases all resident arrays before the next large case.
        with ProcessPoolExecutor(max_workers=1) as pool:result=next(pool.map(run_case,[(c,fp)]))
        results.append(result);output(results,fp)
    output(results,fp)


if __name__=='__main__':main()
