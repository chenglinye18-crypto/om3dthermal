"""Compose complete policies, scientific closures and the v3 result report."""
import csv
import json
import math
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
from run_formal_long_context_v3 import ROOT,OUT,CONFIG,POINTS,save,read


def csv_out(name,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open('w',newline='',encoding='utf-8') as stream:
        w=csv.DictWriter(stream,fieldnames=keys);w.writeheader();w.writerows(rows)


def table(rows,columns):
    def value(v):
        return f'{v:.4f}' if isinstance(v,float) else str(v)
    return '| '+' | '.join(columns)+' |\n|'+'|'.join(['---']*len(columns))+'|\n'+''.join('| '+' | '.join(value(r.get(c,'')) for c in columns)+' |\n' for r in rows)


def main():
    suite=ET.parse(OUT/'wave_tests.xml').getroot().find('testsuite')
    tested=dict(passed=int(suite.attrib['tests'])-int(suite.attrib['failures'])-int(suite.attrib['errors'])-int(suite.attrib['skipped']),
                failed=int(suite.attrib['failures'])+int(suite.attrib['errors']),runtime_s=float(suite.attrib['time']))
    test_report=read(OUT/'test_report.json');test_report['new_targeted']=tested;save(OUT/'test_report.json',test_report)
    dependency_suite=ET.parse(OUT/'physical_dependency_tests.xml').getroot().find('testsuite')
    dependency=dict(passed=int(dependency_suite.attrib['tests'])-int(dependency_suite.attrib['failures'])-int(dependency_suite.attrib['errors'])-int(dependency_suite.attrib['skipped']),
        failed=int(dependency_suite.attrib['failures'])+int(dependency_suite.attrib['errors']),skipped=int(dependency_suite.attrib['skipped']),
        runtime_s=float(dependency_suite.attrib['time']))
    test_report['physical_dependencies']=dependency
    test_report['related_total']={k:test_report['related'][k]+dependency[k] for k in ('passed','failed','skipped','runtime_s')}
    save(OUT/'test_report.json',test_report)
    rows=[];comparison=[];energy=[];latency=[];thermal=[];requests=[];checks=[]
    with (OUT/'capacity_audit.csv').open() as stream:capacity=list(csv.DictReader(stream))
    for m,c,b in POINTS:
        get=lambda p:read(OUT/'candidates'/f'{m}_{c}_B{b}_{p}.json')
        host,wave,best=[get(p) for p in ('HBM_HOST_OFFLOAD','HBM_RESIDENT_WAVE','HBM_BEST')]
        selected=wave if best['selected_HBM_policy']=='HBM_RESIDENT_WAVE' else host
        for k in ('tokens_per_s','tokens_per_J','energy','traffic','requests'):
            assert best[k]==selected[k]
        cap=next(r for r in capacity if (r['model'],r['workload'],int(r['B']))==(m,c,b))
        cap['safe_resident_batch']=wave['safe_resident_batch'];cap['wave_sizes']=json.dumps(wave['wave_sizes']);cap['num_waves']=wave['num_waves']
        cap['HBM_full_batch_fit']=host['waves'][0]['high_water_bytes']<=int(cap['HBM_capacity_bytes'])
        comp=dict(model=m,context=c,batch=b,HBM_overflow=not cap['HBM_full_batch_fit'],safe_resident_batch=wave['safe_resident_batch'],
            wave_sizes=json.dumps(wave['wave_sizes']),num_waves=wave['num_waves'],selected_HBM_policy=best['selected_HBM_policy'])
        for label,r in [('host',host),('wave',wave)]:
            for k in ('tokens_per_s','tokens_per_J','mean_TTFT','P95_TTFT','max_TTFT','mean_TPOT','P95_TPOT','mean_completion_latency','P95_completion_latency','max_completion_latency'):
                comp[label+'_'+k]=r.get(k)
            ident=dict(model=m,context=c,batch=b,path=r['policy'],H=CONFIG['contexts'][c],P=128,G=32)
            if r['status']=='EVALUATED':
                latency.append(ident|{k:r[k] for k in r if k.startswith(('mean_','P95_','max_'))})
                energy.append(ident|r['energy']|{'E2E_J':r['E2E_J']})
                requests.extend(ident|q for q in r['requests'])
        comparison.append(comp)
        for path in CONFIG['paths']:
            r=get(path)
            ident=dict(model=m,context=c,batch=b,path=path,H=CONFIG['contexts'][c],P=128,G=32)
            t=dict(Tmax_C=85.,thermal_feasible=True,thermal_metric='THERMAL_CLOSED_ACTIVE_SERVICE_DESIGN_POINT') if path=='HBM_BEST' else read(OUT/'thermal_rows'/f'{m}_{c}_B{b}_{path}.json')
            row=ident|{k:r[k] for k in ('E2E_s','decode_s','tokens_per_s','Decode_tokens_per_s','E2E_J','J_per_token','tokens_per_J')}
            row.update(E2E_tokens_per_s=r['tokens_per_s'],E2E_tokens_per_J=r['tokens_per_J'])
            row.update({k:r[k] for k in r if k.startswith(('mean_','P95_','max_'))})
            row.update(Tmax_C=t['Tmax_C'],thermal_feasible=t['thermal_feasible'],safe_resident_batch=wave['safe_resident_batch'] if path=='HBM_BEST' else b,
                num_waves=r.get('num_waves',1),wave_sizes=json.dumps(r.get('wave_sizes',[b])),selected_HBM_policy=best['selected_HBM_policy'] if path=='HBM_BEST' else 'NOT_APPLICABLE')
            traffic=r['traffic'];hostbytes=traffic.get('host_read_bytes',0)+traffic.get('host_write_bytes',0)
            row.update(HBM_host_read_GB=traffic.get('host_read_bytes',0)/1e9,wave_admission_GB=traffic.get('admission_bytes',0)/1e9,
                C2C_GB_per_token=hostbytes/1e9/(b*32),local_memory_GB=(traffic.get('HBM_read_bytes',0)+traffic.get('HBM_write_bytes',0)+traffic.get('local_memory_bytes',0))/1e9,
                M3D_boundary_GB=traffic.get('M3D_boundary_bytes',0)/1e9)
            if path!='HBM_BEST':
                row['Decode_local_memory_GB']=row['local_memory_GB']
                row['Decode_M3D_boundary_GB']=row['M3D_boundary_GB']
                pre_bytes=r['prefill']['ledger']['total_memory_bytes']
                row['local_memory_GB']+=pre_bytes/1e9
                row['M3D_boundary_GB']+=pre_bytes/1e9
            row['traffic_phase']='E2E_UNLESS_PREFIXED_DECODE'
            row['normalized_throughput']=r['tokens_per_s']/best['tokens_per_s']
            row['normalized_energy_efficiency']=r['tokens_per_J']/best['tokens_per_J']
            assert math.isclose(r['tokens_per_s']*r['E2E_s'],b*32,rel_tol=1e-12)
            assert math.isclose(sum(r['energy'].values()),r['E2E_J'],rel_tol=1e-12)
            if path=='HBM_BEST':
                assert sum(r['wave_sizes'])==b
                assert row['normalized_throughput']==row['normalized_energy_efficiency']==1
            else:
                assert [s['context'] for s in r['steps']]==list(range(CONFIG['contexts'][c]+128,CONFIG['contexts'][c]+160))
                ev=r['events']
                assert math.isclose(sum(s['local_array_bytes'] for s in r['steps'])*8,ev['array_read_bits']+ev['array_write_bits'],rel_tol=1e-12)
                assert math.isclose(sum(s['boundary_bytes'] for s in r['steps'])*8,ev['interface_bits'],rel_tol=1e-12)
            rows.append(row);energy.append(ident|r['energy']|{'E2E_J':r['E2E_J']})
            latency.append(ident|{k:row[k] for k in row if k.startswith(('mean_','P95_','max_'))})
            thermal.append(ident|t)
            requests.extend(ident|q for q in r['requests'])
            checks.append(ident|dict(throughput_closure='PASS',energy_closure='PASS'))
    aggregates=[];policy_effects=[]
    for group in [*CONFIG['models'],'ALL']:
        for path in CONFIG['paths']:
            subset=[r for r in rows if r['path']==path and (group=='ALL' or r['model']==group)]
            aggregates.append(dict(group=group,path=path,cases=len(subset),**{k:math.exp(sum(math.log(r[k]) for r in subset)/len(subset)) for k in ('normalized_throughput','normalized_energy_efficiency')}))
        subset=[r for r in comparison if group=='ALL' or r['model']==group]
        for mode in ('RESIDENT_WAVE','HBM_BEST'):
            effects=dict(group=group,policy=mode,cases=len(subset))
            for metric in ('tokens_per_s','tokens_per_J','mean_TTFT','mean_TPOT','P95_completion_latency'):
                ratios=[]
                for r in subset:
                    label='wave' if mode=='RESIDENT_WAVE' or r['selected_HBM_policy']=='HBM_RESIDENT_WAVE' else 'host'
                    ratios.append(r[label+'_'+metric]/r['host_'+metric])
                effects[metric+'_geomean_ratio_vs_HOST_OFFLOAD']=math.exp(sum(map(math.log,ratios))/len(ratios))
            policy_effects.append(effects)
    for name,data in [('capacity_audit.csv',capacity),('final_e2e_metrics.csv',rows),('hbm_policy_comparison.csv',comparison),
        ('energy_breakdown.csv',energy),('latency_metrics.csv',latency),('thermal_summary.csv',thermal),
        ('request_latencies.csv',requests),('aggregates.csv',aggregates),('policy_effects.csv',policy_effects),('closure_checks.csv',checks)]:csv_out(name,data)
    metadata=dict(benchmark_id='formal_long_context_v3',operating_points=18,formal_rows=72,old_HEAD='2b693b8a91da5e7a16efcfb04354bc4983832104',
        M3D_capacity_infeasible=sum(r['M3D_status']!='PASS' for r in capacity),
        B1_wave_infeasible=sum(r['batch']==1 and r['safe_resident_batch']==0 for r in comparison),
        thermal_over_85=[{k:r[k] for k in ('model','context','batch','path','Tmax_C')} for r in rows if r['Tmax_C']>85],
        wave_selected=sum(r['selected_HBM_policy']=='HBM_RESIDENT_WAVE' for r in comparison))
    with (ROOT/'runs/no_nmp_geometry_sensitivity_v2/thermal_limits.csv').open() as stream:
        limits=list(csv.DictReader(stream))
    metadata['HBM_thermal_bandwidth_TBps']=float(next(r['Bthermal_TBps'] for r in limits if r['architecture']=='conventional_hbm_2x1'))
    metadata['M3D_GPU_thermal_cap_TBps']=float(next(r['Bthermal_TBps'] for r in limits if r['architecture']=='orthogonal_m3d_igzo'))
    metadata['NMP_physical_boundary_cap_status']='UNCHANGED_FROZEN_IMPLEMENTATION'
    save(OUT/'manifest.json',metadata)
    report='''# Formal long-context v3

18 operating points: Llama-3.1-8B / Qwen2.5-32B base; H=20000/64000/126000; B=1/8/32; P=128; G=32. Each of B requests arrives at t=0. The output numerator is B*G. Historical H is cached KV, never H-token Prefill. Incremental linear/FFN computation covers P only; each Decode step j uses H+P+j and appends local KV.

## HBM resident waves

Persistent weights must fit. The maximum resident batch includes weights, final KV(H+P+G), and the larger of canonical Prefill(P) and final-step Decode workspaces, rounded to 32-byte slots. Canonical workspace includes live activations and runtime buffers; no additional unspecified runtime reserve is invented. Wave 1's historical KV is initially local; all waiting histories are host-valid. Each later wave admits H-token KV once, through the canonical Grace/C2C pipeline bounded also by HBM write bandwidth. Waves execute serially to completion and release KV. There is no weight streaming and no post-completion archival cost in completion latency or primary energy. These results do not use legacy fixed-S wave numbers.

HOST_OFFLOAD preserves the existing traffic-minimizing static extent allocation and optimistic max(local GPU/memory, recurring remote service) overlap. As in v2, external host capacity is sufficiently provisioned; this is not a claim that every point fits one physical 480 GB Grace memory. The canonical Grace/C2C bandwidth and energy coefficients remain unchanged. HBM_BEST selects the entire policy by E2E throughput (ties choose HOST_OFFLOAD); all its energy, traffic and latency fields come from that policy. No per-metric selection.

## Latency and energy definitions

TTFT = queue delay + admission delay + incremental Prefill + first Decode step. Completion latency is arrival-to-final-token. TPOT = active Decode duration / G. P95 uses the linear sample percentile across all B individual requests, including duplicate per-wave values. Decode throughput uses B*G divided by summed active wave Decode durations; E2E throughput uses full batch makespan including admission and Prefill.

GPU static energy is counted once over the serial active system interval, including admission. Waiting requests do not receive additional copies of server static energy. There is no added per-request queue idle-power model; the GPU serves the current wave while others wait. Separate HBM/Grace static power remains zero in the canonical baseline accounting. HBM reads use 3.00 pJ/bit and writes retain 1.9955 pJ/bit. Admission counts host-memory reads, C2C transfer and HBM writes once. All flat energy components close to E2E_J; tokens/J = B*G/E2E_J.

## Model provenance and boundaries

Llama-8B retains the exact v2 matrix-derived parameter count (8,030,261,248), not the rounded legacy registry value. Qwen2.5-32B is the base checkpoint: 64 layers, hidden 5120, FFN 27648, 40 Q heads, 8 KV heads, head dimension 128, vocabulary 152064, untied embeddings, BF16 weights/KV, maximum context 131072, no sliding-window attention. Its exact 32,763,876,352 parameters include QKV biases and RMSNorm; the official safetensors index lists 65,527,752,704 bytes. Parameter formula: 64*(2*5120^2+2*5120*1024+3*5120*27648+2*5120+5120+2*1024)+2*5120*152064+5120.

Sources: https://huggingface.co/Qwen/Qwen2.5-32B/raw/main/config.json and https://huggingface.co/Qwen/Qwen2.5-32B/raw/main/model.safetensors.index.json . The frozen dense analytical evaluator does not separately time projection-bias additions; all parameters occupy capacity, while active execution follows its existing matrix and small-operator accounting. These are model estimates, not measured inference.

## Physical and thermal semantics

The three M3D paths are fully local and use unchanged physical execution/hardware. GPU uses the established GPU_PORT_BALANCED placement, Uniform and CPA use their respective frozen NMP placement. Prefill reuses the canonical cached-prefix ledger and existing v2 path-specific timing: physical GPU memory service for M3D_GPU, corrected aggregate GPU Prefill for NMP systems. No adaptive execution is introduced.

Tmax is peak Decode steady-state temperature, not E2E duty-cycle average. HBM retains the 85 C active-service thermally closed design point. Every M3D workload receives its own GPU/per-die power RHS and FP64 GPU-PCG steady-state solve, using the existing operator cache; memory power is die-grouped and uniform in BEOL. Old v2 workload power solutions are not reused. Hardware, mesh and solver tolerances are unchanged.

## Capacity

'''
    report+=table(capacity,['model','workload','B','peak_bytes','safe_resident_batch','wave_sizes','M3D_status'])
    report+='\n## HBM overflow policy comparison\n\n'+table([r for r in comparison if r['HBM_overflow']],['model','context','batch','safe_resident_batch','wave_sizes','host_tokens_per_s','wave_tokens_per_s','selected_HBM_policy','host_P95_completion_latency','wave_P95_completion_latency'])
    report+='\n## Four-path results\n\n'+table(rows,['model','context','batch','path','tokens_per_s','tokens_per_J','Tmax_C'])
    report+='\n## Geometric means, normalized to HBM_BEST per case\n\n'+table(aggregates,['group','path','cases','normalized_throughput','normalized_energy_efficiency'])
    report+='\n## Wave-policy effects relative to HOST_OFFLOAD\n\n'+table(policy_effects,list(policy_effects[0]))
    report+='\nRatios below 1 improve latency; ratios above 1 improve throughput/energy efficiency. P95 completion need not worsen under waves: eliminating recurring remote traffic can outweigh serialization, while queueing can still delay the first token. No result is adjusted to enforce a narrative.\n'
    report+='\n## Status\n\n```json\n'+json.dumps(metadata,indent=2)+'\n```\n'
    report+='''
## Running and tests

Run these commands from the repository in the native Windows Conda environment `om3dthermal`:

```powershell
conda activate om3dthermal
python scripts/preflight_formal_long_context_v3.py
python scripts/run_formal_long_context_v3.py
python scripts/thermal_formal_long_context_v3.py
python scripts/finalize_formal_long_context_v3.py
python scripts/plot_formal_long_context_v3.py
```

The runner reuses completed v3 candidate files and uses F:/om3dthermal_cache/formal_long_context_v3_shared for temporary read-only execution maps. It never consumes legacy wave numbers or writes v2. Separate runners may own independent cases; wait for all runners before thermal/finalization. The thermal operator cache remains local and is not committed. Detailed candidate/checkpoint archives may be compressed without changing numeric content.

The required related serving/workload run returned 313 passed, 11 failed, 8 skipped in 271.01 seconds. Nine failures require previously deleted legacy result files, one exposes an existing v1/v2 configuration import collision (W64), and one asserts an obsolete HBM thermal bandwidth. Those old files, tests and physical settings were not changed to mask failures. The final new targeted run returned 6 passed in 6.65 seconds, including exact Qwen per-step/per-die event conservation. Full pytest was not run: no shared/public evaluator was modified. See test_report.json and test logs for exact results.
'''
    report=report.replace('6 passed in 6.65 seconds',f"{tested['passed']} passed in {tested['runtime_s']:.2f} seconds")
    report+=f"\nThe additional direct physical/decode dependency suite returned {dependency['passed']} passed, {dependency['failed']} failed, {dependency['skipped']} skipped in {dependency['runtime_s']:.2f} seconds. Combined related suites: {test_report['related_total']['passed']} passed, {test_report['related_total']['failed']} failed, {test_report['related_total']['skipped']} skipped.\n"
    (OUT/'README.md').write_text(report,encoding='utf-8')
    print(json.dumps(metadata),flush=True)


if __name__=='__main__':main()
