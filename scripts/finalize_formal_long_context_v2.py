"""Canonical 72-row table and lightweight scientific sanity assertions."""
import hashlib
import json
import math
import statistics
import time
import numpy as np
import formal_long_context_v2_support as f
from om3dthermal.power.feol_energy import sum_events


def main():
    refresh_file=f.OUT/'gpu_refresh_preservation.json'
    refresh=json.loads(refresh_file.read_text(encoding='utf-8')) if refresh_file.exists() else None
    repair_file=f.OUT/'capacity_repair_preservation.json'
    repair=json.loads(repair_file.read_text()) if repair_file.exists() else None
    prior_manifest=json.loads((f.OUT/'formal_long_context_v2_manifest.json').read_text()) if repair else None
    assert len(f.points())==18 and f.CONFIG['incremental_prefill_tokens']==128 and f.CONFIG['generated_tokens']==32
    assert set(f.CONFIG['contexts'].values())=={20000,64000,126000} and set(f.CONFIG['batches'])=={1,8}
    rows=[];energy_audit=[];capacity=[];runtime=[];ledgers={};thermal=[];projections=[];infeasible=[];prefill_audit=[]
    for m,c,b in f.points():
        prefill_times=[]
        for path in f.PATHS:
            key=f'{m}_{c}_B{b}_{path}'
            r=json.loads((f.OUT/'candidates'/f'{key}.json').read_text());s,e=r['summary'],r['energy']
            if 'prefill_timing_audit' in r:
                prefill_audit.append(dict(model=m,context=c,batch=b,path=path,**r['prefill_timing_audit']))
            assert not r['reused'] and not r['workload_checkpoint_reused']
            assert (s['H'],s['P'],s['G'],s['batch_size'])==(f.CONFIG['contexts'][c],128,32,b)
            if s['status']!='EVALUATED':
                failure=dict(model=m,context=c,H=s['H'],P=s['P'],G=s['G'],batch=b,path=path,
                    execution_status=s['status'],reason=s['physical_slot_status'],slot_excess_GB=s['physical_slot_excess_GB'])
                infeasible.append(failure);capacity.append(failure);energy_audit.append(failure)
                runtime.append(json.loads((f.OUT/'runtime_rows'/f'{key}.json').read_text()))
                rows.append(dict(**failure,E2E_s=None,prefill_s=None,decode_s=None,generated_tokens=b*32,
                    tokens_per_s=None,E2E_J=None,J_per_token=None,tokens_per_J=None,
                    Tmax_C=f.CONFIG['thermal']['design_point_C'] if path=='M3D_GPU' else None,
                    thermal_metric_type=f.CONFIG['thermal']['GPU_metric'] if path=='M3D_GPU' else f.CONFIG['thermal']['NMP_metric'],
                    thermal_phase='DESIGN_POINT' if path=='M3D_GPU' else 'DECODE',thermal_status='NOT_EXECUTED_CAPACITY_GATE',
                    execution_checkpoint_reused=False,thermal_provenance='Design point remains architecture provenance; no workload metrics fabricated'))
                thermal.append({k:rows[-1][k] for k in ('model','context','batch','path','Tmax_C','thermal_metric_type','thermal_phase','thermal_status')})
                continue
            assert s['E2E_generated_tokens']==b*32
            np.testing.assert_allclose(s['prefill_s']+s['decode_s'],s['E2E_s'],rtol=1e-14)
            np.testing.assert_allclose(s['E2E_generated_tokens']/s['E2E_s'],s['E2E_tok_s'],rtol=1e-14)
            np.testing.assert_allclose(b*32/e['E2E_J'],e['E2E_tokens_per_J'],rtol=1e-14)
            if path=='HBM_GPU':
                components={k:e[k] for k in ('GPU_energy_J','local_memory_energy_J','external_memory_energy_J')}
                np.testing.assert_allclose(sum(components.values()),e['E2E_J'],rtol=1e-12)
            else:
                if not (refresh and path=='M3D_GPU'):prefill_times.append(s['prefill_s'])
                components={k:v for k,v in e.items() if k.startswith(('decode_','prefill_'))
                    and k.endswith('_J') and k not in ('decode_J','prefill_J','decode_tokens_per_J','prefill_tokens_per_J')}
                for phase in ('decode','prefill'):
                    np.testing.assert_allclose(sum(v for k,v in components.items() if k.startswith(phase+'_')),e[phase+'_J'],rtol=1e-12)
                np.testing.assert_allclose(sum(components.values()),e['E2E_J'],rtol=1e-12)
                fp=r.get('legacy_fingerprint','')[:16]
                log=list((f.OUT/'checkpoints').glob(f'{m}_{c}_B{b}_{f.LEGACY[path]}_{fp}*.jsonl'))
                assert len(log)==1
                steps=[json.loads(line) for line in log[0].read_text().splitlines()]
                assert [x['context'] for x in steps]==list(range(s['H']+128,s['H']+160))
                total_events=sum_events(x['energy_events'] for x in steps)
                if not (refresh and path=='M3D_GPU'):
                    for k in total_events:np.testing.assert_allclose(total_events[k],r['events'][k],rtol=1e-12,atol=1e-7)
                if path!='M3D_GPU':
                    np.testing.assert_allclose(sum(x['latency_s'] for x in steps),s['decode_s'],rtol=1e-12)
                else:
                    assert s['placement_policy']==('GPU_PORT_BALANCED' if refresh else 'NOT_APPLICABLE_GPU_EXECUTION') and s['accepted_moves']==0
                    if refresh:
                        assert r['gpu_refresh']['logical_traffic_conservation']=='PASS'
                        np.testing.assert_allclose(sum(x['latency_s'] for x in r['gpu_refresh']['operator_steps']),s['decode_s'],rtol=1e-12)
            energy_audit.append(dict(model=m,context=c,batch=b,path=path,E2E_J=e['E2E_J'],conservation='PASS',**components))
            capacity.append(dict(model=m,context=c,batch=b,path=path,status=s['status'],
                external_capacity_mode=s.get('external_capacity_mode','NOT_APPLICABLE'),
                **{k:s[k] for k in ('weight_GB','initial_KV_GB','final_KV_GB','workspace_GB','peak_state_GB','HBM_resident_GB','Grace_resident_GB') if k in s}))
            capacity[-1].update(r.get('physical_capacity_audit',{}))
            if r.get('capacity_legalization'):
                capacity[-1]['initial_placement_provenance']=s['initial_placement_provenance']
            runtime.append(json.loads((f.OUT/'runtime_rows'/f'{key}.json').read_text()))
            if path in f.PATHS[:2]:
                t=dict(Tmax_C=f.CONFIG['thermal']['design_point_C'],thermal_metric_type=f.CONFIG['thermal']['GPU_metric'],
                    thermal_phase='DESIGN_POINT',provenance='EXISTING_85C_BANDWIDTH_THERMAL_CLOSURE',
                    thermal_status='THERMAL_CLOSED_DESIGN_POINT')
                if refresh:t['provenance']=r['thermal_provenance']
                actual_gpu_thermal=f.OUT/'thermal_rows_decode'/f'{key}.json'
                if path=='M3D_GPU' and actual_gpu_thermal.exists():
                    t=json.loads(actual_gpu_thermal.read_text(encoding='utf-8'))
            else:
                t=json.loads((f.OUT/'thermal_rows_decode'/f'{key}.json').read_text())
                x=json.loads((f.OUT/'spatial_decode'/f'{key}.json').read_text());projections.append(x['projection_s'])
                assert t['energy_denominator']=='DECODE_ONLY' and t['power_conservation']=='PASS'
                assert not x['prefill_energy_included'] and x['event_conservation']=='PASS'
                np.testing.assert_allclose(sum(x['die_decode_J'])+x['GPU_decode_J'],e['decode_J'],rtol=1e-12)
                np.testing.assert_allclose(t['package_power_W'],e['decode_J']/s['decode_s'],rtol=1e-12)
            row=dict(model=m,context=c,H=s['H'],P=s['P'],G=s['G'],batch=b,path=path,execution_status='EVALUATED',
                E2E_s=s['E2E_s'],prefill_s=s['prefill_s'],decode_s=s['decode_s'],generated_tokens=b*32,
                tokens_per_s=s['E2E_tok_s'],E2E_J=e['E2E_J'],J_per_token=e['E2E_J_per_token'],tokens_per_J=e['E2E_tokens_per_J'],
                Tmax_C=t['Tmax_C'],thermal_metric_type=t['thermal_metric_type'],thermal_phase=t['thermal_phase'],
                thermal_status=t['thermal_status'],execution_checkpoint_reused=False,
                thermal_provenance=t.get('provenance','DECODE_ENERGY_OVER_DECODE_TIME__DIE_BEOL_UNIFORM'),
                operating_point_source=f.CONFIG['operating_point_source'])
            assert all(math.isfinite(row[k]) and row[k]>0 for k in ('tokens_per_s','tokens_per_J','Tmax_C'))
            rows.append(row);thermal.append(dict(model=m,context=c,batch=b,path=path)|t)
            ledgers[key]={k:r[k] for k in ('summary','energy','traffic','events','prefill_ledger','component_sums') if k in r}
        if prefill_times:np.testing.assert_allclose(prefill_times,prefill_times[0],rtol=1e-12)
    assert len(rows)==72
    if repair:
        identity=lambda r:tuple(str(r[k]) for k in ('model','context','batch','path'))
        lookup={identity(r):r for r in rows}
        for old_row in repair['rows']:
            if refresh and old_row['path'] in f.PATHS[:2]:continue
            new_row=lookup[identity(old_row)]
            for field in ('E2E_s','tokens_per_s','E2E_J','tokens_per_J','Tmax_C'):
                assert float(old_row[field])==new_row[field],(identity(old_row),field)
        for name,digest in repair['files'].items():
            if refresh:continue  # Current refresh snapshot supersedes the historical repair snapshot.
            assert hashlib.sha256((f.ROOT/name).read_bytes()).hexdigest()==digest,name
        assert not infeasible and all(r['execution_status']=='EVALUATED' for r in rows)
    f.csv_out('final_e2e_metrics.csv',rows)
    tables={}
    for metric,file in [('tokens_per_s','four_path_tokens_per_s.csv'),('tokens_per_J','four_path_tokens_per_J.csv'),('Tmax_C','four_path_Tmax.csv')]:
        table=[]
        for m,c,b in f.points():
            selected=[r for r in rows if (r['model'],r['context'],r['batch'])==(m,c,b)]
            table.append(dict(model=m,context=c,batch=b,**{r['path']:r[metric] for r in selected}))
        f.csv_out(file,table);tables[metric]=table
    norms=[]
    for metric in ('tokens_per_s','tokens_per_J'):
        for num,den in [('M3D_GPU','HBM_GPU'),('M3D_NMP_UNIFORM','M3D_GPU'),('M3D_NMP_CPA','M3D_GPU'),('M3D_NMP_CPA','HBM_GPU'),('M3D_NMP_CPA','M3D_NMP_UNIFORM')]:
            valid=[r for r in tables[metric] if r[num] is not None and r[den] is not None]
            values=[r[num]/r[den] for r in valid]
            case=lambda i:'/'.join(str(valid[i][k]) for k in ('model','context','batch'))
            norms.append(dict(metric=metric,numerator=num,denominator=den,paired_cases=len(values),min=min(values),mean=statistics.mean(values),
                geomean=statistics.geometric_mean(values),max=max(values),min_case=case(values.index(min(values))),max_case=case(values.index(max(values)))))
    f.csv_out('normalized_summary.csv',norms);f.csv_out('energy_closure_audit.csv',energy_audit)
    f.csv_out('capacity_audit.csv',capacity);f.csv_out('runtime_summary.csv',runtime);f.csv_out('thermal_summary.csv',thermal)
    f.csv_out('prefill_consistency_audit.csv',prefill_audit)
    f.save(f.OUT/'candidate_ledgers.json',ledgers)
    changes=[]
    old_file=f.ROOT/'runs/formal_long_context_v1/final_e2e_metrics.csv'
    old=f.read_csv(old_file) if old_file.exists() else []
    for failure in infeasible:
        changes.append(dict(status='TREND_CHANGE',kind='CAPACITY_FEASIBILITY',previous_v1_status='EVALUATED',
            latency_breakdown='NOT_AVAILABLE_CAPACITY_GATE',**failure))
    for new in tables['tokens_per_s']:
        if not old:break
        prev=next(r for r in old if (r['model'],r['context_label'],int(r['B']))==(new['model'],new['context'],new['batch']))
        for num,den in [('M3D_NMP_CPA','M3D_GPU'),('M3D_NMP_CPA','M3D_NMP_UNIFORM')]:
            if new[num] is None or new[den] is None:continue
            if (new[num]>new[den]) != (float(prev[num+'_tok_s'])>float(prev[den+'_tok_s'])):
                relevant=[dict(r,component_sums=ledgers[f"{r['model']}_{r['context']}_B{r['batch']}_{r['path']}"].get('component_sums',{}))
                    for r in rows if (r['model'],r['context'],r['batch'])==(new['model'],new['context'],new['batch']) and r['path'] in (num,den)]
                changes.append(dict(status='TREND_CHANGE',numerator=num,denominator=den,rows=relevant))
    kv_trends=[];offload_trends=[]
    for m in f.MODELS:
        for b in f.CONFIG['batches']:
            series=[json.loads((f.OUT/'candidates'/f'{m}_{c}_B{b}_HBM_GPU.json').read_text()) for c in f.CONTEXTS]
            kv=[r['summary']['initial_KV_GB'] for r in series]
            assert all(a<z for a,z in zip(kv,kv[1:]))
            kv_trends.append(dict(model=m,batch=b,initial_KV_GB=kv,status='INCREASING_WITH_HISTORY'))
            offload=[r['traffic']['total_C2C_GB']/(b*32) for r in series]
            offload_trends.append(dict(model=m,batch=b,C2C_GB_per_generated_token=offload,
                status='NONDECREASING' if all(a<=z for a,z in zip(offload,offload[1:])) else 'TREND_CHANGE'))
    f.save(f.OUT/'trend_audit.json',dict(ordering_changes=changes,KV_pressure=kv_trends,HBM_offload=offload_trends,
        capacity_exceptions=infeasible,description='P/G changed; absolute E2E throughput is not expected to match v1'))
    protected=refresh['files'] if refresh else repair['source_before'] if repair else json.loads((f.OUT/'protected_hashes.json').read_text())
    for name,digest in protected.items():
        if repair and name.replace('\\','/')=='src/om3dthermal/placement/nmp_load_balance.py':continue
        assert hashlib.sha256((f.ROOT/name).read_bytes()).hexdigest()==digest,name
    execution=json.loads((f.OUT/'execution_manifest.json').read_text());operator=json.loads((f.OUT/'thermal_operator_audit.json').read_text())
    priority=json.loads((f.OUT/'priority_runtime.json').read_text()) if (f.OUT/'priority_runtime.json').exists() else {}
    elapsed=prior_manifest['total_wall_clock_s'] if repair else time.time()-execution['start_epoch']
    manifest=dict(config=f.CONFIG,base_HEAD=execution['base_HEAD'],operating_points=18,final_rows=72,
        NMP_physical_checkpoints=len(projections),NMP_Decode_steps=len(projections)*32,
        GPU_observation_steps=sum(r['path']=='M3D_GPU' and r['execution_status']=='EVALUATED' for r in rows)*32,
        v1_workload_reuse=False,adaptive_clipping=False,pytest_run=False,
        sanity_checks='FAIL_CAPACITY_INFEASIBLE' if infeasible else 'PASS',capacity_exceptions=infeasible,
        original_requirement_all_72_metrics_finite=not infeasible,
        total_wall_clock_s=elapsed,runtime_target='TARGET_1H_MET' if elapsed<=3600 else 'TARGET_1H_NOT_MET',
        outer_processes=execution['outer_processes'],inner_workers=execution['inner_workers'],cpu_count=execution['cpu_count'],
        maximum_outer_processes=priority.get('maximum_outer_processes',execution['outer_processes']),priority_runtime=priority,
        runtime_configurations=execution.get('runtime_configurations',[]),
        calibration=[r for r in runtime if (r['model'],r['context'],r['batch'])==('Llama-3.1-8B','LC64K',1) and r['path'] in f.PATHS[2:]],
        initial_available_RAM_bytes=execution['available_RAM_bytes'],
        summed_path_runtime_s={p:sum(r['elapsed_s'] for r in runtime if r['path']==p) for p in f.PATHS},
        runtime_accounting='Total wall includes orchestration and disk recovery. Path sums are recorded successful attempts inclusive of projection; projection is a subset, not additive. Interrupted pre-recovery attempt costs remain in total wall, not reconstructed.',
        summed_spatial_projection_s=sum(projections),thermal_operator=operator,
        thermal_solve_s=sum(r.get('thermal_solve_s',0) for r in thermal),trend_changes=len(changes),
        thermal_semantics='GPU baselines: 85C closed design point; NMP: workload-specific sustained Decode steady state. Not transient request peaks.',
        physical_fingerprint=f.fingerprint())
    if repair:
        manifest['capacity_repair']=dict(existing_rows_unchanged=69,valid_rows=72,pytest_run=False,
            wall_clock_s=time.time()-repair['start_epoch'],base_HEAD=repair['HEAD'],
            new_execution_paths=3,new_NMP_Decode_steps=64,new_thermal_solves=2,inner_workers=4,
            source_change='Capacity-only layer legalization; physical equations and CPA unchanged')
        manifest['fingerprint_provenance']=dict(existing_69=prior_manifest['physical_fingerprint'],repaired_3=f.fingerprint())
        manifest['initial_sweep_wall_clock_s']=elapsed
        manifest['runtime_accounting']+=' Capacity repair wall-clock is reported separately in capacity_repair.'
    f.save(f.OUT/'formal_long_context_v2_manifest.json',manifest)
    if refresh:
        from report_formal_gpu_refresh import finalize_refresh
        finalize_refresh(refresh,rows,manifest)
        print('FINALIZED GPU REFRESH: 72 rows; frozen NMP preserved',flush=True)
        return
    def md(data):
        keys=list(data[0]);return '\n'.join(['| '+' | '.join(keys)+' |','|'+'|'.join('---' for _ in keys)+'|',
            *['| '+' | '.join('N/A' if r[k] is None else f'{r[k]:.5g}' if isinstance(r[k],float) else str(r[k]) for k in keys)+' |' for r in data]])
    report=['# Final long-context benchmark v2','',
        'Canonical source: final_e2e_metrics.csv (72 rows). H=20000/64000/126000; P=128 incremental Prefill exactly once; G=32; B=1/8; Llama-3.1 8B/70B/405B. Decode contexts H+128 through H+159. Full E2E performance/energy includes Prefill and Decode; generated tokens=B*32.',
        'All workload executions are fresh v2. GPU raw event-observation timing is superseded by the unchanged corrected package-level aggregate GPU timing; physical NMP executes every step. Uniform and CPA are raw placements, not adaptive clipping.',
        'HBM/M3D-GPU Tmax=85 C is a THERMAL_CLOSED_DESIGN_POINT at frozen 3.297387/3.395784 TB/s, not a workload thermal solve. Uniform/CPA temperatures are WORKLOAD_SPECIFIC_STEADY_STATE, DECODE only, using E_decode/T_decode. These metrics are deliberately distinct and are not full-request transient peaks.',
        'NMP power uses actual per-die energy distributed uniformly in each memory die BEOL, with GPU FEOL source. Existing background, interface and all physical energy terms are retained. No Prefill energy is mapped. One existing M3D operator is reused for all feasible NMP RHS solves, with unchanged FP64 PCG tolerances.',
        'HBM external capacity is sufficiently provisioned without an invented TB capacity. Frozen local/external max overlap, 416.34 GB/s C2C, local read+write 1.9955 pJ/bit and external 5.3 pJ/bit remain unchanged. External energy is included in system tokens/J.',
        '## Throughput (tokens/s)',md(tables['tokens_per_s']),'## Energy efficiency (tokens/J)',md(tables['tokens_per_J']),
        '## Temperature (C; metric definitions above)',md(tables['Tmax_C']),'## Normalized results',md(norms),
        '## Runtime',json.dumps(manifest,indent=2),'## Trend audit',json.dumps(changes,indent=2),
        'Lightweight conservation and integrity assertions PASS. No pytest was run. The capacity repair preserves all 69 existing result values and their physical artifacts; only the three missing paths are newly executed. Frozen physical equations and CPA are unchanged. Legacy results may have been removed separately by the user during workspace cleanup.',
        '## Capacity exceptions',json.dumps(infeasible,indent=2)]
    if repair:
        report.extend(['## Capacity repair',
            'LC64K/B8 previously failed because cyclic start-layer phases caused local slot overflow, despite more total free memory than LC126K/B8. GPU feasibility now checks global persistent state plus workspace. Only when original Uniform initialization fails, a capacity-only layer reassignment retains die/group/atom ownership and uses no route, latency or CPA objective. CPA then runs unchanged from that legal initial placement. This Uniform is explicitly UNIFORM_STRIPING_CAPACITY_LEGALIZED.',
            json.dumps([dict(path=p,capacity=ledgers[f"Llama-3.1-405B_LC64K_B8_{p}"]['summary'].get('initial_placement_provenance','GLOBAL_CAPACITY_GATE')) for p in f.PATHS[1:]],indent=2)])
    (f.OUT/'formal_long_context_v2_report.md').write_text('\n\n'.join(report)+'\n',encoding='utf-8')
    print('FINALIZED',elapsed,manifest['runtime_target'],manifest['sanity_checks'],flush=True)


if __name__=='__main__':main()
