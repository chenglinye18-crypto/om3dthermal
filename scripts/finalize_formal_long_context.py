"""Publish only a complete 18-case four-path LC dataset, with thermal status."""
import json
import math
import statistics
import re
from formal_long_context_support import *
from compose_formal_long_context import energy_breakdown


def main():
    metrics=[];perf=[];eff=[];thermal=[];energy=[];capacity=[];reuse=[];ledgers={}
    for m,c,b in points():
        base=identity(m,c,b);row=dict(base);pr=dict(base);er=dict(base)
        for path in PATHS:
            key=f'{m}_{c}_B{b}_{path}'
            r=json.loads((OUT/'candidates'/f'{key}.json').read_text())
            tr=json.loads((OUT/'thermal_rows'/f'{key}.json').read_text())
            tr['hotspot_coordinate_units']='m'
            s,e=r['summary'],r['energy'];n=b*base['G']
            ledgers[key]={k:r[k] for k in ('summary','energy','traffic','events','prefill_ledger','source_checkpoint','fingerprint') if k in r}
            assert s['E2E_generated_tokens']==n and s['status']=='EVALUATED'
            rate=s['E2E_tok_s'];tj=e['E2E_tokens_per_J'];temp=tr['Tmax_C']
            assert all(math.isfinite(x) and x>0 for x in (rate,tj,temp))
            row.update({path+'_tok_s':rate,path+'_tok_J':tj,path+'_Tmax_C':temp,path+'_thermal_status':tr['thermal_status']})
            pr[path+'_tok_s']=rate;er[path+'_J_per_token']=e['E2E_J_per_token'];er[path+'_tok_per_J']=tj
            thermal.append(tr)
            audit=energy_breakdown(r) if path=='HBM_GPU' else json.loads((OUT/'spatial'/f'{key}.json').read_text())['energy_audit']
            if path=='M3D_GPU':
                # The GPU-only memory interface is active, but is not an NMP
                # execution component. Reclassify it without changing energy.
                audit['local_memory_energy_J']+=audit['NMP_interface_energy_J']
                audit['NMP_interface_energy_J']=0.
            energy.append(dict(**base,path=path,**audit))
            capacity.append(dict(**base,path=path,status=s['status'],workspace_GB=s['workspace_GB'],peak_state_GB=s['peak_state_GB'],
                external_capacity_mode=EXTERNAL_MODE if path=='HBM_GPU' else 'NOT_APPLICABLE',
                external_resident_required_GB=s.get('Grace_resident_GB',0),local_capacity_GB=s.get('HBM_capacity_GB') if path=='HBM_GPU' else s['capacity_GB']))
            reuse.append(dict(**base,path=path,performance_action='RECOMPOSE_HBM_UNBOUNDED_CAPACITY' if path=='HBM_GPU' else 'EXACT_REUSE' if c!='LC64K' else 'NEW_LC64K',
                physical_decode_checkpoint_reused=c!='LC64K' and path in PATHS[2:],
                spatial_action='EXACT_EVENT_INTEGRATION_NO_OLD_DECODE_RERUN' if path!='HBM_GPU' else 'E2E_LOCAL_ACCESS_ENERGY',
                source_checkpoint=r.get('source_checkpoint','HBM_ANALYTICAL_TRAFFIC_LEDGER')))
        for name,num,den in [('M3D_GPU_over_HBM','M3D_GPU','HBM_GPU'),('UNIFORM_over_M3D_GPU','M3D_NMP_UNIFORM','M3D_GPU'),
            ('CPA_over_M3D_GPU','M3D_NMP_CPA','M3D_GPU'),('CPA_over_UNIFORM','M3D_NMP_CPA','M3D_NMP_UNIFORM'),('CPA_over_HBM','M3D_NMP_CPA','HBM_GPU')]:
            pr[name]=row[num+'_tok_s']/row[den+'_tok_s']
            er[name]=row[num+'_tok_J']/row[den+'_tok_J']
        row['metric_status']='THERMAL_FEASIBLE' if all(row[p+'_thermal_status']=='THERMAL_FEASIBLE' for p in PATHS) else 'NOMINAL_WITH_EXPLICIT_THERMAL_CLOSURE_REQUIRED'
        metrics.append(row);perf.append(pr);eff.append(er)
    csv_out('final_e2e_metrics.csv',metrics);csv_out('four_path_tokens_per_s.csv',perf);csv_out('four_path_tokens_per_J.csv',eff)
    csv_out('four_path_Tmax.csv',thermal);csv_out('energy_closure_audit.csv',energy);csv_out('capacity_audit.csv',capacity);csv_out('reuse_audit.csv',reuse)
    save(OUT/'candidate_ledgers.json',ledgers)
    norms=[]
    for label,rows in [('throughput',perf),('energy_efficiency',eff)]:
        for ratio in [k for k in rows[0] if '_over_' in k]:
            low=min(rows,key=lambda r:r[ratio]);high=max(rows,key=lambda r:r[ratio]);v=[r[ratio] for r in rows]
            case=lambda r:f"{r['model']} {r['context_label']} B{r['B']}"
            norms.append(dict(metric=label,ratio=ratio,min=min(v),mean=statistics.mean(v),geomean=statistics.geometric_mean(v),max=max(v),min_case=case(low),max_case=case(high)))
    csv_out('normalized_summary.csv',norms)
    therm=[]
    for path in PATHS:
        rs=[r for r in thermal if r['path']==path];peak=max(rs,key=lambda r:r['Tmax_C'])
        therm.append(dict(path=path,max_Tmax_C=peak['Tmax_C'],mean_Tmax_C=statistics.mean(r['Tmax_C'] for r in rs),
            over_85C=sum(r['Tmax_C']>85 for r in rs),max_case=f"{peak['model']} {peak['context_label']} B{peak['B']}",
            hotspot_statistics=json.dumps({k:sum(r['hotspot_component']==k for r in rs) for k in {r['hotspot_component'] for r in rs}})))
    csv_out('thermal_summary.csv',therm)
    execution=[]
    for m in MODELS:
        raw=(ROOT/'runs'/f'formal_long_context_execution_{m.split("-")[-1].lower()}.log').read_bytes()
        log=raw.decode('utf-16' if raw.startswith((b'\xff\xfe',b'\xfe\xff')) else 'utf-8-sig')
        for b in (1,8):
            for path in PATHS[1:]:
                key=f'{m}_W64_B{b}_{LEGACY[path]}_{fingerprint()[:16]}'
                elapsed=re.findall(re.escape(key)+r': 512/512; ([0-9.]+)s',log)
                if not elapsed:raise RuntimeError('Missing exact completed runtime: '+key)
                execution.append(dict(**identity(m,'LC64K',b),path=path,Decode_steps=512,
                    new_case_checkpoints=1,elapsed_including_placement_s=float(elapsed[-1]),
                    checkpoint=f'checkpoints/{key}.json',runtime_basis='RUN_LOG_COMPLETE_512_STEPS_INCLUDES_PLACEMENT'))
    csv_out('lc64k_execution_audit.csv',execution)
    wins=sum(r['CPA_over_M3D_GPU']>1 for r in perf);ties=sum(r['CPA_over_M3D_GPU']==1 for r in perf)
    cfg,_,platform,hbm,_=setup(ROOT)
    manifest=dict(base_HEAD='b5f98baa692eacdc0897c094c4d25611a4d38856',primary_models=list(MODELS),primary_contexts=list(CONTEXTS),primary_batches=[1,8],
        context_mapping={c:dict(internal_id=v[0],H=v[1],P=v[2],G=v[3],origin='new workload' if c=='LC64K' else 'old '+v[0]) for c,v in CONTEXTS.items()},
        evaluation_paths=list(PATHS),primary_cases=18,excluded_from_primary={'W1_SHORT_CONTEXT':'SHORT_CONTEXT_BOUNDARY_EXPERIMENT','B32_HIGH_CONCURRENCY_STRESS':'HIGH_CONCURRENCY_STRESS_ARCHIVED'},
        external_capacity_mode=EXTERNAL_MODE,hbm_access_energy_pj_per_bit=HBM_ACCESS_PJ,HBM_energy_status='HBM_ENERGY_CLOSED',
        HBM_access_provenance='Representative measured access/transferred-bit energy; arithmetic mean (0.978+3.013)/2; applied to read+write by user instruction',
        C2C_bandwidth_GBps=platform.host_offload.direct_effective_bandwidth_bytes_per_second/1e9,external_access_energy_pj_per_bit=5.3,
        HBM_thermal_bandwidth_TBps=hbm.sustained_bandwidth_bytes_per_s/1e12,M3D_GPU_thermal_bandwidth_TBps=primary.gpu_memory_closure(ROOT)['effective_Bps']/1e12,
        HBM_static_W=0,Grace_static_W=0,NMP_FEOL_background_budget_W_per_slab=0.1,
        thermal_convention='USER_CONFIRMED_DIE_GROUPED_BEOL_UNIFORM__E2E_EQUIVALENT_STEADY_STATE',
        hotspot_coordinate_units='m',
        NMP_power_W_definition='MAC + SRAM + Fabric + NoC + reduction; subset of memory_power_W. FEOL background and shared interface remain in memory_power_W.',
        HBM_thermal_boundary='GPU + local HBM package; external Grace and off-package link energy excluded; no new C2C mapper',
        thermal_setup={'M3D':'EXISTING_VALIDATED_OPERATOR_REUSED','HBM':'USER_AUTHORIZED_REBUILD_ONCE_OF_MISSING_IDENTICAL_OPERATOR'},
        f_NMP_thermal_closure_mechanism='NOT_IMPLEMENTED_IN_REPOSITORY; NO_DVFS_ADDED',
        physical_fingerprint=fingerprint(),existing_operating_points_reused=12,existing_NMP_case_checkpoints_reused=24,existing_NMP_Decode_steps_reused=9216,
        new_LC64K_operating_points=6,new_NMP_case_checkpoints=12,new_NMP_Decode_steps=6144,new_GPU_observation_steps=3072,
        new_LC64K_summed_job_runtime_s=sum(r['elapsed_including_placement_s'] for r in execution),
        simulation_worker_changes=[json.loads(p.read_text(encoding='utf-8-sig')) for p in sorted(OUT.glob('runtime_parallelism_*.json'))],
        CPA_wins=wins,GPU_wins=18-wins-ties,ties=ties,adaptive_clipping=False,
        closure=dict(matrix=True,tokens_per_s=True,tokens_per_J=True,Tmax_computed=True,all_paths_thermal_feasible=all(r['Tmax_C']<=85 for r in thermal)),
        canonical_data_source='runs/formal_long_context_v1/final_e2e_metrics.csv')
    save(OUT/'formal_long_context_manifest.json',manifest)
    overlap_example=next(r for r in thermal if r['model']=='Llama-3.1-70B' and r['context_label']=='LC64K' and r['B']==1 and r['path']=='HBM_GPU')
    def md(rows,keys):
        fmt=lambda v:f'{v:.6g}' if isinstance(v,float) else str(v)
        return '\n'.join(['| '+' | '.join(keys)+' |','|'+'|'.join('---' for _ in keys)+'|',
            *['| '+' | '.join(fmt(r[k]) for k in keys)+' |' for r in rows]])
    report=['# Formal long-context four-path E2E evaluation','',
        'Canonical source: `final_e2e_metrics.csv`. Exactly 18 points; no adaptive clipping. All temperatures are workload-specific E2E-equivalent steady-state solves. Throughput, energy efficiency and their ratios are nominal; per-path thermal-feasibility flags remain binding.',
        'LC20K=(20000,512,256); LC64K=(64000,512,512); LC126K=(126000,512,512). Models 8B/70B/405B, batches 1/8. W1 is retained as short-context boundary evidence; B32 remains archived stress. Historical artifacts are preserved.',
        '', '## Throughput (tokens/s)',md(perf,('model','context_label','B',*[p+'_tok_s' for p in PATHS])),
        '', '## Energy efficiency (tokens/J)',md(eff,('model','context_label','B',*[p+'_tok_per_J' for p in PATHS])),
        '', '## Temperature (C)',md(metrics,('model','context_label','B',*[p+'_Tmax_C' for p in PATHS],'metric_status')),
        '', '## Normalized results',md(norms,tuple(norms[0])),
        '', '## Thermal statistics',md(therm,tuple(therm[0])),
        '', '## Scope and accounting',
        'HBM external storage is sufficiently provisioned, without inventing a physical TB capacity. The frozen persistent-state allocator and optimistic max(local, external) overlap are retained. HBM access energy is 1.9955 pJ/bit for reads+writes; Grace+C2C is 5.3 pJ/bit; C2C is 416.34 GB/s. HBM and Grace static memory power stay zero under the requested system-energy boundary. GPU and all M3D coefficients are unchanged, including the existing 0.1 W/slab active-NMP FEOL budget.',
        'HBM local and external service can overlap. Their aggregate delivered traffic is therefore not bounded by the local HBM bandwidth alone. In 70B/LC64K/B1 this retained optimistic overlap makes HBM nominal throughput exceed M3D-GPU; it is not an HBM-local paired case. The workload GPU power proxy accounts for the delivered traffic. '
        f"Its thermal RHS has {overlap_example['GPU_power_W']:.4f} W GPU, {overlap_example['memory_power_W']:.4f} W local memory, and {overlap_example['excluded_external_power_W']:.4f} W excluded external power, yielding {overlap_example['Tmax_C']:.4f} C at GPU FEOL. "
        'A bandwidth-characterization cap is not a guarantee that every E2E workload power map stays below 85 C.',
        'Energy terms in energy_closure_audit.csv are disjoint. Memory includes array/peripheral/MIV/local routes and the existing FEOL budget. For GPU-only execution its active memory interface is included in local memory, with all NMP columns zero. For NMP paths, Fabric/NoC include their actual wire and router events; interface includes the shared memory-die interface and edge-port routes over the E2E window. Every component sum closes to E2E energy. Generated tokens are B*G.',
        'User clarification explicitly selected per-die BEOL-uniform memory power, retaining the GPU FEOL source. This is not tile-resolved FEOL temperature. Uniform/CPA use their own actual per-die events. Old physical Decode latencies are never reexecuted: resident plans are restored, deterministic CPA traces are checked, and memory events are integrated over the exact atom birth contexts. All integrated global events and Prefill energies are checked against saved results.',
        'In the thermal table, memory_power_W includes every memory-die component. NMP_power_W is the MAC+SRAM+Fabric+NoC+reduction subset, not an additional source; the existing FEOL budget and shared interface remain included in memory_power_W.',
        'CPA still minimizes the modeled critical path. Faster execution can increase E2E-average power and Tmax even when total joules fall. Temperature and energy were not used to retune placement or select a different execution path.',
        'M3D operator signature must match the existing cached geometry/mesh/material/BC. HBM original cache was absent (including no_nmp_geometry_sensitivity_v2); the user authorized one identical rebuild. No dense factorization, mesh alteration, solver tolerance change, or transient model is introduced. FP64 GPU-PCG/Jacobi, true KCL residual and no full-vector D2H during iteration are retained.',
        'HBM package temperature excludes external Grace and off-package link energy. The full system-energy denominator still includes both. No separate GPU-side C2C thermal mapper exists, so none is invented.',
        '', '## Execution verdict',f'CPA faster than GPU: {wins}/18; GPU faster: {18-wins-ties}/18; ties: {ties}.',
        'Within the formal long-context benchmark, the NMP+CPA path is consistently selected; GPU fallback remains only a short-context boundary mechanism.' if wins==18 else 'Exceptions are retained in the raw candidate results; no model or selector is tuned.',
        '', '## Thermal closure verdict',
        'All nominal rows have finite workload-specific Tmax. Any >85 C row is explicitly THERMAL_CLOSURE_REQUIRED; its throughput/energy remain nominal and must not be labelled thermally feasible. There is no existing f_NMP closure mechanism, and no DVFS model has been added.',
        f"All paths thermal-feasible: {manifest['closure']['all_paths_thermal_feasible']}. See per-path thermal statuses in canonical data.",
        '', '## Reuse and new work',
        '12 existing points reused; 24 NMP case checkpoints / 9216 physical Decode steps reused. LC64K adds 6 points, 12 NMP case checkpoints / 6144 Decode steps, plus GPU memory-event observation checkpoints. No partial Decode extrapolation. Raw checkpoints and placement caches remain local.',
        md(execution,('model','context_label','B','path','Decode_steps','elapsed_including_placement_s')),
        '', 'Tests and final Git verification are recorded after the complete suite.']
    (OUT/'formal_long_context_report.md').write_text('\n\n'.join(report)+'\n',encoding='utf-8')
    print(json.dumps(manifest['closure']),flush=True)


if __name__=='__main__':main()
