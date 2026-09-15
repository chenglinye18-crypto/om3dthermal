"""Report and preservation gate for the canonical formal-v2 GPU refresh."""
import math
import statistics
from collections import Counter
import formal_long_context_v2_support as f
from diagnose_m3d_gpu_port_balanced import read, sha, loadcsv, OUT as DIAG

def identity(r):return (r['model'],r['context'],int(r['batch']),r['path'])

def markdown(rows):
    keys=list(rows[0])
    return '\n'.join(['| '+' | '.join(keys)+' |','|'+'|'.join('---' for _ in keys)+'|',
        *['| '+' | '.join(f'{v:.6g}' if isinstance(v,float) else str(v) for v in r.values())+' |' for r in rows]])

def finalize_refresh(snapshot,rows,manifest):
    validation_file=f.OUT/'gpu_refresh_validation.json'
    checks=read(validation_file).get('refresh_checks') if validation_file.exists() else None
    assert Counter(r['path'] for r in rows)=={p:18 for p in f.PATHS}
    old={identity(r):r for r in snapshot['rows']};new={identity(r):r for r in rows}
    # All canonical fields, not just selected numeric metrics, remain unchanged.
    for key,before in old.items():
        if key[3] not in f.PATHS[2:]:
            assert any(float(before[k])!=new[key][k] for k in ('E2E_s','E2E_J'))
            continue
        after=new[key]
        for field,value in before.items():
            if isinstance(after[field],(float,int)) and not isinstance(after[field],bool):
                assert float(value)==after[field],(key,field,value,after[field])
            else:assert value==str(after[field]),(key,field)
    for name,digest in snapshot['files'].items():assert sha(f.ROOT/name)==digest,name
    preservation=dict(old_HEAD=snapshot['old_HEAD'],HBM_GPU='18/18 updated',M3D_GPU='18/18 updated',
        M3D_NMP_UNIFORM='18/18 unchanged',M3D_NMP_CPA='18/18 unchanged',
        protected_artifact_count=len(snapshot['files']),candidate_bytes='IDENTICAL',all_NMP_row_fields='IDENTICAL',status='PASS')
    if checks:preservation['refresh_checks']=checks
    f.save(validation_file,preservation)
    hbm=[];gpu=[];aggregate=[]
    diagnostic={(r['model'],r['context'],int(r['batch'])):r for r in loadcsv(DIAG/'comparison.csv')}
    for m,c,b in f.points():
        a=new[m,c,b,'HBM_GPU'];g=new[m,c,b,'M3D_GPU'];d=diagnostic[m,c,b]
        hbm.append(dict(model=m,context=c,batch=b,tokens_per_s=a['tokens_per_s'],tokens_per_J=a['tokens_per_J'],J_per_token=a['J_per_token']))
        gpu.append(dict(model=m,context=c,batch=b,old_formal_tokens_per_s=float(old[m,c,b,'M3D_GPU']['tokens_per_s']),
            new_tokens_per_s=g['tokens_per_s'],old_formal_bandwidth_TBps=3.3957835581187483,
            old_bandwidth_semantics='AGGREGATE_CLOSURE_NOT_MEASURED',old_uniform_achieved_TBps=float(d['old_achieved_BW_TBps']),
            new_achieved_TBps=float(d['new_achieved_BW_TBps']),new_utilization_pct=float(d['new_utilization_pct']),tokens_per_J=g['tokens_per_J']))
    for model in ['ALL',*f.MODELS]:
        for path in f.PATHS:
            selected=[r for r in rows if r['path']==path and (model=='ALL' or r['model']==model)]
            gm=statistics.geometric_mean
            aggregate.append(dict(model=model,path=path,cases=len(selected),aggregation='GEOMETRIC_MEAN',
                decode_tokens_per_s=gm(r['generated_tokens']/r['decode_s'] for r in selected),
                E2E_tokens_per_s=gm(r['tokens_per_s'] for r in selected),E2E_tokens_per_J=gm(r['tokens_per_J'] for r in selected),
                speedup_vs_HBM=gm(r['tokens_per_s']/new[r['model'],r['context'],r['batch'],'HBM_GPU']['tokens_per_s'] for r in selected)))
    f.csv_out('gpu_refresh_hbm_results.csv',hbm);f.csv_out('gpu_refresh_m3d_results.csv',gpu);f.csv_out('formal_aggregate_comparison.csv',aggregate)
    manifest.update(gpu_refresh=preservation,HBM_thermal_source='runs/no_nmp_geometry_sensitivity_v2/thermal_limits.csv',
        HBM_bandwidth_TBps=read(f.OUT/'candidates/Llama-3.1-8B_LC20K_B1_HBM_GPU.json')['summary']['effective_bandwidth_TBps'],
        HBM_read_energy_pj_per_bit=3.,HBM_write_energy_pj_per_bit=1.9955,
        M3D_GPU_placement='GPU_PORT_BALANCED',M3D_GPU_timing_source='m3d_gpu_port_balanced_diagnostic',
        GPU_Decode_steps_reused=576,M3D_thermal_resolved=False,
        GPU_event_method='EXACT_ATOM_BIRTH_CONTEXT_INTEGRATION_WITH_FROZEN_COEFFICIENTS',
        Prefill='M3D GPU diagnostic physical timing reused; NMP aggregate Prefill frozen',
        NMP_preservation='36/36 byte-identical candidates and unchanged canonical rows')
    if checks:manifest.update(refresh_checks=checks,pytest_run=True)
    actual_thermal=f.OUT/'m3d_gpu_thermal_update.json'
    if actual_thermal.exists():
        manifest['M3D_GPU_thermal_update']=read(actual_thermal)
        manifest['M3D_thermal_resolved']=True
        manifest['M3D_thermal_cap_reswept']=False
        manifest['thermal_semantics']='HBM: 85C design point; M3D_GPU and NMP: actual Decode steady-state'
    f.save(f.OUT/'formal_long_context_v2_manifest.json',manifest)
    report=['# Formal long-context v2 GPU refresh',
        '72 canonical rows; H=20K/64K/126K, P=128, G=32, B=1/8. Only HBM_GPU and M3D_GPU were refreshed. NMP results and physical artifacts are frozen.',
        'HBM: thermal cap 3.2973872924850354 -> 3.0865643306417803 TB/s from runs/no_nmp_geometry_sensitivity_v2/thermal_limits.csv. Read energy 1.9955 -> 3.00 pJ/bit. The existing write coefficient remains 1.9955 pJ/bit; the read-only rebaseline does not invent a write model. C2C and all logical traffic/compute/capacity remain unchanged. This is an optimistic full-utilization thermally closed design point.',
        'M3D_GPU: GPU_PORT_BALANCED physical operator-level data-chain and contention-aware external-port service; all 576 saved Decode steps reused. Exact atom-birth integration regenerates route-sensitive events without reexecuting timing. Frozen event coefficients account for real FEOL route changes and new static duration. No CPA or NMP MAC activity.',
        'Prefill scope: M3D_GPU uses the diagnostic\'s reused physical Prefill time, rather than the old formal aggregate Prefill time. Its Prefill energy is recomputed with port-balanced routing. The two NMP paths retain their frozen aggregate Prefill times and energy. Therefore this formal comparison contains that explicitly documented Prefill model difference.',
        'GPU temperatures remain 85 C THERMAL_CLOSED_DESIGN_POINT, not new workload solves. NMP temperatures remain their unchanged Decode-only steady-state results. No thermal operator was loaded or thermal sweep rerun.',
        'Old formal M3D bandwidth was an aggregate closure, not a measured achieved bandwidth. The table separately labels old Uniform achieved bandwidth. NMP absolute values are unchanged; normalized NMP bars necessarily change when their HBM denominator changes.',
        '## Updated HBM',markdown(hbm),'## M3D GPU comparison',markdown(gpu),
        '## Four-path aggregates', 'Geometric means across paired operating points (not aggregate serving throughput).',markdown(aggregate),
        '## Preservation',str(preservation)]
    report.extend(['## Reproduction',
        'In the om3dthermal Conda environment: python scripts/refresh_formal_gpu_results.py; python scripts/finalize_formal_long_context_v2.py; python scripts/plot_formal_long_context_v2.py. The refresh reuses completed candidates on subsequent invocations. No thermal sweep or NMP execution is dispatched.'])
    if checks:report.extend(['## Validation and plots',str(checks)])
    if actual_thermal.exists():
        report=[p.replace('GPU temperatures remain 85 C THERMAL_CLOSED_DESIGN_POINT, not new workload solves. NMP temperatures remain their unchanged Decode-only steady-state results. No thermal operator was loaded or thermal sweep rerun.',
            'HBM remains an 85 C design point. M3D_GPU and NMP use actual Decode steady-state temperatures; M3D_GPU reuses the existing operator with port-balanced die power. No thermal bandwidth sweep was rerun.') for p in report]
    (f.OUT/'formal_long_context_v2_report.md').write_text('\n\n'.join(report)+'\n',encoding='utf-8')
