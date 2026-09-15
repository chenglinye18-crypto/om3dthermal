"""Actual port-balanced GPU Decode temperatures on the existing thermal setup."""
import argparse
import concurrent.futures
import json
import subprocess
import sys
import time
import numpy as np
import formal_long_context_v2_support as f
from diagnose_m3d_gpu_port_balanced import read

def stable_source_power(mapper,gpu_W,die_W):
    """Same volume-uniform map, using pairwise sums instead of serial bincount.

    Serial accumulation across many GPU cells can exceed the existing 1e-12
    conservation tolerance by rounding alone. Keep that tolerance unchanged.
    """
    ids=mapper.source[mapper.active]
    if not hasattr(mapper,'stable_groups'):
        mapper.stable_groups=[np.flatnonzero(ids==i) for i in range(319)]
        mapper.stable_volumes=np.array([mapper.volume[g].sum() for g in mapper.stable_groups])
    values=np.r_[gpu_W,die_W]
    active=values[ids]*mapper.volume/mapper.stable_volumes[ids]
    mapped=np.array([active[g].sum() for g in mapper.stable_groups])
    np.testing.assert_allclose(mapped,values,rtol=1e-12,atol=1e-12)
    result=np.zeros(len(mapper.source));result[mapper.active]=active
    return result

def compose():
    from diagnose_m3d_gpu_port_balanced import loadcsv
    temperatures={}
    ledger=read(f.OUT/'candidate_ledgers.json')
    for m,c,b in f.points():
        key=f'{m}_{c}_B{b}_M3D_GPU';row=read(f.OUT/'thermal_rows_decode'/f'{key}.json')
        temperatures[m,c,b]=row
        p=f.OUT/'candidates'/f'{key}.json';r=read(p);r['thermal_result']=row;f.save(p,r)
        ledger[key]['thermal_result']=row
    f.save(f.OUT/'candidate_ledgers.json',ledger)
    rows=loadcsv(f.OUT/'final_e2e_metrics.csv')
    for r in rows:
        if r['path']!='M3D_GPU':continue
        t=temperatures[r['model'],r['context'],int(r['batch'])]
        r.update({k:t[k] for k in ('Tmax_C','thermal_metric_type','thermal_phase','thermal_status')})
        r['thermal_provenance']=t['provenance']
    f.csv_out('final_e2e_metrics.csv',rows)
    rows=loadcsv(f.OUT/'thermal_summary.csv')
    for r in rows:
        if r['path']=='M3D_GPU':r.update(temperatures[r['model'],r['context'],int(r['batch'])])
    f.csv_out('thermal_summary.csv',rows)
    rows=loadcsv(f.OUT/'four_path_Tmax.csv')
    for r in rows:r['M3D_GPU']=temperatures[r['model'],r['context'],int(r['batch'])]['Tmax_C']
    f.csv_out('four_path_Tmax.csv',rows)
    f.csv_out('m3d_gpu_actual_temperatures.csv',list(temperatures.values()))
    p=f.OUT/'formal_long_context_v2_manifest.json';manifest=read(p)
    manifest['M3D_GPU_thermal_update']=read(f.OUT/'m3d_gpu_thermal_update.json')
    manifest['M3D_thermal_resolved']=True
    manifest['M3D_thermal_cap_reswept']=False
    manifest['thermal_semantics']='HBM: 85C design point; M3D_GPU and NMP: actual Decode steady-state, die-grouped BEOL-uniform mapping'
    f.save(p,manifest)
    p=f.OUT/'formal_long_context_v2_report.md';report=p.read_text(encoding='utf-8')
    report=report.replace('GPU temperatures remain 85 C THERMAL_CLOSED_DESIGN_POINT, not new workload solves. NMP temperatures remain their unchanged Decode-only steady-state results. No thermal operator was loaded or thermal sweep rerun.',
        'HBM temperature remains the 85 C design point. M3D_GPU temperatures now use actual Decode steady-state solves on the existing operator, with port-balanced per-die power. NMP temperatures remain unchanged. No bandwidth thermal sweep was rerun.')
    report=report.replace('## Validation and plots','## Prior performance-refresh validation and plots')
    marker='\n## Actual M3D-GPU Decode temperatures\n'
    report=report.split(marker)[0]+marker+'\n| Model | Context | B | Tmax C |\n|---|---|---:|---:|\n'
    report+='\n'.join(f"| {m} | {c} | {b} | {t['Tmax_C']:.4f} |" for (m,c,b),t in temperatures.items())
    p.write_text(report+'\n\nExisting FP64 GPU-PCG setup reused for all 18 RHS solves. No pytest or additional benchmark audit was run for this temperature update.\n',encoding='utf-8')
    print('UPDATED 18 M3D-GPU TEMPERATURES',flush=True)

def project(index):
    from long_context_spatial_events import project_decode
    from om3dthermal.placement.nmp_load_balance import PhysicalResidentPlacement
    from om3dthermal.architecture.feol_floorplan import resolve_feol_floorplan
    from om3dthermal.power.feol_energy import FEOLEnergyModel
    m,c,b=f.points()[index];key=f'{m}_{c}_B{b}_M3D_GPU'
    target=f.OUT/'spatial_decode'/f'{key}.json'
    if target.exists():return
    r=read(f.OUT/'candidates'/f'{key}.json');s,e=r['summary'],r['energy']
    w,cw,_=f.legacy.inputs(m,c,b,f.ROOT)
    p=PhysicalResidentPlacement(w,resolve_feol_floorplan(f.ROOT),'GPU_PORT_BALANCED')
    slab=project_decode(p,cw,nmp=False)
    model=FEOLEnergyModel(p.floorplan,f.setup(f.ROOT)[2]);die=[]
    for i in range(p.dies):
        comp=model.account({k:v[i].tolist() for k,v in slab.items()},1,phase='decode',policy='NO_NMP')['components']
        die.append(sum(v for k,v in comp.items() if not k.startswith('gpu_')))
    gpu=e['decode_gpu_dynamic_J']+e['decode_gpu_static_J'];duration=s['decode_s']
    f.save(target,dict(model=m,context=c,batch=b,path='M3D_GPU',placement_policy='GPU_PORT_BALANCED',
        phase='DECODE',prefill_energy_included=False,decode_s=duration,decode_J=e['decode_J'],GPU_decode_J=gpu,
        die_decode_J=die,GPU_power_W=gpu/duration,die_power_W=[x/duration for x in die]))
    print('POWER READY',m,c,b,flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--project',type=int);parser.add_argument('--compose-only',action='store_true');args=parser.parse_args()
    if args.compose_only:return compose()
    if args.project is not None:return project(args.project)
    if all((f.OUT/'thermal_rows_decode'/f'{m}_{c}_B{b}_M3D_GPU.json').exists() for m,c,b in f.points()):return compose()
    def child(i):
        subprocess.run([sys.executable,__file__,'--project',str(i)],check=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(child,range(18)))
    from thermal_formal_long_context import load_setup_cache,GPUPCGOperator,require_cupy,SlabPowerMapper,solve_pcg_gpu,hot
    signature=read(f.OUT/'thermal_operator_audit.json')['physical_signature']
    cache=f.ROOT/f.CONFIG['thermal']['operator_cache']
    print('LOAD EXISTING THERMAL SETUP',flush=True)
    setup,load_s,status=load_setup_cache(cache,signature)
    if setup is None:raise RuntimeError('Existing M3D thermal setup unavailable')
    gpu=GPUPCGOperator.from_cpu(setup.operator_template,require_cupy());mapper=SlabPowerMapper(setup.cells)
    print('THERMAL SETUP READY',load_s,flush=True)
    for m,c,b in f.points():
        key=f'{m}_{c}_B{b}_M3D_GPU';target=f.OUT/'thermal_rows_decode'/f'{key}.json'
        if target.exists():continue
        x=read(f.OUT/'spatial_decode'/f'{key}.json')
        power=stable_source_power(mapper,x['GPU_power_W'],x['die_power_W'])
        operator=setup.operator_template.with_power(power)
        result=solve_pcg_gpu(operator,np.full(operator.cell_count,293.15),setup.boundary_table,
            relative_residual_tolerance=1e-3,max_temperature_update_tolerance=1e-2,
            max_iterations=100000,check_interval=10,gpu_operator=gpu)
        if not result.converged:raise RuntimeError(f'{key}: thermal solve did not converge')
        row=dict(model=m,context=c,batch=b,path='M3D_GPU',thermal_metric_type='WORKLOAD_SPECIFIC_STEADY_STATE',
            thermal_phase='DECODE',GPU_power_W=x['GPU_power_W'],memory_power_W=sum(x['die_power_W']),
            package_power_W=float(power.sum()),energy_denominator='DECODE_ONLY',decode_J=x['decode_J'],decode_s=x['decode_s'],
            operator_reused=True,physical_signature=signature,
            provenance='GPU_PORT_BALANCED_DECODE__DIE_BEOL_UNIFORM__ACTUAL_STEADY_STATE_SOLVE',**hot(result,setup))
        f.save(target,row)
        print('THERMAL',m,c,b,row['Tmax_C'],flush=True)
    f.save(f.OUT/'m3d_gpu_thermal_update.json',dict(cases=18,operator_reused=True,operator_builds=0,
        cache=str(cache),cache_load_s=load_s,phase='DECODE',pytest_run=False))
    compose()

if __name__=='__main__':main()
