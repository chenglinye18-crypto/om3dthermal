"""New workload RHS solves on the unchanged v2 geometry/operator cache."""
import json
import numpy as np
from run_formal_long_context_v3 import ROOT,OUT,CONFIG,POINTS,save,read


def main():
    from thermal_formal_long_context import load_setup_cache,GPUPCGOperator,require_cupy,SlabPowerMapper,solve_pcg_gpu,hot
    from thermal_m3d_gpu_formal_v2 import stable_source_power
    signature=json.loads((ROOT/'runs/formal_long_context_v2/thermal_operator_audit.json').read_text())['physical_signature']
    cache=ROOT/CONFIG['thermal']['operator_cache']
    print('LOAD EXISTING THERMAL OPERATOR',flush=True)
    setup,load_s,status=load_setup_cache(cache,signature)
    if setup is None:raise RuntimeError('Canonical geometry/operator cache unavailable')
    gpu=GPUPCGOperator.from_cpu(setup.operator_template,require_cupy())
    mapper=SlabPowerMapper(setup.cells)
    for m,c,b in POINTS:
        for path in CONFIG['paths'][1:]:
            key=f'{m}_{c}_B{b}_{path}'
            target=OUT/'thermal_rows'/f'{key}.json'
            if target.exists():continue
            r=read(OUT/'candidates'/f'{key}.json')
            power=stable_source_power(mapper,r['GPU_power_W'],r['die_power_W'])
            operator=setup.operator_template.with_power(power)
            result=solve_pcg_gpu(operator,np.full(operator.cell_count,293.15),setup.boundary_table,
                relative_residual_tolerance=1e-3,max_temperature_update_tolerance=1e-2,
                max_iterations=100000,check_interval=10,gpu_operator=gpu)
            if not result.converged:raise RuntimeError(f'{key}: thermal solve failed')
            row=dict(model=m,context=c,batch=b,path=path,thermal_metric='PEAK_DECODE_STEADY_STATE',
                GPU_power_W=r['GPU_power_W'],memory_power_W=sum(r['die_power_W']),
                mapping='DIE_GROUPED_BEOL_UNIFORM',operator_signature=signature,**hot(result,setup))
            row['thermal_feasible']=row['Tmax_C']<=85
            save(target,row)
            print('THERMAL',key,row['Tmax_C'],flush=True)
    save(OUT/'thermal_operator_reuse.json',dict(cache=str(cache),signature=signature,load_s=load_s,
         new_operator_builds=0,new_workload_solutions=54))


if __name__=='__main__':main()
