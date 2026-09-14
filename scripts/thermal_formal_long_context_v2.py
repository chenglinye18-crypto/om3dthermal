"""36 Decode-only NMP RHS solves on one unchanged M3D operator."""
import argparse
import json
import time
import numpy as np
import formal_long_context_v2_support as f
from thermal_formal_long_context import (ARCHITECTURES,_base_case,build_scene_and_signature,
    load_setup_cache,GPUPCGOperator,require_cupy,SlabPowerMapper,solve_pcg_gpu,hot)


def main(wait=False,repair_only=False):
    start=time.time()
    _,simulation,_=_base_case(f.ROOT,ARCHITECTURES[1])
    _,signature=build_scene_and_signature(simulation)
    reference=json.loads((f.ROOT/'runs/no_nmp_geometry_sensitivity/orthogonal_m3d_igzo.json').read_text())
    assert signature==reference['physical_signature']
    cache=f.ROOT/f.CONFIG['thermal']['operator_cache']
    stamp=(cache.stat().st_size,cache.stat().st_mtime_ns)
    setup,load_s,status=load_setup_cache(cache,signature)
    assert setup is not None,'Existing workload-independent M3D operator must load; no rebuild permitted'
    gpu=GPUPCGOperator.from_cpu(setup.operator_template,require_cupy());mapper=SlabPowerMapper(setup.cells)
    points=[('Llama-3.1-405B','LC64K',8)] if repair_only else f.points()
    for m,c,b in points:
        for path in f.PATHS[2:]:
            key=f'{m}_{c}_B{b}_{path}';target=f.OUT/'thermal_rows_decode'/f'{key}.json'
            if target.exists():continue
            candidate_file=f.OUT/'candidates'/f'{key}.json'
            while wait and (not candidate_file.exists() or (repair_only and json.loads(candidate_file.read_text())['summary']['status']!='EVALUATED')):time.sleep(2)
            candidate=json.loads(candidate_file.read_text())
            if candidate['summary']['status']!='EVALUATED':continue
            x=json.loads((f.OUT/'spatial_decode'/f'{key}.json').read_text())
            assert x['phase']=='DECODE' and not x['prefill_energy_included']
            power=mapper.power(x['GPU_power_W'],x['die_power_W'])
            np.testing.assert_allclose(power.sum(),x['decode_J']/x['decode_s'],rtol=1e-12,atol=1e-8)
            operator=setup.operator_template.with_power(power)
            result=solve_pcg_gpu(operator,np.full(operator.cell_count,293.15),setup.boundary_table,
                relative_residual_tolerance=1e-3,max_temperature_update_tolerance=1e-2,
                max_iterations=100000,check_interval=10,gpu_operator=gpu)
            assert result.converged and result.solver_info['dtype']=='float64'
            assert result.solver_info['full_vector_d2h_during_iteration']==0
            row=dict(model=m,context=c,batch=b,path=path,thermal_metric_type=f.CONFIG['thermal']['NMP_metric'],
                thermal_phase=f.CONFIG['thermal']['NMP_phase'],GPU_power_W=x['GPU_power_W'],
                memory_power_W=sum(x['die_power_W']),package_power_W=float(power.sum()),
                energy_denominator='DECODE_ONLY',decode_J=x['decode_J'],decode_s=x['decode_s'],
                power_conservation='PASS',operator_reused=True,physical_signature=signature,**hot(result,setup))
            f.save(target,row);print('THERMAL',key,row['Tmax_C'],flush=True)
    assert stamp==(cache.stat().st_size,cache.stat().st_mtime_ns)
    f.save(f.OUT/'thermal_operator_audit.json',dict(operator_reused=True,operator_builds=0,
        physical_signature=signature,cache_status=status,cache_load_s=load_s,cache=str(cache),
        elapsed_s=time.time()-start,cache_file_unchanged=True,wait_for_candidates=wait,repair_only=repair_only))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--wait-for-candidates',action='store_true');parser.add_argument('--repair-only',action='store_true')
    args=parser.parse_args();main(args.wait_for_candidates,args.repair_only)
