"""User-specified die-grouped BEOL-uniform E2E steady-state thermal RHS."""
import argparse
import gc
import json
import time
import numpy as np
from formal_long_context_support import *
from om3dthermal.bandwidth_thermal_sweep import _base_case,ARCHITECTURES
from om3dthermal.config import ThermalPowerSourcesConfig
from om3dthermal.thermal.setup_cache import build_scene_and_signature,load_setup_cache
from om3dthermal.thermal.placement_diagnostic import SlabPowerMapper
from om3dthermal.thermal.gpu_pcg import GPUPCGOperator,require_cupy,solve_pcg_gpu
from om3dthermal.thermal.power import map_power_sources
from om3dthermal.case_runner import run_steady_pipeline

CONVENTION='USER_SPECIFIED_DIE_RESOLVED_BEOL_UNIFORM__E2E_EQUIVALENT_STEADY_STATE'


def hot(result,setup):
    cell=setup.cells[int(result.temperature_K.argmax())]
    return dict(Tmax_C=result.max_temperature_K-273.15,hotspot_component=f'{cell.component}/{cell.material}',
        hotspot_x=(cell.x0+cell.x1)/2,hotspot_y=(cell.y0+cell.y1)/2,hotspot_z=(cell.z0+cell.z1)/2,
        hotspot_die=cell.tags.get('die_index'),thermal_solve_s=result.solve_seconds,
        iterations=result.iterations,relative_residual=result.final_relative_residual,
        max_temperature_update_K=result.max_temperature_update,
        thermal_status='THERMAL_FEASIBLE' if result.max_temperature_K<=358.15 else 'THERMAL_CLOSURE_REQUIRED')


def run(family):
    spec=ARCHITECTURES[0 if family=='HBM' else 1]
    _,simulation,memory=_base_case(ROOT,spec)
    _,signature=build_scene_and_signature(simulation)
    cache=(OUT/'thermal_cache/conventional_hbm_2x1.pkl' if family=='HBM' else ROOT/'runs/placement_ablation_b1_b8_v1/thermal_setup.pkl')
    history=json.loads((ROOT/'runs/no_nmp_geometry_sensitivity'/f'{spec.architecture}.json').read_text())
    assert signature==history['physical_signature'],'Frozen thermal physics changed'
    started=time.perf_counter();setup,load_s,status=load_setup_cache(cache,signature)
    if family=='M3D' and setup is None:raise RuntimeError('Existing M3D operator must be reused')
    audit=dict(family=family,cache=str(cache),physical_signature=signature,cache_status=status,
        setup_reused=setup is not None,setup_load_s=load_s,operator_builds=0,
        HBM_rebuild_authorization='USER_CONFIRMED_MISSING_CACHE_REBUILD_ONCE',convention=CONVENTION)
    save(OUT/'thermal_setup_audit'/f'{family}.json',audit)
    gpu=None;mapper=None
    if setup is not None:
        gpu=GPUPCGOperator.from_cpu(setup.operator_template,require_cupy())
        if family=='M3D':mapper=SlabPowerMapper(setup.cells)
    for m,c,b in points():
        for path in (('HBM_GPU',) if family=='HBM' else PATHS[1:]):
            target=OUT/'thermal_rows'/f'{m}_{c}_B{b}_{path}.json'
            if target.exists():continue
            if family=='HBM':
                r=hbm(m,c,b);save(OUT/'candidates'/f'{m}_{c}_B{b}_{path}.json',r)
                s,e=r['summary'],r['energy'];duration=s['E2E_s']
                GPU=e['GPU_energy_J']/duration;mem=e['local_memory_energy_J']/duration;NMP=0.;external=e['external_memory_energy_J']/duration
                fraction=memory.E_base_route_pj_bit/memory.E_access_total_pj_bit
                powers=dict(gpu=GPU,dram_group_0=mem*(1-fraction)/2,dram_group_1=mem*(1-fraction)/2,
                    base_route_group_0=mem*fraction/2,base_route_group_1=mem*fraction/2)
                sources=[x.model_copy(update={'total_power':powers[x.name]}) for x in simulation.thermal_power_sources.sources]
                point=simulation.model_copy(update={'thermal_power_sources':ThermalPowerSourcesConfig(sources=sources)})
                if setup is None:
                    print('REBUILD_MISSING_HBM_OPERATOR_ONCE',flush=True)
                    pipeline=run_steady_pipeline(point,backend='gpu_pcg',setup_cache_path=cache,
                        rtol=1e-3,max_delta_t_K=1e-2,max_iterations=100000,check_interval=10,initial_temperature_K=293.15)
                    setup=pipeline.reusable_setup;result=pipeline.result;power=pipeline.power.power_W
                    gpu=GPUPCGOperator.from_cpu(setup.operator_template,require_cupy())
                    audit.update(operator_builds=1,setup_build_s=pipeline.setup_build_seconds)
                    save(OUT/'thermal_setup_audit'/f'{family}.json',audit)
                    reused=False
                else:
                    power=map_power_sources(setup.cells,point.thermal_power_sources).power_W;result=None;reused=True
                total=e['E2E_J']/duration
            else:
                file=OUT/'spatial'/f'{m}_{c}_B{b}_{path}.json'
                if not file.exists():continue
                x=json.loads(file.read_text());duration=x['E2E_s'];GPU=x['GPU_power_W'];mem=sum(x['die_power_W']);external=0.
                NMP=sum(x['energy_audit'][k] for k in ('NMP_MAC_energy_J','NMP_SRAM_energy_J','NMP_fabric_energy_J','NMP_NoC_energy_J','NMP_reduction_energy_J'))/duration
                power=mapper.power(GPU,x['die_power_W']);result=None;reused=True;total=x['E2E_J']/duration
            expected=total-external
            np.testing.assert_allclose(power.sum(),expected,rtol=1e-12,atol=1e-9)
            if result is None:
                operator=setup.operator_template.with_power(power)
                result=solve_pcg_gpu(operator,np.full(operator.cell_count,293.15),setup.boundary_table,
                    relative_residual_tolerance=1e-3,max_temperature_update_tolerance=1e-2,
                    max_iterations=100000,check_interval=10,gpu_operator=gpu)
            assert result.converged and result.solver_info['dtype']=='float64'
            assert result.solver_info['full_vector_d2h_during_iteration']==0
            row=dict(**identity(m,c,b),path=path,E2E_time_s=duration,total_average_power_W=total,GPU_power_W=GPU,
                memory_power_W=mem,NMP_power_W=NMP,mapped_package_power_W=float(power.sum()),excluded_external_power_W=external,
                thermal_operator_reused=reused,thermal_convention=CONVENTION,physical_signature=signature,**hot(result,setup))
            row['power_columns']='memory_power_W includes all memory-die components; NMP_power_W is a subset, not additive'
            row['thermal_boundary']='GPU + local HBM; external Grace and off-package link energy excluded; no separate GPU-side C2C mapper' if family=='HBM' else 'GPU + 318 memory slabs'
            save(target,row);print('THERMAL',m,c,b,path,row['Tmax_C'],row['thermal_status'],flush=True)
    del setup,gpu,mapper;gc.collect()
    require_cupy().get_default_memory_pool().free_all_blocks()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--family',choices=('HBM','M3D'),required=True)
    run(p.parse_args().family)
