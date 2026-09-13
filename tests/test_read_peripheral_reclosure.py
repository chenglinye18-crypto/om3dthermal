"""Read-side replacement, unchanged events, and power-independent closure."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from om3dthermal.architecture.feol_floorplan import resolve_feol_floorplan
from om3dthermal.platform import load_platform_spec_file
from om3dthermal.power.feol_energy import FEOLEnergyModel, memory_events
from om3dthermal.thermal_sensitivity import (
    SENSITIVITY_ARCHITECTURES, prepare_case, no_nmp_read_energy, crossing_85,
)
from om3dthermal.bandwidth_thermal_sweep import _point_simulation
from om3dthermal.thermal.setup_cache import build_scene_and_signature

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def model():
    return FEOLEnergyModel(resolve_feol_floorplan(ROOT),load_platform_spec_file(
        ROOT/'configs/platform/gpu_package_h200_reference.yaml'))


@pytest.mark.parametrize('payload_bytes',(1,32,33,64))
def test_read_payload_budget_and_no_duplicate_selector(model,payload_bytes):
    b=np.zeros((2,8),dtype=np.int64);b[0,3]=payload_bytes
    e=memory_events(b,write=False);before=deepcopy(e)
    c=model.account(e,1,phase='decode',policy='NO_NMP')['components']
    assert c['read_peripheral_J']==pytest.approx(payload_bytes*8*.04e-12,rel=1e-15)
    if payload_bytes==32:assert c['read_peripheral_J']==pytest.approx(10.24e-12)
    assert 'sense_amplifier_J' not in c
    assert c['row_select_J']==c['column_select_J']==0
    assert e==before
    assert e['read_services']==(payload_bytes+31)//32
    assert e['sa_sensed_bits']==e['read_services']*256
    assert e['row_select_events']==e['column_select_events']==e['read_services']
    assert e['array_read_bits']==sum(e['miv_read_bits_by_layer'])==payload_bytes*8


def test_write_coefficients_and_physical_events_unchanged(model):
    b=np.zeros((2,8),dtype=np.int64);b[0,0]=33;b[1,7]=32
    e=memory_events(b,write=True)
    c=model.account(e,1,phase='decode',policy='NO_NMP')['components']
    assert e['write_services']==3 and e['sa_sensed_bits']==0
    assert e['write_driver_bits']==65*8
    assert c['read_peripheral_J']==0
    assert c['row_select_J']==c['column_select_J']==3*32e-12
    assert c['write_driver_J']==65*8*.1e-12
    assert c['array_write_J']==65*8*model.write_pj*1e-12


@pytest.mark.parametrize('index,slabs',((2,318),(3,106)))
def test_physical_read_stress_and_cache_signature(index,slabs):
    case,sim,memory,*_=prepare_case(ROOT,SENSITIVITY_ARCHITECTURES[index])
    audit=no_nmp_read_energy(ROOT,case,memory);e=audit['events'];p=audit['effective']
    assert audit['slabs']==slabs
    assert e['read_services']==slabs*70*8
    assert e['array_read_bits']==e['interface_bits']==slabs*70*8*256
    assert sum(e['miv_read_bits_by_layer'])==e['array_read_bits']
    for key in ('mac_operations','sram_read32_accesses','sram_write32_accesses',
                'router_bit_traversals','noc_link_bit_um','fp32_reduction_adds'):
        assert e[key]==0
    f=resolve_feol_floorplan(ROOT)
    assert e['sa_to_edge_bit_um']==pytest.approx(slabs*8*256*f.sa_edge_um.sum(),rel=1e-14)
    assert p['total_memory_side_pJ_per_bit']==sum(v for k,v in p.items() if not k.startswith('total_'))
    _,signature=build_scene_and_signature(sim)
    for coeff in (p['total_memory_side_pJ_per_bit'],p['total_memory_side_pJ_per_bit']+1):
        point,gpu,mem=_point_simulation(sim,'orthogonal_m3d_igzo',SimpleNamespace(E_access_total_pj_bit=coeff),4.8)
        assert build_scene_and_signature(point)[1]==signature
        assert sum(s.total_power for s in point.thermal_power_sources.sources)==pytest.approx(gpu+mem,rel=1e-14)
        assert gpu==pytest.approx(74+4.8*8*11.68)


def test_crossing_requires_bracket():
    rows=[dict(bandwidth_TBps=b,Tmax_C=25+10*b) for b in (0,4,8)]
    assert crossing_85(rows)==6
    with pytest.raises(ValueError):crossing_85(rows[:2])


def test_cached_and_fresh_power_solve_equivalent(tmp_path):
    from om3dthermal.config import load_config
    from om3dthermal.config import CellSizeConfig
    from om3dthermal.case_runner import run_steady_pipeline
    config=load_config(ROOT/'tests/fixtures/toy_1box.yaml')
    # Resolve gradients on a small mesh rather than the two-eigenvalue fixture.
    config=config.model_copy(update={'discretization':config.discretization.model_copy(update={
        'max_cell_size':CellSizeConfig(x=.0031,y=.0027,z=.0013)})})
    options=dict(backend='gpu_pcg',rtol=1e-8,max_delta_t_K=1e-8,max_iterations=10000,check_interval=1)
    first=run_steady_pipeline(config,setup_cache_path=tmp_path/'operator.pkl',**options)
    sources=config.thermal_power_sources
    changed=config.model_copy(update={'thermal_power_sources':sources.model_copy(update={
        'sources':[s.model_copy(update={'total_power':s.total_power*1.1}) for s in sources.sources]})})
    cached=run_steady_pipeline(changed,setup_cache_path=tmp_path/'operator.pkl',**options)
    fresh=run_steady_pipeline(changed,**options)
    assert cached.cache_status=='HIT' and cached.setup_build_seconds==0
    assert first.result.converged and cached.result.converged and fresh.result.converged
    assert cached.result.max_temperature_K==pytest.approx(fresh.result.max_temperature_K,abs=1e-10)
    assert cached.result.final_relative_residual==pytest.approx(fresh.result.final_relative_residual,abs=1e-12)
    from om3dthermal.thermal.gpu_pcg import GPUPCGOperator, require_cupy
    prepared=GPUPCGOperator.from_cpu(cached.reusable_setup.operator_template,require_cupy())
    reused=run_steady_pipeline(changed,reusable_setup=cached.reusable_setup,gpu_operator=prepared,**options)
    np.testing.assert_array_equal(reused.result.temperature_K,fresh.result.temperature_K)
    assert prepared.operator_h2d_copy_count==0
    with pytest.raises(ValueError,match='identical immutable'):
        run_steady_pipeline(changed,reusable_setup=fresh.reusable_setup,gpu_operator=prepared,**options)
