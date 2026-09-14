"""LC scope, HBM accounting, spatial projection and canonical artifact gates."""
import json
import math
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from formal_long_context_support import *
from long_context_spatial_events import integrated_layers
from om3dthermal.placement.nmp_load_balance import ResidentOperator
from om3dthermal.placement.critical_path import MigratedOperator


def test_formal_long_context_matrix_has_18_cases():
    assert len(points())==len(set(points()))==18
    assert set(CONTEXTS)=={'LC20K','LC64K','LC126K'}
    assert CONTEXTS['LC20K']==('W2',20000,512,256)
    assert CONTEXTS['LC126K']==('W3',126000,512,512)


def test_lc64k_definition():
    assert CONTEXTS['LC64K']==('W64',64000,512,512)
    w,cw,_=legacy.inputs(MODELS[0],'W64',8,ROOT)
    assert list(cw.contexts)==list(range(64512,65024))
    assert w.context_length==65024


def test_w1_excluded_from_formal_primary():
    with pytest.raises(ValueError):identity(MODELS[0],'W1',1)
    from om3dthermal.serving.workload_matrix import setup as public
    assert 'W64' not in public(ROOT)[0]['workloads']
    assert public(ROOT)[0]['grace_capacity_bytes']==480000000000


def test_b32_excluded_from_formal_primary():
    with pytest.raises(ValueError):identity(MODELS[0],'LC20K',32)


def test_hbm_external_capacity_sufficient():
    for m,c,b in points():
        row=cap(m,c,b,'HBM_GPU')
        assert row['status']=='EVALUATED'
        assert row['external_capacity_mode']==EXTERNAL_MODE
        assert 'Grace_capacity_GB' not in row


@pytest.mark.parametrize('c',CONTEXTS)
@pytest.mark.parametrize('b',(1,8))
def test_hbm_405b_is_evaluable(c,b):
    r=hbm(MODELS[-1],c,b)
    assert r['summary']['status']=='EVALUATED'
    assert r['summary']['external_capacity_mode']==EXTERNAL_MODE
    assert 'Grace_capacity_GB' not in r['summary']
    assert math.isfinite(r['summary']['E2E_tok_s']) and r['summary']['E2E_tok_s']>0
    assert r['energy']['E2E_tokens_per_J']>0


def test_hbm_access_energy_read_write():
    r=hbm(MODELS[0],'LC20K',1);e=r['energy'];t=r['traffic']
    nread=(t['HBM_read_GB']+t['prefill_HBM_read_GB'])*1e9
    nwrite=(t['HBM_write_GB']+t['prefill_HBM_write_GB'])*1e9
    assert nwrite>0
    assert e['hbm_access_energy_pj_per_bit']==1.9955
    assert e['local_memory_energy_J']==pytest.approx((nread+nwrite)*8*1.9955e-12,rel=1e-12)
    assert e['external_memory_energy_J']==pytest.approx(t['total_C2C_GB']*1e9*8*5.3e-12,rel=1e-12)
    assert e['energy_status']=='HBM_ENERGY_CLOSED'
    assert e['E2E_J']==e['GPU_energy_J']+e['local_memory_energy_J']+e['external_memory_energy_J']


@pytest.mark.parametrize('m',MODELS[:2])
@pytest.mark.parametrize('c',('LC20K','LC126K'))
@pytest.mark.parametrize('b',(1,8))
def test_existing_hbm_performance_unchanged(m,c,b):
    old=next(r for r in read_csv(OLD/'primary_summary.csv') if r['model']==m and r['workload_id']==CONTEXTS[c][0] and int(r['batch_size'])==b and r['system']=='HBM_GPU')
    assert hbm(m,c,b)['summary']['E2E_tok_s']==float(old['E2E_tok_s'])


@pytest.mark.parametrize('migrated',(False,True))
def test_exact_spatial_atom_integration_not_decode_extrapolation(migrated):
    p=ResidentOperator(None,256,500,0,np.arange(140,dtype=np.int32),np.arange(140,dtype=np.uint8)%8,
                       np.zeros(140,dtype=np.int32),2)
    if migrated:p=MigratedOperator.build(p,[0],[2],[1])
    expected=sum(p.layer_bytes(c*2) for c in range(160,171))
    np.testing.assert_array_equal(integrated_layers(p,160,170,2),expected)


def complete():
    file=OUT/'final_e2e_metrics.csv'
    if not file.exists():pytest.skip('Canonical artifact gates await completed authorized LC runs')
    return read_csv(file)


def test_four_path_performance_closed():
    rows=complete();assert len(rows)==18
    ledgers=json.loads((OUT/'candidate_ledgers.json').read_text())
    for r in rows:
        identity(r['model'],r['context_label'],int(r['B']))
        for p in PATHS:assert math.isfinite(float(r[p+'_tok_s'])) and float(r[p+'_tok_s'])>0
        prefill=[ledgers[f"{r['model']}_{r['context_label']}_B{r['B']}_{p}"]['summary']['prefill_s'] for p in PATHS[1:]]
        assert prefill[0]==prefill[1]==prefill[2]


def test_four_path_energy_closed():
    rows=complete()
    for r in rows:
        for p in PATHS:assert math.isfinite(float(r[p+'_tok_J'])) and float(r[p+'_tok_J'])>0
    rows=read_csv(OUT/'energy_closure_audit.csv');assert len(rows)==72
    for r in rows:
        components=[float(v) for k,v in r.items() if k.endswith('_energy_J') and k!='total_E2E_energy_J']
        assert sum(components)==pytest.approx(float(r['total_E2E_energy_J']),rel=1e-12)
        if r['path'] in ('HBM_GPU','M3D_GPU'):
            assert all(float(v)==0 for k,v in r.items() if k.startswith('NMP_') and k.endswith('_energy_J'))


def test_existing_nmp_checkpoint_regression():
    rows=complete()
    if not (OLD/'checkpoints').exists():pytest.skip('Local physical checkpoints are not distributed')
    for r in rows:
        if r['context_label']=='LC64K':continue
        for p in PATHS[1:]:
            old=existing(r['model'],r['context_label'],int(r['B']),p)
            assert float(r[p+'_tok_s'])==old['summary']['E2E_tok_s']
            assert float(r[p+'_tok_J'])==old['energy']['E2E_tokens_per_J']


def test_workload_thermal_power_conservation():
    complete()
    rows=read_csv(OUT/'four_path_Tmax.csv');assert len(rows)==72
    for r in rows:
        key=f"{r['model']}_{r['context_label']}_B{r['B']}_{r['path']}"
        candidate=json.loads((OUT/'candidate_ledgers.json').read_text())[key]
        assert float(r['total_average_power_W'])==pytest.approx(candidate['energy']['E2E_J']/candidate['summary']['E2E_s'],rel=1e-12)
        assert float(r['E2E_time_s'])==candidate['summary']['E2E_s']
        if r['path']!='HBM_GPU':
            spatial=json.loads((OUT/'spatial'/f'{key}.json').read_text())
            assert len(spatial['die_energy_J'])==318
            assert min(spatial['die_energy_J'])>=0
            assert sum(spatial['die_energy_J'])+spatial['GPU_energy_J']==pytest.approx(candidate['energy']['E2E_J'],rel=1e-12)
            np.testing.assert_allclose(spatial['die_power_W'],np.asarray(spatial['die_energy_J'])/float(r['E2E_time_s']),rtol=1e-12)
            assert float(r['memory_power_W'])==pytest.approx(sum(spatial['die_power_W']),rel=1e-12)
        assert math.isfinite(float(r['Tmax_C']))
        assert float(r['relative_residual'])<=1e-3
        assert float(r['max_temperature_update_K'])<=1e-2
        assert float(r['mapped_package_power_W'])+float(r['excluded_external_power_W'])==pytest.approx(float(r['total_average_power_W']),rel=1e-12)
        assert r['thermal_status']=='THERMAL_FEASIBLE' if float(r['Tmax_C'])<=85 else r['thermal_status']=='THERMAL_CLOSURE_REQUIRED'


def test_lc64k_nmp_event_conservation():
    complete()
    if not (OUT/'checkpoints').exists():pytest.skip('Local physical checkpoints are not distributed')
    from om3dthermal.power.feol_energy import sum_events
    count=0
    for m in MODELS:
        for b in (1,8):
            for p in PATHS[2:]:
                key=f'{m}_LC64K_B{b}_{p}'
                candidate=json.loads((OUT/'candidate_ledgers.json').read_text())[key]
                checkpoint=ROOT/candidate['source_checkpoint']
                steps=[json.loads(line) for line in checkpoint.with_suffix('.jsonl').read_text().splitlines()]
                assert [s['context'] for s in steps]==list(range(64512,65024))
                events=sum_events(s['energy_events'] for s in steps)
                for field,value in events.items():
                    np.testing.assert_allclose(value,candidate['events'][field],rtol=1e-12,atol=1e-7)
                assert sum(s['latency_s'] for s in steps)==candidate['summary']['decode_s']
                spatial=json.loads((OUT/'spatial'/f'{key}.json').read_text())
                assert spatial['event_conservation']=='PASS'
                assert len(spatial['die_energy_J'])==318
                count+=len(steps)
    assert count==6144


def test_thermal_operator_cache_reuse():
    complete()
    rows=read_csv(OUT/'four_path_Tmax.csv')
    assert all(r['thermal_operator_reused']=='True' for r in rows if r['path']!='HBM_GPU')
    assert sum(r['thermal_operator_reused']=='False' for r in rows)<=1


def test_historical_runs_preserved():
    from probe_attention_nmp import digest
    frozen=json.loads((ROOT/'runs/attention_nmp_probe_8b_w1_b8/frozen_integrity.json').read_text())
    if not all((ROOT/p).exists() for p in frozen):pytest.skip('Local historical checkpoints are not distributed')
    for path,sha in frozen.items():assert digest(ROOT/path)==sha,path
