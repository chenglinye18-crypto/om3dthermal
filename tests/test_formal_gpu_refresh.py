"""Scientific gates for formal-v2 GPU refresh, with no thermal solves."""
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import refresh_formal_gpu_results as refresh
from diagnose_m3d_gpu_port_balanced import read, loadcsv, sha
from om3dthermal.serving.decode_policy import DecodePolicyModel, CachedWorkload
from om3dthermal.power.feol_energy import sum_events, SCALARS, LAYERS

def test_port_balanced_integrated_events_match_physical_steps():
    w,cw,_=refresh.f.legacy.inputs('Llama-3.1-8B','LC20K',1,ROOT)
    short=CachedWorkload(history=cw.history,prompt=cw.prompt,generated=2)
    model=DecodePolicyModel(w,project_root=ROOT,placement_policy='GPU_PORT_BALANCED',record_energy=True,
                            external_bandwidth_cap=refresh.f.setup(ROOT)[4])
    actual=sum_events(model.step(c,'NO_NMP')['energy_events'] for c in short.contexts)
    projected=refresh.global_events(refresh.project_decode(model.placement,short,nmp=False))
    for key in SCALARS+list(LAYERS):
        np.testing.assert_allclose(projected[key],actual[key],rtol=1e-12,atol=1e-7,err_msg=key)

def test_formal_gpu_refresh_closure_and_preservation():
    snapshot=read(refresh.SNAP)
    rows=loadcsv(refresh.f.OUT/'final_e2e_metrics.csv')
    assert len(rows)==72
    for p in refresh.f.PATHS:assert sum(r['path']==p for r in rows)==18
    for name,digest in snapshot['files'].items():assert sha(ROOT/name)==digest
    old={(r['model'],r['context'],r['batch'],r['path']):r for r in snapshot['rows']}
    for row in rows:
        key=tuple(row[k] for k in ('model','context','batch','path'))
        if row['path'] in refresh.f.PATHS[2:]:assert row==old[key]
        else:
            r=read(refresh.f.OUT/'candidates'/f"{row['model']}_{row['context']}_B{row['batch']}_{row['path']}.json")
            s,e=r['summary'],r['energy'];n=int(row['batch'])*32
            np.testing.assert_allclose(e['E2E_J'],e['decode_J']+e['prefill_J'],rtol=1e-13)
            np.testing.assert_allclose(n/s['E2E_s'],float(row['tokens_per_s']),rtol=1e-13)
            np.testing.assert_allclose(n/e['E2E_J'],float(row['tokens_per_J']),rtol=1e-13)
            if row['path']=='HBM_GPU':
                assert s['effective_bandwidth_TBps']==refresh.hbm_bandwidth()/1e12
                assert e['hbm_read_energy_pj_per_bit']==3.
            else:
                assert s['placement_policy']=='GPU_PORT_BALANCED'
                assert s['accepted_moves']==0 and r['cpa_audit']==[]
                for phase in ('prefill','decode'):
                    np.testing.assert_allclose(e[phase+'_gpu_static_J'],s[phase+'_s']*refresh.f.setup(ROOT)[2].gpu_decode_power.static_power_W,rtol=1e-13)
                for key in ('mac_operations','fp32_reduction_adds','noc_link_bit_um'):
                    assert r['events'][key]==0
                cached=read(refresh.DIAG/'cases'/f"{row['model']}_{row['context']}_B{row['batch']}.json")['new']['summary']
                if 'decode_J' in cached:
                    np.testing.assert_allclose(e['decode_J'],cached['decode_J'],rtol=1e-12)
                np.testing.assert_allclose(s['E2E_tok_s'],cached['tokens_per_s'],rtol=1e-13)

def test_hbm_logical_traffic_unchanged():
    import json,subprocess
    snap=read(refresh.SNAP)
    ledgers=json.loads(subprocess.check_output(['git','show',snap['old_HEAD']+':runs/formal_long_context_v2/candidate_ledgers.json']))
    for m,c,b in refresh.f.points():
        name=f'runs/formal_long_context_v2/candidates/{m}_{c}_B{b}_HBM_GPU.json'
        old=ledgers[f'{m}_{c}_B{b}_HBM_GPU']
        new=read(ROOT/name)
        assert old['prefill_ledger']==new['prefill_ledger']
        for key,value in old['traffic'].items():
            if key!='host_critical_path_fraction':assert value==new['traffic'][key]
        for path in refresh.f.PATHS[2:]:
            candidate=read(refresh.f.OUT/'candidates'/f'{m}_{c}_B{b}_{path}.json')
            for field,value in ledgers[f'{m}_{c}_B{b}_{path}'].items():
                assert candidate[field]==value
