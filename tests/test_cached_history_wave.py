"""Scientific gates for the isolated cached-history wave policy."""
from pathlib import Path
import math
import pytest

from om3dthermal.serving.cached_history_wave import (
    evaluate_hbm_policy, high_water, resident_limit, select_hbm_best,
)
from om3dthermal.serving.decode_policy import CachedWorkload, llama31_models
from om3dthermal.serving.workload_matrix import setup
from om3dthermal.workload.model_registry import load_dense_model_spec

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def case():
    spec=load_dense_model_spec(ROOT/'configs/workload/models/llama31_8b.yaml')
    spec=spec.model_copy(update={'n_param':llama31_models()['Llama-3.1-8B'].n_param})
    _,ws,platform,hbm,_=setup(ROOT)
    return spec,CachedWorkload(126000,128,32),ws,platform,hbm


def run(case,b,policy='HBM_RESIDENT_WAVE'):
    spec,cw,ws,platform,hbm=case
    return evaluate_hbm_policy(spec,cw,b,ws,platform,hbm,policy=policy)


def test_cached_history_wave_high_water_and_partition(case):
    spec,cw,ws,_,hbm=case
    limit=resident_limit(spec,cw,ws,hbm.capacity_bytes)
    assert high_water(spec,cw.history,cw.prompt,cw.generated,limit,ws)[0]<=hbm.capacity_bytes
    assert high_water(spec,cw.history,cw.prompt,cw.generated,limit+1,ws)[0]>hbm.capacity_bytes
    r=run(case,32)
    assert sum(r['wave_sizes'])==32 and r['num_waves']>1
    assert all(w['high_water_bytes']<=hbm.capacity_bytes for w in r['waves'])
    for w in r['waves']:
        assert [s['context'] for s in w['steps']]==list(range(126128,126160))
        assert w['prefill_ledger']['new_prompt_tokens']==128
        assert w['prefill_ledger']['linear_ffn_tokens_computed_per_request']==128
        assert w['prefill_ledger']['cached_history_tokens']==126000


def test_cached_history_wave_admission_latency_and_closure(case):
    spec,cw,_,platform,_=case
    r=run(case,32)
    kv=2*spec.n_layers*spec.n_heads_kv*(spec.d_model//spec.n_heads_q)*2
    assert r['waves'][0]['admission_bytes']==0
    assert r['traffic']['admission_bytes']==sum(r['wave_sizes'][1:])*cw.history*kv
    assert r['traffic']['host_read_bytes']==r['traffic']['admission_bytes']
    assert all(w['traffic']['host_read_bytes']==0 for w in r['waves'])
    assert r['P95_completion_latency']==r['max_completion_latency']==r['E2E_s']
    assert r['requests'][-1]['queue_delay']>0
    assert r['max_TTFT']>r['requests'][0]['prefill_latency']+r['requests'][0]['first_decode_step']
    assert math.isclose(r['tokens_per_s']*r['E2E_s'],32*32,rel_tol=1e-12)
    assert math.isclose(sum(r['energy'].values()),r['E2E_J'],rel_tol=1e-12)
    assert math.isclose(sum(v for k,v in r['energy'].items() if k.endswith('gpu_static_J')),
                        r['E2E_s']*platform.gpu_decode_power.static_power_W,rel_tol=1e-12)


def test_local_hbm_wave_matches_offload_when_fit(case):
    wave=run(case,1);host=run(case,1,'HBM_HOST_OFFLOAD')
    assert wave['num_waves']==1 and wave['traffic']['host_read_bytes']==0
    assert host['traffic']['host_read_bytes']==host['traffic']['host_write_bytes']==0
    assert wave['tokens_per_s']==host['tokens_per_s']
    assert wave['E2E_J']==host['E2E_J']


def test_hbm_best_uses_complete_selected_policy(case):
    host=run(case,32,'HBM_HOST_OFFLOAD');wave=run(case,32)
    best=select_hbm_best(host,wave)
    selected=wave if wave['tokens_per_s']>host['tokens_per_s'] else host
    for key in ('energy','traffic','requests','tokens_per_J','P95_completion_latency'):
        assert best[key]==selected[key]
    assert select_hbm_best(host,{'status':'CAPACITY_INFEASIBLE'})['selected_HBM_policy']=='HBM_HOST_OFFLOAD'


def test_weights_cannot_be_streamed_by_resident_wave(case):
    spec,cw,ws,platform,hbm=case
    hbm=hbm.model_copy(update={'capacity_bytes':spec.n_param*2-1})
    r=evaluate_hbm_policy(spec,cw,1,ws,platform,hbm,policy='HBM_RESIDENT_WAVE')
    assert r['status']=='CAPACITY_INFEASIBLE'


def test_qwen_official_parameter_count_and_exact_slab_accounting():
    import numpy as np
    from om3dthermal.serving.decode_policy import DecodePolicyModel
    from om3dthermal.power.feol_energy import SCALARS,LAYERS
    spec=load_dense_model_spec(ROOT/'configs/workload/models/qwen25_32b.yaml')
    d,k=spec.d_model,spec.n_heads_kv*(spec.d_model//spec.n_heads_q)
    parameters=spec.n_layers*(2*d*d+2*d*k+3*d*spec.d_ff+2*d+d+2*k)+2*d*spec.vocab_size+d
    assert parameters==spec.n_param==65527752704//2
    _,workspace,_,hbm,_=setup(ROOT)
    cw=CachedWorkload(20000,128,32)
    peak,scratch=high_water(spec,20000,128,32,15,workspace)
    assert peak-scratch<=hbm.capacity_bytes<peak
    assert resident_limit(spec,cw,workspace,hbm.capacity_bytes)==14
    w=spec.decode_input(batch_size=1,context_length=20160)
    engine=DecodePolicyModel(w,project_root=ROOT,placement_policy='UNIFORM_STRIPING',record_energy=True,record_slabs=True)
    r=engine.step(20128,'MAC_NMP')
    for field in SCALARS+list(LAYERS):
        np.testing.assert_allclose(np.asarray(r['slab_events'][field]).sum(axis=0),r['energy_events'][field],rtol=1e-12,atol=1e-7)
