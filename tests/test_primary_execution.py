"""Scientific regression gates for the B1/B8 GPU/adaptive rebaseline."""
from copy import deepcopy
from dataclasses import replace
import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from om3dthermal.serving.primary_execution import (
    ARCHITECTURES, GPU_PLACEMENT, SELECTOR_STATUS, adaptive_result,
    candidate_timing, corrected_gpu_result, gpu_memory_closure, gpu_roofline)
from om3dthermal.serving.workload_matrix import conventional, setup
from om3dthermal.serving.decode_policy import llama31_models
from om3dthermal.workload import evaluate_llm_decode

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'runs/formal_iom3d_workload_sweep_v1'
sys.path.insert(0,str(ROOT/'scripts'))
from rebaseline_primary_workloads import check_integrity, OUTPUTS


@pytest.mark.parametrize('name',llama31_models())
@pytest.mark.parametrize('wid',('W1','W2','W3'))
@pytest.mark.parametrize('batch',(1,8))
def test_gpu_only_paired_consistency(name,wid,batch):
    candidate = candidate_timing(name,wid,batch,ROOT)
    cfg,_,platform,hbm,_ = setup(ROOT)
    c = candidate['closure']
    assert c['effective_Bps'] == min(c[k] for k in ('thermal_Bps','internal_Bps','boundary_Bps','GPU_Bps'))
    assert c['parallel_service_lanes'] == 318*50
    assert c['internal_Bps'] == c['parallel_service_lanes']*c['payload_bytes_per_service']/(c['service_cycle_ns']*1e-9)
    w = llama31_models()[name].model_copy(update={'batch_size':batch})
    hbm_times = []
    for row in candidate['steps']:
        # Independent canonical HBM workload accounting; not a second call to
        # the streaming implementation under test.
        m = evaluate_llm_decode(w.model_copy(update={'context_length':row['context']}))
        assert row['flops'] == batch*m.flops_per_token
        assert row['read_bytes'] == batch*m.read_bytes_per_token
        assert row['write_bytes'] == batch*m.write_bytes_per_token
        compute = batch*m.flops_per_token/platform.gpu_compute_power.peak_compute_BF16_dense_flops_per_s
        assert row['compute_s'] == compute
        reference = max(compute,batch*(m.read_bytes_per_token+m.write_bytes_per_token)/hbm.sustained_bandwidth_bytes_per_s)
        assert row['latency_s'] <= reference
        hbm_times.append(reference)
    # This is a capacity-conditional HBM-local relation, not a fabricated
    # formal HBM result for capacity-infeasible 405B points.
    assert candidate['timing']['decode_s'] <= sum(hbm_times)
    p = platform.gpu_prefill_compute; pre = candidate['ledger']
    compute = (pre['linear_flops']+pre['lm_head_flops'])/(p.large_gemm_effective_tflops*1e12)+pre['attention_flops']/(p.causal_attention_effective_tflops*1e12)
    assert candidate['prefill_compute_s'] == compute
    assert candidate['timing']['prefill_s'] <= max(compute,pre['total_memory_bytes']/hbm.sustained_bandwidth_bytes_per_s)


def test_streaming_has_no_duplicate_serialization_and_respects_other_limits():
    c = gpu_memory_closure(ROOT)
    size = 1e9
    assert gpu_roofline(0,size,c) == size/c['effective_Bps']
    for limit in ('internal_Bps','boundary_Bps','GPU_Bps','thermal_Bps'):
        constrained = {**c,limit:1e9}
        constrained['effective_Bps'] = min(constrained[k] for k in ('thermal_Bps','internal_Bps','boundary_Bps','GPU_Bps'))
        assert gpu_roofline(0,size,constrained) == 1
    assert gpu_roofline(2,size,c) == 2
    assert gpu_roofline(0,size,{**c,'startup_s':.01}) == .01+size/c['effective_Bps']


@pytest.mark.parametrize('field',('internal_bandwidth_average_bytes_per_s','coil_bandwidth_bytes_per_s'))
def test_closure_resolves_internal_and_boundary_before_thermal(field,monkeypatch):
    import om3dthermal.serving.primary_execution as execution
    architecture=execution.resolve_m3d_architecture_backend(ROOT)
    bandwidth=replace(architecture.bandwidth,**{field:1e12})
    monkeypatch.setattr(execution,'resolve_m3d_architecture_backend',lambda root:SimpleNamespace(bandwidth=bandwidth))
    resolved=execution.gpu_memory_closure.__wrapped__(ROOT)
    assert resolved['effective_Bps']==1e12


def candidates(gpu_time,nmp_time):
    def make(system,time,placement):
        return dict(summary=dict(model='synthetic',workload_id='arbitrary',batch_size=1,H=100,P=20,G=5,
            status='EVALUATED',system=system,prefill_s=2.,decode_s=time,E2E_s=time+2,
            E2E_tok_s=5/(time+2),placement_policy=placement,accepted_moves=3 if system=='NMP_CPA' else 0,
            optimizer_runtime_s=9 if system=='NMP_CPA' else 0),
            traffic={'bytes':13 if system=='NMP_CPA' else 19},
            energy={'E2E_J':100 if system=='NMP_CPA' else 10},
            events={'mac':1 if system=='NMP_CPA' else 0},
            cpa_audit=[{'moves':3}] if system=='NMP_CPA' else [])
    return make('M3D_GPU',gpu_time,GPU_PLACEMENT),make('NMP_CPA',nmp_time,'CRITICAL_PATH_AWARE')


@pytest.mark.parametrize('gpu_cost,nmp_cost,expected',[(1,2,'GPU'),(2,1,'NMP_CPA'),(1,1,'GPU')])
@pytest.mark.parametrize('batch',(1,8))
def test_adaptive_execution_selection_exact_copy(gpu_cost,nmp_cost,expected,batch,monkeypatch):
    import om3dthermal.placement.critical_path as placement
    def forbidden(*args,**kwargs):
        pytest.fail('Selector/GPU execution must not invoke CPA')
    monkeypatch.setattr(placement,'refine',forbidden)
    gpu,nmp=candidates(gpu_cost,nmp_cost)
    gpu['summary']['batch_size']=nmp['summary']['batch_size']=batch
    before=deepcopy((gpu,nmp))
    result=adaptive_result(gpu,nmp);s=result['summary']
    selected=gpu if expected=='GPU' else nmp
    assert (gpu,nmp)==before
    assert s['selected_executor']==expected
    assert s['decode_s']==min(gpu_cost,nmp_cost)
    assert s['E2E_s']==min(gpu['summary']['E2E_s'],nmp['summary']['E2E_s'])
    assert s['E2E_tok_s']/gpu['summary']['E2E_tok_s']>=1
    for field in ('traffic','energy','events','cpa_audit'):
        assert result[field]==selected[field]
    assert s['selector_overhead_status']==SELECTOR_STATUS
    assert s['selector_latency_s']==s['selector_energy_J']==0
    if expected=='GPU':
        assert s['placement_policy']==GPU_PLACEMENT
        assert s['CPA_moves']==s['CPA_optimizer_runtime']==0
    else:
        assert s['placement_policy']=='CRITICAL_PATH_AWARE'
        assert s['CPA_moves']==3
        # Faster NMP can consume more energy: selection is latency-only.
        assert result['energy']['E2E_J']>gpu['energy']['E2E_J']


def test_b32_rejected_before_candidate_computation(monkeypatch):
    import om3dthermal.serving.primary_execution as execution
    def forbidden(*args,**kwargs):
        pytest.fail('B32 must be rejected before workload/capacity evaluation')
    monkeypatch.setattr(execution,'inputs',forbidden)
    with pytest.raises(ValueError,match='B32'):
        candidate_timing('unused','unused',32,ROOT)
    gpu,nmp=candidates(1,2)
    gpu['summary']['batch_size']=nmp['summary']['batch_size']=32
    with pytest.raises(ValueError,match='B32'):
        adaptive_result(gpu,nmp)


def test_uniform_not_an_architecture_or_adaptive_candidate():
    assert ARCHITECTURES==('HBM_GPU','M3D_GPU','M3D_MAC_NMP')
    gpu,nmp=candidates(1,2)
    nmp['summary']['placement_policy']='UNIFORM_STRIPING'
    with pytest.raises(ValueError,match='CPA'):
        adaptive_result(gpu,nmp)


def test_invalid_candidates_fail_closed():
    for field,value in [('prefill_s',3),('decode_s',float('nan')),('decode_s',0),('status','STOPPED_BY_USER'),('G',6)]:
        gpu,nmp=candidates(1,2);nmp['summary'][field]=value
        with pytest.raises(ValueError):adaptive_result(gpu,nmp)


def test_integrity_fails_on_changed_source_or_stopped_state(tmp_path):
    from rebaseline_primary_workloads import sha
    source=tmp_path/'physical.py';source.write_text('frozen = True\n')
    result=tmp_path/'B32.json';result.write_text('{"status":"STOPPED_BY_USER","latency":1.234}')
    integrity=dict(frozen_sources={'physical.py':sha(source,text=True)},preserved_artifacts={'B32.json':sha(result)})
    check_integrity(tmp_path,integrity)
    result.write_text('{"status":"EVALUATED","latency":1.234}')
    with pytest.raises(ValueError,match='Preserved'):check_integrity(tmp_path,integrity)
    assert not ({'summary.csv','normalized.csv','b32_stress.csv','capacity.csv','energy.csv','traffic.csv'} & OUTPUTS)


def test_reviewed_diagnostic_change_cannot_waive_decode_source_validation(tmp_path):
    integrity=dict(frozen_sources={},preserved_artifacts={},reviewed_non_decode_source_changes={
        'src/om3dthermal/serving/decode_policy.py':{}})
    with pytest.raises(ValueError,match='Unreviewed physical source'):
        check_integrity(tmp_path,integrity)


def test_local_b32_artifacts_and_physical_sources_untouched():
    path=OUT/'rebaseline_integrity.json'
    if not path.exists():pytest.skip('Local historical artifacts not distributed')
    integrity=json.loads(path.read_text())
    if not all((ROOT/p).exists() for p in integrity['preserved_artifacts']):
        pytest.skip('Local physical checkpoints not distributed')
    check_integrity(ROOT,integrity)
    assert any('_B32_' in p for p in integrity['preserved_artifacts'])


def test_local_formal_outputs_and_gpu_energy_unchanged_events():
    path=OUT/'primary_validation.json'
    if not path.exists():pytest.skip('Rebaseline outputs unavailable')
    validation=json.loads(path.read_text())
    assert validation['NMP_physical_Decode_checkpoints_reused']==36
    assert validation['GPU_only_recomputed_cases']==18
    rows=list(csv.DictReader((OUT/'primary_summary.csv').open()))
    assert len(rows)==54
    assert {r['system'] for r in rows}==set(ARCHITECTURES)
    assert {int(r['batch_size']) for r in rows}=={1,8}
    integrity=json.loads((OUT/'rebaseline_integrity.json').read_text())
    fp=integrity['legacy_source_fingerprint']
    p=OUT/'checkpoints'/f'Llama-3.1-8B_W1_B1_IOM3D_NO_NMP_{fp[:16]}.json'
    if not p.exists():pytest.skip('Local physical checkpoints not distributed')
    old=json.loads(p.read_text())
    gpu=corrected_gpu_result(old,candidate_timing('Llama-3.1-8B','W1',1,ROOT),ROOT)
    assert gpu['events']==old['events']
    for key,value in old['energy'].items():
        if key.startswith('prefill_') or (key.startswith('decode_') and key.endswith('_J') and key not in ('decode_J','decode_gpu_static_J','decode_tokens_per_J')):
            assert gpu['energy'][key]==value
