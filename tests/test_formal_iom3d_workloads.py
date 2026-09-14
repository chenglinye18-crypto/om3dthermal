"""Formal matrix semantics and frozen physical execution contracts."""
from pathlib import Path
from copy import deepcopy
import pytest
import numpy as np
from om3dthermal.serving.workload_matrix import *
from om3dthermal.placement.critical_path import AggregateCandidateModel
from om3dthermal.power.nmp_die_activity import PhysicalStageModel
from om3dthermal.power.batched_physical import merge_request_stages

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('wid,H,P,G',(('W1',2000,512,256),('W2',20000,512,256),('W3',126000,512,512)))
@pytest.mark.parametrize('batch',(1,8,32))
def test_cached_workload(wid,H,P,G,batch):
    w,cw,spec=inputs('Llama-3.1-8B',wid,batch,ROOT)
    pre=evaluate_cached_prefix_incremental_prefill(spec.prefill_input(batch_size=batch,prompt_length=P),cached_history_tokens=H)
    assert pre.prefill_input_tokens==batch*P
    assert pre.linear_ffn_tokens_computed_per_request==P
    assert pre.cached_history_tokens==H and not pre.cached_kv_rewritten
    assert len(cw.contexts)==G and cw.contexts.start==H+P and cw.contexts.stop==H+P+G
    assert w.context_length==H+P+G<=131072
    result=timing(1,[.1]*G,batch*G)
    assert result['decode_generated_tokens']==batch*G
    assert result['E2E_tok_s']==pytest.approx(batch*G/(1+.1*G))


def test_thermal_sources_and_grace_frozen():
    config,_,platform,hbm,bw=setup(ROOT)
    assert hbm.sustained_bandwidth_bytes_per_s==3.2973872924850354e12
    assert bw==3.3957835581187483e12
    assert platform.host_offload.direct_effective_bandwidth_bytes_per_second==416.34e9
    assert platform.host_offload.total_dynamic_J_per_bit==5.3e-12
    assert config['grace_capacity_bytes']==480e9


def test_hbm_thermal_cap_changes_the_formal_execution(monkeypatch):
    import om3dthermal.serving.workload_matrix as matrix
    cap=capacity('Llama-3.1-8B','W1',1,'HBM_GRACE_C2C',ROOT)
    thermal=conventional('Llama-3.1-8B','W1',1,ROOT,cap)
    cfg,ws,platform,hbm,bw=setup(ROOT)
    peak=hbm.model_copy(update={'sustained_bandwidth_bytes_per_s':4.8e12})
    monkeypatch.setattr(matrix,'setup',lambda root:(cfg,ws,platform,peak,bw))
    reference=conventional('Llama-3.1-8B','W1',1,ROOT,cap)
    assert thermal['traffic']==reference['traffic']
    assert thermal['summary']['decode_s']/reference['summary']['decode_s']==pytest.approx(4.8e12/hbm.sustained_bandwidth_bytes_per_s)


@pytest.mark.parametrize('name,wid,batch',(('Llama-3.1-8B','W1',1),('Llama-3.1-8B','W3',8),('Llama-3.1-70B','W1',8)))
def test_host_conservation_and_recurring(name,wid,batch):
    cap=capacity(name,wid,batch,'HBM_GRACE_C2C',ROOT)
    r=conventional(name,wid,batch,ROOT,cap);s=r['summary'];t=r['traffic']
    objects=r['host_audit']
    assert sum(o['size'] for o in objects)==pytest.approx((cap['weight_GB']+cap['final_KV_GB'])*1e9)
    assert sum(o['size']-o['local'] for o in objects)==pytest.approx(s['Grace_resident_GB']*1e9)
    assert s['HBM_resident_GB']+cap['workspace_GB']<=cap['HBM_capacity_GB']+1e-12
    assert t['total_C2C_GB']==pytest.approx(sum(t[k] for k in ('Grace_weight_read_GB','Grace_KV_read_GB','Grace_KV_write_GB','migration_GB','prefill_Grace_read_GB','prefill_Grace_write_GB')))
    assert t['Grace_weight_read_GB']==pytest.approx(sum(o['size']-o['local'] for o in objects if 'weights' in o['name'] and o['decode_reads'])*cap['G']/1e9)
    assert t['Grace_KV_write_GB']==pytest.approx(sum(o['size']-o['local'] for o in objects if o['write_phase']=='decode')/1e9)
    assert t['Grace_KV_read_GB']==pytest.approx(sum((o['size']-o['local'])*o['decode_reads'] for o in objects if 'KV' in o['name'])/1e9)
    assert t['migration_GB']==0
    if cap['HBM_only_fit']:assert t['total_C2C_GB']==0
    else:assert t['total_C2C_GB']>0
    assert r['energy']['E2E_tokens_per_J'] is None
    assert r==conventional(name,wid,batch,ROOT,cap)
    benefits=lambda o:o['prefill_reads']+o['decode_reads']+bool(o['write_phase'])
    assert min(benefits(o) for o in objects if o['local'])>=max(benefits(o) for o in objects if o['size']>o['local']) if not cap['HBM_only_fit'] else True


def test_capacity_infeasible_and_shared_state():
    a=capacity('Llama-3.1-8B','W1',1,'HBM_GRACE_C2C',ROOT)
    b=capacity('Llama-3.1-8B','W1',8,'HBM_GRACE_C2C',ROOT)
    assert a['weight_GB']==b['weight_GB'] and b['final_KV_GB']==8*a['final_KV_GB']
    assert capacity('Llama-3.1-405B','W1',1,'HBM_GRACE_C2C',ROOT)['status']=='SYSTEM_CAPACITY_INFEASIBLE'
    assert capacity('Llama-3.1-405B','W3',32,'IOM3D_MAC_NMP_CPA',ROOT)['status']=='CAPACITY_INFEASIBLE'


def test_no_nmp_cap_enters_actual_execution():
    w,cw,_=inputs('Llama-3.1-8B','W1',1,ROOT)
    _,_,_,_,bw=setup(ROOT)
    e=DecodePolicyModel(w,project_root=ROOT,placement_policy='UNIFORM_STRIPING',record_energy=True,external_bandwidth_cap=bw)
    assert e.floorplan.external_Bps==bw
    s=e.step(cw.contexts.start,'NO_NMP')
    m=FEOLEnergyModel(e.floorplan,e.platform).account(s['energy_events'],s['latency_s'],phase='decode',policy='NO_NMP')
    assert m['components']['read_peripheral_J']==s['energy_events']['array_read_bits']*.04e-12
    assert m['total_J']==sum(m['components'].values())
    assert resolve_feol_floorplan(ROOT).external_Bps==3.4e12


def test_batched_cpa_candidate_uses_aggregate_physics():
    w,cw,_=inputs('Llama-3.1-8B','W1',8,ROOT)
    e=DecodePolicyModel(w,project_root=ROOT,placement_policy='UNIFORM_STRIPING')
    model=PhysicalStageModel(e.floorplan,e.platform,w)
    aggregate=AggregateCandidateModel(e.placement,model)
    entry=e.placement.get(0,'ATTENTION_QK',0);atoms=cw.contexts.start*w.n_heads_kv
    actual=aggregate.evaluate(entry,atoms,nmp=True,resources=True)
    expected=merge_request_stages(e.floorplan,[model.evaluate(e.placement.get(0,'ATTENTION_QK',r),atoms,nmp=True,resources=True) for r in range(8)])
    assert actual['latency_s']==expected['latency_s']
    assert actual['nmp_flops']==expected['nmp_flops']
    np.testing.assert_array_equal(actual['resources']['group_times'],expected['resources']['group_times'])
