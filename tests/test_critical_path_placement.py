"""Frozen Uniform, legal CPA moves, deterministic acceptance and events."""
import hashlib,json
from pathlib import Path
import numpy as np
import pytest
from om3dthermal.serving.decode_policy import DecodePolicyModel,ExecutionPolicy,llama31_models,CachedWorkload
from om3dthermal.placement.critical_path import MigratedOperator,MAX_ITERATIONS,proposal,resident_proposal
from om3dthermal.power.feol_energy import FEOLEnergyModel,SCALARS,LAYERS

ROOT=Path(__file__).resolve().parents[1]


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,default=lambda a:a.tolist() if isinstance(a,np.ndarray) else a.item()).encode()).hexdigest()


@pytest.mark.parametrize('name',list(llama31_models()))
def test_uniform_exact_reference(name):
    frozen=json.loads((ROOT/'tests/data/cpa_uniform_920efd2.json').read_text())['cases'][name]
    e=DecodePolicyModel(llama31_models()[name],project_root=ROOT,placement_policy='UNIFORM_STRIPING',record_energy=True,record_slabs=True)
    placement=[{k:getattr(v,k) for k in ('lanes','start_layers','tile_ids','atom_count','atom_bytes')} for v in e.placement.operators.values()]
    assert digest(placement)==frozen['placement_hash']
    assert digest(e.prefill(CachedWorkload()))==frozen['prefill_hash']
    for context,expected in frozen['step_hashes'].items():
        assert digest(e.step(int(context),ExecutionPolicy.MAC_NMP))==expected
    saved=json.loads((ROOT/'runs/placement_ablation_b1_b8_v1/case_results'/f'{name}_B1_UNIFORM_STRIPING.json').read_text()) if (ROOT/'runs/placement_ablation_b1_b8_v1/case_results'/f'{name}_B1_UNIFORM_STRIPING.json').exists() else None
    if saved is not None:
        for key in ('summary','energy','events','thermal','stage_hash'):assert saved[key]==frozen[key]


@pytest.fixture(scope='module')
def pair():
    w=llama31_models()['Llama-3.1-8B']
    return [DecodePolicyModel(w,project_root=ROOT,placement_policy=p,record_energy=True,record_slabs=True)
            for p in ('UNIFORM_STRIPING','CRITICAL_PATH_AWARE')]


def test_capacity_atomic_locality_and_pairing(pair):
    u,c=pair
    assert u.placement.audit()['resident_bytes']==c.placement.audit()['resident_bytes']
    assert c.placement.audit()['slot_capacity_violations']==0
    total=0
    for key,e in c.placement.operators.items():
        b=e.layer_bytes(e.atom_count)
        assert np.all(b%e.atom_bytes==0)
        assert b.sum()==e.atom_count*e.atom_bytes
        assert np.all(e.tile_ids//8==e.region_ids)
        total+=int(b.sum())
    assert total==c.placement.audit()['resident_bytes']
    for l in range(c.workload.n_layers):
        q=c.placement.get(l,'ATTENTION_QK',0);v=c.placement.get(l,'ATTENTION_AV',0)
        assert q.unit.request_id==v.unit.request_id==0
        for n in (1,1007,126000*8,126999*8,127000*8):
            np.testing.assert_array_equal(q.prefix_counts(n),v.prefix_counts(n))


def test_real_migration_prefix_and_append_conservation(pair):
    e=pair[0].placement.get(0,'ATTENTION_QK',0)
    source=0
    dest=int(np.flatnonzero((e.die_ids==e.die_ids[source])&(e.group_ids!=e.group_ids[source]))[0])
    m=MigratedOperator.build(e,[source],[dest],[5])
    for n in (0,1,20000,126000*8,126999*8):
        assert m.prefix_counts(n).sum()==n
        assert m.layer_bytes(n).sum()==n*m.atom_bytes
        np.testing.assert_array_equal(m.layer_bytes(n+8,begin=n),m.layer_bytes(n+8)-m.layer_bytes(n))
    assert m.die_ids[source]==m.die_ids[dest]
    old=pair[0].physical.evaluate(e,126000*8,nmp=True)
    new=pair[0].physical.evaluate(m,126000*8,nmp=True)
    assert old['nmp_flops']==new['nmp_flops']
    assert old['local_array_bytes']==new['local_array_bytes']


def test_acceptance_and_termination(pair):
    a=pair[1].placement.optimizer_audit
    assert a
    for r in a:
        assert r['accepted_moves']<=MAX_ITERATIONS+1
        assert r['after_s']<=r['before_s']
        assert all(after<before for before,after in r['accepted_history'])
        assert 0<=r['moved_atom_fraction']<=1


def test_cross_slab_move_is_rejected(pair):
    e=pair[0].placement.get(0,'ATTENTION_QK',0)
    destination=int(np.flatnonzero(e.die_ids!=e.die_ids[0])[0])
    with pytest.raises(AssertionError):MigratedOperator.build(e,[0],[destination],[1])


def test_cpa_scope_does_not_expand_to_b8():
    w=llama31_models()['Llama-3.1-8B'].model_copy(update={'batch_size':8})
    with pytest.raises(ValueError,match='B=1'):
        DecodePolicyModel(w,project_root=ROOT,placement_policy='CRITICAL_PATH_AWARE')


def test_large_operator_candidate_does_not_overflow_int32():
    e=DecodePolicyModel(llama31_models()['Llama-3.1-405B'],project_root=ROOT,placement_policy='UNIFORM_STRIPING')
    entry=e.placement.get(e.workload.n_layers,'LM_HEAD')
    assert entry.unit.local_flops>2**31
    stage=e.physical.evaluate(entry,entry.atom_count,nmp=True,resources=True)
    targets=proposal(entry,stage,e.floorplan,8)
    assert np.all(targets//8==entry.region_ids)


def test_source_follows_actual_critical_resource_not_fabric_bytes(pair):
    e=pair[0];entry=e.placement.get(0,'ATTENTION_QK',0)
    stage=e.physical.evaluate(entry,126999*8,nmp=True,resources=True)
    r=stage['resources']
    r['fabric_bytes'][:]=0;r['fabric_bytes'][:,2]=1e12
    r['group_times'][:]=0;r['group_times'][:,5]=1.
    stage['components'].update(ARRAY=10.,LOCAL_FABRIC=1.,MAC=1.)
    moved=resident_proposal(entry,stage,e.floorplan,0)
    assert len(moved.source)>0
    assert np.all(entry.group_ids[moved.source]==5)
    r['tile_loads'][:]=0;r['tile_loads'][:,16]=1e12
    stage['components'].update(ARRAY=1.,LOCAL_FABRIC=1.,MAC=10.)
    moved=resident_proposal(entry,stage,e.floorplan,0)
    assert np.all(entry.region_ids[moved.source]==2)


def test_external_limited_tile_search_is_skipped(pair):
    e=pair[0];entry=e.placement.get(0,'Q')
    stage=e.physical.evaluate(entry,entry.atom_count,nmp=True,resources=True)
    stage['bottleneck']='EXTERNAL_BOUNDARY'
    np.testing.assert_array_equal(proposal(entry,stage,e.floorplan,1),entry.tile_ids)


def test_optimizer_deterministic(pair):
    e=DecodePolicyModel(pair[1].workload,project_root=ROOT,placement_policy='CRITICAL_PATH_AWARE')
    for key,v in e.placement.operators.items():
        other=pair[1].placement.operators[key]
        np.testing.assert_array_equal(v.tile_ids,other.tile_ids)
        np.testing.assert_array_equal(v.layer_bytes(v.atom_count),other.layer_bytes(other.atom_count))
    def deterministic(a):return [{k:v for k,v in r.items() if k!='optimizer_runtime_s'} for r in a]
    assert deterministic(e.placement.optimizer_audit)==deterministic(pair[1].placement.optimizer_audit)


@pytest.mark.parametrize('context',[126000,126499,126999])
def test_work_and_energy_conservation(pair,context):
    u,c=pair
    a=u.step(context,ExecutionPolicy.MAC_NMP)
    b=c.step(context,ExecutionPolicy.MAC_NMP)
    assert b['latency_s']<=a['latency_s']
    for k in ('array_read_bits','array_write_bits','mac_operations','read_services','write_services','sa_sensed_bits'):
        assert b['energy_events'][k]==a['energy_events'][k]
    for k in SCALARS:np.testing.assert_allclose(b['slab_events'][k].sum(),b['energy_events'][k],rtol=1e-12,atol=1e-7)
    for k in LAYERS:np.testing.assert_allclose(b['slab_events'][k].sum(axis=0),b['energy_events'][k],rtol=1e-12)
    model=FEOLEnergyModel(c.floorplan,c.platform)
    energy=model.account(b['energy_events'],b['latency_s'],phase='decode',policy='MAC_NMP')
    assert sum(energy['components'].values())==energy['total_J']
    gpu=sum(v for k,v in energy['components'].items() if k.startswith('gpu_'))
    slab_total=0
    for s in range(318):
        events={k:np.asarray(v[s]).tolist() for k,v in b['slab_events'].items()}
        components=model.account(events,b['latency_s'],phase='decode',policy='NO_NMP')['components']
        slab_total+=sum(v for k,v in components.items() if not k.startswith('gpu_'))
    np.testing.assert_allclose(slab_total+gpu+energy['components']['feol_unresolved_J'],energy['total_J'],rtol=1e-12)
