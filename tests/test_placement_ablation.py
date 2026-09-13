"""Batch/resource aggregation, placement legality and slab energy conservation."""
from pathlib import Path
import numpy as np
import pytest
from om3dthermal.serving.decode_policy import DecodePolicyModel,ExecutionPolicy,llama31_models,CachedWorkload
from om3dthermal.placement.nmp_load_balance import PlacementPolicy
from om3dthermal.power.feol_energy import FEOLEnergyModel,SCALARS,LAYERS
from om3dthermal.power.batched_physical import merge_request_stages

ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture(scope="module")
def engines():
    w=llama31_models()["Llama-3.1-8B"]
    return {b:DecodePolicyModel(w.model_copy(update={"batch_size":b}),project_root=ROOT,record_energy=True,record_slabs=True) for b in (1,8)}


def test_shared_weights_and_request_local_kv(engines):
    a,b=(engines[i].placement.audit() for i in (1,8))
    assert a["shared_weight_bytes"]==b["shared_weight_bytes"]
    assert b["total_kv_bytes"]==8*a["total_kv_bytes"]
    e=engines[8]
    for layer in range(e.workload.n_layers):
        qks=[]
        for request in range(8):
            qk=e.placement.get(layer,"ATTENTION_QK",request);av=e.placement.get(layer,"ATTENTION_AV",request)
            assert qk.unit.request_id==av.unit.request_id==request
            assert qk.lane_offset==av.lane_offset
            qks.append(qk)
        assert len({id(q) for q in qks})==8


@pytest.mark.parametrize("op",("Q","K","V","O","FFN_GATE","FFN_UP","FFN_DOWN","LM_HEAD"))
def test_linear_shared_read_batched_mac_and_activation(engines,op):
    layer=engines[1].workload.n_layers if op=="LM_HEAD" else 0
    rows=[e._evaluate_operator(op,layer,e.placement.get(layer,op).atom_count,nmp=True) for e in engines.values()]
    a,b=rows
    assert b["local_array_bytes"]==a["local_array_bytes"]
    assert b["nmp_flops"]==8*a["nmp_flops"]
    assert b["output_boundary_bytes"]==8*a["output_boundary_bytes"]
    assert b["energy_events"]["mac_operations"]==b["nmp_flops"]/2
    assert b["energy_events"]["activation_sram_bytes"]==engines[8].placement.get(layer,op).unit.activation_input_bytes*b["active_mac_tiles"]


def test_requests_contend_in_one_physical_stage(engines):
    e=engines[8];atoms=126000*8
    singles=[e.physical.evaluate(e.placement.get(0,"ATTENTION_QK",r),atoms,nmp=True,resources=True) for r in range(8)]
    aggregate=merge_request_stages(e.floorplan,singles)
    np.testing.assert_array_equal(aggregate["resources"]["group_times"],sum(s["resources"]["group_times"] for s in singles))
    np.testing.assert_array_equal(aggregate["resources"]["tile_loads"],sum(s["resources"]["tile_loads"] for s in singles))
    assert aggregate["latency_s"]!=sum(s["latency_s"] for s in singles)
    assert aggregate["latency_s"]>max(s["latency_s"] for s in singles)
    assert aggregate["boundary_bytes"]==sum(s["boundary_bytes"] for s in singles)
    step=e.step(126000,ExecutionPolicy.MAC_NMP,include_stages=True)
    assert step["nmp_flops"]==8*engines[1].step(126000,ExecutionPolicy.MAC_NMP)["nmp_flops"]
    qk=next(s for s in step["stages"] if s["operator"]=="ATTENTION_QK")
    av=next(s for s in step["stages"] if s["operator"]=="ATTENTION_AV")
    assert qk["request_ids"]==av["request_ids"]==list(range(8))
    reduction=next(s for s in step["stages"] if s["operator"]=="AV_REDUCTION")
    assert reduction["latency_s"]==(av["output_boundary_bytes"]+8*e.workload.d_model*2)/e.physical.gpu_bw


@pytest.mark.parametrize("batch",(1,8))
def test_slab_events_and_package_energy_conserve(engines,batch):
    e=engines[batch];s=e.step(126000,ExecutionPolicy.MAC_NMP)
    for k in SCALARS:np.testing.assert_allclose(s["slab_events"][k].sum(),s["energy_events"][k],rtol=1e-13,atol=1e-8,err_msg=k)
    for k in LAYERS:np.testing.assert_allclose(s["slab_events"][k].sum(axis=0),s["energy_events"][k],rtol=1e-13)
    m=FEOLEnergyModel(e.floorplan,e.platform)
    package=m.account(s["energy_events"],s["latency_s"],phase="decode",policy="MAC_NMP")
    slab_total=0.
    for slab in range(318):
        ev={k:np.asarray(v[slab]).tolist() for k,v in s["slab_events"].items()}
        c=m.account(ev,s["latency_s"],phase="decode",policy="NO_NMP")["components"]
        slab_total+=sum(v for k,v in c.items() if not k.startswith("gpu_"))
    expected=package["total_J"]-package["components"]["gpu_dynamic_J"]-package["components"]["gpu_static_J"]-package["components"]["feol_unresolved_J"]
    assert slab_total==pytest.approx(expected,rel=1e-13)
    assert sum(e.placement.audit()["per_slab_resident_bytes"])==e.placement.audit()["resident_bytes"]


def test_prefill_aggregate_and_generated_numerator(engines):
    a,b=(e.prefill(CachedWorkload()) for e in engines.values())
    assert b["ledger"]["total_flops"]==8*a["ledger"]["total_flops"]
    assert b["latency_s"]>a["latency_s"]
    assert engines[8].workload.batch_size*len(CachedWorkload().contexts)==8000


@pytest.mark.parametrize("policy",[PlacementPolicy.COMPACT_FIRST_FIT,PlacementPolicy.UNIFORM_STRIPING,PlacementPolicy.BALANCED])
@pytest.mark.parametrize("batch",(1,8))
def test_baseline_capacity_and_deterministic_mapping(policy,batch):
    w=llama31_models()["Llama-3.1-8B"].model_copy(update={"batch_size":batch})
    e=DecodePolicyModel(w,project_root=ROOT,placement_policy=policy)
    a=e.placement.audit()
    assert a["slot_capacity_violations"]==0
    assert sum(a["per_slab_resident_bytes"])==a["resident_bytes"]
    if policy==PlacementPolicy.COMPACT_FIRST_FIT:
        used=np.flatnonzero(a["per_slab_resident_bytes"])
        np.testing.assert_array_equal(used,np.arange(len(used)))
        assert len(used)<318
    else:assert np.count_nonzero(a["per_slab_resident_bytes"])==318
    if batch==1:
        again=DecodePolicyModel(w,project_root=ROOT,placement_policy=policy)
        assert a==again.placement.audit()
        np.testing.assert_array_equal(e.placement.get(0,"Q").tile_ids,again.placement.get(0,"Q").tile_ids)


def test_thermal_sources_use_actual_slab_power():
    from om3dthermal.thermal.placement_diagnostic import map_slab_power
    # Mapping needs only model_copy; geometry itself is reused in the real solver.
    class Simulation:
        def model_copy(self,update):return update
    powers=np.arange(318,dtype=float)/100
    result=map_slab_power(Simulation(),123.,powers)["thermal_power_sources"]
    assert sum(x.total_power for x in result.sources)==pytest.approx(123+powers.sum())
    for slab,source in enumerate(result.sources[1:]):
        assert source.total_power==powers[slab]
        assert source.selector.tags=={"role":"m3d_bitcell_beol_stack","die_index":slab+1}


def test_prefill_temporary_traffic_can_reuse_free_slots(engines):
    p=engines[1].placement
    before=p.slot_used.copy()
    free=int(((p.floorplan.layout.slot_capacity_bytes-p.slot_used)*4).sum())
    traffic=free*2+64
    mapped=p.transient_layer_bytes(traffic)
    assert mapped.sum()==traffic
    np.testing.assert_array_equal(p.slot_used,before)


@pytest.mark.parametrize("name",list(llama31_models()))
def test_frozen_b1_balanced_first_last_steps(name):
    import json
    frozen=json.loads((ROOT/"tests/data/placement_ablation_ae14f09.json").read_text())
    old=next(r for r in frozen["summary"] if r["model"]==name)
    e=DecodePolicyModel(llama31_models()[name],project_root=ROOT,record_energy=True,record_slabs=True)
    for context,key in ((126000,"first_step_ms"),(126999,"last_step_ms")):
        point=e.step(context,ExecutionPolicy.MAC_NMP)
        assert point["latency_s"]*1000==float(old[key])


def test_cached_thermal_selection_matches_canonical_mapping():
    from types import SimpleNamespace
    from om3dthermal.thermal.placement_diagnostic import SlabPowerMapper,map_slab_power
    from om3dthermal.thermal.power import map_power_sources
    class Simulation:
        def model_copy(self,update):return update
    cells=[SimpleNamespace(component="gpu",material="FEOL",parent_box_name="gpu",tags={},volume=3.)]
    for slab in range(318):
        for v in (1.,2.):
            cells.append(SimpleNamespace(component=f"slab{slab}",material="BEOL",parent_box_name="beol",tags={"role":"m3d_bitcell_beol_stack","die_index":slab+1},volume=v))
    powers=np.arange(318)/100
    config=map_slab_power(Simulation(),74.,powers)["thermal_power_sources"]
    expected=map_power_sources(cells,config).power_W
    np.testing.assert_array_equal(SlabPowerMapper(cells).power(74.,powers),expected)

@pytest.mark.parametrize("policy",[PlacementPolicy.COMPACT_FIRST_FIT,PlacementPolicy.UNIFORM_STRIPING,PlacementPolicy.BALANCED])
def test_b8_deterministic_step_rerun(policy):
    import hashlib,json
    w=llama31_models()["Llama-3.1-8B"].model_copy(update={"batch_size":8})
    e=DecodePolicyModel(w,project_root=ROOT,placement_policy=policy,record_energy=True,record_slabs=True)
    def digest(step):
        return hashlib.sha256(json.dumps(step,sort_keys=True,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item()).encode()).hexdigest()
    first=digest(e.step(126000,ExecutionPolicy.MAC_NMP))
    e.step(126999,ExecutionPolicy.MAC_NMP)
    assert digest(e.step(126000,ExecutionPolicy.MAC_NMP))==first
