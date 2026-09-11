"""Physical FEOL execution gates; obsolete die-only/energy gates are removed."""
from pathlib import Path
from types import SimpleNamespace
import inspect

import numpy as np
import pytest

from om3dthermal.architecture.feol_floorplan import resolve_feol_floorplan, manhattan
from om3dthermal.placement.nmp_load_balance import PhysicalResidentPlacement, ResidentOperator
from om3dthermal.power.nmp_die_activity import group_service_seconds, structured_noc
from om3dthermal.power.feol_route import distributed_elmore_delay_ns, calculate_feol_route
from om3dthermal.serving.decode_policy import CachedWorkload, DecodePolicyModel, ExecutionPolicy, llama31_models

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def f():
    return resolve_feol_floorplan(ROOT)


@pytest.fixture(scope="module")
def engines():
    return {name: DecodePolicyModel(w, project_root=ROOT) for name,w in llama31_models().items()}


def test_clusters_and_capacity(f):
    assert (f.topology.cluster_count_x, f.topology.cluster_count_y) == (35, 8)
    assert len(f.clusters) == 280
    assert f.layout.layers_per_cluster == 8
    assert f.layout.physical_slot_count == 318*280*8
    assert f.layout.total_capacity_bytes == 1493843312640
    assert f.case.geometry.orthogonal.slab_pitch_x_um == 100


@pytest.mark.parametrize("g", range(70))
def test_physical_group_members_and_sa(f, g):
    group = f.groups[g]
    members = [f.clusters[i] for i in group["cluster_members"]]
    assert len(members) == 4
    assert len({x["column"] for x in members}) == 1
    assert [x["row"] for x in members] in ([0,1,2,3], [4,5,6,7])
    assert group["center_um"] == np.mean([x["center_um"] for x in members], axis=0).tolist()
    assert f.audit()["sa_banks"][g]["center_um"] == group["center_um"]


@pytest.mark.parametrize("r,count", tuple(enumerate((18,18,18,16))))
def test_regions_tiles_and_buffer(f, r, count):
    region = f.regions[r]
    assert len(region["groups"]) == count
    tiles = [t for t in f.tiles if t["region_id"] == r]
    assert len(tiles) == 8 and sum(t["macs"] for t in tiles) == 128
    x0,y0,x1,y1 = region["bounds_um"]
    assert region["center_um"] == [(x0+x1)/2, (y0+y1)/2]
    assert len({t["center_um"][0] for t in tiles}) == 4
    assert len({t["center_um"][1] for t in tiles}) == 2
    for g in region["groups"]:
        x,y = f.groups[g]["center_um"]
        assert x0 <= x <= x1 and y0 <= y <= y1
    assert f.config["region_buffer_bytes"] == 128*1024


def test_hardware_caps(f):
    assert len(f.tiles) == 32
    assert f.tile_flops == 32e9
    assert 318*32*f.tile_flops == 325.632e12
    assert f.fabric_Bps == 64e9 and f.link_Bps == 32e9
    assert len(f.ports) == 50 and all(p[1] == 0 for p in f.ports)
    assert np.diff([p[0] for p in f.ports]) == pytest.approx(np.full(49, 440.0))
    assert f.config["m3d_external_bandwidth"]["raw_interface_bytes_per_s"] == 15.9e12
    assert f.external_rate(1e30) == f.external_Bps == 3.4e12
    assert f.external_rate(1e9) == 1e9


def test_rc_shared_primitive_and_pipeline(f):
    wire = f.case.architecture.feol_route.wire
    assert (wire.resistance_ohm_per_um, wire.capacitance_fF_per_um, wire.voltage_V) == (2,.2,.8)
    assert (wire.fixed_driver_resistance_ohm, wire.fixed_load_pF) == (100,.006)
    old_route = calculate_feol_route(f.case.architecture, f.topology)
    for length, delay in zip(old_route.feol_route_length_per_cluster_um, old_route.feol_delay_per_cluster_ns):
        assert distributed_elmore_delay_ns(length, wire) == pytest.approx(delay, rel=1e-12)
    for link in f.links:
        a,b = link["regions"]
        assert link["length_um"] == manhattan(f.regions[a]["center_um"], f.regions[b]["center_um"])
        assert link["wire_pipeline_cycles"] == np.ceil(link["rc_ns"])
        assert link["hop_ns"] == 1+link["wire_pipeline_cycles"]
        assert link["wire_pipeline_cycles"] > 1
    assert np.all(f.service_ns >= 10) and np.all(f.service_ns < 10.02)


def test_single_inactive_hot_group(f):
    b = np.zeros((70,8), dtype=np.int64)
    assert not np.any(group_service_seconds(b, f.service_ns))
    b[3,0] = 33
    t = group_service_seconds(b, f.service_ns)
    assert np.count_nonzero(t) == 1
    assert t[3] == 2*f.service_ns[3,0]*1e-9
    b[4,7] = 32*1000
    assert np.argmax(group_service_seconds(b, f.service_ns)) == 4


def test_structured_noc_conservation(f):
    multicast = structured_noc(f, np.array([[32,32,32,32]]), mode="multicast")
    assert multicast["hop_bytes"] == 3*32
    assert multicast["root_regions"].tolist() == [1]
    reduction = structured_noc(f, np.full((1,4),16), mode="reduce", vector_bytes=16)
    assert reduction["hop_bytes"] == 3*16
    assert reduction["link_bytes"].shape == (1,3,2)
    with pytest.raises(ValueError):
        structured_noc(f, np.ones((1,4)), mode="arbitrary_mesh")


@pytest.mark.parametrize("name", list(llama31_models()))
def test_resident_slot_group_die_atomic_closure(engines, name):
    p = engines[name].placement
    a = p.audit()
    w = p.workload
    assert a["resident_bytes"] == w.n_param*2 + 2*w.n_layers*127000*8*128*2
    assert a["max_slot_bytes"] <= a["slot_capacity_bytes"]
    assert a["max_group_bytes"] <= a["group_capacity_bytes"]
    assert a["max_die_bytes"] <= a["die_capacity_bytes"]
    assert a["slot_capacity_violations"] == 0
    for e in p.operators.values():
        assert e.layer_bytes(e.atom_count).sum() == e.atom_count*e.atom_bytes
        if e.unit.operator_type in ("ATTENTION_QK", "ATTENTION_AV"):
            # Each appended head vector stays in one resident group/layer.
            b = e.layer_bytes(126001*8, begin=126000*8)
            assert b.sum() == 8*256
            assert np.count_nonzero(b) == 8
            assert np.all(b[b>0] == 256)


@pytest.mark.parametrize("name", list(llama31_models()))
def test_policy_placement_traffic_caps_and_context(engines, name):
    e = engines[name]
    for policy in ExecutionPolicy:
        first = e.step(126000, policy, include_stages=True)
        last = e.step(126999, policy)
        assert last["latency_s"] > first["latency_s"]
        assert last["boundary_bytes"]/last["latency_s"] <= 3.4e12
        nmp = {s["operator"] for s in first["stages"] if s["executor"] == "NMP"}
        if policy == ExecutionPolicy.ATTENTION_NMP:
            assert nmp == {"ATTENTION_QK","ATTENTION_AV"}
        elif policy == ExecutionPolicy.NO_NMP:
            assert not nmp and first["NMP_average_utilization"] == 0
        else:
            assert {"Q","K","V","O","FFN_GATE","FFN_UP","FFN_DOWN","LM_HEAD"} <= nmp
        traffic = first["traffic_bytes"]
        assert (traffic["historical_K"] > 0) == (policy == ExecutionPolicy.NO_NMP)
        assert (traffic["historical_V"] > 0) == (policy == ExecutionPolicy.NO_NMP)
        assert (traffic["weight"] > 0) == (policy != ExecutionPolicy.MAC_NMP)
        assert traffic["KV_append"] == traffic["local_write"] == e.workload.n_layers*8*128*4
        assert first["fabric_region_peak_Bps"] <= 64e9
        assert 0 <= first["NMP_peak_utilization"] <= 1
        assert 0 <= first["noc_max_link_utilization"] <= 1
        assert not any("_J" in k or "energy" in k for k in last)


def test_compiled_service_equals_slot_sum(engines):
    e = engines["Llama-3.1-8B"]
    for context in (126000,126501,126999):
        op = e.placement.operators[0,"ATTENTION_QK"]
        point = e.physical.evaluate(op, context*8, nmp=True)
        explicit = group_service_seconds(op.layer_bytes(context*8), e.floorplan.service_ns[op.group_ids])
        assert point["array_service_s"] == pytest.approx(explicit.max(), rel=1e-12)


def test_active_tile_compute_not_die_pool(engines):
    e = engines["Llama-3.1-8B"]
    unit = SimpleNamespace(operator_type="Q", layer_id=0, local_flops=1024, activation_input_bytes=2, partial_output_bytes=2)
    op = ResidentOperator(unit,32,1,0,np.array([0]),np.array([0]),np.array([0]),318)
    stage = e.physical.evaluate(op,1,nmp=True)
    assert stage["active_groups"] == stage["active_mac_tiles"] == 1
    assert stage["components"]["MAC"] == 1024/32e9


def test_prefill_and_determinism(engines):
    for e in engines.values():
        pref = [e.prefill(CachedWorkload()) for _ in ExecutionPolicy]
        assert pref[0] == pref[1] == pref[2]
        assert pref[0]["external_bandwidth_cap_TBps"] == 3.4
        assert pref[0]["ledger"]["attention_pairs_per_request"] == 125500500
        for p in ExecutionPolicy:
            assert e.step(126000,p) == e.step(126000,p)
    assert list(CachedWorkload().contexts) == list(range(126000,127000))


def test_no_old_shortcut_or_energy_columns():
    from scripts import compare_decode_policies
    from om3dthermal.serving import decode_policy
    for module in (compare_decode_policies, decode_policy):
        source = inspect.getsource(module)
        assert "NMP_BANK_TO_LOCAL_ROUTE_DELAY_NS" not in source
        assert "203.5" not in source and "64.7e12" not in source
        assert "e_decode_J_per_bit" not in source
        assert "tok_J" not in source and '"gpu_J"' not in source


def test_model_architectures():
    for w,dims in zip(llama31_models().values(), ((4096,32,32,14336),(8192,80,64,28672),(16384,126,128,53248))):
        assert (w.d_model,w.n_layers,w.n_heads_q,w.d_ff) == dims
        assert w.n_heads_kv == 8 and w.d_head == 128
        assert w.weight_bits == w.kv_bits == 16
