"""Distributed port routing, physical caps and unchanged payload semantics."""
from pathlib import Path
import inspect
import numpy as np
import pytest
from om3dthermal.architecture.feol_floorplan import resolve_feol_floorplan, manhattan
from om3dthermal.power.nmp_die_activity import external_service, structured_noc, ingress_regions
from om3dthermal.serving.decode_policy import DecodePolicyModel, llama31_models, ExecutionPolicy

ROOT = Path(__file__).resolve().parents[1]
@pytest.fixture(scope="module")
def f():
    return resolve_feol_floorplan(ROOT)

@pytest.fixture(scope="module")
def engine():
    return DecodePolicyModel(llama31_models()["Llama-3.1-8B"], project_root=ROOT)


def test_port_ownership_and_all_routes(f):
    assert sorted(p for ids in f.region_port_ids for p in ids) == list(range(50))
    assert all(f.region_port_ids)
    for r,ids in enumerate(f.region_port_ids):
        x0,_,x1,_ = f.regions[r]["bounds_um"]
        for p in ids:
            assert x0 <= f.ports[p][0] <= x1
            length = manhattan(f.regions[r]["center_um"],f.ports[p])
            assert f.root_port_route_um[r,p] == length
            assert f.root_port_route_ns[r,p] == f.wire_ns(length)


@pytest.mark.parametrize("r",range(4))
def test_region_bulk_uses_whole_pool_and_slowest_route(f,r):
    b=np.zeros((318,4)); b[0,r]=1000000
    t=external_service(f,b,mode="REGION_DIRECT",details=True)
    assert t["active_port_count"] == len(f.region_port_ids[r]) > 1
    assert sum(p["bytes"] for p in t["ports"]) == pytest.approx(1000000)
    assert t["max_port_bytes"] == pytest.approx(1000000/len(f.region_port_ids[r]))
    expected=max(p["bytes"]/1e9+p["route_startup_s"] for p in t["ports"])
    assert t["external_service_s"] == expected
    assert t["external_service_s"] < 1000000/1e9
    assert t["limiting_reason"] == "PORT_SERIALIZATION"
    assert t["max_port_utilization"] <= 1


def test_global_cap_and_group_single_port(f):
    b=np.full((318,4),1000000.)
    t=external_service(f,b,mode="REGION_DIRECT")
    assert t["external_service_s"] == b.sum()/3.4e12
    assert t["active_port_count"] == 318*50
    assert t["limiting_reason"] == "GLOBAL_THERMAL_CAP"
    g=np.zeros((318,70)); g[0,0]=1000000
    t=external_service(f,g,mode="GROUP_DIRECT",details=True)
    assert t["active_port_count"] == 1
    assert t["external_service_s"] == 1e6/1e9+f.sa_edge_ns[0]*1e-9


def test_group_contention_and_route_startup(f):
    g=np.zeros((318,70)); g[0,:2]=1
    t=external_service(f,g,mode="GROUP_DIRECT",details=True)
    assert t["active_port_count"] == 1 and t["max_port_bytes"] == 2
    assert t["external_service_s"] == 2/1e9+max(f.sa_edge_ns[:2])*1e-9
    b=np.zeros((318,4)); b[0,2]=1
    t=external_service(f,b,mode="REGION_DIRECT")
    assert t["limiting_reason"] == "ROUTE_STARTUP"


@pytest.mark.parametrize("mask",range(1,16))
def test_one_copy_broadcast_and_deterministic_ingress(f,mask):
    active=np.array([[bool(mask&(1<<r)) for r in range(4)]])
    b=np.zeros((318,4)); b[0]=active[0]*8192
    t=external_service(f,b,mode="REGION_BROADCAST_INGRESS",details=True)
    root=int(ingress_regions(f,active)[0])
    assert active[0,root]
    positions=np.r_[0,np.cumsum([l["hop_ns"] for l in f.links])]
    dest=np.flatnonzero(active[0])
    assert root == min(dest,key=lambda r:(max(abs(positions[r]-positions[dest])),max(dest)-min(dest),r))
    assert t["total_boundary_bytes"] == 8192
    assert {p["port"] for p in t["ports"]} == set(f.region_port_ids[root])
    noc=structured_noc(f,b,mode="multicast")
    assert noc["hop_bytes"] == (max(dest)-min(dest))*8192
    reduction=structured_noc(f,active*16,mode="reduce",vector_bytes=16)
    assert reduction["root_regions"][0] in dest
    assert reduction["link_bytes"].shape == (1,3,2)


def test_operator_handoffs(engine):
    e=engine
    row=e.step(126000,ExecutionPolicy.MAC_NMP,include_stages=True)
    for s in row["stages"]:
        if "external_transfers" not in s: continue
        modes=[t["mode"] for t in s["external_transfers"]]
        if s["operator"]=="KV_APPEND":
            assert modes==["REGION_DIRECT"] and s["noc_bytes"]==0
        elif s["operator"]=="ATTENTION_AV":
            assert modes==["REGION_DIRECT","REGION_DIRECT"]
            assert s["output_boundary_bytes"]==s["active_dies"]*e.workload.d_model*4
            assert s["noc_bytes"]==s["active_dies"]*3*e.workload.d_model*4
            assert s["components"]["INTER_REGION_NOC"]>0
        else:
            assert modes==["REGION_BROADCAST_INGRESS","REGION_DIRECT"]
            op=e.placement.operators[s["layer"],s["operator"]]
            size=e.workload.d_model*2 if s["operator"]=="ATTENTION_QK" else op.unit.activation_input_bytes
            assert s["input_boundary_bytes"]==s["active_dies"]*size
            assert s["noc_bytes"] <= s["active_dies"]*size*3
            if s["operator"]=="ATTENTION_QK":
                assert s["output_boundary_bytes"]==126000*e.workload.n_heads_q*2


@pytest.mark.parametrize("name,expected",[
 ("Llama-3.1-8B",[31589998592,15777683456,1565917696]),
 ("Llama-3.1-70B",[180455129088,142844909568,7820067328]),
 ("Llama-3.1-405B",[872782229504,819595923968,24295316480])])
def test_payload_matches_pre_fix_matrix(name,expected):
    e=DecodePolicyModel(llama31_models()[name],project_root=ROOT)
    for p,want in zip(ExecutionPolicy,expected):
        first=e.step(126000,p); last=e.step(126999,p)
        assert (first["boundary_bytes"]+last["boundary_bytes"])/2==want
        for row in (first,last):
            traffic=row["traffic_bytes"]
            assert sum(v for k,v in traffic.items() if k not in ("local_read","local_write"))==row["boundary_bytes"]
            assert (traffic["historical_K"]>0)==(p==ExecutionPolicy.NO_NMP)
            assert (traffic["weight"]>0)==(p!=ExecutionPolicy.MAC_NMP)
        assert last==e.step(126999,p)


def test_no_hardcoded_handoff_root():
    from om3dthermal.power import nmp_die_activity
    source=inspect.getsource(nmp_die_activity)
    assert "input_root" not in source and "output_root" not in source
    assert "root_bytes" not in inspect.signature(external_service).parameters
