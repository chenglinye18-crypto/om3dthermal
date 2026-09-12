"""Frozen performance and event-based accounting conservation gates."""
import json
from pathlib import Path
import numpy as np
import pytest
from om3dthermal.serving.decode_policy import DecodePolicyModel,ExecutionPolicy,llama31_models,CachedWorkload
from om3dthermal.power.feol_energy import FEOLEnergyModel,memory_events,sram_events,sum_events,power_groups
from om3dthermal.power.nmp_die_activity import group_service_seconds

ROOT=Path(__file__).resolve().parents[1]
@pytest.fixture(scope="module")
def engines():
    return {name:DecodePolicyModel(w,project_root=ROOT,record_energy=True) for name,w in llama31_models().items()}

@pytest.mark.parametrize("name",list(llama31_models()))
def test_observer_leaves_all_policies_and_prefill_identical(engines,name):
    e=engines[name]; off=DecodePolicyModel(e.workload,project_root=ROOT)
    m=FEOLEnergyModel(e.floorplan,e.platform)
    for p in ExecutionPolicy:
        for context in (126000,126999):
            before=off.step(context,p); after=e.step(context,p)
            assert before=={k:v for k,v in after.items() if k!="energy_events"}
            for adverse in (False,True):
                m.account(after["energy_events"],after["latency_s"],phase="decode",policy=p,adverse=adverse)
                assert before=={k:v for k,v in e.step(context,p).items() if k!="energy_events"}
    a=off.prefill(CachedWorkload()); b=e.prefill(CachedWorkload())
    assert a=={k:v for k,v in b.items() if k!="energy_events"}
    assert b==e.prefill(CachedWorkload())


def test_memory_service_events_and_layer_conservation():
    b=np.zeros((3,8),dtype=int);b[0,0]=33;b[1,7]=64
    read=memory_events(b,write=False);write=memory_events(b,write=True)
    assert read["read_services"]==4 and read["sa_sensed_bits"]==4*256
    assert read["row_select_events"]==read["column_select_events"]==4
    assert read["array_read_bits"]==sum(read["miv_read_bits_by_layer"])==97*8
    assert write["write_services"]==4 and write["sa_sensed_bits"]==0
    assert write["row_select_events"]==write["column_select_events"]==4
    assert write["write_driver_bits"]==write["array_write_bits"]==sum(write["miv_write_bits_by_layer"])==97*8
    tiny=np.zeros((2,8),dtype=int);tiny[:,0]=1
    assert memory_events(tiny,write=False)["read_services"]==2  # aggregate ceil would incorrectly give one
    with pytest.raises(ValueError): memory_events(-b,write=False)


def test_pe_capacity_chunking_per_tile_ceil(engines):
    e=next(iter(engines.values()));f=e.floorplan;c=f.energy_config
    assert c["pe_sram_bytes"]==1024 and f.config["macs_per_tile"]==16
    assert c["pe_sram_bytes"]*16==16384
    assert len(f.tiles)==32 and f.config["region_buffer_bytes"]==1024*16*8
    assert 32*16*1024==512*1024
    assert 318*32*16*1024==159*1024*1024
    assert c["psum_policy"]=="NO_PSUM_SPILL"
    v=sram_events([1,1,16385],capacity_bytes=16384,port_Bps=64e9,stage_s=1)
    assert v["sram_read32_accesses"]==v["sram_write32_accesses"]==1+1+4097
    assert v["max_tile_activation_chunk_bytes"]==16384 and v["max_tile_sram_utilization"]==1


@pytest.mark.parametrize("name",list(llama31_models()))
def test_physical_stage_events_and_streaming_bypass(engines,name):
    engine=engines[name];f=engine.floorplan;w=engine.workload
    point=engine.step(126000,ExecutionPolicy.MAC_NMP,include_stages=True)
    for s in point["stages"]:
        if "energy_events" not in s:continue
        e=s["energy_events"]
        assert e["interface_bits"]==e["gpu_decode_proxy_bits"]==s["boundary_bytes"]*8
        assert sum(e["miv_read_bits_by_layer"])+sum(e["miv_write_bits_by_layer"])==s["local_array_bytes"]*8
        assert e["mac_operations"]==s["nmp_flops"]/2
        assert e["sa_sensed_bits"]==e["read_services"]*256
        if s["operator"] in ("TOKEN_EMBED_LOOKUP","KV_APPEND"):
            assert e["mac_operations"]==e["sram_read32_accesses"]==e["sram_write32_accesses"]==0
        if s["operator"]=="KV_APPEND":
            assert e["sa_sensed_bits"]==0
            continue
        entry=engine.placement.operators[s["layer"],s["operator"]]
        atoms=126000*8 if s["operator"].startswith("ATTENTION") else 1 if s["operator"]=="TOKEN_EMBED_LOOKUP" else entry.atom_count
        b=entry.layer_bytes(atoms)
        assert int(((b+31)//32).sum())==e["read_services"]
        assert group_service_seconds(b,f.service_ns[entry.group_ids]).max()==pytest.approx(s["array_service_s"],rel=1e-12)
        if s["operator"] not in ("ATTENTION_AV","TOKEN_EMBED_LOOKUP"):
            size=w.d_model*2 if s["operator"]=="ATTENTION_QK" else entry.unit.activation_input_bytes
            assert e["activation_sram_bytes"]==size*s["active_mac_tiles"]
        if s["operator"] in ("ATTENTION_QK","ATTENTION_AV"):
            assert e["kv_bypass_sram_bytes"]==s["local_array_bytes"]
        elif s["operator"]!="TOKEN_EMBED_LOOKUP":
            assert e["weight_bypass_sram_bytes"]==s["local_array_bytes"]
        assert e["sram_read32_accesses"]==e["sram_write32_accesses"]
        assert e["max_tile_activation_chunk_bytes"]<=16384
        links=s["noc_link_busy_s"]*f.link_Bps
        assert e["noc_link_bit_um"]==pytest.approx(float(np.sum(links*np.array([x["length_um"] for x in f.links])[None,:,None]))*8)
        assert e["pipeline_register_bit_stages"]==pytest.approx(float(np.sum(links*np.array([max(x["wire_pipeline_cycles"]-1,0) for x in f.links])[None,:,None]))*8)
        output_source_bytes=s["active_mac_tiles"]*w.d_model*4 if s["operator"]=="ATTENTION_AV" else s["output_boundary_bytes"]
        assert e["router_bit_traversals"]==(s["local_array_bytes"]+e["activation_sram_bytes"]+output_source_bytes+s["noc_bytes"])*8
        counts=entry.prefix_counts(atoms)
        expected_wire=float(np.sum(counts*entry.atom_bytes*8*f.sa_tile_um[entry.group_ids,entry.tile_ids]))
        assert e["sa_to_tile_bit_um"]==pytest.approx(expected_wire,rel=1e-14)
        if s["operator"]=="ATTENTION_AV":
            assert e["fp32_reduction_adds"]==(s["active_mac_tiles"]-s["active_dies"])*w.d_model


@pytest.mark.parametrize("name",list(llama31_models()))
@pytest.mark.parametrize("policy",list(ExecutionPolicy))
def test_gpu_proxy_unresolved_components_and_adverse(engines,name,policy):
    e=engines[name];m=FEOLEnergyModel(e.floorplan,e.platform)
    s=e.step(126000,policy);events=s["energy_events"];t=s["latency_s"]
    a=m.account(events,t,phase="decode",policy=policy)
    b=m.account(events,t,phase="decode",policy=policy,adverse=True)
    c=a["components"]
    assert c["gpu_dynamic_J"]==events["interface_bits"]*11.68e-12
    assert c["gpu_static_J"]==74*t
    assert c["mac_J"]==events["mac_operations"]*2e-12
    assert c["sram_read_J"]==events["sram_read32_accesses"]*12e-12
    assert c["reduction_J"]==events["fp32_reduction_adds"]*1e-12
    assert sum(c.values())==a["total_J"]
    assert power_groups(c,t)["total_W"]==pytest.approx(a["total_J"]/t,rel=1e-14)
    if policy==ExecutionPolicy.NO_NMP:
        assert a==b and c["feol_unresolved_J"]==c["mac_J"]==c["sram_read_J"]==c["router_J"]==0
    else:
        assert c["feol_unresolved_J"]==pytest.approx(31.8*t)
        assert b["components"]["feol_unresolved_J"]==pytest.approx(95.4*t)
        assert b["total_J"]>a["total_J"]
        for k in ("array_read_J","sense_amplifier_J","miv_J","feol_wire_J","interface_J","gpu_dynamic_J"):
            assert c[k]==b["components"][k]
    assert a==m.account(events,t,phase="decode",policy=policy)


@pytest.mark.parametrize("name",list(llama31_models()))
def test_prefill_components_and_no_decode_proxy(engines,name):
    e=engines[name];m=FEOLEnergyModel(e.floorplan,e.platform);p=e.prefill(CachedWorkload())
    rows=[m.account(p["energy_events"],p["latency_s"],phase="prefill",policy=x,total_flops=p["ledger"]["total_flops"]) for x in ExecutionPolicy]
    assert rows[0]==rows[1]==rows[2]
    c=rows[0]["components"]
    assert c["gpu_dynamic_J"]==p["ledger"]["total_flops"]*(4.557857503789793e-13+6.326427488630622e-13)/2
    assert c["gpu_static_J"]==74*p["latency_s"]
    assert c["interface_J"]==p["ledger"]["total_memory_bytes"]*8*.5e-12
    assert p["energy_events"]["gpu_decode_proxy_bits"]==0
    for k in ("mac_J","sram_read_J","sram_write_J","router_J","pipeline_register_J","reduction_J","feol_unresolved_J"):assert c[k]==0
    assert rows[0]["total_J"]==sum(c.values())
    assert p["energy_events"]["array_read_bits"]==p["ledger"]["total_read_bytes"]*8
    assert p["energy_events"]["array_write_bits"]==p["ledger"]["total_write_bytes"]*8


def test_coefficients_miv_and_boundaries(engines):
    e=next(iter(engines.values()));f=e.floorplan;m=FEOLEnergyModel(f,e.platform)
    assert m.read_pj==pytest.approx(.18430) and m.write_pj==pytest.approx(.0003725)
    assert not f.case.memory.cell_model.operation_energy_provenance.sensing_included
    assert np.all(np.diff(f.miv_pj_per_bit_by_layer)>0)
    assert f.energy_config["memory_static_W"]==f.energy_config["refresh_background_W"]==0
    for v in m.audit()["parameters"].values():
        assert {"value","unit","classification","source","accounting_boundary","included_components","excluded_components"} <= v.keys()


def test_external_wire_uses_actual_port_striping(engines):
    from om3dthermal.power.nmp_die_activity import external_service
    f=next(iter(engines.values())).floorplan
    b=np.zeros((318,4));b[0,2]=13000
    t=external_service(f,b,mode="REGION_DIRECT",details=True,record_events=True)
    expected=sum(p["bytes"]*8*f.root_port_route_um[2,p["port"]] for p in t["ports"])
    assert t["wire_bit_um"]==pytest.approx(expected,rel=1e-14)
    g=np.zeros((318,70));g[0,4]=32
    t=external_service(f,g,mode="GROUP_DIRECT",record_events=True)
    assert t["wire_bit_um"]==32*8*f.sa_edge_um[4]
