"""Canonical NMP hardware and physical-die activity closure."""
from __future__ import annotations
import math
import pytest
from scripts.evaluate_nmp_locality_placement import ROOT, run
from om3dthermal.power.nmp_die_activity import canonical_nmp_hardware

@pytest.fixture(scope="module")
def payload(tmp_path_factory):
    return run(tmp_path_factory.mktemp("nmp_die_activity"))

def test_hardware_freeze_and_local_route(payload):
    row=payload["rows"][0]; h=row["canonical_nmp_hardware"]; a=row["canonical_die_activity"]
    assert h["macs_per_die"]==h["active_mac_ceiling_per_die"]==512
    assert h["clock_hz"]==1e9
    assert h["peak_flops_per_die"]==512*1e9*2==1.024e12
    assert h["aggregate_peak_flops"]==h["peak_flops_per_die"]*98
    assert a["local_route_delay_ns"]==1.0
    old_bw=(70*32/((a["local_access_latency_ns"]-1.0)*1e-9))*98
    assert a["aggregate_local_bandwidth_bytes_per_s"] < old_bw
    assert payload["rows"][0]["non_nmp_gpu"]["timing"]["external_bandwidth_bytes_per_s"]==4.9e12

def test_three_performance_references_are_explicit(payload):
    for row in payload["rows"]:
        ideal=row["IDEAL_AGGREGATE_UPPER_BOUND"]
        locality=row["LOCALITY_ONLY_BASELINE"]
        canonical=row["A_FINAL_CANONICAL_GAIN"]
        assert ideal["nmp_step_ms"]==pytest.approx(max(
            ideal["nmp_local_memory_ms"],ideal["nmp_compute_ms"]
        )+ideal["nmp_remaining_external_ms"])
        assert canonical["nmp_step_ms"]==pytest.approx(canonical["activity"]["decode_step_interval_ms"])
        assert canonical["combined_A_gain"]>1 and math.isfinite(canonical["combined_A_gain"])
        assert ideal["timing_semantics"]=="IDEAL_AGGREGATE_BALANCED_UPPER_BOUND"
        assert locality["timing_semantics"]=="REALIZED_DIE_LEVEL_LOCALITY_ONLY"
        assert "PERFORMANCE_BALANCED" in canonical["timing_semantics"]

def test_performance_balanced_constraints_and_closures(payload):
    for row in payload["rows"]:
        locality=row["LOCALITY_ONLY_BASELINE"]; balanced=row["A_FINAL_CANONICAL_GAIN"]
        lp=locality["placement"]; bp=balanced["placement"]
        assert bp["capacity_violations"]==0
        assert sum(bp["resident_used_bytes_per_die"])==pytest.approx(row["logical_working_set_bytes"])
        assert sum(bp["traffic_bytes_per_die"])==pytest.approx(sum(x["total_local_memory_bytes"] for x in balanced["activity"]["activities"]))
        assert sum(bp["flops_per_die"])==pytest.approx(sum(x["nmp_flops"] for x in balanced["activity"]["activities"]))
        assert bp["operator_die_spans"]==lp["operator_die_spans"]
        assert balanced["remaining_external_bytes"]==locality["remaining_external_bytes"]
        assert max(bp["service_time_ms_per_die"])<=max(lp["service_time_ms_per_die"])*(1+1e-12)
        assert max(bp["service_time_ms_per_die"])/(sum(bp["service_time_ms_per_die"])/98)<=max(lp["service_time_ms_per_die"])/(sum(lp["service_time_ms_per_die"])/98)*(1+1e-12)
        assert row["points"][0]["locality_aware"]["traffic"]["direct_die_to_die_bytes"]==0

def test_performance_balanced_allocator_is_deterministic(payload,tmp_path):
    repeated=run(tmp_path/"repeat")
    assert [r["A_FINAL_CANONICAL_GAIN"]["placement"]["ownership"] for r in repeated["rows"]]==[
        r["A_FINAL_CANONICAL_GAIN"]["placement"]["ownership"] for r in payload["rows"]]

@pytest.mark.parametrize("index",[0,1,2])
def test_per_die_activity_energy_and_service_closure(payload,index):
    row=payload["rows"][index]; a=row["canonical_die_activity"]; activities=a["activities"]; h=a["hardware"]
    demand=row["non_nmp_gpu"]["traffic"]
    expected_bytes=demand["weight_bulk_external_bytes"]+demand["kv_bulk_external_bytes"]
    assert sum(x["total_local_memory_bytes"] for x in activities)==pytest.approx(expected_bytes)
    expected_flops=sum(x["nmp_flops"] for x in activities)
    assert expected_flops>0
    assert a["global_nmp_stage_time_ms"]==pytest.approx(max(max(x["memory_service_time_ms"],x["compute_service_time_ms"]) for x in activities))
    assert all(0<=x["memory_utilization"]<=1 and 0<=x["compute_utilization"]<=1 for x in activities)
    for x in activities:
        assert x["memory_service_time_ms"]==pytest.approx(x["total_local_memory_bytes"]/a["local_bandwidth_per_die_bytes_per_s"]*1e3)
        assert x["compute_service_time_ms"]==pytest.approx(x["nmp_flops"]/h["peak_flops_per_die"]*1e3)
        assert math.isfinite(x["power"]["compute_dynamic_W"]) and x["power"]["compute_dynamic_W"]>=0
    assert sum(x["compute_energy_j"] for x in activities)==pytest.approx(a["aggregate_compute_energy_j"])
    assert sum(x["power"]["compute_dynamic_W"] for x in activities)==pytest.approx(a["aggregate_compute_dynamic_W"])
