"""Canonical NMP hardware and physical-die activity closure."""
from __future__ import annotations
import math
import pytest
from scripts.evaluate_nmp_locality_placement import ROOT, run
from om3dthermal.power.nmp_die_activity import canonical_nmp_hardware
from om3dthermal.experiment import load_workload_spec
from om3dthermal.workload.llm_decode import evaluate_llm_decode
from scripts.evaluate_die_local_placement import _architecture
from om3dthermal.workload import build_m3d_workload_page_demand
from om3dthermal.placement.nmp_load_balance import build_performance_balanced_placement

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

def test_performance_balanced_allocator_is_deterministic(payload):
    layout,_=_architecture()
    workload=load_workload_spec(ROOT/"configs/workload/llama31_8b_decode_b1_s131072.yaml",project_root=ROOT).decode
    demand=build_m3d_workload_page_demand(workload,layout)
    kwargs={"bandwidth_per_die_bytes_per_s":payload["rows"][0]["canonical_die_activity"]["local_bandwidth_per_die_bytes_per_s"],
            "compute_per_die_flops_per_s":1.024e12}
    first=build_performance_balanced_placement(workload,demand,layout,**kwargs)
    second=build_performance_balanced_placement(workload,demand,layout,**kwargs)
    assert first.ownership==second.ownership

def test_request_level_kv_unit_and_workload_closures(payload):
    base=load_workload_spec(ROOT/"configs/workload/llama31_8b_decode_b1_s131072.yaml",project_root=ROOT).decode
    for row in payload["rows"]:
        workload=base.model_copy(update={"batch_size":row["requests"]}); metrics=evaluate_llm_decode(workload)
        loads=row["A_FINAL_CANONICAL_GAIN"]["placement"]["unit_loads"]
        request_units=[x for x in loads if x["unit"]["placement_scope"]=="REQUEST_LOCAL"]
        shared=[x for x in loads if x["unit"]["placement_scope"]=="SHARED_BATCH"]
        expected_one_layer=2*workload.context_length*workload.n_heads_kv*workload.d_head*workload.kv_bits/8
        assert len(request_units)==workload.n_layers*workload.batch_size
        assert len(shared)==workload.n_layers*7
        assert all(x["unit"]["request_id"] is None for x in shared)
        assert all(x["unit"]["kv_bytes"]==expected_one_layer for x in request_units)
        assert all(x["minimum_die_span"]==1 for x in request_units)
        assert sum(x["unit"]["kv_bytes"] for x in request_units)==pytest.approx(metrics.kv_footprint_bytes)
        assert sum(x["kv_read_bytes"] for x in request_units)==pytest.approx(workload.batch_size*metrics.kv_read_bytes_per_token)
        assert sum(x["kv_write_bytes"] for x in request_units)==pytest.approx(workload.batch_size*metrics.kv_write_bytes_per_token)
        expected_attention=workload.batch_size*workload.n_layers*4*workload.n_heads_q*workload.context_length*workload.d_head
        assert sum(x["nmp_flops"] for x in request_units)==expected_attention
        assert sum(x["local_memory_traffic_bytes"] for x in loads)==pytest.approx(sum(a["total_local_memory_bytes"] for a in row["A_FINAL_CANONICAL_GAIN"]["activity"]["activities"]))
        assert sum(x["nmp_flops"] for x in loads)==pytest.approx(sum(a["nmp_flops"] for a in row["A_FINAL_CANONICAL_GAIN"]["activity"]["activities"]))

def test_power_primitives_reuse_feol_and_exclude_external_bulk_terms(payload):
    primitive=payload["rows"][0]["B_PREP_DIE_POWER_MAP"]["primitives"]
    assert primitive["local_route_length_um"]==pytest.approx(.5*(primitive["cluster_width_um"]+primitive["cluster_height_um"]))
    expected=primitive["feol_activity_factor"]*primitive["feol_capacitance_fF_per_um"]*primitive["local_route_length_um"]*primitive["feol_voltage_V"]**2*1e-3
    assert primitive["local_route_energy_pj_per_bit"]==pytest.approx(expected)
    assert primitive["local_route_energy_pj_per_bit"]<primitive["long_feol_pj_per_bit"]
    assert primitive["local_read_total_pj_per_bit"]==pytest.approx(primitive["igzo_local_read_and_global_control_pj_per_bit"]+primitive["vertical_miv_pj_per_bit"]+primitive["local_route_energy_pj_per_bit"])
    assert primitive["local_write_total_pj_per_bit"]==pytest.approx(primitive["igzo_weighted_write_pj_per_bit"]+primitive["vertical_miv_pj_per_bit"]+primitive["local_route_energy_pj_per_bit"])
    assert primitive["nmp_logic_overhead_factor"]==1.0

def test_per_die_power_and_energy_closure(payload):
    refresh=[]
    for row in payload["rows"]:
        power=row["B_PREP_DIE_POWER_MAP"]; primitive=power["primitives"]; die=power["die_powers"]
        activity=row["A_FINAL_CANONICAL_GAIN"]["activity"]["activities"]
        assert len(die)==98 and power["power_component_double_count_gate"]=="PASS"
        assert sum(x["refresh_W"] for x in die)==pytest.approx(power["refresh_total_W"])
        refresh.append(tuple(x["refresh_W"] for x in die))
        assert all(x["nmp_logic_overhead_factor"]==1 and x["nmp_dynamic_W"]==pytest.approx(x["mac_dynamic_W"]) for x in die)
        assert all(math.isfinite(x["total_W"]) and x["total_W"]>=0 for x in die)
        assert sum(x["total_W"] for x in die)==pytest.approx(power["aggregate_total_W"])
        assert sum(x["memory_read_dynamic_W"] for x in die)==pytest.approx(power["aggregate_memory_read_dynamic_W"])
        assert sum(x["memory_write_dynamic_W"] for x in die)==pytest.approx(power["aggregate_memory_write_dynamic_W"])
        assert sum(x["mac_dynamic_W"] for x in die)==pytest.approx(power["aggregate_mac_dynamic_W"])
        interval=power["decode_step_interval_ms"]*1e-3
        expected_read=sum(8*(x["weight_read_bytes"]+x["kv_read_bytes"]) for x in activity)*primitive["local_read_total_pj_per_bit"]*1e-12/interval
        expected_write=sum(8*x["kv_write_bytes"] for x in activity)*primitive["local_write_total_pj_per_bit"]*1e-12/interval
        assert power["aggregate_memory_read_dynamic_W"]==pytest.approx(expected_read)
        assert power["aggregate_memory_write_dynamic_W"]==pytest.approx(expected_write)
        expected_mac_j=sum(x["nmp_flops"] for x in activity)/2*primitive["mac_energy_pj_per_mac"]*1e-12
        assert power["aggregate_mac_dynamic_W"]*interval==pytest.approx(expected_mac_j)
        assert power["residual_external_bytes"]==pytest.approx(row["A_FINAL_CANONICAL_GAIN"]["remaining_external_bytes"])
        expected_external=power["residual_external_bytes"]*8*(primitive["long_feol_pj_per_bit"]+primitive["interface_pj_per_bit"])*1e-12/(power["decode_step_interval_ms"]*1e-3)
        assert power["aggregate_residual_external_W"]==pytest.approx(expected_external)
    assert refresh[0]==refresh[1]==refresh[2]

def test_a_canonical_gains_unchanged_by_power_map(payload):
    assert [r["A_FINAL_CANONICAL_GAIN"]["combined_A_gain"] for r in payload["rows"]]==pytest.approx([2.565,3.950,3.744],abs=.001)

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
