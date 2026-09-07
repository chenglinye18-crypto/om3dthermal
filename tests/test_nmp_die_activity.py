"""Analytical dense attention E2E closure; no architecture/throughput sweep."""
import math
import pytest
from scripts.evaluate_nmp_locality_placement import run


@pytest.fixture(scope="module")
def payload(tmp_path_factory):
    return run(tmp_path_factory.mktemp("nmp_attention"))


def test_nominal_attention_boundary_and_softmax(payload):
    a=payload["activity"]; s=payload["summary"]
    assert s["qk_flops_per_token"] == s["av_flops_per_token"] == 2*32*131072*128*32
    assert a["score_bytes"] == a["probability_bytes"] == 32*131072*2*32
    placement=payload["placement"]
    expected_partial=0
    for layer in range(32):
        owners=set(d for x,o in zip(placement["unit_loads"],placement["ownership"])
                   if x["unit"]["layer_id"]==layer and x["unit"]["operator_type"]=="ATTENTION_AV" for d in o)
        expected_partial+=len(owners)*4096*4
    assert a["partial_bytes"] == expected_partial
    assert a["attention_boundary_bytes"] == a["score_bytes"]+a["probability_bytes"]+a["partial_bytes"]
    assert a["softmax_local_bytes"] == a["score_bytes"]+a["probability_bytes"]
    assert a["softmax_time_ms"] == pytest.approx(a["softmax_local_bytes"]/2.4e12*1e3)
    assert a["softmax_dynamic_energy_j"] == pytest.approx(8*a["softmax_local_bytes"]*15.29e-12)
    transfer=a["transfer"]
    assert transfer["memory_capability_bytes_per_s"] == 5.3e12
    assert transfer["bandwidth_actual_bytes_per_s"] == 2.4e12
    assert transfer["bandwidth_actual_bytes_per_s"] == min(transfer[k] for k in (
        "bandwidth_demand_bytes_per_s","memory_capability_bytes_per_s","gpu_peak_bandwidth_bytes_per_s"))
    assert a["boundary_time_ms"] == pytest.approx(a["residual_boundary_bytes"]/2.4e12*1e3)


def test_operator_and_die_closure(payload):
    a=payload["activity"]; p=payload["placement"]
    units=p["unit_loads"]
    assert sum(x["unit"]["weight_bytes"] for x in units) == 16e9
    assert sum(x["weight_read_bytes"] for x in units) == 15_009_316_864
    other=next(x for x in units if x["unit"]["operator_type"]=="OTHER_WEIGHT")
    assert other["unit"]["weight_bytes"] == 990_683_136
    assert other["weight_read_bytes"] == other["nmp_flops"] == 0
    assert other["unit"]["traffic_provenance"] == "RESIDENT_FOOTPRINT_RESIDUAL_ONLY__NOT_ACTIVE_FULL_READ_TRAFFIC"
    assert sum(x["kv_read_bytes"] for x in units) == 2*32*131072*8*128*2
    assert sum(x["kv_write_bytes"] for x in units) == 2*32*8*128*2
    assert sum(x["nmp_flops"] for x in units) == sum(x["nmp_flops"] for x in a["activities"])
    assert p["capacity_violations"] == 0
    assert sum(p["resident_used_bytes_per_die"]) == pytest.approx(16e9+2*32*131072*8*128*2+payload["workload"]["runtime_bytes"])
    hw=a["hardware"]
    assert hw["macs_per_die"] == 512 and hw["clock_hz"] == 1e9
    assert hw["aggregate_peak_flops"] == 108.544e12
    assert hw["mac_energy_pj"] == .604
    for row in a["activities"]:
        assert row["memory_service_time_ms"] == pytest.approx(row["total_local_memory_bytes"]/a["local_bandwidth_per_die_bytes_per_s"]*1e3)
        assert row["compute_service_time_ms"] == pytest.approx(row["nmp_flops"]/1.024e12*1e3)
        assert row["active_service_time_ms"] == max(row["memory_service_time_ms"],row["compute_service_time_ms"])


def test_row_and_kv_shards_close_without_splitting_atomic_vectors(payload):
    p=payload["placement"]
    for load,shards in zip(p["unit_loads"],p["shard_assignments"],strict=True):
        assert sum(x["resident_bytes"] for x in shards)==pytest.approx(load["resident_bytes"])
        assert sum(x["weight_read_bytes"] for x in shards)==pytest.approx(load["weight_read_bytes"])
        assert sum(x["nmp_flops"] for x in shards)==pytest.approx(load["nmp_flops"])
        unit=load["unit"]
        if unit["shard_mode"] in ("ROW_PARALLEL","KV_ATOMIC"):
            assert sum(x["shard_count"] for x in shards)==unit["atomic_count"]
        if unit["shard_mode"]=="ROW_PARALLEL":
            assert unit["output_rows"]==sum(x["shard_count"] for x in shards)
        if unit["shard_mode"]=="KV_ATOMIC" and unit["atomic_count"]:
            for shard in shards:
                assert shard["kv_read_bytes"]==pytest.approx(
                    shard["shard_count"]*unit["atomic_locality_bytes"])
    keyed={(x["unit"]["layer_id"],x["unit"]["request_id"],x["unit"]["operator_type"]):o
           for x,o in zip(p["unit_loads"],p["ownership"],strict=True)}
    for layer in range(32):
        assert keyed[layer,0,"ATTENTION_QK"]==keyed[layer,0,"ATTENTION_AV"]


def test_stage_parallelism_regression_gates(payload):
    p=payload["placement"]
    qk=next((load,index) for index,load in enumerate(p["unit_loads"])
            if load["unit"]["operator_type"]=="ATTENTION_QK")
    load,index=qk
    assert p["operator_die_spans"][index] > load["minimum_die_span"]
    assert payload["activity"]["mean_exec_die_span"]>1
    # Fixed stage workload must partition monotonically as span grows.
    total=load["local_memory_traffic_bytes"]
    flops=load["nmp_flops"]
    atomic=load["unit"]["atomic_count"]
    times=[]
    for span in (1,2,4,8,16,32,64,106):
        q,r=divmod(atomic,span)
        fraction=(q+(r>0))/atomic
        times.append(max(total*fraction/payload["activity"]["local_bandwidth_per_die_bytes_per_s"],
                         flops*fraction/1.024e12))
    assert times==sorted(times,reverse=True)


def test_serial_stage_dependencies(payload):
    a=payload["activity"]
    expected=["Q","K","V","ATTENTION_QK","SCORE_TRANSFER","GPU_SOFTMAX",
              "PROBABILITY_TRANSFER","ATTENTION_AV","PARTIAL_TRANSFER","O","FFN_GATE","FFN_UP","FFN_DOWN"]
    for layer in range(32):
        rows=[x for x in a["stages"] if x["layer"]==layer]
        assert [x["operator"] for x in rows if not x["operator"].endswith("_GPU_BOUNDARY")] == expected
        for row in rows:
            if "memory_ms" in row:
                assert row["time_ms"] == max(row["memory_ms"],row["compute_ms"])
    assert a["decode_step_interval_ms"] == pytest.approx(sum(x["time_ms"] for x in a["stages"]))
    assert a["decode_step_interval_ms"] == pytest.approx(a["global_nmp_stage_time_ms"]+a["boundary_time_ms"]+a["softmax_time_ms"])
    assert a["global_nmp_stage_time_ms"] > max(x["active_service_time_ms"] for x in a["activities"])


def test_power_boundaries_and_static_once(payload):
    a=payload["activity"]; p=payload["power_map"]; primitive=p["primitives"]
    seconds=a["decode_step_interval_ms"]*1e-3
    assert a["gpu_static_energy_j"] == pytest.approx(74*seconds)
    assert p["residual_external_bytes"] == pytest.approx(a["residual_boundary_bytes"])
    assert p["aggregate_residual_external_W"]*seconds == pytest.approx(8*a["residual_boundary_bytes"]*(primitive["long_feol_pj_per_bit"]+primitive["interface_pj_per_bit"])*1e-12)
    assert primitive["local_read_total_pj_per_bit"] == pytest.approx(primitive["igzo_local_read_and_global_control_pj_per_bit"]+primitive["vertical_miv_pj_per_bit"]+primitive["local_route_energy_pj_per_bit"])
    active_read=sum(x["weight_read_bytes"]+x["kv_read_bytes"] for x in a["activities"])
    assert p["aggregate_memory_read_dynamic_W"]*seconds == pytest.approx(
        8*active_read*primitive["local_read_total_pj_per_bit"]*1e-12)
    assert p["aggregate_mac_dynamic_W"]*seconds == pytest.approx(sum(x["nmp_flops"] for x in a["activities"])/2*.604e-12)
    assert p["aggregate_total_W"] == pytest.approx(sum(p[k] for k in ("aggregate_memory_read_dynamic_W","aggregate_memory_write_dynamic_W","aggregate_mac_dynamic_W","aggregate_refresh_W","aggregate_residual_external_W")))
    assert payload["summary"]["J_per_token"] == pytest.approx(p["aggregate_total_W"]*seconds+a["softmax_dynamic_energy_j"]+74*seconds)
    assert p["power_component_double_count_gate"] == "PASS"
    assert math.isfinite(payload["summary"]["energy_efficiency_gain"])
