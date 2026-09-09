"""Canonical per-die NMP activity and power closure."""
from pathlib import Path
from dataclasses import replace

import pytest

from om3dthermal.experiment import load_workload_spec
from om3dthermal.serving import evaluate_nmp_decode_batch
from om3dthermal.power.nmp_die_activity import evaluate_nmp_die_activity
from om3dthermal.workload.dense_decode_ledger import (
    attention_boundary_by_layer,
    boundary_bytes_per_die,
    build_dense_decode_handoffs,
    build_dense_decode_placement_units,
    build_dense_decode_small_ops,
)
from om3dthermal.workload import evaluate_llm_decode


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def payload():
    workload = load_workload_spec(
        ROOT / "configs/workload/llama31_8b_decode_b1_s131072.yaml",
        project_root=ROOT,
    ).decode
    result = evaluate_nmp_decode_batch(workload, project_root=ROOT)
    trace = result.execution_trace
    assert trace is not None
    return {
        "activity": trace.activity.as_dict(),
        "placement": trace.placement.as_dict(),
        "power_map": trace.power_map.as_dict(),
        "summary": result.model_dump(mode="json"),
        "workload": workload.model_dump(mode="json"),
        "trace": trace,
    }


def test_nominal_attention_boundary_and_softmax(payload):
    a=payload["activity"]; s=payload["summary"]
    assert s["qk_flops_per_step"] == s["av_flops_per_step"] == 2*32*131072*128*32
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
    assert transfer["memory_capability_bytes_per_s"] == 15.9e12
    assert transfer["bandwidth_actual_bytes_per_s"] == 2.4e12
    assert transfer["bandwidth_actual_bytes_per_s"] == min(transfer[k] for k in (
        "bandwidth_demand_bytes_per_s","memory_capability_bytes_per_s","gpu_peak_bandwidth_bytes_per_s"))
    assert a["boundary_time_ms"] == pytest.approx(a["residual_boundary_bytes"]/2.4e12*1e3)


def test_operator_and_die_closure(payload):
    a=payload["activity"]; p=payload["placement"]
    assert a["local_route_delay_ns"] == 1.0
    assert a["local_route_provenance"] == (
        "MODELING_CHOICE_FIXED_LOCAL_NMP_ROUTE_DELAY__NOT_PHYSICALLY_"
        "EXTRACTED__NOT_OPTIMIZED__NOT_POSITION_DEPENDENT")
    units=p["unit_loads"]
    assert sum(x["unit"]["weight_bytes"] for x in units) == 16e9
    assert sum(x["weight_read_bytes"] for x in units) == 15_009_325_056
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
        if unit["shard_mode"] in ("ROW_PARALLEL","KV_ATOMIC","LOCAL_LOOKUP"):
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
    expected=["RMSNORM_PRE_ATTENTION","RMSNORM_TO_Q_TRANSFER","RMSNORM_TO_K_TRANSFER",
              "RMSNORM_TO_V_TRANSFER","Q","Q_TO_ROPE_TRANSFER","K","K_TO_ROPE_TRANSFER",
              "V","V_TO_KV_REPACK_TRANSFER","V_KV_REPACK_RETURN_TRANSFER","ROPE",
              "ROPE_Q_RETURN_TRANSFER","ROPE_K_RETURN_TRANSFER","ATTENTION_QK",
              "SCORE_TRANSFER","GPU_SOFTMAX","PROBABILITY_TRANSFER","ATTENTION_AV",
              "PARTIAL_TRANSFER","AV_REDUCTION","AV_REDUCTION_TO_O_TRANSFER","O",
              "O_TO_RESIDUAL_TRANSFER","RESIDUAL_ADD_ATTENTION","RMSNORM_PRE_FFN",
              "RMSNORM_TO_FFN_GATE_TRANSFER","RMSNORM_TO_FFN_UP_TRANSFER","FFN_GATE",
              "FFN_GATE_TO_SWIGLU_TRANSFER","FFN_UP","FFN_UP_TO_SWIGLU_TRANSFER",
              "SWIGLU","SWIGLU_TO_FFN_DOWN_TRANSFER","FFN_DOWN",
              "FFN_DOWN_TO_RESIDUAL_TRANSFER","RESIDUAL_ADD_FFN"]
    for layer in range(32):
        rows=[x for x in a["stages"] if x["layer"]==layer]
        assert [x["operator"] for x in rows] == expected
        for row in rows:
            if "memory_ms" in row:
                assert row["time_ms"] == max(row["memory_ms"],row["compute_ms"])
    assert a["decode_step_interval_ms"] == pytest.approx(sum(x["time_ms"] for x in a["stages"]))
    assert a["decode_step_interval_ms"] == pytest.approx(a["global_nmp_stage_time_ms"]+a["boundary_time_ms"]+a["softmax_time_ms"]+a["gpu_remaining_time_ms"])
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
    assert payload["summary"]["J_per_token"] == pytest.approx(p["aggregate_total_W"]*seconds+a["softmax_dynamic_energy_j"]+a["gpu_remaining_dynamic_energy_j"]+74*seconds)
    assert p["power_component_double_count_gate"] == "PASS"


def test_small_op_dimensions_and_energy(payload):
    a=payload["activity"]
    rows=a["small_ops"]
    def total(operator): return sum(x["gpu_local_total_bytes"] for x in rows if x["operator"]==operator)
    assert total("RMSNORM")+total("FINAL_RMSNORM")==1_597_440
    assert sum(x["execution_count"] for x in rows if x["operator"] in ("RMSNORM","FINAL_RMSNORM"))==65
    assert total("ROPE")==655_360
    assert total("SWIGLU")==2_752_512
    assert total("RESIDUAL_ADD")==1_572_864
    assert total("AV_REDUCTION")==55_836_672
    assert total("SAMPLING")==256_516
    assert a["gpu_remaining_local_bytes"]==62_671_364
    assert a["gpu_remaining_time_ms"]==pytest.approx(a["gpu_remaining_local_bytes"]/2.4e12*1e3)
    assert a["gpu_remaining_dynamic_energy_j"]==pytest.approx(8*a["gpu_remaining_local_bytes"]*15.29e-12)
    assert a["embedding_local_read_bytes"]==8192


def test_explicit_handoffs_are_unique_and_rope_is_closed(payload):
    handoffs=payload["activity"]["handoffs"]
    keys=[(x["producer"],x["consumer"],x["layer_id"],x["request_id"]) for x in handoffs]
    assert len(keys)==len(set(keys))
    assert all(x["request_id"] is not None for x in handoffs)
    layer0=[x for x in handoffs if x["layer_id"]==0]
    assert sum(x["bytes"] for x in layer0 if x["producer"]=="Q" and x["consumer"]=="ROPE")==8192
    assert sum(x["bytes"] for x in layer0 if x["producer"]=="K" and x["consumer"]=="ROPE")==2048
    assert sum(x["bytes"] for x in layer0 if x["producer"]=="ROPE_Q")==106*8192
    assert sum(x["bytes"] for x in layer0 if x["producer"]=="ROPE_K")==2048
    assert not any(x["producer"]=="V" and x["consumer"]=="ROPE" for x in handoffs)
    assert any(x["producer"]=="V" and "not RoPE" in x["reason"] for x in handoffs)
    assert len([x for x in layer0 if x["producer"]=="FFN_GATE" and x["consumer"]=="SWIGLU"])==1
    assert len([x for x in layer0 if x["producer"]=="FFN_UP" and x["consumer"]=="SWIGLU"])==1
    assert len([x for x in layer0 if x["producer"]=="SWIGLU" and x["consumer"]=="FFN_DOWN"])==1
    assert len([x for x in handoffs if x["producer"]=="TOKEN_EMBED_LOOKUP"])==1
    assert sum(x["bytes"] for x in handoffs)==pytest.approx(payload["activity"]["residual_boundary_bytes"])
    logits=[x for x in handoffs if x["producer"]=="LM_HEAD" and x["consumer"]=="SAMPLING"]
    assert len(logits)==1 and logits[0]["bytes"]==128256*2
    sampling=next(x for x in payload["activity"]["small_ops"] if x["operator"]=="SAMPLING")
    assert sampling["gpu_local_read_bytes"]==logits[0]["bytes"]
    assert sampling["gpu_local_total_bytes"]==logits[0]["bytes"]+4


def test_local_service_is_independent_of_external_gpu_bandwidth(payload):
    trace = payload["trace"]
    activity = trace.activity
    reduced = replace(
        trace.architecture.bandwidth, coil_bandwidth_bytes_per_s=1.2e12)
    changed = evaluate_nmp_die_activity(
        trace.workload, trace.demand, trace.architecture.layout, reduced,
        local_access_latency_ns=activity.local_access_latency_ns,
        bandwidth_demand_bytes_per_s=1.2e12,
        ownership=trace.placement.ownership,
    )
    assert changed.local_bandwidth_per_die_bytes_per_s == pytest.approx(
        activity.local_bandwidth_per_die_bytes_per_s)
    assert changed.global_nmp_stage_time_ms == pytest.approx(
        activity.global_nmp_stage_time_ms)
    assert changed.boundary_time_ms == pytest.approx(2 * activity.boundary_time_ms)


def test_external_boundary_uses_canonical_transfer_resolver(payload, monkeypatch):
    import om3dthermal.power.nmp_die_activity as module
    from om3dthermal.platform import resolve_local_memory_gpu_transfer

    calls = []

    def record(**kwargs):
        calls.append(kwargs)
        return resolve_local_memory_gpu_transfer(**kwargs)

    monkeypatch.setattr(module, "resolve_local_memory_gpu_transfer", record)
    trace = payload["trace"]
    changed = evaluate_nmp_die_activity(
        trace.workload, trace.demand, trace.architecture.layout,
        trace.architecture.bandwidth,
        local_access_latency_ns=trace.activity.local_access_latency_ns,
        bandwidth_demand_bytes_per_s=1e12,
        ownership=trace.placement.ownership,
    )
    assert calls and calls[0]["bandwidth_demand_bytes_per_s"] == 1e12
    assert changed.transfer["bandwidth_actual_bytes_per_s"] == 1e12
    assert changed.boundary_time_ms == pytest.approx(
        changed.residual_boundary_bytes / 1e12 * 1e3)


def test_av_partial_is_resolved_per_layer_request(payload):
    trace = payload["trace"]
    workload = trace.workload.model_copy(
        update={"batch_size": 2, "context_length": 17})
    units = build_dense_decode_placement_units(workload)
    ownership = tuple(
        (0, 1) if unit.operator_type == "ATTENTION_AV" and unit.request_id == 0
        else (1, 2, 3) if unit.operator_type == "ATTENTION_AV"
        else (0,)
        for unit in units)
    layers = attention_boundary_by_layer(units, ownership)
    expected = workload.n_layers * (2 + 3) * workload.d_model * 4
    assert sum(row["partial_bytes"] for row in layers.values()) == expected
    assert all(
        row["partial_bytes"] == (2 + 3) * workload.d_model * 4
        for row in layers.values())
    small = build_dense_decode_small_ops(
        workload, units, ownership, trace.architecture.layout.slab_count)
    reductions = [row for row in small if row.operator == "AV_REDUCTION"]
    assert sum(row.gpu_local_total_bytes for row in reductions) == (
        workload.n_layers
        * ((2 + 3) * workload.d_model * 4 + 2 * workload.d_model * 2))


@pytest.mark.parametrize(
    "batch,context,kv_bits", [(1, 131072, 16), (2, 17, 8), (1, 0, 16)])
def test_workload_ledger_formulas(payload, batch, context, kv_bits):
    workload = payload["trace"].workload.model_copy(update={
        "batch_size": batch, "context_length": context, "kv_bits": kv_bits})
    metrics = evaluate_llm_decode(workload)
    units = build_dense_decode_placement_units(workload)
    qk = [unit for unit in units if unit.operator_type == "ATTENTION_QK"]
    av = [unit for unit in units if unit.operator_type == "ATTENTION_AV"]
    expected = (
        2 * batch * workload.n_heads_q * context * workload.d_head
        * workload.n_layers)
    assert sum(unit.local_flops for unit in qk) == expected
    assert sum(unit.local_flops for unit in av) == expected
    assert sum(
        unit.local_flops
        * (batch if unit.placement_scope == "SHARED_BATCH" else 1)
        for unit in units) == batch * metrics.flops_per_token
    assert sum(unit.weight_bytes for unit in units) == metrics.weight_footprint_bytes
    assert sum(unit.kv_bytes for unit in units) == (
        batch * metrics.kv_read_bytes_per_token)
    assert sum(unit.kv_write_bytes for unit in units) == (
        batch * metrics.kv_write_bytes_per_token)
    ownership = tuple((0, 1) for _ in units)
    handoffs = build_dense_decode_handoffs(units, ownership, 2)
    assert sum(boundary_bytes_per_die(units, ownership, 2)) == pytest.approx(
        sum(row.bytes for row in handoffs))
    keys = {
        (row.producer, row.consumer, row.layer_id, row.request_id)
        for row in handoffs}
    assert len(keys) == len(handoffs)
    assert all(row.request_id is not None for row in handoffs)
