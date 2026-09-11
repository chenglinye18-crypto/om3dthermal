"""Scientific placement, traffic, precision, and canonical execution gates."""
from pathlib import Path

import pytest

from om3dthermal.serving.decode_policy import CachedWorkload, DecodePolicyModel, ExecutionPolicy, llama31_models
from om3dthermal.power.nmp_die_activity import evaluate_nmp_die_activity
from om3dthermal.workload import build_m3d_workload_page_demand

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def engines():
    return {name: DecodePolicyModel(w, project_root=ROOT) for name, w in llama31_models().items()}


@pytest.mark.parametrize("name", list(llama31_models()))
def test_attention_placement_and_traffic(engines, name):
    engine = engines[name]
    row = engine.step(126000, ExecutionPolicy.ATTENTION_NMP, include_stages=True)
    nmp = {s["operator"] for s in row["stages"] if s["kind"] == "NMP_STAGE"}
    gpu = {s["operator"] for s in row["stages"] if s["kind"] == "GPU_STAGE"}
    assert nmp == {"ATTENTION_QK", "ATTENTION_AV"}
    assert {"Q", "K", "V", "O", "FFN_GATE", "FFN_UP", "FFN_DOWN", "LM_HEAD", "TOKEN_EMBED_LOOKUP"} <= gpu
    t = row["traffic_bytes"]
    assert t["historical_K"] == t["historical_V"] == 0
    assert t["weight"] == engine.canonical.weight_read_bytes_per_step
    assert all(t[k] > 0 for k in ("Q", "score_T", "softmax_S", "attention_O", "KV_append"))
    assert t["FFN_activation"] == 0
    assert t["local_write"] == engine.canonical.kv_write_bytes_per_step


@pytest.mark.parametrize("name", list(llama31_models()))
def test_no_nmp_and_energy_closure(engines, name):
    engine = engines[name]
    w = engine.workload
    row = engine.step(126000, ExecutionPolicy.NO_NMP, include_stages=True)
    assert not any(s["kind"] == "NMP_STAGE" for s in row["stages"])
    assert row["nmp_J"] == row["average_nmp_utilization"] == 0
    t = row["traffic_bytes"]
    assert t["historical_K"] == t["historical_V"] == w.n_layers*126000*w.n_heads_kv*w.d_head*2
    assert t["weight"] == engine.canonical.weight_read_bytes_per_step
    for policy in ExecutionPolicy:
        r = engine.step(126000, policy)
        assert r["total_J"] == sum(r[k] for k in ("gpu_J", "memory_J", "boundary_J", "nmp_J", "memory_static_refresh_J"))
        assert r["memory_static_refresh_J"] == 0
        assert r["traffic_bytes"]["local_read"] == t["local_read"]
        assert r["traffic_bytes"]["local_write"] == t["local_write"]


@pytest.mark.parametrize("name", list(llama31_models()))
def test_mac_canonical_regression(engines, name):
    engine = engines[name]
    row = engine.step(127000, ExecutionPolicy.MAC_NMP, include_stages=True)
    c = engine.canonical
    assert row["latency_s"] == pytest.approx(c.decode_step_time_ms*1e-3, rel=1e-12)
    assert row["boundary_bytes"] == c.boundary_bytes_per_step
    assert row["memory_J"] == pytest.approx(c.memory_dynamic_J_per_step, rel=1e-12)
    assert row["nmp_J"] == pytest.approx(c.mac_dynamic_J_per_step, rel=1e-12)
    assert row["gpu_dynamic_J"] == pytest.approx(c.gpu_dynamic_J_per_step, rel=1e-12)
    assert row["total_J"] == pytest.approx(c.total_J_per_step-c.refresh_J_per_step, rel=1e-12)
    assert row["traffic_bytes"]["weight"] == 0
    assert len(row["stages"]) == len(engine.activity.stages)
    for got, original in zip(row["stages"], engine.activity.stages, strict=True):
        assert got["operator"] == original["operator"]
        assert got["time_ms"] == pytest.approx(original["time_ms"], rel=1e-12)


def test_growing_context_against_uncompiled_canonical(engines):
    e = engines["Llama-3.1-8B"]
    trace = e.canonical.execution_trace
    for context in (126000, 126501, 126999):
        w = e.workload.model_copy(update={"context_length": context})
        demand = build_m3d_workload_page_demand(w, e.architecture.layout)
        activity = evaluate_nmp_die_activity(
            w, demand, e.architecture.layout, e.architecture.bandwidth,
            local_access_latency_ns=e.activity.local_access_latency_ns,
            bandwidth_demand_bytes_per_s=e.architecture.bandwidth.coil_bandwidth_bytes_per_s,
            ownership=trace.placement.ownership)
        compiled = e.step(context, ExecutionPolicy.MAC_NMP)
        assert compiled["latency_s"] == pytest.approx(activity.decode_step_interval_ms*1e-3, rel=1e-12)
        assert compiled["boundary_bytes"] == activity.residual_boundary_bytes


def test_workload_precision_capacity_and_determinism(engines):
    workload = CachedWorkload()
    assert len(workload.contexts) == 1000
    assert workload.contexts[0] == 126000 and workload.contexts[-1] == 126999
    for e in engines.values():
        assert e.workload.weight_bits == e.workload.kv_bits == 16
        assert e.activity.hardware.precision == "FP16"
        assert e.activity.hardware.macs_per_die == 512
        assert e.activity.hardware.aggregate_peak_flops == 325.632e12
        assert e.architecture.layout.slab_count == 318
        assert e.architecture.case.geometry.orthogonal.slab_pitch_x_um == 100
        assert e.canonical.capacity_status == "FEASIBLE"
        assert e.canonical.physical_page_rounded_capacity_GB < 1493.84331264
        prefill = [e.prefill(workload) for _ in ExecutionPolicy]
        assert prefill[0] == prefill[1] == prefill[2]
        ledger = prefill[0]["ledger"]
        assert ledger["linear_ffn_tokens_computed_per_request"] == 1000
        assert ledger["cached_history_tokens"] == 125000
        assert ledger["attention_pairs_per_request"] == 125500500
        for policy in ExecutionPolicy:
            first = e.step(workload.contexts[0], policy)
            last = e.step(workload.contexts[-1], policy)
            assert last == e.step(workload.contexts[-1], policy)
            assert last["latency_s"] > first["latency_s"]
            assert last["traffic_bytes"]["local_read"] > first["traffic_bytes"]["local_read"]


def test_meta_architectures():
    for w, expected in zip(llama31_models().values(),
                           ((4096, 32, 32, 14336), (8192, 80, 64, 28672), (16384, 126, 128, 53248)), strict=True):
        assert (w.d_model, w.n_layers, w.n_heads_q, w.d_ff) == expected
        assert w.n_heads_kv == 8 and w.d_head == 128
        assert w.context_length < 131072


def test_prerun_capacity_audit():
    from scripts.compare_decode_policies import audit
    setup = audit()
    assert setup["bytes_weight"] == setup["bytes_KV"] == setup["bytes_activation"] == 2
    for capacity in setup["capacities"].values():
        assert capacity["required_resident_GB"] < capacity["available_GB"]
        assert capacity["workspace_GB"] >= capacity["prefill_workspace_GB"]
        assert capacity["workspace_GB"] >= capacity["decode_workspace_GB"]


def test_reject_mixed_precision(engines):
    w = engines["Llama-3.1-8B"].workload.model_copy(update={"weight_bits": 1})
    with pytest.raises(ValueError, match="uniform 16-bit"):
        DecodePolicyModel(w, project_root=ROOT)
