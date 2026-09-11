"""Run the fixed 3-model x 3-placement cached-history comparison."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import yaml

from om3dthermal.power.nmp_die_activity import canonical_nmp_hardware
from om3dthermal.platform import load_platform_spec_file
from om3dthermal.serving.decode_policy import CachedWorkload, DecodePolicyModel, ExecutionPolicy, llama31_models
from om3dthermal.serving.nmp_decode import resolve_m3d_architecture_backend
from om3dthermal.serving.workspace import WorkspaceExecutionConfig, evaluate_decode_workspace, evaluate_prefill_workspace
from om3dthermal.workload import DenseLLMModelSpec

ROOT = Path(__file__).resolve().parents[1]


def audit():
    a = resolve_m3d_architecture_backend(ROOT)
    platform = load_platform_spec_file(ROOT / "configs/platform/gpu_package_h200_reference.yaml")
    hw = canonical_nmp_hardware(a.layout.slab_count)
    assert a.layout.slab_count == 318
    assert a.case.geometry.orthogonal.slab_pitch_x_um == 100
    assert a.bandwidth.coil_bandwidth_bytes_per_s == 15.9e12
    models = llama31_models()
    config_path = ROOT/"configs/experiment/formal_e2e_benchmark.yaml"
    workspace_config = WorkspaceExecutionConfig(**yaml.safe_load(config_path.read_text(encoding="utf-8"))["workspace"])
    capacities = {}
    for name, w in models.items():
        kv_token = 2*w.n_layers*w.n_heads_kv*w.d_head*2
        model_spec = DenseLLMModelSpec(model_id=name, model_spec_status="RESOLVED", context_status="131072",
                                      **{k:v for k,v in w.model_dump().items() if k in DenseLLMModelSpec.model_fields})
        prefill_workspace = evaluate_prefill_workspace(model_spec, batch_size=1, context_length=1000, config=workspace_config)
        decode_workspace = evaluate_decode_workspace(model_spec, batch_size=1, context_length=127000,
                                                      config=workspace_config, proposed_nmp=True, nmp_die_count=318)
        workspace = max(prefill_workspace.peak_bytes, decode_workspace.peak_bytes)
        total = w.n_param*2+kv_token*127000+workspace
        assert total <= a.layout.total_capacity_bytes
        capacities[name] = dict(weights_GB=w.n_param*2/1e9, kv_126K_GB=kv_token*126000/1e9,
                                kv_127K_GB=kv_token*127000/1e9, workspace_GB=workspace/1e9,
                                prefill_workspace_GB=prefill_workspace.peak_bytes/1e9,
                                decode_workspace_GB=decode_workspace.peak_bytes/1e9,
                                required_resident_GB=total/1e9, available_GB=a.layout.total_capacity_bytes/1e9)
    return dict(git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                geometry=a.case.name, slab_thickness_um=100, slab_count=a.layout.slab_count,
                capacity_GB=a.layout.total_capacity_bytes/1e9, raw_BW_TB_s=a.bandwidth.coil_bandwidth_bytes_per_s/1e12,
                gpu=platform.gpu_compute_power.model_dump(), gpu_decode=platform.gpu_decode_power.model_dump(),
                gpu_prefill=platform.gpu_prefill_compute.model_dump(), nmp_MAC_per_slab=hw.macs_per_die,
                nmp_clock_Hz=hw.clock_hz, nmp_peak_TFLOP_s=hw.aggregate_peak_flops/1e12,
                nmp_pJ_MAC=hw.mac_energy_pj, MAC_precision=hw.precision, GPU_precision="BF16 dense",
                bytes_weight=2, bytes_KV=2, bytes_activation=2, bytes_partial_accumulator=4,
                memory_static_refresh_W=0, memory_static_policy="frozen llm_decode_workload_power exclusion",
                workload="B1; resident cached 125000 + GPU Prefill 1000 + Decode 1000; S_j=126000+j, j=0..999",
                max_context=131072, policies=list(ExecutionPolicy), workspace_config=workspace_config.model_dump(),
                model_source="https://github.com/meta-llama/llama-models/blob/main/models/sku_list.py",
                models={name:w.model_dump() for name,w in models.items()}, capacities=capacities)


def run(out: Path):
    setup = audit()
    print("PRE-RUN AUDIT\n"+json.dumps(setup, indent=2), flush=True)
    out.mkdir(parents=True, exist_ok=True)
    (out/"audit.json").write_text(json.dumps(setup, indent=2)+"\n", encoding="utf-8")
    workload = CachedWorkload()
    summary, traffic_rows, step_rows, prefill_rows = [], [], [], {}
    for name, model in llama31_models().items():
        print(f"Preparing canonical resident placement: {name}", flush=True)
        engine = DecodePolicyModel(model, project_root=ROOT)
        prefill = engine.prefill(workload)
        prefill_rows[name] = prefill
        for policy in ExecutionPolicy:
            steps = [engine.step(context, policy) for context in workload.contexts]
            repeat = engine.step(workload.contexts[-1], policy)
            assert repeat == steps[-1], "deterministic rerun failed"
            decode_s = sum(s["latency_s"] for s in steps)
            decode_j = sum(s["total_J"] for s in steps)
            e2e_s = prefill["latency_s"]+decode_s
            # Keep the existing compute-energy envelope. Main columns use its
            # upper-energy endpoint; no new nominal coefficient is selected.
            e2e_j = decode_j+prefill["total_J_max"]
            row = dict(model=name, policy=policy, resident_GB=setup["capacities"][name]["required_resident_GB"],
                       prefill_s=prefill["latency_s"], decode_s=decode_s,
                       decode_tok_s=1000/decode_s, e2e_tok_s=1000/e2e_s,
                       decode_J=decode_j, prefill_J=prefill["total_J_max"], e2e_J=e2e_j,
                       prefill_J_min=prefill["total_J_min"], e2e_J_min=decode_j+prefill["total_J_min"],
                       decode_tok_J=1000/decode_j, e2e_tok_J=1000/e2e_j,
                       gpu_J=sum(s["gpu_J"] for s in steps), memory_J=sum(s["memory_J"] for s in steps),
                       boundary_J=sum(s["boundary_J"] for s in steps), nmp_J=sum(s["nmp_J"] for s in steps),
                       boundary_GB_token=sum(s["boundary_bytes"] for s in steps)/1000/1e9,
                       nmp_peak_util=max(s["peak_nmp_utilization"] for s in steps),
                       nmp_average_util=sum(s["nmp_flops"] for s in steps)/decode_s/engine.activity.hardware.aggregate_peak_flops,
                       first_step_ms=steps[0]["latency_s"]*1000, last_step_ms=steps[-1]["latency_s"]*1000)
            summary.append(row)
            traffic = {k:sum(s["traffic_bytes"][k] for s in steps)/1000 for k in steps[0]["traffic_bytes"]}
            assert (traffic["historical_K"] > 0) == (policy == ExecutionPolicy.NO_NMP)
            assert (traffic["historical_V"] > 0) == (policy == ExecutionPolicy.NO_NMP)
            assert (traffic["weight"] > 0) == (policy != ExecutionPolicy.MAC_NMP)
            traffic_rows.append(dict(model=name, policy=policy, **traffic))
            step_rows.extend(dict(model=name, policy=policy, context=s["context"], latency_s=s["latency_s"],
                                  total_J=s["total_J"]) for s in steps)
            print(json.dumps(row), flush=True)
    normalized = []
    for name in llama31_models():
        rows = {r["policy"]:r for r in summary if r["model"] == name}
        base, attention, mac = (rows[p] for p in ExecutionPolicy)
        for row in (attention, mac):
            normalized.append(dict(model=name, policy=row["policy"],
                                   speedup=row["decode_tok_s"]/base["decode_tok_s"],
                                   tokens_J_improvement=row["decode_tok_J"]/base["decode_tok_J"]))
        normalized.append(dict(model=name, policy="MAC_NMP / ATTENTION_NMP",
                               speedup=mac["decode_tok_s"]/attention["decode_tok_s"],
                               tokens_J_improvement=mac["decode_tok_J"]/attention["decode_tok_J"]))
    for filename, rows in (("summary.csv", summary), ("traffic_bytes_per_token.csv", traffic_rows),
                           ("decode_steps.csv", step_rows), ("normalized.csv", normalized)):
        with (out/filename).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (out/"prefill.json").write_text(json.dumps(prefill_rows, indent=2)+"\n", encoding="utf-8")
    print("NORMALIZED\n"+json.dumps(normalized, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"runs/decode_policy_318")
    args = parser.parse_args()
    run(args.output)
