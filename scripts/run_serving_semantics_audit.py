"""Run narrow non-thermal state/capacity/placement semantic audit smokes."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from om3dthermal.serving import (
    MixedPhaseServingCase,
    evaluate_conventional_prefill_first_state_window,
    evaluate_m3d_growing_kv_capacity,
    evaluate_nmp_decode_batch,
)
from om3dthermal.workload import load_dense_model_registry


ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"runs/serving_semantics_audit"


def _hash(path:Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main()->None:
    registry=load_dense_model_registry(ROOT/"configs/workload/models")
    qcase=MixedPhaseServingCase(model_id="qwen25_7b",context_length=131072,
        batch_size=28,prefill_requests=14,decode_requests=14)
    qwen=evaluate_conventional_prefill_first_state_window(
        project_root=ROOT,model=registry["qwen25_7b"],case=qcase)
    llama=[]
    for prefill,decode in ((1,27),(14,14)):
        case=MixedPhaseServingCase(model_id="llama31_8b",context_length=131072,
            batch_size=28,prefill_requests=prefill,decode_requests=decode)
        llama.append(evaluate_m3d_growing_kv_capacity(
            project_root=ROOT,model=registry["llama31_8b"],case=case,
            generated_decode_steps=512).model_dump(mode="json"))
    nmp=[]
    for decode,active in ((1,1),(14,28)):
        row=evaluate_nmp_decode_batch(registry["llama31_8b"].decode_input(
            batch_size=decode,context_length=131072),project_root=ROOT,
            active_capacity_requests=active)
        nmp.append(row.model_dump(mode="json",exclude={"execution_trace"}))
    head=subprocess.check_output(
        ["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    payload={
        "commit":head,
        "scope":"NARROW_NON_THERMAL_SERVING_SEMANTICS_AUDIT",
        "config_hashes":{
            "final_dense_e2e_matrix.yaml":_hash(
                ROOT/"configs/experiment/final_dense_e2e_matrix.yaml")},
        "model_hashes":{name:_hash(ROOT/"configs/workload/models"/f"{name}.yaml")
                        for name in ("llama31_8b","qwen25_7b","llama2_7b")},
        "qwen_14_14_prefill_first_mixed_window":qwen.model_dump(mode="json"),
        "llama31_b28_g512_growing_capacity":llama,
        "nmp_fixed_context_smokes":nmp,
        "comparison_status":(
            "CROSS_SYSTEM_GROWING_KV_RATIO_NOT_COMPUTED__"
            "HETEROGENEOUS_P_D_PERSISTENT_HORIZON_EXECUTION_UNRESOLVED"),
        "workspace_status":"WORKSPACE_UNRESOLVED__NOT_ASSUMED_ZERO",
        "thermal":None}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"serving_semantics_audit.json").write_text(
        json.dumps(payload,indent=2),encoding="utf-8")
    lines=["# Serving semantics audit","",
        f"- Commit: `{head}`",
        "- Scope: `NARROW_NON_THERMAL_SERVING_SEMANTICS_AUDIT`",
        "- Cross-system G ratio: not computed; matched heterogeneous P/D persistent horizon execution remains unresolved.","",
        "## Qwen 14:14 PREFILL_FIRST mixed window","",
        f"- resident Decode: {qwen.resident_decode_requests}",
        f"- resident Prefill KV: {qwen.resident_prefill_requests}",
        f"- host Prefill KV: {qwen.host_prefill_requests}",
        f"- host write: {qwen.host_prefill_kv_write_GB:.9f} GB",
        f"- host transfer: {qwen.host_prefill_transfer_time_ms/1e3:.9f} s","",
        "## Llama-3.1 B28 G512 high water","",
        "| P:D | growth GB | rounded GB | margin GB | status |",
        "|---|---:|---:|---:|---|" ]
    for row in llama:
        lines.append(f'| {row["retained_prefill_requests"]}:{row["growing_decode_requests"]} | '
            f'{row["growing_kv_bytes"]/1e9:.9f} | {row["high_water_page_rounded_capacity_GB"]:.9f} | '
            f'{row["capacity_margin_GB"]:.9f} | {row["capacity_status"]} |')
    (OUT/"serving_semantics_audit.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"qwen_host_prefill_GB":qwen.host_prefill_kv_write_GB,
                      "llama_capacity_rows":len(llama),"nmp_smokes":len(nmp)}))


if __name__=="__main__":
    main()
