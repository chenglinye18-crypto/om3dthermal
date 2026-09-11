"""Fixed performance-only 318-slab physical FEOL comparison, with rerun gate."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from om3dthermal.architecture.feol_floorplan import resolve_feol_floorplan
from om3dthermal.serving.decode_policy import CachedWorkload, DecodePolicyModel, ExecutionPolicy, llama31_models
from om3dthermal.power.nmp_die_activity import group_service_seconds

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"runs/decode_policy_318_feol_v1"


def write_csv(name, rows):
    with (OUT/name).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def audit():
    f = resolve_feol_floorplan(ROOT)
    a = f.audit()
    assert a["cluster_grid"] == [35, 8] and a["slabs"] == 318
    assert a["region_group_counts"] == [18, 18, 18, 16]
    assert a["macs_per_slab"] == 512 and len(a["io_ports_um"]) == 50
    assert f.fabric_Bps == 64e9 and f.link_Bps == 32e9 and f.external_Bps == 3.4e12
    a.update(git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
             workload="B1, resident history 125000 + GPU Prefill 1000 + Decode contexts 126000..126999; final KV 127000",
             bytes_weight=2, bytes_KV=2, bytes_activation=2, bytes_partial=4,
             policies=list(ExecutionPolicy), models={name:w.model_dump() for name,w in llama31_models().items()})
    print("PRE-RUN FLOORPLAN AUDIT", flush=True)
    for k in ("git_head", "slab_dimensions_um", "slabs", "thickness_um", "layers", "capacity_GB", "cluster_grid",
              "region_group_counts", "region_port_counts", "region_port_ids", "root_local_port_length_um", "root_local_port_rc_ns", "macs_per_slab", "fabric_GBps_per_region", "noc_GBps_per_direction",
              "external_saturated_TBps", "config", "sa_nearest_mac_length_um", "sa_nearest_edge_length_um",
              "noc_links", "workload", "models"):
        print(k+": "+json.dumps(a[k]), flush=True)
    print("70 distributed 256-bit SA banks; 4 regions x 8 tiles x 16 MAC; 50 single-edge 8-Gb/s ports", flush=True)
    return a


def evaluate_chunk(task):
    name, start, stop = task
    engine = DecodePolicyModel(llama31_models()[name], project_root=ROOT)
    cases = {p:[] for p in ExecutionPolicy}
    for j, context in enumerate(range(start, stop)):
        for policy in ExecutionPolicy:
            row = engine.step(context, policy)
            cases[policy].append(row)
        if (j+1) % 100 == 0:
            print(f"  {name} contexts {start}..{context} done", flush=True)
    return cases


def evaluate(name, pool):
    tasks = [(name, 126000+i*250, 126000+(i+1)*250) for i in range(4)]
    cases = {p:[] for p in ExecutionPolicy}
    for chunk in pool.map(evaluate_chunk, tasks):
        for policy in ExecutionPolicy:
            cases[policy].extend(chunk[policy])
    digest = hashlib.sha256()
    for j in range(1000):
        for policy in ExecutionPolicy:
            digest.update(json.dumps(cases[policy][j], sort_keys=True).encode())
    return cases, digest.hexdigest()


def group_audit(engine):
    f = engine.floorplan
    placement = engine.placement
    total_group = placement.slot_used.sum(axis=1).reshape(70, 318)*4
    examples = {}
    for op in ("Q", "ATTENTION_QK", "ATTENTION_AV", "FFN_GATE"):
        e = placement.operators[0, op]
        atoms = 126000*8 if op.startswith("ATTENTION") else e.atom_count
        b = e.layer_bytes(atoms)
        g = e.lanes//318
        seconds = group_service_seconds(b, f.service_ns[g])
        examples[op] = [dict(group_id=gid, active_read_bytes=int(b[g == gid].sum()),
                            max_group_service_s=float(seconds[g == gid].max(initial=0)),
                            active_die_count=int(np.count_nonzero(b[g == gid].sum(axis=1))),
                            active_layer_slots=int(np.count_nonzero(b[g == gid]))) for gid in range(70)]
    return dict(groups=[dict(group_id=g, resident_bytes=int(total_group[g].sum()),
                             max_die_group_resident_bytes=int(total_group[g].max()),
                             resident_layer_slots=int(np.count_nonzero(placement.slot_used[g*318:(g+1)*318]))) for g in range(70)],
                layer0_context126000_examples=examples,
                conservation={k:v for k,v in placement.audit().items() if not k.startswith("group_")})


def main():
    setup = audit()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/"feol_floorplan_audit.json").write_text(json.dumps(setup, indent=2)+"\n", encoding="utf-8")
    summary, traffic_rows, bottlenecks, reruns, prefill_rows, groups = [], [], [], {}, {}, {}
    for name, w in llama31_models().items():
        print("Building physical resident layout: "+name, flush=True)
        engine = DecodePolicyModel(w, project_root=ROOT)
        pref = engine.prefill(CachedWorkload())
        prefill_rows[name] = pref
        groups[name] = group_audit(engine)
        print("Capacity: "+json.dumps(groups[name]["conservation"]), flush=True)
        with ProcessPoolExecutor(max_workers=4) as pool:
            cases, first_hash = evaluate(name, pool)
        print("Deterministic fresh-layout rerun: "+name, flush=True)
        fresh = DecodePolicyModel(w, project_root=ROOT)
        assert fresh.prefill(CachedWorkload()) == pref
        with ProcessPoolExecutor(max_workers=4) as pool:
            repeated, second_hash = evaluate(name, pool)
        assert first_hash == second_hash, "deterministic per-step rerun mismatch"
        del fresh, repeated
        reruns[name] = first_hash
        for policy, steps in cases.items():
            seconds = sum(s["latency_s"] for s in steps)
            group_counts = [g for s in steps for g in s["active_group_counts"]]
            region_counts = [g for s in steps for g in s["active_region_counts"]]
            tiles = [g for s in steps for g in s["active_tile_counts"]]
            total = lambda k: sum(s[k] for s in steps)
            boundary = total("boundary_bytes")
            fabric_realized = sum(s["fabric_region_average_Bps"]*s["latency_s"] for s in steps)/seconds
            row = dict(model=name, policy=policy, resident_GB=engine.placement.audit()["resident_bytes"]/1e9,
                prefill_s=pref["latency_s"], decode_s=seconds, decode_tok_s=1000/seconds,
                e2e_tok_s=1000/(seconds+pref["latency_s"]), first_step_ms=steps[0]["latency_s"]*1000,
                last_step_ms=steps[-1]["latency_s"]*1000,
                boundary_GB_per_token=boundary/1000/1e9,
                external_bandwidth_cap_TBps=engine.floorplan.external_Bps/1e12,
                external_bandwidth_realized_TBps=boundary/seconds/1e12,
                local_array_bytes_per_token=total("local_array_bytes")/1000,
                local_array_realized_TBps=total("local_array_bytes")/seconds/1e12,
                mean_active_service_groups_per_stage=float(np.mean(group_counts)),
                p90_active_service_groups_per_stage=float(np.percentile(group_counts, 90)),
                max_active_service_groups_per_stage=max(group_counts),
                mean_active_regions_per_stage=float(np.mean(region_counts)),
                mean_active_mac_tiles_per_stage=float(np.mean(tiles)),
                region_fabric_peak_GBps=64.0, region_fabric_realized_GBps=fabric_realized/1e9,
                region_fabric_utilization=fabric_realized/64e9,
                inter_region_noc_bytes_per_token=total("noc_bytes")/1000,
                inter_region_noc_max_link_utilization=float(np.sum([s["noc_link_busy_s"] for s in steps], axis=0).max())/seconds,
                inter_region_noc_latency_ms_per_token=total("noc_s"),
                NMP_peak_utilization=max(s["NMP_peak_utilization"] for s in steps),
                NMP_average_utilization=total("nmp_flops")/seconds/(318*32*engine.floorplan.tile_flops))
            port_counts = [v for s in steps for v in s["active_external_port_counts"]]
            row.update(mean_active_external_ports_per_stage=float(np.mean(port_counts)),
                       p90_active_external_ports_per_stage=float(np.percentile(port_counts,90)),
                       max_active_external_ports_per_stage=max(port_counts),
                       max_external_port_utilization=max(s["max_external_port_utilization"] for s in steps),
                       external_global_cap_limited_fraction=sum(s["external_limit_counts"]["GLOBAL_THERMAL_CAP"] for s in steps)/len(port_counts),
                       external_port_limited_fraction=sum(s["external_limit_counts"]["PORT_SERIALIZATION"] for s in steps)/len(port_counts))
            for cause,label in (("ARRAY","ARRAY"),("LOCAL_FABRIC","LOCAL_FABRIC"),("MAC","MAC"),
                                ("INTER_REGION_NOC","INTER_REGION_NOC"),("EXTERNAL_BOUNDARY","EXTERNAL"),("GPU_COMPUTE","GPU")):
                row["sum_"+label+"_component_s"] = sum(s["component_sums"][cause] for s in steps)
            summary.append(row)
            traffic = {k:sum(s["traffic_bytes"][k] for s in steps)/1000 for k in steps[0]["traffic_bytes"]}
            assert (traffic["historical_K"] > 0) == (policy == ExecutionPolicy.NO_NMP)
            assert (traffic["historical_V"] > 0) == (policy == ExecutionPolicy.NO_NMP)
            assert (traffic["weight"] > 0) == (policy != ExecutionPolicy.MAC_NMP)
            traffic_rows.append(dict(model=name, policy=policy, **traffic))
            for cause in ("ARRAY", "LOCAL_FABRIC", "MAC", "INTER_REGION_NOC", "EXTERNAL_BOUNDARY", "GPU_COMPUTE"):
                count = sum(s["bottlenecks"].get(cause, {}).get("count", 0) for s in steps)
                time = sum(s["bottlenecks"].get(cause, {}).get("time_s", 0) for s in steps)
                bottlenecks.append(dict(model=name, policy=policy, bottleneck=cause, stage_count=count,
                                        dominant_stage_attribution_s=time, dominant_stage_attribution_fraction=time/seconds))
            print(json.dumps(row), flush=True)
    for row in summary:
        base = next(r for r in summary if r["model"] == row["model"] and r["policy"] == ExecutionPolicy.NO_NMP)
        attention = next(r for r in summary if r["model"] == row["model"] and r["policy"] == ExecutionPolicy.ATTENTION_NMP)
        row["speedup_vs_no_nmp"] = base["decode_s"]/row["decode_s"]
        row["speedup_vs_attention_nmp"] = attention["decode_s"]/row["decode_s"]
    write_csv("summary.csv", summary)
    write_csv("normalized.csv", [{k:r[k] for k in ("model", "policy", "speedup_vs_no_nmp", "speedup_vs_attention_nmp")} for r in summary])
    write_csv("traffic_bytes_per_token.csv", traffic_rows)
    write_csv("stage_bottlenecks.csv", bottlenecks)
    for filename, obj in (("resident_group_audit.json", groups), ("prefill.json", prefill_rows), ("deterministic_rerun.json", reruns)):
        (OUT/filename).write_text(json.dumps(obj, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
