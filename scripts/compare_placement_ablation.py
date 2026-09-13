"""MAC-NMP B1/B8 placement ablation; frozen event energy and slab thermal diagnostic."""
import argparse,csv,hashlib,json,pickle,subprocess,time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from om3dthermal.serving.decode_policy import DecodePolicyModel,ExecutionPolicy,llama31_models,CachedWorkload
from om3dthermal.placement.nmp_load_balance import PlacementPolicy
from om3dthermal.power.feol_energy import FEOLEnergyModel,sum_events,sum_slab_events,SCALARS,LAYERS,MAXIMA,power_groups
from om3dthermal.thermal.placement_diagnostic import PlacementThermalDiagnostic,THERMAL_MODEL

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"runs/placement_ablation_b1_b8_v1"


def run_fingerprints():
    paths=[*sorted((ROOT/"src/om3dthermal").rglob("*.py")),*sorted((ROOT/"configs").rglob("*.yaml"))]
    inputs=b"".join(p.read_bytes() for p in paths)
    return (hashlib.sha256(Path(__file__).read_bytes()+inputs).hexdigest(),
            hashlib.sha256(inputs).hexdigest())


def serial(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    raise TypeError(type(x))


def engine(name,batch,placement):
    w=llama31_models()[name].model_copy(update={"batch_size":batch})
    return DecodePolicyModel(w,project_root=ROOT,placement_policy=placement,record_energy=True,record_slabs=True)


def chunk(args):
    name,batch,placement,start,end,computation_fp=args
    runner_fp,_=run_fingerprints()
    cache=OUT/"step_checkpoints"/f"{name}_B{batch}_{placement}"/f"{start}_{end}.pkl"
    rows=[]
    if cache.exists():
        with cache.open("rb") as stream:saved=pickle.load(stream)
        if saved["computation_fingerprint"]==computation_fp and saved["runner_fingerprint"]==runner_fp:
            rows=saved["rows"]
            assert [s["context"] for s in rows]==list(range(start,start+len(rows)))
            assert start+len(rows)<=end
            print(f"PERFORMANCE RESUME {name} B{batch} {placement}: {len(rows)}/{end-start} saved steps",flush=True)
    if len(rows)==end-start:return rows
    cache.parent.mkdir(parents=True,exist_ok=True)
    def save():
        temporary=cache.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            pickle.dump(dict(computation_fingerprint=computation_fp,runner_fingerprint=runner_fp,rows=rows),stream,protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(cache)
    e=engine(name,batch,placement);started=time.perf_counter()
    for context in range(start+len(rows),end):
        s=e.step(context,ExecutionPolicy.MAC_NMP)
        rows.append({k:s[k] for k in ("context","latency_s","boundary_bytes","energy_events","slab_events","slab_service_s","component_sums")})
        if len(rows)%10==0:print(f"PERFORMANCE {name} B{batch} {placement} {start}..{context}; worker elapsed {time.perf_counter()-started:.1f}s",flush=True)
        if len(rows)%25==0:save()
    save()
    return rows


def frozen_gate(name,pref,steps,energy):
    frozen=json.loads((ROOT/"tests/data/placement_ablation_ae14f09.json").read_text())
    old=next(r for r in frozen["energy_summary"] if r["model"]==name)
    seconds=sum(s["latency_s"] for s in steps)
    checks=dict(prefill_s=pref["latency_s"],decode_s=seconds,decode_tok_s=1000/seconds,E2E_tok_s=1000/(seconds+pref["latency_s"]),decode_J=energy["total_J"])
    for k,v in checks.items():assert v==float(old[k]),(name,k,v,old[k],"FROZEN_B1_CHANGED_STOP")
    events=sum_events(s["energy_events"] for s in steps)
    old_events=next(r for r in frozen["energy_events"] if r["model"]==name)
    for k,v in events.items():
        if k in LAYERS:
            for i,x in enumerate(v):assert x==float(old_events[k+"_"+str(i+1)]),(name,k,i,"FROZEN_EVENTS_CHANGED_STOP")
        else:assert v==float(old_events[k]),(name,k,"FROZEN_EVENTS_CHANGED_STOP")
    old_summary=next(r for r in frozen["summary"] if r["model"]==name)
    assert sum(s["boundary_bytes"] for s in steps)/1e12==float(old_summary["boundary_GB_per_token"])
    print(f"FROZEN B1 BALANCED EXACT GATE PASSED: {name}",flush=True)


def evaluate(name,batch,placement,workers):
    e=engine(name,batch,placement);audit=e.placement.audit()
    assert audit["slot_capacity_violations"]==0
    assert audit["resident_bytes"]<=e.floorplan.layout.total_capacity_bytes
    print("CAPACITY "+json.dumps(dict(model=name,batch=batch,placement=placement,resident_GB=audit["resident_bytes"]/1e9,shared_weight_GB=audit["shared_weight_bytes"]/1e9,KV_GB=audit["total_kv_bytes"]/1e9)),flush=True)
    pref=e.prefill(CachedWorkload());m=FEOLEnergyModel(e.floorplan,e.platform)
    pe=m.account(pref["energy_events"],pref["latency_s"],phase="prefill",policy="MAC_NMP",total_flops=pref["ledger"]["total_flops"])
    cuts=np.linspace(126000,127000,workers+1,dtype=int)
    _,computation_fp=run_fingerprints()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pieces=list(pool.map(chunk,[(name,batch,placement,int(a),int(b),computation_fp) for a,b in zip(cuts[:-1],cuts[1:])]))
    steps=[s for part in pieces for s in part]
    assert [s["context"] for s in steps]==list(range(126000,127000))
    seconds=sum(s["latency_s"] for s in steps)
    events=sum_events(s["energy_events"] for s in steps)
    se=sum_slab_events(s["slab_events"] for s in steps)
    service=sum(s["slab_service_s"] for s in steps)
    energy=m.account(events,seconds,phase="decode",policy="MAC_NMP")
    if batch==1 and placement==PlacementPolicy.BALANCED:
        frozen_gate(name,pref,steps,energy)
        old=json.loads((ROOT/"tests/data/placement_ablation_ae14f09.json").read_text())
        assert pe["total_J"]==float(next(r for r in old["prefill_energy_summary"] if r["model"]==name)["prefill_total_J"])
    for k in SCALARS:np.testing.assert_allclose(se[k].sum(),events[k],rtol=1e-12,atol=1e-7,err_msg=k)
    for k in LAYERS:np.testing.assert_allclose(se[k].sum(axis=0),events[k],rtol=1e-12)
    gpu=energy["components"]["gpu_dynamic_J"]+energy["components"]["gpu_static_J"]
    slab_rows=[];powers=[]
    for slab in range(318):
        ev={k:np.asarray(v[slab]).tolist() for k,v in se.items()}
        c=m.account(ev,seconds,phase="decode",policy="NO_NMP")["components"]
        dynamic=sum(v for k,v in c.items() if not k.startswith("gpu_"))
        unresolved=energy["components"]["feol_unresolved_J"]/318
        total=dynamic+unresolved;power=total/seconds;powers.append(power)
        slab_rows.append(dict(model=name,batch_size=batch,placement=placement,slab_id=slab,
            resident_bytes=audit["per_slab_resident_bytes"][slab],array_read_bytes=ev["array_read_bits"]/8,array_write_bytes=ev["array_write_bits"]/8,
            NMP_FLOPs=ev["mac_operations"]*2,SRAM_read32=ev["sram_read32_accesses"],SRAM_write32=ev["sram_write32_accesses"],
            router_bit_traversals=ev["router_bit_traversals"],noc_link_bit_um=ev["noc_link_bit_um"],interface_bits=ev["interface_bits"],
            dynamic_energy_J=dynamic,unresolved_energy_J=unresolved,total_energy_J=total,average_power_W=power,service_s=float(service[slab])))
    powers=np.asarray(powers)
    assert np.isclose(powers.sum()*seconds+gpu,energy["total_J"],rtol=1e-12)
    generated=1000*batch;e2e_J=pe["total_J"]+energy["total_J"];e2e_s=pref["latency_s"]+seconds
    active=(se["array_read_bits"]+se["array_write_bits"])>0
    summary=dict(model=name,batch_size=batch,placement=placement,resident_GB=audit["resident_bytes"]/1e9,
        capacity_utilization=audit["resident_bytes"]/e.floorplan.layout.total_capacity_bytes,
        decode_s=seconds,decode_generated_tokens=generated,decode_tok_s=generated/seconds,
        first_step_ms=steps[0]["latency_s"]*1000,last_step_ms=steps[-1]["latency_s"]*1000,
        prefill_s=pref["latency_s"],prefill_J=pe["total_J"],e2e_s=e2e_s,e2e_tok_s=generated/e2e_s,
        decode_J=energy["total_J"],decode_J_per_token=energy["total_J"]/generated,decode_tokens_per_J=generated/energy["total_J"],
        e2e_J=e2e_J,e2e_J_per_token=e2e_J/generated,e2e_tokens_per_J=generated/e2e_J,
        average_decode_power_W=energy["total_J"]/seconds,max_slab_power_W=float(powers.max()),mean_slab_power_W=float(powers.mean()),
        slab_power_max_mean_ratio=float(powers.max()/powers.mean()),slab_power_cv=float(powers.std()/powers.mean()),
        active_slab_count=int(active.sum()),active_slab_fraction=float(active.mean()),max_slab_service_ms=float(service.max()),mean_slab_service_ms=float(service.mean()),
        boundary_GB_per_generated_token=sum(s["boundary_bytes"] for s in steps)/generated/1e9,
        GPU_power_W=gpu/seconds)
    for k in steps[0]["component_sums"]:summary["sum_"+k+"_component_s"]=sum(s["component_sums"][k] for s in steps)
    stage_hash=hashlib.sha256(json.dumps([{k:v for k,v in s.items() if k not in ("slab_events","slab_service_s")} for s in steps],sort_keys=True,default=serial).encode()).hexdigest()
    return dict(summary=summary,placement_audit=audit,slabs=slab_rows,events=events,prefill=dict(latency_s=pref["latency_s"],ledger=pref["ledger"],energy=pe),
        energy=energy,stage_hash=stage_hash)


def write_csv(name,rows):
    with (OUT/name).open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--phase",choices=("all","performance","thermal"),default="all")
    parser.add_argument("--models",nargs="*",default=list(llama31_models()))
    parser.add_argument("--batches",nargs="*",type=int,default=[1,8])
    parser.add_argument("--placements",nargs="*",default=[p.value for p in (PlacementPolicy.BALANCED,PlacementPolicy.COMPACT_FIRST_FIT,PlacementPolicy.UNIFORM_STRIPING)])
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True);checkpoints=OUT/"case_results";checkpoints.mkdir(exist_ok=True)
    fingerprint,computation_fp=run_fingerprints()
    results=[]
    for name in args.models:
        for batch in args.batches:
            for placement in args.placements:
                file=checkpoints/f"{name}_B{batch}_{placement}.json"
                result=json.loads(file.read_text()) if file.exists() else None
                valid=(result is not None and result.get("computation_fingerprint")==computation_fp
                       and fingerprint in (result.get("fingerprint"),result.get("runner_reuse_fingerprint")))
                if not valid:
                    if args.phase=="thermal":raise RuntimeError(f"Missing validated performance checkpoint: {file}; thermal-only will not run performance")
                    print(f"PERFORMANCE CASE START {name} B{batch} {placement}",flush=True)
                    result=evaluate(name,batch,placement,args.workers);result["fingerprint"]=fingerprint
                    result["computation_fingerprint"]=computation_fp
                    file.write_text(json.dumps(result,default=serial)+"\n")
                else:print(f"PERFORMANCE CHECKPOINT HIT {name} B{batch} {placement}",flush=True)
                results.append((file,result))
    if args.phase=="performance":
        print(f"PERFORMANCE COMPLETE: {len(results)} cases; thermal not requested",flush=True)
        return
    thermal=None
    for file,result in results:
        if "thermal" not in result:
            if thermal is None:
                if not (OUT/"thermal_setup.pkl").is_file():raise RuntimeError("Thermal setup cache missing; refusing an implicit setup rebuild")
                started=time.perf_counter();thermal=PlacementThermalDiagnostic(ROOT,OUT)
                if thermal.setup is None:raise RuntimeError("Thermal setup cache invalid; refusing an implicit setup rebuild")
                print(f"THERMAL SETUP HIT: loaded in {time.perf_counter()-started:.1f}s; {thermal.setup.operator_template.cell_count} cells; fixed mesh/operator reused",flush=True)
            row=result["summary"];powers=[s["average_power_W"] for s in result["slabs"]]
            print(f"THERMAL CASE START {row['model']} B{row['batch_size']} {row['placement']}",flush=True)
            result["thermal"]=thermal.run(row["GPU_power_W"],powers)
            file.write_text(json.dumps(result,default=serial)+"\n")
            print("THERMAL "+json.dumps({**{k:row[k] for k in ("model","batch_size","placement")},**result["thermal"]}),flush=True)
    summaries=[{**r["summary"],**r["thermal"]} for _,r in results]
    write_csv("summary.csv",summaries)
    write_csv("thermal_summary.csv",[{**{k:r["summary"][k] for k in ("model","batch_size","placement","max_slab_power_W","mean_slab_power_W","slab_power_max_mean_ratio","slab_power_cv")},**r["thermal"]} for _,r in results])
    slabs=[s for _,r in results for s in r["slabs"]];write_csv("slab_activity.csv",slabs)
    write_csv("slab_power.csv",[{k:s[k] for k in ("model","batch_size","placement","slab_id","dynamic_energy_J","unresolved_energy_J","total_energy_J","average_power_W")} for s in slabs])
    write_csv("placement_audit.csv",[{**{k:r["summary"][k] for k in ("model","batch_size","placement")},**{k:v for k,v in r["placement_audit"].items() if not isinstance(v,list)}} for _,r in results])
    write_csv("energy_breakdown.csv",[{**{k:r["summary"][k] for k in ("model","batch_size","placement")},**r["energy"]["components"],**power_groups(r["energy"]["components"],r["summary"]["decode_s"])} for _,r in results])
    norm=[]
    for r in summaries:
        if r["placement"]=="BALANCED":continue
        bal=next((b for b in summaries if b["model"]==r["model"] and b["batch_size"]==r["batch_size"] and b["placement"]=="BALANCED"),None)
        if bal is None:continue
        norm.append(dict(model=r["model"],batch_size=r["batch_size"],baseline=r["placement"],BALANCED_E2E_speedup=bal["e2e_tok_s"]/r["e2e_tok_s"],
            BALANCED_E2E_energy_efficiency=bal["e2e_tokens_per_J"]/r["e2e_tokens_per_J"],delta_Tmax_C=bal["Tmax_C"]-r["Tmax_C"],
            delta_slab_power_max_mean_ratio=bal["slab_power_max_mean_ratio"]-r["slab_power_max_mean_ratio"]))
    if norm:write_csv("normalized.csv",norm)
    manifest=dict(git_head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),fingerprint=fingerprint,computation_fingerprint=computation_fp,
        cases=len(results),batch_sizes=args.batches,execution_policy="MAC_NMP",thermal_model=THERMAL_MODEL,
        workload=dict(history=125000,prompt=1000,generated_steps=1000,contexts=[126000,126999]),
        energy_config_sha256=hashlib.sha256((ROOT/"configs/architecture/m3d_feol_energy_v1.yaml").read_bytes()).hexdigest(),
        thermal_status="PLACEMENT_DIAGNOSTIC_ONLY__FORMAL_FEOL_THERMAL_RECLOSURE_PENDING",
        batch_semantics="one shared weight stream; B MAC consumers; request-local KV; concurrent resource aggregation",
        embedding_lookup="deterministic distinct resident row=request_id; one shared embedding matrix",
        slab_service_definition="sum per-stage local resource completion, excluding global-cap/GPU waiting; ms averaged over 1000 aggregate steps",
        stage_hashes={f.name:r["stage_hash"] for f,r in results})
    (OUT/"experiment_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(summaries,indent=2),flush=True)

if __name__=="__main__":main()
