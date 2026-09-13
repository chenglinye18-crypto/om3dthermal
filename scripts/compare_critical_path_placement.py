"""Three B1 CPA cases, with immutable saved Uniform comparison and cached heat."""
import argparse,hashlib,json,pickle,subprocess,time
from pathlib import Path
import compare_placement_ablation as base
from om3dthermal.serving.decode_policy import DecodePolicyModel,llama31_models
from om3dthermal.thermal.placement_diagnostic import PlacementThermalDiagnostic,THERMAL_MODEL

ROOT=base.ROOT
OUT=ROOT/'runs/critical_path_placement_b1_v1'
REFERENCE=ROOT/'runs/placement_ablation_b1_b8_v1'
base.OUT=OUT
original_fingerprints=base.run_fingerprints


def fingerprints():
    runner,physical=original_fingerprints()
    return hashlib.sha256(Path(__file__).read_bytes()+runner.encode()).hexdigest(),physical


def engine(name,batch,placement):
    assert batch==1 and placement=='CRITICAL_PATH_AWARE'
    fp=fingerprints()[0]
    path=OUT/'plans'/f'{name}.pkl'
    plan=None
    if path.exists():
        with path.open('rb') as stream: saved=pickle.load(stream)
        if saved['fingerprint']==fp: plan=saved['placement']
    if plan is None:
        started=time.perf_counter()
        e=DecodePolicyModel(llama31_models()[name],project_root=ROOT,placement_policy=placement)
        plan=e.placement
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('wb') as stream:pickle.dump(dict(fingerprint=fp,placement=plan),stream,pickle.HIGHEST_PROTOCOL)
        print(f'OPTIMIZER {name}: {time.perf_counter()-started:.2f}s; accepted {sum(r["accepted_moves"] for r in plan.optimizer_audit)}',flush=True)
    e=DecodePolicyModel(llama31_models()[name],project_root=ROOT,placement_policy='UNIFORM_STRIPING',record_energy=True,record_slabs=True)
    # Resident lane set is unchanged; the shared schedule/small-op owner set
    # remains valid. All physical reads and events use the refined placement.
    e.placement=plan
    return e


base.engine=engine
base.run_fingerprints=fingerprints


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--phase',choices=('all','performance','thermal'),default='all')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    fp,physical=fingerprints()
    frozen=json.loads((ROOT/'tests/data/cpa_uniform_920efd2.json').read_text())
    results=[];references=[];audits=[]
    for name in llama31_models():
        reference_path=REFERENCE/'case_results'/f'{name}_B1_UNIFORM_STRIPING.json'
        ref=json.loads(reference_path.read_text())
        for key in ('summary','events','energy','thermal','stage_hash'):
            assert ref[key]==frozen['cases'][name][key],(name,key,'UNIFORM_REFERENCE_CHANGED_STOP')
        references.append(ref)
        path=OUT/f'{name}.json'
        result=json.loads(path.read_text()) if path.exists() else None
        if result is None or result.get('fingerprint')!=fp:
            if args.phase=='thermal':raise RuntimeError('Missing validated CPA performance; no implicit rerun')
            result=base.evaluate(name,1,'CRITICAL_PATH_AWARE',args.workers)
            result.update(fingerprint=fp,computation_fingerprint=physical)
            result['optimizer_audit']=engine(name,1,'CRITICAL_PATH_AWARE').placement.optimizer_audit
            path.write_text(json.dumps(result,default=base.serial)+'\n')
        else:print(f'PERFORMANCE CHECKPOINT HIT {name}',flush=True)
        results.append((path,result))
        audits.extend(dict(model=name,**r) for r in result['optimizer_audit'])
    if args.phase=='performance':return
    thermal=None
    for path,result in results:
        if 'thermal' not in result:
            if thermal is None:
                if not (REFERENCE/'thermal_setup.pkl').is_file():raise RuntimeError('Existing setup required')
                started=time.perf_counter();thermal=PlacementThermalDiagnostic(ROOT,REFERENCE)
                if thermal.setup is None:raise RuntimeError('Invalid setup; rebuilding is forbidden')
                print(f'THERMAL CACHE HIT {time.perf_counter()-started:.2f}s',flush=True)
            s=result['summary']
            result['thermal']=thermal.run(s['GPU_power_W'],[r['average_power_W'] for r in result['slabs']])
            path.write_text(json.dumps(result,default=base.serial)+'\n')
            print('THERMAL '+s['model']+' '+str(result['thermal']['Tmax_C']),flush=True)
    rows=[r for _,r in results]
    base.write_csv('summary.csv',[{**r['summary'],**r['thermal']} for r in rows])
    base.write_csv('normalized.csv',[dict(model=r['summary']['model'],
        uniform_e2e_tok_s=u['summary']['e2e_tok_s'],CPA_e2e_tok_s=r['summary']['e2e_tok_s'],
        speedup=r['summary']['e2e_tok_s']/u['summary']['e2e_tok_s'],
        uniform_e2e_tok_J=u['summary']['e2e_tokens_per_J'],CPA_e2e_tok_J=r['summary']['e2e_tokens_per_J'],
        uniform_Tmax_C=u['thermal']['Tmax_C'],CPA_Tmax_C=r['thermal']['Tmax_C'],
        delta_Tmax_C=r['thermal']['Tmax_C']-u['thermal']['Tmax_C']) for u,r in zip(references,rows)])
    base.write_csv('optimizer_audit.csv',[dict(model=r['summary']['model'],
        candidate_evaluations=sum(a['candidate_evaluations'] for a in r['optimizer_audit']),
        accepted_moves=sum(a['accepted_moves'] for a in r['optimizer_audit']),
        optimizer_runtime_s=sum(a['optimizer_runtime_s'] for a in r['optimizer_audit']),
        resident_atoms_moved=sum(a['resident_atoms_moved'] for a in r['optimizer_audit']),
        compute_chunks_moved=sum(a['compute_chunks_moved'] for a in r['optimizer_audit'])) for r in rows])
    base.write_csv('operator_before_after.csv',[{k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in a.items()} for a in audits])
    base.write_csv('placement_audit.csv',[dict(model=r['summary']['model'],**{k:v for k,v in r['placement_audit'].items() if not isinstance(v,list)}) for r in rows])
    base.write_csv('slab_power.csv',[s for r in rows for s in r['slabs']])
    base.write_csv('energy_breakdown.csv',[dict(model=r['summary']['model'],**r['energy']['components']) for r in rows])
    base.write_csv('thermal_summary.csv',[dict(model=r['summary']['model'],**r['thermal']) for r in rows])
    manifest=dict(git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),fingerprint=fp,
        baseline_head=frozen['git_head'],thermal_model=THERMAL_MODEL,thermal_cache=str(REFERENCE/'thermal_setup.pkl'),
        contexts=[126000,126999],batch_size=1,execution_policy='MAC_NMP',
        hardware_energy_parameters='UNCHANGED',objective='full physical stage latency',
        search='bounded resident group-chunk migration then region-local group-stream tile reassignment',
        oracle_boundary='fixed active topology relaxation; not a global placement bound',
        stage_hashes={r['summary']['model']:r['stage_hash'] for r in rows})
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print((OUT/'normalized.csv').read_text(),flush=True)


if __name__=='__main__':main()
