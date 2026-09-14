"""Optional context-parallel checkpoint producer for the same formal matrix.

Run the canonical capacity gate first. The ordinary matrix runner consumes
these identical physical checkpoints and remains the single table writer.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import run_formal_iom3d_workloads as runner
from formal_parallel_runtime import ParallelDecodeModel
from om3dthermal.serving.workload_matrix import capacity


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',required=True)
    parser.add_argument('--workload',required=True,choices=('W1','W2','W3'))
    parser.add_argument('--batch',required=True,type=int,choices=(1,8,32))
    parser.add_argument('--system',required=True,choices=('IOM3D_NO_NMP','IOM3D_MAC_NMP_UNIFORM','IOM3D_MAC_NMP_CPA'))
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    fp=runner.fingerprint()
    manifest_path=runner.OUT/'manifest.json'
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if manifest.get('source_fingerprint')==fp:
        rows=list(csv.DictReader((runner.OUT/'capacity.csv').open()))
        row=next(r for r in rows if (r['model'],r['workload_id'],int(r['batch_size']),r['system'])==(args.model,args.workload,args.batch,args.system))
    else:
        row=capacity(args.model,args.workload,args.batch,args.system,runner.ROOT)
    for k in ('H','P','G','batch_size'):row[k]=int(row[k])
    for k in row:
        if k.endswith('_GB') or k in ('capacity_utilization','kv_state_fraction'):row[k]=float(row[k])
    row['HBM_only_fit']=str(row['HBM_only_fit'])=='True'
    ParallelDecodeModel.workers=args.workers
    runner.DecodePolicyModel=ParallelDecodeModel
    try:
        runner.run_case((row,fp))
    finally:
        for engine in list(ParallelDecodeModel.active):engine.close()
    sources=[Path(__file__),Path(__file__).with_name('formal_parallel_runtime.py')]
    runtime_hash=hashlib.sha256(b''.join(p.read_bytes() for p in sources)).hexdigest()
    key=f'{args.model}_{args.workload}_B{args.batch}_{args.system}_{fp[:16]}_{runtime_hash[:16]}'
    runner.save(runner.OUT/'runtime_audit'/f'{key}.json',dict(physical_source_fingerprint=fp,
        runtime_source_fingerprint=runtime_hash,context_workers=args.workers,
        engine='UNCHANGED_DecodePolicyModel',scheduling='READ_ONLY_MMAP__ORDERED_COMPLETE_CONTEXTS',
        cache_contract='PHYSICAL_RESULTS_SHARED_ONLY_AFTER_EXACT_RUNTIME_EQUIVALENCE_VALIDATION'))


if __name__=='__main__':main()
