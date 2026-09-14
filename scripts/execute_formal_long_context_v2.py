"""Conservative host-only case orchestration; no sampling or pytest."""
import concurrent.futures
import ctypes
import hashlib
import json
import os
import subprocess
import sys
import time
import formal_long_context_v2_support as f


class MemoryStatus(ctypes.Structure):
    _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in
        ('total','available','total_page','available_page','total_virtual','available_virtual','extended')]


def main():
    started=time.time();f.OUT.mkdir(parents=True,exist_ok=True)
    memory=MemoryStatus();memory.length=ctypes.sizeof(memory)
    assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory))
    cpu=os.cpu_count()
    # CPA planning is parent-side; favor independent cases over context workers.
    outer=4 if memory.available>=16e9 and cpu>=16 else 2 if memory.available>=12e9 and cpu>=8 else 1
    inner=1 if outer==4 else max(1,min(4,cpu//(outer+1)))
    assert outer*inner<=cpu
    files=subprocess.check_output(['git','ls-files','--','src','configs','runs'],text=True).splitlines()
    protected={p:hashlib.sha256((f.ROOT/p).read_bytes()).hexdigest() for p in files
               if not p.startswith('runs/formal_long_context_v2/')}
    f.save(f.OUT/'protected_hashes.json',protected)
    prior=list((f.OUT/'runtime_rows').glob('*.json'))
    earliest=min([started,*[p.stat().st_mtime-json.loads(p.read_text())['elapsed_s'] for p in prior]])
    previous=json.loads((f.OUT/'execution_manifest.json').read_text()) if (f.OUT/'execution_manifest.json').exists() else {}
    history=previous.get('runtime_configurations',[])
    history.append(dict(start_epoch=started,outer_processes=outer,inner_workers=inner,available_RAM_bytes=memory.available))
    metadata=dict(start_epoch=min(earliest,previous.get('start_epoch',earliest)),orchestration_start_epoch=started,cpu_count=cpu,
        total_RAM_bytes=memory.total,available_RAM_bytes=memory.available,outer_processes=outer,inner_workers=inner,
        v1_execution_reuse=False,pytest_run=False,runtime_configurations=history,
        base_HEAD=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    f.save(f.OUT/'execution_manifest.json',metadata)
    jobs=[(m,c,b,p) for p in f.PATHS for m,c,b in f.points()
          if not (f.OUT/'candidates'/f'{m}_{c}_B{b}_{p}.json').exists()]

    def launch(job):
        m,c,b,p=job;key=f'{m}_{c}_B{b}_{p}'
        log=f.OUT/'logs'/f'{key}.log';log.parent.mkdir(exist_ok=True)
        with log.open('w',encoding='utf-8') as stream:
            result=subprocess.run([sys.executable,'-u',str(f.ROOT/'scripts/run_formal_long_context_v2.py'),
                '--model',m,'--context',c,'--batch',str(b),'--path',p,'--workers',str(inner)],
                stdout=stream,stderr=subprocess.STDOUT,env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
        if result.returncode:
            raise RuntimeError(f'{key}: {log.read_text(encoding="utf-8")[-4000:]}')
        print('COMPLETE',key,flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=outer) as pool:
        for _ in pool.map(launch,jobs):pass
    metadata['physical_runs_end_epoch']=time.time()
    f.save(f.OUT/'execution_manifest.json',metadata)
    print('ALL_PHYSICAL_RUNS_COMPLETE',flush=True)


if __name__=='__main__':main()
