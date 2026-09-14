"""Isolated formal LC matrix; legacy W1/B32 producers and physics stay frozen."""
from copy import deepcopy
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/formal_long_context_v1'
OLD=ROOT/'runs/formal_iom3d_workload_sweep_v1'
MODELS=('Llama-3.1-8B','Llama-3.1-70B','Llama-3.1-405B')
CONTEXTS={'LC20K':('W2',20000,512,256),'LC64K':('W64',64000,512,512),'LC126K':('W3',126000,512,512)}
PATHS=('HBM_GPU','M3D_GPU','M3D_NMP_UNIFORM','M3D_NMP_CPA')
LEGACY={'HBM_GPU':'HBM_GRACE_C2C','M3D_GPU':'IOM3D_NO_NMP','M3D_NMP_UNIFORM':'IOM3D_MAC_NMP_UNIFORM','M3D_NMP_CPA':'IOM3D_MAC_NMP_CPA'}
EXTERNAL_MODE='SUFFICIENTLY_PROVISIONED_FOR_FORMAL_BASELINE'
HBM_ACCESS_PJ=1.9955


def private_module(stem):
    # Execute the unchanged legacy helpers in an isolated namespace. Their
    # config overlay cannot change callers of the public legacy modules.
    name='om3dthermal.serving._formal_lc_'+stem
    spec=importlib.util.spec_from_file_location(name,ROOT/'src/om3dthermal/serving'/f'{stem}.py')
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


legacy=private_module('workload_matrix')
base_setup=legacy.setup


def setup(root):
    config,*rest=base_setup(root)
    config=deepcopy(config)
    config['workloads']['W64']=dict(history=64000,prompt=512,generated=512)
    # Symbolic unbounded capacity only, never an invented external TB number.
    # The existing allocator, overlap, C2C BW and access coefficients are exact.
    config['grace_capacity_bytes']=math.inf
    return config,*rest


legacy.setup=setup
primary=private_module('primary_execution')
primary.inputs=legacy.inputs;primary.setup=setup


def points():
    return [(m,c,b) for m in MODELS for c in CONTEXTS for b in (1,8)]


def identity(m,c,b):
    if (m,c,b) not in points():raise ValueError('Not a formal B1/B8 long-context case')
    _,h,p,g=CONTEXTS[c]
    return dict(model=m,context_label=c,H=h,P=p,G=g,B=b)


def fingerprint():
    paths=[*sorted((ROOT/'src').rglob('*.py')),*sorted((ROOT/'configs').rglob('*.yaml'))]
    return hashlib.sha256(b''.join(p.read_bytes() for p in paths)+json.dumps(CONTEXTS,sort_keys=True).encode()).hexdigest()


def cap(m,c,b,path):
    identity(m,c,b)
    row=legacy.capacity(m,CONTEXTS[c][0],b,LEGACY[path],ROOT)
    if path=='HBM_GPU':
        row.update(external_capacity_mode=EXTERNAL_MODE)
        for k in ('capacity_GB','Grace_capacity_GB','capacity_utilization'):row.pop(k,None)
    return row


def hbm(m,c,b):
    row=cap(m,c,b,'HBM_GPU')
    assert row['status']=='EVALUATED'
    r=legacy.conventional(m,CONTEXTS[c][0],b,ROOT,row)
    t,s,e=r['traffic'],r['summary'],r['energy']
    platform=setup(ROOT)[2]
    local=(t['HBM_read_GB']+t['HBM_write_GB']+t['prefill_HBM_read_GB']+t['prefill_HBM_write_GB'])*1e9*8*HBM_ACCESS_PJ*1e-12
    external=t['total_C2C_GB']*1e9*8*5.3e-12
    pc=platform.gpu_compute_power
    gpu=(e['GPU_dynamic_J']+s['E2E_s']*platform.gpu_decode_power.static_power_W+
        r['prefill_ledger']['total_flops']*(pc.e_compute_dynamic_J_per_FLOP_min+pc.e_compute_dynamic_J_per_FLOP_max)/2)
    n=b*CONTEXTS[c][3];total=gpu+local+external
    r['energy']=dict(energy_status='HBM_ENERGY_CLOSED',hbm_access_energy_pj_per_bit=HBM_ACCESS_PJ,
        GPU_energy_J=gpu,local_memory_energy_J=local,external_memory_energy_J=external,
        E2E_J=total,E2E_J_per_token=total/n,E2E_tokens_per_J=n/total,
        HBM_static_power_W=0,Grace_static_power_W=0)
    r['summary']['system']='HBM_GPU'
    return r


def read_csv(path):
    with path.open(encoding='utf-8') as f:return list(csv.DictReader(f))


def parse(r):
    out={}
    for k,v in r.items():
        if v=='':continue
        try:out[k]=float(v)
        except ValueError:out[k]=v
    return out


def existing(m,c,b,path):
    assert c in ('LC20K','LC126K')
    wid=CONTEXTS[c][0]
    f=next((OLD/'checkpoints').glob(f'{m}_{wid}_B{b}_{LEGACY[path]}_*.json'))
    r=json.loads(f.read_text())
    if path=='M3D_GPU':
        match=lambda x:x['model']==m and x['workload_id']==wid and int(x['batch_size'])==b and x['system']=='M3D_GPU'
        r['summary']=parse(next(x for x in read_csv(OLD/'primary_summary.csv') if match(x)))
        r['energy']=parse(next(x for x in read_csv(OLD/'primary_energy.csv') if match(x)))
    r['summary']['system']=path
    r['source_checkpoint']=str(f.relative_to(ROOT));r['reused']=True
    return r


def save(path,obj):
    from run_formal_iom3d_workloads import save as atomic_save
    atomic_save(path,obj)


def csv_out(name,rows):
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader();w.writerows(rows)
