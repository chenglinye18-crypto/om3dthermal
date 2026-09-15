"""Extract one frozen CPA Decode step; never execute a benchmark or optimizer."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
FORMAL = ROOT / 'runs/formal_long_context_v2'
OUT = ROOT / 'runs/physical_decode_execution_anatomy'
TAG = 'Llama-3.1-8B_LC20K_B1'
KEY = TAG + '_M3D_NMP_CPA'
CACHE = Path('F:/om3dthermal_cache/cpa_feol_frequency_sweep') / TAG
RESOURCES = ('ARRAY', 'MAC', 'LOCAL_FABRIC', 'INTER_REGION_NOC', 'EXTERNAL_BOUNDARY', 'GPU_COMPUTE')


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as s:
        for b in iter(lambda: s.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def save(p, x):
    p.write_text(json.dumps(x, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def table(p, rows):
    with p.open('w', encoding='utf-8', newline='') as s:
        w = csv.DictWriter(s, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def rows(p):
    with p.open(encoding='utf-8', newline='') as s: return list(csv.DictReader(s))


def make_timeline(stages, layer=0):
    result = []; start = 0.0; local_start = None
    for index, s in enumerate(stages):
        end = start + s['latency_s']
        if s['layer'] == layer:
            if local_start is None: local_start = start
            result.append(dict(stage_index=index, operator=s['operator'], layer=layer,
                executor=s['executor'], start_s=start, end_s=end,
                layer_start_s=start-local_start, layer_end_s=end-local_start,
                latency_s=s['latency_s'], dominant_resource=max(s['components'], key=s['components'].get)))
        start = end
    return result


def subset_statistics():
    data = rows(FORMAL/'final_e2e_metrics.csv'); capacity = rows(FORMAL/'capacity_audit.csv')
    lookup = {(r['model'],r['context'],int(r['batch']),r['path']):r for r in data}
    caps = {(r['model'],r['context'],int(r['batch'])):r for r in capacity if r['path']=='HBM_GPU'}
    fits = {}
    for key, c in caps.items():
        candidate = json.loads((FORMAL/'candidates'/f'{key[0]}_{key[1]}_B{key[2]}_HBM_GPU.json').read_text(encoding='utf-8'))
        fits[key] = candidate['summary']['HBM_only_fit']
        assert fits[key] == (float(c['Grace_resident_GB']) == 0)
    pairs = [('M3D_GPU','HBM_GPU'),('M3D_NMP_UNIFORM','HBM_GPU'),('M3D_NMP_CPA','HBM_GPU'),
             ('M3D_NMP_UNIFORM','M3D_GPU'),('M3D_NMP_CPA','M3D_GPU')]
    stats=[]
    for subset in ('HBM_RESIDENT','HBM_OVERFLOW','ALL'):
        keys=[k for k in caps if subset=='ALL' or fits[k]==(subset=='HBM_RESIDENT')]
        for a,b in pairs:
            for metric in ('tokens_per_s','tokens_per_J'):
                value=math.exp(math.fsum(math.log(float(lookup[*k,a][metric])/float(lookup[*k,b][metric])) for k in keys)/len(keys))
                stats.append(dict(subset=subset,cases=len(keys),numerator=a,denominator=b,metric=metric,geometric_mean_ratio=value))
    table(OUT/'subset_statistics.csv',stats)
    table(OUT/'subset_membership.csv',[dict(model=k[0],context=k[1],batch=k[2],HBM_only_fit=fits[k],
        peak_state_GB=caps[k]['peak_state_GB'],Grace_resident_GB=caps[k]['Grace_resident_GB']) for k in caps])
    absolute=[]
    for key in [('Llama-3.1-8B','LC20K',1),('Llama-3.1-70B','LC20K',8),('Llama-3.1-405B','LC20K',1)]:
        for path in ('HBM_GPU','M3D_GPU','M3D_NMP_UNIFORM','M3D_NMP_CPA'):
            r=lookup[*key,path]
            c=json.loads((FORMAL/'candidates'/f'{key[0]}_{key[1]}_B{key[2]}_{path}.json').read_text(encoding='utf-8'))['summary']
            absolute.append(dict(model=key[0],context=key[1],batch=key[2],path=path,HBM_only_fit=fits[key],
                E2E_tokens_per_s=float(r['tokens_per_s']),mean_TPOT_ms=c['mean_TPOT_ms'],TTFT_s=c['TTFT']))
    table(OUT/'representative_absolute_metrics.csv',absolute)
    return stats


def extract():
    import formal_parallel_runtime as shared
    candidate=FORMAL/'candidates'/f'{KEY}.json'
    c=json.loads(candidate.read_text(encoding='utf-8'))
    fingerprint=c.get('legacy_fingerprint',c['fingerprint'])[:16]
    checkpoint=FORMAL/'checkpoints'/f'{TAG}_IOM3D_MAC_NMP_CPA_{fingerprint}.jsonl'
    with checkpoint.open(encoding='utf-8') as s: expected=json.loads(s.readline())
    sources=[candidate,checkpoint,CACHE/'engine.pkl',CACHE/'arrays.bin',
        ROOT/'src/om3dthermal/serving/decode_policy.py',ROOT/'src/om3dthermal/power/nmp_die_activity.py',
        ROOT/'src/om3dthermal/power/batched_physical.py',ROOT/'src/om3dthermal/placement/critical_path.py',
        ROOT/'configs/architecture/m3d_feol_execution.yaml']
    hashes={str(p):sha(p) for p in sources}
    shared.initialize(CACHE); e=shared.ENGINE
    assert e.floorplan.config['clock_hz']==1e9 and e.floorplan.external_Bps==3.4e12
    # One step only; do not instantiate an optimizer or invoke the sweep runner.
    actual=e.step(expected['context'],'MAC_NMP',include_stages=True)
    for k in ('latency_s','boundary_bytes','local_array_bytes','nmp_flops','external_service_s','array_service_s','noc_bytes'):
        assert math.isclose(actual[k],expected[k],rel_tol=1e-12,abs_tol=1e-15),(k,actual[k],expected[k])
    for k,v in expected['component_sums'].items():
        assert math.isclose(actual['component_sums'][k],v,rel_tol=1e-12,abs_tol=1e-15)
    for k,v in expected['energy_events'].items():
        np.testing.assert_allclose(actual['energy_events'][k],v,rtol=1e-12,atol=1e-7)
    stages=[]
    for s in actual['stages']:
        stages.append({k:s[k] for k in ('operator','layer','executor','latency_s','components','bottleneck')})
    save(OUT/'replayed_step.json',dict(context=actual['context'],latency_s=actual['latency_s'],stages=stages))
    timeline=make_timeline(stages); table(OUT/'timeline.csv',timeline)
    matrix=[]
    for row in timeline:
        s=stages[row['stage_index']]; peak=max(s['components'].values())
        for resource in RESOURCES:
            duration=s['components'].get(resource,0.0)
            matrix.append(dict(stage_index=row['stage_index'],operator=s['operator'],layer=0,resource=resource,
                service_s=duration,normalized_service=duration/peak if peak else 0,
                dominant=resource==row['dominant_resource']))
    table(OUT/'operator_resource_matrix.csv',matrix)
    provenance=dict(source_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        model='Llama-3.1-8B',context='LC20K',H=20000,P=128,G=32,B=1,path='M3D_NMP_CPA',
        decode_step_one_based=1,decode_context=actual['context'],layer_zero_based=0,
        replayed_checkpoints=1,optimizer_runs=0,checkpoint_comparison='PASS',source_sha256=hashes,
        timeline='RECONSTRUCTED_DEPENDENCY_ORDERED_OPERATOR_INTERVALS',
        resource_lane_semantics='Dominant service component of the entire operator; not resource occupancy intervals.',
        metric='service_s(resource,operator)/max_resource_service_s(operator)',
        NOC='INTER_REGION_NOC includes modeled reduction; not pure NoC link busy time.',
        GPU='GPU_COMPUTE is a legacy label; small GPU operators here use GPU-local byte service.',
        full_step_s=actual['latency_s'],layer_s=sum(r['latency_s'] for r in timeline),
        achieved_boundary_TBps=actual['boundary_bytes']/actual['external_service_s']/1e12)
    save(OUT/'provenance.json',provenance)
    print(json.dumps(provenance,indent=2))


def plot():
    timeline=rows(OUT/'timeline.csv'); matrix=rows(OUT/'operator_resource_matrix.csv')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'axes.linewidth':.6,'svg.fonttype':'none','svg.hashsalt':'physical-decode-anatomy-v1',
        'pdf.fonttype':42,'savefig.facecolor':'white'})
    fig,(ax,hm)=plt.subplots(1,2,figsize=(7.15,2.65),gridspec_kw={'width_ratios':[1.5,1]})
    lane={'ARRAY':0,'MAC':1,'LOCAL_FABRIC':2,'INTER_REGION_NOC':2,'EXTERNAL_BOUNDARY':3,'GPU_COMPUTE':4}
    colors=['#4276A5','#BE7540','#7B659C','#D0AA42','#54917A']
    short={'ATTENTION_QK':'QK','ATTENTION_AV':'AV','FFN_GATE':'Gate','FFN_UP':'Up','FFN_DOWN':'Down'}
    for r in timeline:
        x=float(r['layer_start_s'])*1e6; width=float(r['latency_s'])*1e6; y=lane[r['dominant_resource']]
        ax.broken_barh([(x,width)],(y-.27,.54),facecolors=colors[y],edgecolors='white',linewidth=.3)
        if width>2: ax.text(x+width/2,y,short.get(r['operator'],r['operator']),ha='center',va='center',fontsize=6,color='white')
    ax.set_yticks(range(5),['Array + MIV','MAC','Fabric / NoC*','Boundary','GPU'])
    ax.invert_yaxis(); ax.set_ylim(4.65,-.65)
    ax.set_xlim(0,float(timeline[-1]['layer_end_s'])*1e6)
    ax.set_xlabel('Time from layer start (µs)')
    ax.set_title('(a) Dependency-ordered operator intervals',fontsize=7,pad=7)
    for side in ('top','right'): ax.spines[side].set_visible(False)
    ax.text(.0,-.28,'Lanes identify the dominant component, not busy intervals.',transform=ax.transAxes,fontsize=6)
    chosen=[r for r in timeline if float(r['latency_s'])>=.1e-6]
    lookup={(int(r['stage_index']),r['resource']):r for r in matrix}
    values=np.array([[float(lookup[int(r['stage_index']),k]['normalized_service']) for k in RESOURCES] for r in chosen])
    im=hm.imshow(values,vmin=0,vmax=1,cmap='YlGnBu',aspect='auto',interpolation='nearest')
    hm.set_yticks(range(len(chosen)),[short.get(r['operator'],r['operator']) for r in chosen],fontsize=6)
    hm.set_xticks(range(6),['Array\n+ MIV','MAC','Fabric','NoC*','Bdry.','GPU'],fontsize=6)
    hm.tick_params(length=0)
    for i,r in enumerate(chosen):
        j=RESOURCES.index(r['dominant_resource']); hm.add_patch(Rectangle((j-.49,i-.49),.98,.98,fill=False,edgecolor='black',lw=.8))
    hm.set_title('(b) Resource service / operator maximum',fontsize=7,pad=7)
    cb=fig.colorbar(im,ax=hm,fraction=.04,pad=.025,ticks=[0,.5,1]); cb.ax.tick_params(labelsize=6)
    hm.text(0,-.28,'*NoC includes reduction.\nBlack box: dominant component.',transform=hm.transAxes,fontsize=6)
    fig.subplots_adjust(left=.105,right=.94,bottom=.24,top=.88,wspace=.48)
    for ext in ('svg','pdf'):
        fig.savefig(OUT/f'figure.{ext}',metadata={'Date':None} if ext=='svg' else {'CreationDate':None,'ModDate':None})
    svg=OUT/'figure.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--extract',action='store_true');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    protected=[p for p in FORMAL.rglob('*') if p.is_file() and p.suffix in ('.csv','.json','.jsonl')]
    before={str(p):sha(p) for p in protected}
    if args.extract or not (OUT/'replayed_step.json').exists(): extract()
    stats=subset_statistics(); plot()
    assert before=={str(p):sha(p) for p in protected},'Canonical artifacts changed'
    save(OUT/'canonical_preservation.json',dict(status='BYTE_IDENTICAL',files=before))
    print('canonical files unchanged:',len(before))
    for r in stats:
        if r['metric']=='tokens_per_s':print(r)


if __name__=='__main__':main()
