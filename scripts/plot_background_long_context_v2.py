"""Background workload properties from frozen v3 ledgers; no execution simulation."""
from pathlib import Path
import csv
import hashlib
import json
import sys
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from run_formal_long_context_v3 import CONFIG, POINTS, inputs
from preflight_formal_long_context_v3 import footprint
from om3dthermal.workload import evaluate_cached_prefix_incremental_prefill, evaluate_llm_decode
from om3dthermal.serving.workload_matrix import setup

OUT = ROOT / 'runs/background_long_context_v2'
FORMAL = ROOT / 'runs/formal_long_context_v3'
MODELS = list(CONFIG['models'])

def read_csv(path):
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def write_csv(name, rows):
    with (OUT/name).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def hashes():
    return {str(p.relative_to(FORMAL)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in FORMAL.rglob('*') if p.is_file()}

def font(size=9, bold=False):
    p = FontProperties(fname=r'C:\Windows\Fonts\timesbd.ttf' if bold else r'C:\Windows\Fonts\times.ttf', size=size)
    p.set_family('Times New Roman'); p.set_weight('bold' if bold else 'normal')
    return p

def style(ax):
    ax.tick_params(direction='in', top=True, right=True, width=.65)
    for t in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        t.set_fontproperties(font(8.5))
    ax.grid(axis='y', linestyle='--', color='#dddddd', linewidth=.45, zorder=0)
    ax.set_axisbelow(True)

def save(fig, name):
    for ext in ('pdf','svg'):
        p=OUT/f'{name}.{ext}'
        try: fig.savefig(p)
        except PermissionError:
            p=OUT/f'{name}_new.{ext}'; print(f'WARNING: locked output, saved {p}'); fig.savefig(p)
        if ext == 'svg':
            p.write_text('\n'.join(line.rstrip() for line in p.read_text(encoding='utf-8').splitlines())+'\n', encoding='utf-8')
    plt.close(fig)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    before=hashes()
    workspace=setup(ROOT)[1]
    capacity=[]
    audit=read_csv(FORMAL/'capacity_audit.csv')
    for m in MODELS:
        for b in CONFIG['batches']:
            r=next(r for r in audit if r['model']==m and r['workload']=='LC126K' and int(r['B'])==b)
            weights,kv,cap=map(float,(r['weight_bytes'],r['final_KV_bytes'],r['HBM_capacity_bytes']))
            _,_,_,spec,cw=inputs(POINTS.index((m,'LC126K',b)))
            current=footprint(spec,cw.history,b,cw.prompt,cw.generated,workspace)
            assert current['weight_bytes']==weights and current['final_KV_bytes']==kv
            capacity.append(dict(model=m,H=126000,P=128,G=32,B=b,weight_GB=weights/1e9,KV_GB=kv/1e9,
                total_GB=(weights+kv)/1e9,HBM_usable_GB=cap/1e9,
                persistent_status='RESIDENT' if weights+kv<=cap else 'OVERFLOW',formal_fit=r['HBM_full_batch_fit']))
        group=capacity[-3:]
        assert all(math.isclose(r['KV_GB']/r['B'], group[0]['KV_GB'], rel_tol=1e-12) for r in group)
    ai=[]
    for m in MODELS:
        for c in CONFIG['contexts']:
            _,_,b,spec,cw=inputs(POINTS.index((m,c,1)))
            pre=evaluate_cached_prefix_incremental_prefill(spec.prefill_input(batch_size=b,prompt_length=cw.prompt),cached_history_tokens=cw.history)
            canonical=json.loads((FORMAL/'candidates'/f'{m}_{c}_B1_M3D_GPU.json').read_text())['prefill']['ledger']
            assert math.isclose(pre.total_flops,canonical['total_flops'],rel_tol=1e-12)
            assert math.isclose(pre.total_memory_bytes,canonical['total_memory_bytes'],rel_tol=1e-12)
            dec=[evaluate_llm_decode(spec.decode_input(batch_size=b,context_length=k)) for k in cw.contexts]
            for phase,f,d in [('Prefill',pre.total_flops,pre.total_memory_bytes),('Decode',sum(x.flops_per_token*b for x in dec),sum((x.read_bytes_per_token+x.write_bytes_per_token)*b for x in dec))]:
                ai.append(dict(model=m,H=cw.history,P=cw.prompt,G=cw.generated,B=b,phase=phase,FLOPs=f,bytes=d,AI_FLOP_per_byte=f/d))
    _,_,platform,backend,_=setup(ROOT)
    bw=float(next(r for r in read_csv(ROOT/'runs/no_nmp_geometry_sensitivity_v2/thermal_limits.csv') if r['architecture']=='conventional_hbm_2x1')['Bthermal_TBps'])*1e12
    compute=platform.gpu_prefill_compute.large_gemm_effective_tflops*1e12
    assert compute==platform.gpu_prefill_compute.causal_attention_effective_tflops*1e12
    assert backend.capacity_bytes==capacity[0]['HBM_usable_GB']*1e9
    ridge=dict(effective_compute_FLOP_s=compute,effective_HBM_bytes_s=bw,AI_ridge_FLOP_per_byte=compute/bw,
        decode_compute_ceiling_FLOP_s=platform.gpu_compute_power.peak_compute_BF16_dense_flops_per_s,
        compute_source='configs/platform/gpu_package_h200_reference.yaml: gpu_prefill_compute',
        bandwidth_source='runs/no_nmp_geometry_sensitivity_v2/thermal_limits.csv: conventional_hbm_2x1',
        definition='Effective Prefill GPU roofline; Decode evaluator separately uses configured dense peak ceiling')
    write_csv('capacity_data.csv',capacity); write_csv('arithmetic_intensity_data.csv',ai)
    (OUT/'ridge_point.json').write_text(json.dumps(ridge,indent=2)+'\n')
    plt.rcParams.update({'font.family':'Times New Roman','svg.fonttype':'none','pdf.fonttype':42,'axes.linewidth':.65,'figure.facecolor':'white'})
    fig,ax=plt.subplots(figsize=(3.65,2.65))
    x=[0,1,2,3.6,4.6,5.6]
    ax.bar(x,[r['weight_GB'] for r in capacity],.68,color='#806589',edgecolor='#444444',linewidth=.4,label='Model weights')
    ax.bar(x,[r['KV_GB'] for r in capacity],.68,bottom=[r['weight_GB'] for r in capacity],color='#D19A5A',edgecolor='#444444',linewidth=.4,label='KV cache')
    for xx,r in zip(x,capacity): ax.text(xx,r['total_GB']+20,f"{r['total_GB']:.1f}",ha='center',fontproperties=font(8))
    cap=capacity[0]['HBM_usable_GB']
    ax.axhline(cap,color='#555555',linestyle='--',linewidth=.8)
    ax.text(.39,cap+15,'H200 HBM',transform=ax.get_yaxis_transform(),fontproperties=font(8))
    ax.set_xticks(x,['B1','B8','B32']*2)
    for center,m in zip((1,4.6),MODELS): ax.text(center,-.18,m,transform=ax.get_xaxis_transform(),ha='center',fontproperties=font(8.5))
    ax.set_ylabel('Working-set Capacity (GB)',fontproperties=font(9))
    ax.set_ylim(0,1280); style(ax)
    ax.legend(loc='lower center',bbox_to_anchor=(.5,1.01),ncol=2,frameon=False,prop=font(8.5))
    fig.subplots_adjust(left=.18,right=.98,bottom=.25,top=.84);save(fig,'fig1_capacity')
    fig,ax=plt.subplots(figsize=(3.65,2.65))
    for m,color in zip(MODELS,('#527DA2','#956483')):
        for phase,marker,ls in [('Decode','o','-'),('Prefill','s','--')]:
            rows=[r for r in ai if r['model']==m and r['phase']==phase]
            ax.plot(range(3),[r['AI_FLOP_per_byte'] for r in rows],color=color,marker=marker,linestyle=ls,linewidth=1,markersize=4,label=f"{'8B' if m==MODELS[0] else '32B'} {phase}")
    level=compute/bw
    ax.axhline(level,color='#555555',linestyle='--',linewidth=.8)
    ax.text(.02,level*1.12,'GPU Ridge Point',transform=ax.get_yaxis_transform(),fontproperties=font(7.5))
    ax.text(.99,.96,'Compute-bound',transform=ax.transAxes,ha='right',va='top',fontproperties=font(7.5))
    ax.text(.99,.65,'Bandwidth-bound',transform=ax.transAxes,ha='right',fontproperties=font(7.5))
    ax.set_yscale('log');ax.set_ylim(.7,max(level*2.7,max(r['AI_FLOP_per_byte'] for r in ai)*1.5))
    ax.set_xticks(range(3),['20K','64K','126K']);ax.set_xlabel('Context Length',fontproperties=font(9))
    ax.set_ylabel('Arithmetic Intensity (FLOP/Byte)',fontproperties=font(9));style(ax)
    ax.legend(loc='lower center',bbox_to_anchor=(.5,1.01),ncol=2,frameon=False,prop=font(7.5),columnspacing=.9)
    fig.subplots_adjust(left=.18,right=.98,bottom=.21,top=.77);save(fig,'fig1_arithmetic_intensity')
    readme='''# Long-context background Fig. 1 v2

Fig.1 data are derived from the same model/workload definitions used by the new formal benchmark.

Capacity: Llama-3.1-8B and Qwen2.5-32B, H=126000, P=128, G=32, B={1,8,32}. Weights and final KV(H+P+G) are copied from formal_long_context_v3/capacity_audit.csv. Decimal GB = 1e9 bytes. This figure shows the major persistent working-set components; formal capacity preflight additionally includes workspace/runtime state. H200 usable capacity is taken from the same audit and checked against the canonical resolved backend. Resident/overflow in capacity_data.csv refers to persistent components; formal_fit preserves the complete preflight verdict. At 126K, 8B B8 already slightly overflows; it must not be represented as resident.

AI: B=1 for all four series (8B=Llama-3.1-8B; 32B=Qwen2.5-32B), H={20000,64000,126000}, P=128, G=32. Prefill AI = canonical incremental-prefill total_flops / total_memory_bytes, using evaluate_cached_prefix_incremental_prefill, with H cached and only P new tokens projected. Prefill values are checked against saved M3D_GPU candidate ledgers. Decode AI = sum of canonical evaluate_llm_decode FLOPs across G steps / sum of read+write bytes across G steps, contexts H+P+j. This is semantic local-memory traffic, not remote offload or physical duplicated traffic. Both phases use the same formal model specs via run_formal_long_context_v3.inputs, including the exact Llama parameter override. No model parameters are guessed.

Ridge: effective Prefill compute from gpu_package_h200_reference.yaml divided by canonical thermal-closed HBM bandwidth in no_nmp_geometry_sensitivity_v2/thermal_limits.csv, the same bandwidth used by run_formal_long_context_v3.hbm. Both Prefill compute families use 700 TFLOP/s. Decode separately uses the configured 989.5 TFLOP/s dense peak ceiling; this distinction is preserved in ridge_point.json. The effective ridge is a GPU roofline reference, not a claim that Prefill must be compute-bound. Lines connect three sampled contexts only as a guide to the eye. Actual AI values and ridge-side findings must be reported without adjusting the workload.

Outputs are separate vector PDF/SVG, Times New Roman, no panel titles. AI legend uses two compact rows to remain readable at single-column width. Benchmark runs=0; thermal runs=0; CPA runs=0. Canonical data are read-only.
'''
    (OUT/'README.md').write_text(readme,encoding='utf-8')
    assert before==hashes()
    checks=dict(formal_data_modified=False,formal_files=len(before),benchmark_runs=0,thermal_runs=0,CPA_runs=0,capacity_linear_scaling='PASS',prefill_canonical_ledger='PASS',AI_unit_closure='PASS')
    (OUT/'validation.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(dict(capacity=capacity,AI=ai,ridge=ridge,checks=checks),indent=2))

if __name__=='__main__': main()
