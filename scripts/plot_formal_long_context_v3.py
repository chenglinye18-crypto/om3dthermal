"""Paper figures for v3; read-only benchmark data, vector SVG/PDF output."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/formal_long_context_v3'
STYLES=(('HBM_BEST','HBM','#B8BEC5','///'),('M3D_GPU','M3D-GPU','#6C98C4',''),
        ('M3D_NMP_UNIFORM','+DNS','#E6B65C','..'),('M3D_NMP_CPA','+CPA','#4A998B','\\\\\\\\'))
MODELS=('Llama-3.1-8B','Qwen2.5-32B')
CONTEXTS=('LC20K','LC64K','LC126K')
BATCHES=(1,8,32)
CASES=[(m,c,b) for m in MODELS for c in CONTEXTS for b in BATCHES]
CENTERS=[mi*10+ci*3+bi*.8 for mi in range(2) for ci in range(3) for bi in range(3)]


def load(name):
    with (OUT/name).open(encoding='utf-8',newline='') as s:return list(csv.DictReader(s))


def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.2,'axes.labelsize':9,
        'xtick.labelsize':7,'ytick.labelsize':7.8,'legend.fontsize':7.6,'axes.linewidth':1.,
        'xtick.major.width':.7,'ytick.major.width':.7,'xtick.major.size':3,'ytick.major.size':3,
        'hatch.linewidth':.35,'svg.fonttype':'none','pdf.fonttype':42,
        'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white'})


def decorate(ax):
    ax.set_axisbelow(True)
    ax.grid(axis='y',color='#D8D8D8',linewidth=.4,linestyle='--',dashes=(2,2))
    for spine in ax.spines.values():spine.set_linewidth(1);spine.set_color('#202020')


def save(fig,name):
    out=OUT/'figures';out.mkdir(parents=True,exist_ok=True)
    for extension in ('svg','pdf'):fig.savefig(out/f'{name}.{extension}',format=extension)
    plt.close(fig)


def main():
    style()
    rows=load('final_e2e_metrics.csv')
    data={(r['model'],r['context'],int(r['batch']),r['path']):r for r in rows}
    if len(data)!=72:raise ValueError('Expected 72 unique formal rows')
    handles=[Patch(facecolor=color,edgecolor='#303030',linewidth=.5,hatch=hatch,label=label) for _,label,color,hatch in STYLES]
    for metric,ylabel,name in [('tokens_per_s','Normalized E2E Throughput','fig_normalized_throughput'),
        ('tokens_per_J','Normalized Energy Efficiency','fig_normalized_energy_efficiency'),
        ('Tmax_C','Peak Temperature (°C)','fig_temperature')]:
        fig=plt.figure(figsize=(7.4,2.7));ax=fig.add_axes((.09,.27,.89,.58))
        maximum=0;minimum=float('inf')
        for i,(path,label,color,hatch) in enumerate(STYLES):
            values=[]
            for m,c,b in CASES:
                value=float(data[m,c,b,path][metric])
                if metric!='Tmax_C':
                    value/=float(data[m,c,b,'HBM_BEST'][metric])
                    if path=='HBM_BEST':assert value==1
                values.append(value)
            maximum=max(maximum,max(values))
            minimum=min(minimum,min(values))
            ax.bar([x+(i-1.5)*.17 for x in CENTERS],values,width=.17,color=color,edgecolor='#303030',linewidth=.45,hatch=hatch,zorder=3)
        ax.set_xlim(-.55,18.15);ax.set_xticks(CENTERS,[f'B{b}' for _,_,b in CASES])
        transform=ax.get_xaxis_transform()
        for mi,model in enumerate(('8B','32B')):
            for ci,context in enumerate(('20K','64K','126K')):
                ax.text(mi*10+ci*3+.8,-.17,context,ha='center',va='top',transform=transform,fontsize=8)
            ax.text(mi*10+3.8,-.31,model,ha='center',va='top',transform=transform,fontsize=9,fontweight='bold')
            for x in (mi*10+2.3,mi*10+5.3):ax.plot([x,x],[0,-.15],transform=transform,clip_on=False,color='#444444',linewidth=.5)
        for x in (-.55,8.8,18.15):ax.plot([x,x],[0,-.29],transform=transform,clip_on=False,color='#303030',linewidth=.85)
        decorate(ax)
        ax.set_ylabel(ylabel,fontweight='bold')
        if metric=='Tmax_C':
            ax.set_ylim(0,max(100,maximum*1.08));ax.set_yticks((0,25,50,75,100))
            ax.axhline(85,color='#A63732',linewidth=.9,linestyle=(0,(4,2)),zorder=4)
            ax.text(18,86.5,'85°C Thermal Limit',ha='right',va='bottom',fontsize=7,color='#A63732')
        elif metric=='tokens_per_s':
            ax.set_yscale('log')
            ax.set_ylim(min(.8,minimum*.9),maximum*1.15)
            ticks=[v for v in (.5,1,2,5,10,20,50,100) if minimum*.8<=v<=maximum*1.15]
            ax.set_yticks(ticks,[f'{v:g}' for v in ticks])
        else:
            ax.set_ylim(0,maximum*1.1);ax.yaxis.set_major_locator(MaxNLocator(nbins=5,min_n_ticks=3))
        ax.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,1.015),ncol=4,frameon=False,
            handlelength=1.5,columnspacing=1.2,borderaxespad=0)
        save(fig,name)
    overflow=[r for r in load('hbm_policy_comparison.csv') if r['HBM_overflow']=='True']
    fig,axes=plt.subplots(1,2,figsize=(7.4,2.7));fig.subplots_adjust(left=.09,right=.98,bottom=.3,top=.82,wspace=.3)
    for ax,metric,title in zip(axes,('tokens_per_s','P95_completion_latency'),('(a) HBM overflow throughput','(b) P95 completion latency')):
        for i,(policy,label,color,hatch) in enumerate([('host','HOST_OFFLOAD','#B8BEC5','///'),('wave','RESIDENT_WAVE','#6C98C4','')]):
            values=[float(r[policy+'_'+metric]) for r in overflow]
            ax.bar([x+(i-.5)*.35 for x in range(len(overflow))],values,width=.35,color=color,edgecolor='#303030',linewidth=.45,hatch=hatch,label=label,zorder=3)
        ax.set_xticks(range(len(overflow)),[f"{'8B' if r['model']==MODELS[0] else '32B'}\n{r['context'][2:]}\nB{r['batch']}" for r in overflow],fontsize=6.5)
        ax.set_ylabel('Tokens/s' if metric=='tokens_per_s' else 'Seconds');ax.set_title(title,fontsize=8.5)
        decorate(ax);ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=2,frameon=False)
    save(fig,'fig_hbm_overflow_policy')
    print('72 rows; HBM normalized bars exactly 1; four SVG/PDF pairs generated',flush=True)


if __name__=='__main__':main()
