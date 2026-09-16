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
# Match the compact, continuous single-column layout used by the final v2 plot.
MODEL_STRIDE=6.6
LOCAL_CENTERS=(0.,.55,1.10,1.75,2.30,2.85,3.50,4.05,4.60)
CENTERS=[mi*MODEL_STRIDE+x for mi in range(2) for x in LOCAL_CENTERS]
BAR_WIDTH=.13
MODEL_BOUNDARIES=(-.48,5.08,11.68)
CONTEXT_BOUNDARIES=(1.42,3.17,8.02,9.77)


def load(name):
    with (OUT/name).open(encoding='utf-8',newline='') as s:return list(csv.DictReader(s))


def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.2,'axes.labelsize':9,
        'xtick.labelsize':5.5,'ytick.labelsize':7.8,'legend.fontsize':7.6,'axes.linewidth':1.,
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
        fig=plt.figure(figsize=(3.5,1.5));ax=fig.add_axes((.13,.27,.84,.58))
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
            ax.bar([x+(i-1.5)*BAR_WIDTH for x in CENTERS],values,width=BAR_WIDTH,color=color,edgecolor='#303030',linewidth=.45,hatch=hatch,zorder=3)
        ax.set_xlim(MODEL_BOUNDARIES[0],MODEL_BOUNDARIES[-1]);ax.set_xticks(CENTERS,[f'B{b}' for _,_,b in CASES])
        transform=ax.get_xaxis_transform()
        for mi,model in enumerate(('8B','32B')):
            for center,context in zip((.55,2.30,4.05),('20K','64K','126K')):
                ax.text(mi*MODEL_STRIDE+center,-.20,context,ha='center',va='top',transform=transform,fontsize=7)
            ax.text(mi*MODEL_STRIDE+2.30,-.30,model,ha='center',va='top',transform=transform,fontsize=8.5,fontweight='bold')
        for x in CONTEXT_BOUNDARIES:ax.plot([x,x],[0,-.175],transform=transform,clip_on=False,color='#444444',linewidth=.5)
        for x in MODEL_BOUNDARIES:ax.plot([x,x],[0,-.265],transform=transform,clip_on=False,color='#303030',linewidth=.85)
        decorate(ax)
        ax.set_ylabel(ylabel,fontsize=6,labelpad=3,fontweight='bold')
        if metric=='Tmax_C':
            ax.set_ylim(0,max(100,maximum*1.08));ax.set_yticks((0,25,50,75,100))
            ax.axhline(85,color='#A63732',linewidth=.9,linestyle=(0,(4,2)),zorder=4)
            ax.text(MODEL_BOUNDARIES[-1]-.18,87,'85°C Thermal Limit',ha='right',va='bottom',fontsize=6.8,color='#A63732')
        elif metric=='tokens_per_s':
            ax.set_yscale('log')
            ax.set_ylim(min(.8,minimum*.9),maximum*1.15)
            ticks=[v for v in (.5,1,2,5,10,20,50,100) if minimum*.8<=v<=maximum*1.15]
            ax.set_yticks(ticks,[f'{v:g}' for v in ticks])
        else:
            ax.set_ylim(0,maximum*1.1);ax.yaxis.set_major_locator(MaxNLocator(nbins=4,min_n_ticks=3))
        ax.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,1.015),ncol=4,frameon=False,
            handlelength=1.5,handleheight=1.,columnspacing=1.2,labelspacing=.35,borderaxespad=0)
        save(fig,name)
    overflow=[r for r in load('hbm_policy_comparison.csv') if r['HBM_overflow']=='True']
    fig,axes=plt.subplots(1,2,figsize=(3.5,1.5));fig.subplots_adjust(left=.12,right=.98,bottom=.33,top=.78,wspace=.38)
    for ax,metric,title in zip(axes,('tokens_per_s','P95_completion_latency'),('(a) HBM overflow throughput','(b) P95 completion latency')):
        for i,(policy,label,color,hatch) in enumerate([('host','HOST_OFFLOAD','#B8BEC5','///'),('wave','RESIDENT_WAVE','#6C98C4','')]):
            values=[float(r[policy+'_'+metric]) for r in overflow]
            ax.bar([x+(i-.5)*.35 for x in range(len(overflow))],values,width=.35,color=color,edgecolor='#303030',linewidth=.45,hatch=hatch,label=label,zorder=3)
        ax.set_xticks(range(len(overflow)),[f"{'8B' if r['model']==MODELS[0] else '32B'}\n{r['context'][2:]}\nB{r['batch']}" for r in overflow],fontsize=4.8)
        ax.set_ylabel('Tokens/s' if metric=='tokens_per_s' else 'Seconds',fontsize=6,labelpad=2,fontweight='bold');ax.set_title(title,fontsize=6.8)
        decorate(ax);ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=2,frameon=False)
    save(fig,'fig_hbm_overflow_policy')
    print('72 rows; HBM normalized bars exactly 1; four SVG/PDF pairs generated',flush=True)


if __name__=='__main__':main()
