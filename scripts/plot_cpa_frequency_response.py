"""Plot measured CPA response; infeasible samples remain visible as gray crosses."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/cpa_feol_frequency_sweep'


def main():
    with (OUT / 'frequency_response.csv').open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 108
    plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'axes.linewidth': .6,
        'svg.fonttype': 'none', 'pdf.fonttype': 42, 'figure.facecolor': 'white'})
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 4.0), sharex=True, sharey=True)
    styles = [('LC20K', '20K', '#6C98C4', 'o'), ('LC64K', '64K', '#E6B65C', 's'),
              ('LC126K', '126K', '#4A998B', '^')]
    for col, model in enumerate(('8B', '70B', '405B')):
        for ridx, batch in enumerate((1, 8)):
            ax = axes[ridx, col]
            for context, label, color, marker in styles:
                group = sorted((r for r in rows if r['model'] == f'Llama-3.1-{model}'
                    and int(r['batch']) == batch and r['context'] == context),
                    key=lambda r: float(r['feol_frequency_ghz']))
                assert len(group) == 6
                x = np.array([float(r['feol_frequency_ghz']) for r in group])
                y = np.array([float(r['normalized_throughput']) for r in group])
                good = np.array([r['thermal_feasible'].lower() == 'true' for r in group])
                assert y[0] == 1
                ax.plot(x, np.where(good, y, np.nan), color=color, marker=marker,
                        markersize=3.5, linewidth=1, label=label)
                ax.scatter(x[~good], y[~good], color='#777777', marker='x', s=24, linewidths=1)
            ax.set_title(f'{model}, B={batch}', fontsize=9)
            ax.set_xlim(.95, 3.05)
            ax.set_xticks([1, 1.5, 2, 2.5, 3])
            ax.spines[['top', 'right']].set_visible(False)
            ax.grid(axis='y', color='#DDDDDD', linewidth=.4)
            if ridx == 1:
                ax.set_xlabel('FEOL Frequency (GHz)')
            if col == 0:
                ax.set_ylabel('Normalized Throughput')
    handles = [Line2D([], [], color=c, marker=m, linewidth=1, markersize=4, label=l)
               for _, l, c, m in styles]
    handles.append(Line2D([], [], color='#777777', marker='x', linestyle='none', label='Thermal infeasible (>85°C)'))
    fig.legend(handles=handles, loc='upper center', ncol=4, frameon=False, bbox_to_anchor=(.5, 1.01))
    fig.tight_layout(rect=(0, 0, 1, .92), h_pad=1.3)
    directory = OUT / 'figures'
    directory.mkdir(parents=True, exist_ok=True)
    for ext in ('svg', 'pdf'):
        fig.savefig(directory / f'fig_cpa_frequency_response.{ext}')
    plt.close(fig)


if __name__ == '__main__':
    main()
