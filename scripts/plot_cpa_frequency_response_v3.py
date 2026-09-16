"""Plot measured CPA response; infeasible samples remain visible as gray crosses."""

import csv
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/cpa_feol_frequency_sweep_v3'


def main():
    with (OUT / 'frequency_response.csv').open(
        encoding='utf-8',
        newline=''
    ) as stream:
        rows = list(csv.DictReader(stream))

    assert len(rows) == 108

    # ============================================================
    # Global figure style
    # ============================================================
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 11,
        'font.weight': 'bold',

        'axes.labelsize': 12,
        'axes.labelweight': 'bold',
        'axes.titlesize': 12,
        'axes.titleweight': 'bold',
        'axes.linewidth': 0.8,

        'legend.fontsize': 11,

        'xtick.labelsize': 11,
        'ytick.labelsize': 11,

        # Force math text to Times New Roman
        'mathtext.fontset': 'custom',
        'mathtext.rm': 'Times New Roman',
        'mathtext.it': 'Times New Roman:italic',
        'mathtext.bf': 'Times New Roman:bold',

        'svg.fonttype': 'none',
        'pdf.fonttype': 42,

        'figure.facecolor': 'white',
    })

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(7.8, 6.0),
        sharex=True,
        sharey=True
    )

    # ============================================================
    # Context styles
    # ============================================================
    styles = [
        ('LC20K',  '20K',  '#1f77b4', 'o'),
        ('LC64K',  '64K',  '#ff7f0e', 's'),
        ('LC126K', '126K', '#2ca02c', '^'),
    ]

    # ============================================================
    # Six panels
    # ============================================================
    for col, (model, full_model) in enumerate((('8B', 'Llama-3.1-8B'), ('32B', 'Qwen2.5-32B'))):
        for ridx, batch in enumerate((1, 8, 32)):
            ax = axes[ridx, col]

            for context, label, color, marker in styles:
                group = sorted(
                    (
                        r for r in rows
                        if r['model'] == full_model
                        and int(r['batch']) == batch
                        and r['context'] == context
                    ),
                    key=lambda r: float(r['feol_frequency_ghz'])
                )

                assert len(group) == 6

                x = np.array([
                    float(r['feol_frequency_ghz'])
                    for r in group
                ])

                y = np.array([
                    float(r['normalized_throughput'])
                    for r in group
                ])

                good = np.array([
                    r['thermal_feasible'].lower() == 'true'
                    for r in group
                ])

                assert y[0] == 1

                # Thermal-feasible points
                ax.plot(
                    x,
                    np.where(good, y, np.nan),
                    color=color,
                    marker=marker,
                    markersize=5.2,
                    markeredgewidth=1.0,
                    linewidth=1.8,
                    label=label
                )

                # Thermal-infeasible points
                ax.scatter(
                    x[~good],
                    y[~good],
                    color='#777777',
                    marker='x',
                    s=38,
                    linewidths=1.4,
                    zorder=4
                )

            # ----------------------------------------------------
            # Panel title
            # ----------------------------------------------------
            ax.set_title(
                f'{model}, B={batch}',
                fontfamily='Times New Roman',
                fontweight='bold'
            )

            # ----------------------------------------------------
            # X axis
            # ----------------------------------------------------
            ax.set_xlim(0.95, 3.05)
            ax.set_xticks([1, 1.5, 2, 2.5, 3])

            # ----------------------------------------------------
            # Full boxed axes
            # ----------------------------------------------------
            for side in ('top', 'right', 'bottom', 'left'):
                ax.spines[side].set_visible(True)
                ax.spines[side].set_linewidth(0.8)

            ax.tick_params(
                axis='both',
                which='both',
                direction='in',
                top=True,
                right=True,
                width=0.8,
                length=3.8
            )

            # Bold Times New Roman ticks
            for tick in ax.get_xticklabels() + ax.get_yticklabels():
                tick.set_fontfamily('Times New Roman')
                tick.set_fontweight('bold')

            # ----------------------------------------------------
            # Horizontal grid
            # ----------------------------------------------------
            ax.grid(
                axis='y',
                color='#DDDDDD',
                linewidth=0.45
            )

            # ----------------------------------------------------
            # X label only on bottom row
            # ----------------------------------------------------
            if ridx == 2:
                ax.set_xlabel(
                    'FEOL Frequency (GHz)',
                    fontfamily='Times New Roman',
                    fontweight='bold'
                )

    # ============================================================
    # Shared Y-axis title
    # ============================================================
    fig.supylabel(
        r'Normalized Throughput '
        r'($TPS_f/TPS_{1GHz}$)',
        x=0.08,  # move right
        y=0.44,
        fontsize=12,
        fontfamily='Times New Roman',
        fontweight='bold'
    )

    # ============================================================
    # Shared legend
    # ============================================================
    handles = [
        Line2D(
            [],
            [],
            color=color,
            marker=marker,
            linewidth=1.8,
            markersize=5.8,
            markeredgewidth=1.0,
            label=label
        )
        for _, label, color, marker in styles
    ]

    handles.append(
        Line2D(
            [],
            [],
            color='#777777',
            marker='x',
            linestyle='none',
            markersize=6.5,
            markeredgewidth=1.4,
            label='Thermal infeasible (>85°C)'
        )
    )

    # Legend moved further downward
    fig.legend(
        handles=handles,
        loc='upper center',
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.94),
        columnspacing=1.4,
        handletextpad=0.5,
        prop={
            'family': 'Times New Roman',
            'weight': 'bold',
            'size': 11
        }
    )

    # ============================================================
    # Layout
    # ============================================================
    fig.tight_layout(
        rect=(0.050, 0.0, 1.0, 0.90),
        h_pad=1.4,
        w_pad=1.0
    )

    # ============================================================
    # Save
    # ============================================================
    directory = OUT / 'figures'
    directory.mkdir(parents=True, exist_ok=True)

    for ext in ('svg', 'pdf'):
        fig.savefig(
            directory / f'fig_cpa_frequency_response.{ext}',
            bbox_inches='tight'
        )

    plt.close(fig)


if __name__ == '__main__':
    main()
