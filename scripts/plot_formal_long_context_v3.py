"""Paper figures for v3; read-only benchmark data, vector SVG/PDF output."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Patch
from matplotlib.ticker import NullFormatter


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/formal_long_context_v3'

TNR_PATH = Path(r'C:\Windows\Fonts\times.ttf')
TNR_BOLD_PATH = Path(r'C:\Windows\Fonts\timesbd.ttf')

if not TNR_PATH.is_file() or not TNR_BOLD_PATH.is_file():
    raise FileNotFoundError(
        'Required Windows Times New Roman fonts were not found: '
        f'{TNR_PATH}, {TNR_BOLD_PATH}'
    )

TNR = FontProperties(fname=str(TNR_PATH))
TNR_BOLD = FontProperties(fname=str(TNR_BOLD_PATH))
TNR.set_family('Times New Roman')
TNR.set_weight('normal')
TNR_BOLD.set_family('Times New Roman')
TNR_BOLD.set_weight('bold')
TNR_BOLD_LEGEND = TNR_BOLD.copy()
TNR_BOLD_LEGEND.set_size(8.8)


def bold_font(size):
    prop = TNR_BOLD.copy()
    prop.set_size(size)
    return prop


# ============================================================
# Data definitions
# ============================================================

STYLES = (
    ('HBM_BEST', 'HoG', '#B8BEC5', '///'),
    ('M3D_GPU', 'M3D-GPU', '#6C98C4', ''),
    ('M3D_NMP_UNIFORM', '+DNS', '#E6B65C', '..'),
    ('M3D_NMP_CPA', '+CPA', '#4A998B', '\\\\\\\\'),
)

MODELS = (
    'Llama-3.1-8B',
    'Qwen2.5-32B',
)

MODEL_LABELS = (
    'Llama-3.1-8B',
    'Qwen2.5-32B',
)

CONTEXTS = (
    'LC20K',
    'LC64K',
    'LC126K',
)

CONTEXT_LABELS = (
    '20K',
    '64K',
    '126K',
)

BATCHES = (1, 8, 32)

CASES = [
    (m, c, b)
    for m in MODELS
    for c in CONTEXTS
    for b in BATCHES
]


# ============================================================
# Grouped x-axis geometry
# ============================================================
#
# Batch:     closely spaced
# Context:   slightly separated
# Model:     clearly separated
#
# No manually hard-coded centers.
# ============================================================

BATCH_STEP = 0.78
CONTEXT_STEP = 1.05
MODEL_STEP = 1.90

BAR_WIDTH = 0.18


def build_geometry():
    centers = []
    context_groups = []
    model_groups = []

    x = 0.0

    for mi, model in enumerate(MODELS):

        model_start = None
        model_end = None

        for ci, context in enumerate(CONTEXTS):

            context_positions = []

            for bi, batch in enumerate(BATCHES):

                centers.append(x)
                context_positions.append(x)

                if model_start is None:
                    model_start = x

                model_end = x

                if bi != len(BATCHES) - 1:
                    x += BATCH_STEP

            context_groups.append({
                'model': model,
                'context': context,
                'center': (
                    context_positions[0] +
                    context_positions[-1]
                ) / 2.0,
                'start': context_positions[0],
                'end': context_positions[-1],
            })

            if ci != len(CONTEXTS) - 1:
                x += CONTEXT_STEP

        model_groups.append({
            'model': model,
            'center': (model_start + model_end) / 2.0,
            'start': model_start,
            'end': model_end,
        })

        if mi != len(MODELS) - 1:
            x += MODEL_STEP

    return centers, context_groups, model_groups


CENTERS, CONTEXT_GROUPS, MODEL_GROUPS = build_geometry()


# ============================================================
# Utilities
# ============================================================

def load(name):
    with (OUT / name).open(
        encoding='utf-8',
        newline=''
    ) as stream:
        return list(csv.DictReader(stream))


def style():
    plt.rcParams.update({
        'font.family': TNR.get_name(),
        'font.size': 8.5,

        'axes.labelsize': 9,

        'xtick.labelsize': 6.5,
        'ytick.labelsize': 8,

        'legend.fontsize': 7.8,

        'axes.linewidth': 1.0,

        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,

        'xtick.major.size': 3,
        'ytick.major.size': 3,

        'hatch.linewidth': 0.35,

        'svg.fonttype': 'none',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,

        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
    })


def decorate(ax):
    ax.set_axisbelow(True)

    ax.grid(
        axis='y',
        color='#D8D8D8',
        linewidth=0.45,
        linestyle='--',
        dashes=(2, 2)
    )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color('#202020')

    ax.tick_params(
        axis='both',
        direction='in',
        top=True,
        right=True
    )

def apply_axis_fonts(ax):
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        size = tick.get_fontsize()
        tick.set_fontproperties(TNR_BOLD)
        tick.set_fontsize(size)

    ax.xaxis.get_offset_text().set_fontproperties(TNR_BOLD)
    ax.yaxis.get_offset_text().set_fontproperties(TNR_BOLD)


def save(fig, name):
    out = OUT / 'figures'
    out.mkdir(parents=True, exist_ok=True)

    generated = []

    for extension in ('svg', 'pdf'):
        target = out / f'{name}.{extension}'

        try:
            fig.savefig(
                target,
                format=extension,
                bbox_inches='tight'
            )
            generated.append(target)
        except PermissionError:
            fallback = out / f'{name}_new.{extension}'
            print(
                f'WARNING: {target} is locked; saving {fallback} instead.',
                flush=True
            )
            fig.savefig(
                fallback,
                format=extension,
                bbox_inches='tight'
            )
            generated.append(fallback)

    plt.close(fig)
    return generated


# ============================================================
# Hierarchical grouped x-axis
# ============================================================

def grouped_xaxis(ax):
    """
    Level 1: B1 / B8 / B32
    Level 2: 20K / 64K / 126K
    Level 3: model
    """

    # --------------------------------------------------------
    # Batch labels
    # --------------------------------------------------------
    ax.set_xticks(
        CENTERS,
        [f'B{b}' for _, _, b in CASES]
    )
    ax.tick_params(axis='x', labelsize=9)

    transform = ax.get_xaxis_transform()

    # --------------------------------------------------------
    # Context labels
    # Keep them close to batch labels.
    # --------------------------------------------------------
    for group, label in zip(
        CONTEXT_GROUPS,
        CONTEXT_LABELS * len(MODELS)
    ):
        ax.text(
            group['center'],
            -0.155,
            label,
            ha='center',
            va='top',
            transform=transform,
            fontproperties=bold_font(8.0),
            clip_on=False
        )

    # --------------------------------------------------------
    # Model labels
    # Model groups get more separation than contexts.
    # --------------------------------------------------------
    for group, label in zip(
        MODEL_GROUPS,
        MODEL_LABELS
    ):
        ax.text(
            group['center'],
            -0.275,
            label,
            ha='center',
            va='top',
            transform=transform,
            fontproperties=bold_font(9.2),
            clip_on=False
        )

    # --------------------------------------------------------
    # Small separators between contexts
    # --------------------------------------------------------
    for mi in range(len(MODELS)):

        base = mi * len(CONTEXTS)

        for ci in range(len(CONTEXTS) - 1):

            left = CONTEXT_GROUPS[base + ci]
            right = CONTEXT_GROUPS[base + ci + 1]

            boundary = (
                left['end'] +
                right['start']
            ) / 2.0

            ax.plot(
                [boundary, boundary],
                [0, -0.12],
                transform=transform,
                clip_on=False,
                color='#555555',
                linewidth=0.55
            )

    # --------------------------------------------------------
    # Strong separator between models
    # --------------------------------------------------------
    if len(MODEL_GROUPS) > 1:

        for i in range(len(MODEL_GROUPS) - 1):

            boundary = (
                MODEL_GROUPS[i]['end'] +
                MODEL_GROUPS[i + 1]['start']
            ) / 2.0

            ax.plot(
                [boundary, boundary],
                [0, -0.245],
                transform=transform,
                clip_on=False,
                color='#202020',
                linewidth=0.9
            )


# ============================================================
# Main
# ============================================================

def main():

    style()

    rows = load('final_e2e_metrics.csv')

    data = {
        (
            r['model'],
            r['context'],
            int(r['batch']),
            r['path']
        ): r
        for r in rows
    }

    if len(data) != 72:
        raise ValueError(
            'Expected 72 unique formal rows'
        )

    handles = [
        Patch(
            facecolor=color,
            edgecolor='#303030',
            linewidth=0.5,
            hatch=hatch,
            label=label
        )
        for _, label, color, hatch in STYLES
    ]

    generated_files = []

    metrics = (
        (
            'tokens_per_s',
            'Normalized Throughput',
            'fig_normalized_throughput'
        ),
        (
            'tokens_per_J',
            'Normalized Energy Efficiency',
            'fig_normalized_energy_efficiency'
        ),
        (
            'Tmax_C',
            'Peak Temperature (°C)',
            'fig_temperature'
        ),
    )

    # ========================================================
    # Main three figures
    # ========================================================

    for metric, ylabel, name in metrics:

        # Slightly wider than the old 3.5" version.
        # Still compact enough for paper layout.
        fig = plt.figure(
            figsize=(7.0, 2.35)
        )

        ax = fig.add_axes(
            (0.085, 0.32, 0.90, 0.55)
        )

        maximum = 0.0
        minimum = float('inf')

        # ----------------------------------------------------
        # Bars
        # ----------------------------------------------------
        for i, (path, label, color, hatch) in enumerate(STYLES):

            values = []

            for m, c, b in CASES:

                value = float(
                    data[m, c, b, path][metric]
                )

                if metric != 'Tmax_C':

                    value /= float(
                        data[
                            m,
                            c,
                            b,
                            'HBM_BEST'
                        ][metric]
                    )

                    if path == 'HBM_BEST':
                        assert value == 1

                values.append(value)

            maximum = max(
                maximum,
                max(values)
            )

            minimum = min(
                minimum,
                min(values)
            )

            xpos = [
                x + (i - 1.5) * BAR_WIDTH
                for x in CENTERS
            ]

            ax.bar(
                xpos,
                values,
                width=BAR_WIDTH,
                color=color,
                edgecolor='#303030',
                linewidth=0.5,
                hatch=hatch,
                zorder=3
            )

        # ----------------------------------------------------
        # X geometry
        # ----------------------------------------------------
        margin = 0.65

        ax.set_xlim(
            CENTERS[0] - margin,
            CENTERS[-1] + margin
        )

        grouped_xaxis(ax)

        # ----------------------------------------------------
        # Common styling
        # ----------------------------------------------------
        decorate(ax)

        ax.set_ylabel(
            ylabel,
            labelpad=5,
            fontproperties=bold_font(9)
        )

        # ----------------------------------------------------
        # Temperature
        # ----------------------------------------------------
        if metric == 'Tmax_C':

            ymax = max(
                100,
                maximum * 1.08
            )

            ax.set_ylim(
                0,
                ymax
            )

            ax.set_yticks(
                (0, 25, 50, 75, 100)
            )

            ax.axhline(
                85,
                color='#A63732',
                linewidth=1.0,
                linestyle=(0, (4, 2)),
                zorder=4
            )

            ax.text(
                CENTERS[-1] + 0.40,
                87,
                '85°C Thermal Limit',
                ha='right',
                va='bottom',
                fontproperties=bold_font(7.5),
                color='#A63732'
            )

        # ----------------------------------------------------
        # Throughput
        # ----------------------------------------------------
        elif metric == 'tokens_per_s':

            ax.set_yscale('log')

            ax.set_ylim(
                min(0.8, minimum * 0.9),
                maximum * 1.15
            )

            candidate_ticks = (
                0.5,
                1,
                2,
                5,
                10,
                20,
                50,
                100
            )

            ticks = [
                value
                for value in candidate_ticks
                if minimum * 0.8
                <= value
                <= maximum * 1.15
            ]

            ax.set_yticks(
                ticks,
                [f'{v:g}' for v in ticks]
            )
            ax.yaxis.set_minor_formatter(NullFormatter())

        # ----------------------------------------------------
        # Energy efficiency
        #
        # Deliberately use normal Matplotlib linear autoscaling.
        # No MaxNLocator / manual ticks.
        # ----------------------------------------------------
        else:

            ax.set_ylim(
                bottom=0
            )

            ax.margins(
                y=0.08
            )

            ax.ticklabel_format(
                axis='y',
                style='plain',
                useOffset=False
            )
            ax.yaxis.get_offset_text().set_visible(False)

        apply_axis_fonts(ax)

        # ----------------------------------------------------
        # Legend
        # ----------------------------------------------------
        ax.legend(
            handles=handles,
            loc='lower center',
            bbox_to_anchor=(0.5, 1.02),
            ncol=4,
            frameon=False,
            handlelength=1.6,
            handleheight=1.0,
            columnspacing=1.4,
            labelspacing=0.35,
            borderaxespad=0,
            prop=TNR_BOLD_LEGEND
        )

        generated_files.extend(save(fig, name))

    # ========================================================
    # HBM overflow policy figure
    # ========================================================

    overflow = [
        r
        for r in load(
            'hbm_policy_comparison.csv'
        )
        if r['HBM_overflow'] == 'True'
    ]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.0, 2.35)
    )

    fig.subplots_adjust(
        left=0.09,
        right=0.985,
        bottom=0.30,
        top=0.78,
        wspace=0.30
    )

    overflow_styles = (
        (
            'host',
            'HOST_OFFLOAD',
            '#B8BEC5',
            '///'
        ),
        (
            'wave',
            'RESIDENT_WAVE',
            '#6C98C4',
            ''
        ),
    )

    overflow_metrics = (
        (
            'tokens_per_s',
            '(a) HBM overflow throughput'
        ),
        (
            'P95_completion_latency',
            '(b) P95 completion latency'
        ),
    )

    for ax, (
        metric,
        title
    ) in zip(
        axes,
        overflow_metrics
    ):

        for i, (
            policy,
            label,
            color,
            hatch
        ) in enumerate(
            overflow_styles
        ):

            values = [
                float(
                    r[
                        policy +
                        '_' +
                        metric
                    ]
                )
                for r in overflow
            ]

            xpos = [
                x + (i - 0.5) * 0.35
                for x in range(
                    len(overflow)
                )
            ]

            ax.bar(
                xpos,
                values,
                width=0.35,
                color=color,
                edgecolor='#303030',
                linewidth=0.5,
                hatch=hatch,
                label=label,
                zorder=3
            )

        ax.set_xticks(
            range(len(overflow)),
            [
                (
                    f"{'8B' if r['model'] == MODELS[0] else '32B'}\n"
                    f"{r['context'][2:]}\n"
                    f"B{r['batch']}"
                )
                for r in overflow
            ],
            fontsize=7.0
        )

        ax.set_ylabel(
            (
                'Tokens/s'
                if metric == 'tokens_per_s'
                else 'Seconds'
            ),
            labelpad=4,
            fontproperties=bold_font(8.5)
        )

        ax.set_title(
            title,
            fontproperties=bold_font(9)
        )

        decorate(ax)
        apply_axis_fonts(ax)

    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc='upper center',
        ncol=2,
        frameon=False,
        prop=TNR_BOLD_LEGEND
    )

    generated_files.extend(save(fig, 'fig_hbm_overflow_policy'))

    print(
        '72 rows; HBM normalized bars exactly 1; '
        'four SVG/PDF pairs generated',
        flush=True
    )

    for path in generated_files:
        print(path, flush=True)


if __name__ == '__main__':
    main()
