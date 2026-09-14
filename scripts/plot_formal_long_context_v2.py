"""Paper figures from frozen formal_long_context_v2 results; no simulation."""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "runs/formal_long_context_v2"
MODELS = ("8B", "70B", "405B")
CONTEXTS = ("LC20K", "LC64K", "LC126K")
CASES = tuple((model, context, batch) for model in MODELS
              for context in CONTEXTS for batch in (1, 8))
STYLES = (
    ("HBM_GPU", "HBM-GPU", "#B8BEC5", "///"),
    ("M3D_GPU", "M3D-GPU", "#6C98C4", ""),
    ("M3D_NMP_UNIFORM", "+DNS", "#E6B65C", ".."),
    ("M3D_NMP_CPA", "+CPA", "#4A998B", "\\\\"),
)
CENTERS = tuple(model * 8.0 + offset for model in range(3)
                for offset in (0.0, 1.0, 2.5, 3.5, 5.0, 6.0))
BAR_WIDTH = 0.18


def main():
    with (RESULTS / "final_e2e_metrics.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 72, f"Expected 72 rows, found {len(rows)}"
    data = {(r["model"], r["context"], int(r["batch"]), r["path"]): r for r in rows}
    assert len(data) == 72
    output = RESULTS / "figures"
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7.5,
        "axes.labelsize": 8, "axes.titlesize": 9,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 8, "axes.linewidth": 0.6,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "hatch.linewidth": 0.35, "svg.fonttype": "none",
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })
    legend = [Patch(facecolor=color, edgecolor="#303030", linewidth=0.4,
                    hatch=hatch, label=label) for _, label, color, hatch in STYLES]
    specs = (
        ("tokens_per_s", "Normalized Throughput", "fig_normalized_throughput"),
        ("tokens_per_J", "Normalized Energy Efficiency", "fig_normalized_energy_efficiency"),
        ("Tmax_C", "Temperature (°C)", "fig_temperature"),
    )
    for metric, ylabel, name in specs:
        normalized = metric != "Tmax_C"
        fig = plt.figure(figsize=(7.4, 2.75))
        ax = fig.add_axes((0.085, 0.29, 0.9, 0.53))
        fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.53, 0.985),
                   ncol=4, frameon=False, handlelength=1.8, columnspacing=1.8)
        ymax = 0.0
        for index, (path, _, color, hatch) in enumerate(STYLES):
            values = []
            for model, context, batch in CASES:
                model_id = f"Llama-3.1-{model}"
                value = float(data[model_id, context, batch, path][metric])
                if normalized:
                    value /= float(data[model_id, context, batch, "HBM_GPU"][metric])
                    if path == "HBM_GPU":
                        assert value == 1.0, "HBM normalization must equal one"
                values.append(value)
            ymax = max(ymax, max(values))
            positions = [center + (index - 1.5) * BAR_WIDTH for center in CENTERS]
            ax.bar(positions, values, width=BAR_WIDTH, color=color,
                   edgecolor="#303030", linewidth=0.4, hatch=hatch, zorder=3)
        ax.set_xlim(-0.6, CENTERS[-1] + 0.6)
        ax.set_xticks(CENTERS, [f"B{batch}" for _, _, batch in CASES])
        transform = ax.get_xaxis_transform()
        for model_index, model in enumerate(MODELS):
            offset = model_index * 8.0
            for center, context in zip((0.5, 3.0, 5.5), ("20K", "64K", "126K")):
                ax.text(offset + center, -0.19, context, ha="center", va="top",
                        transform=transform, fontsize=7.5)
            ax.text(offset + 3.0, -0.34, model, ha="center", va="top",
                    transform=transform, fontsize=8)
        for boundary in range(2, len(CASES), 2):
            x = (CENTERS[boundary - 1] + CENTERS[boundary]) / 2
            model_boundary = boundary % 6 == 0
            ax.plot([x, x], [0, -0.43 if model_boundary else -0.28],
                    transform=transform, clip_on=False, color="#555555",
                    linewidth=0.65 if model_boundary else 0.45)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#DEDEDE", linewidth=0.4)
        if normalized:
            ax.set_ylim(0, ymax * 1.12)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
        else:
            ax.set_ylim(0, 100)
            ax.set_yticks((0, 25, 50, 75, 100))
            ax.axhline(85, color="#A63732", linewidth=0.8,
                       linestyle=(0, (4, 2)), zorder=4)
            ax.text(CENTERS[-1] + 0.5, 87.2, "85°C Thermal Limit", ha="right",
                    va="bottom", fontsize=6.5, color="#A63732")
        ax.set_ylabel(ylabel, labelpad=5)
        for extension in ("svg", "pdf"):
            target = output / f"{name}.{extension}"
            fig.savefig(target, format=extension)
            assert target.is_file() and target.stat().st_size > 0
        plt.close(fig)
    print("PASS: 72 rows; HBM normalized values all 1; three SVG/PDF pairs generated.")
    print(output)


if __name__ == "__main__":
    main()
