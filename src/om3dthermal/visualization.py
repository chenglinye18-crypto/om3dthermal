"""Matplotlib views of the axis-aligned scene."""

from __future__ import annotations

from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

from .config import SimulationConfig
from .geometry.scene import Scene


def _colors(scene: Scene) -> dict[str, tuple]:
    palette = plt.get_cmap("tab10").colors
    return {material: palette[index % len(palette)]
            for index, material in enumerate(sorted({box.material for box in scene.boxes}))}


def _legend(ax, colors: dict[str, tuple]) -> None:
    ax.legend(handles=[Patch(facecolor=color, edgecolor="black", label=material)
                       for material, color in colors.items()],
              loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)


def plot_top_view(scene: Scene, config: SimulationConfig, path: str | Path) -> None:
    colors = _colors(scene)
    fig, ax = plt.subplots(figsize=(9, 7))
    for box in sorted(scene.boxes, key=lambda item: (item.tags.get("priority", 0), item.z1)):
        ax.add_patch(Rectangle((box.x0 * 1e3, box.y0 * 1e3),
                               (box.x1 - box.x0) * 1e3, (box.y1 - box.y0) * 1e3,
                               facecolor=colors[box.material], edgecolor="black", alpha=0.45))
    coincident: dict[tuple[float, float], list] = defaultdict(list)
    for footprint in config.footprints.values():
        coincident[(footprint.center_x, footprint.center_y)].append(footprint)
    label_offsets = {}
    for footprints in coincident.values():
        for index, footprint in enumerate(footprints):
            label_offsets[footprint.name] = (index - (len(footprints) - 1) / 2) * 18
    for footprint in config.footprints.values():
        ax.add_patch(Rectangle((footprint.x0 * 1e3, footprint.y0 * 1e3),
                               footprint.size_x * 1e3, footprint.size_y * 1e3,
                               fill=False, edgecolor="black", linewidth=0.8, linestyle="--"))
        ax.annotate(f"{footprint.name}\n({footprint.center_x*1e3:g}, {footprint.center_y*1e3:g}) mm",
                    (footprint.center_x * 1e3, footprint.center_y * 1e3),
                    xytext=(0, label_offsets[footprint.name]), textcoords="offset points",
                    ha="center", va="center", fontsize=6,
                    bbox={"facecolor": "white", "alpha": 0.45, "edgecolor": "none", "pad": 1})
    ax.set(xlabel="x (mm)", ylabel="y (mm)", title="Top view — footprints")
    ax.set_aspect("equal")
    ax.autoscale_view()
    _legend(ax, colors)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _section(scene: Scene, path: str | Path, axis: str, coordinate: float = 0.0) -> None:
    colors = _colors(scene)
    horizontal = "x" if axis == "y" else "y"
    selected = [box for box in scene.boxes
                if getattr(box, axis + "0") <= coordinate <= getattr(box, axis + "1")]
    selected.sort(key=lambda item: item.tags.get("priority", 0))
    fig, ax = plt.subplots(figsize=(11, 6))
    for box in selected:
        h0, h1 = getattr(box, horizontal + "0"), getattr(box, horizontal + "1")
        ax.add_patch(Rectangle((h0 * 1e3, box.z0 * 1e6), (h1 - h0) * 1e3,
                               (box.z1 - box.z0) * 1e6, facecolor=colors[box.material],
                               edgecolor="black", linewidth=0.45))
    ax.set(xlabel=f"{horizontal} (mm)", ylabel="z (um)",
           title=f"{horizontal}z section at {axis}={coordinate*1e3:g} mm")
    ax.autoscale_view()
    _legend(ax, colors)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_visualizations(scene: Scene, config: SimulationConfig, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_top_view(scene, config, output_dir / "top_view.png")
    _section(scene, output_dir / "xz_section.png", "y")
    _section(scene, output_dir / "yz_section.png", "x")
