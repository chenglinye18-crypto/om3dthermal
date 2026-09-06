"""Schematic: MAC/GPU operator division and interface-bandwidth amplification.

LLaMA-3.1-8B decode, B=1, S=128K, fp16. Paper-ready English labels.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))
from daimon_runtime import setup_plot  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

setup_plot()

OUT = Path("docs/research/figures")
OUT.mkdir(parents=True, exist_ok=True)

GPU_C = "#4C72B0"
MEM_C = "#55A868"
MAC_C = "#DD8452"
RED = "#C44E52"
GRAY = "#666666"

fig = plt.figure(figsize=(13.5, 9.2))
gs = fig.add_gridspec(2, 2, height_ratios=[1.55, 1.0], hspace=0.30, wspace=0.18)


def box(ax, x, y, w, h, color, title, lines, title_fs=10.5, body_fs=8.8, alpha=0.18):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                fc=color, ec=color, alpha=alpha, lw=1.6))
    ax.text(x + w / 2, y + h - 0.045, title, ha="center", va="top",
            fontsize=title_fs, fontweight="bold", color=color)
    ax.text(x + w / 2, y + h - 0.115, "\n".join(lines), ha="center", va="top",
            fontsize=body_fs, color="#222222", linespacing=1.45)


def arrow(ax, x1, y1, x2, y2, color, label, lw=3.2, label_dy=0.035, fs=9.2,
          style="-|>", ls="-", label_side=1):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=18, lw=lw, color=color,
                                 linestyle=ls, zorder=5))
    ax.text((x1 + x2) / 2, (y1 + y2) / 2 + label_dy * label_side, label,
            ha="center", va="bottom" if label_side > 0 else "top",
            fontsize=fs, color=color, fontweight="bold")


# ---------------- Panel A: baseline ----------------
axA = fig.add_subplot(gs[0, 0])
axA.set_xlim(0, 1); axA.set_ylim(0, 1); axA.axis("off")
axA.set_title("A. Baseline: Conventional HBM-on-GPU\n(every byte crosses the interface)",
              fontsize=11.5, fontweight="bold", loc="left")

box(axA, 0.06, 0.42, 0.36, 0.42, GPU_C, "GPU  (300 W)",
    ["ALL decode compute", "83.7 GFLOP/token", "", "weights GEMV  15.0 G",
     "attention       68.7 G"])
box(axA, 0.58, 0.42, 0.36, 0.42, MEM_C, "HBM stacks",
    ["weights   16.0 GB", "KV cache 17.18 GB", "",
     "read every single token"])

arrow(axA, 0.58, 0.63, 0.42, 0.63, RED, "", lw=6.0)
axA.text(0.50, 0.685, "33.18 GB/token", ha="center", va="bottom",
         fontsize=9.4, color=RED, fontweight="bold")
axA.text(0.50, 0.575, "@ 39.2 Tb/s", ha="center", va="top",
         fontsize=8.8, color=RED, fontweight="bold")
axA.text(0.5, 0.30,
         "interface time = 33.18 GB / 4.9 TB/s = 6.77 ms\n"
         "throughput = 147.7 tok/s   (interface-bound)",
         ha="center", fontsize=10, color=RED, fontweight="bold",
         bbox=dict(fc="white", ec=RED, alpha=0.9, boxstyle="round,pad=0.45"))
axA.text(0.5, 0.10, "interface carries 100% of consumed bytes",
         ha="center", fontsize=9.5, color=GRAY, style="italic")

# ---------------- Panel B: IOM3D-HBM ----------------
axB = fig.add_subplot(gs[0, 1])
axB.set_xlim(0, 1); axB.set_ylim(0, 1); axB.axis("off")
axB.set_title("B. IOM3D-HBM: FEOL-MAC takes weight streaming\n(weights stay local; only KV crosses)",
              fontsize=11.5, fontweight="bold", loc="left")

box(axB, 0.05, 0.42, 0.33, 0.42, GPU_C, "GPU  (attention engine)",
    ["QK$^T$ + softmax + PV", "68.7 GFLOP/token (82%)", "",
     "tensor cores, reductions", "non-linearity"])
box(axB, 0.44, 0.42, 0.24, 0.42, MAC_C, "FEOL-MAC",
    ["Q/K/V/O + FFN", "weight GEMV", "15.0 GFLOP/token",
     "(18%)"])
box(axB, 0.72, 0.42, 0.24, 0.42, MEM_C, "M3D arrays",
    ["weights 16.0 GB", "", "KV cache", "17.18 GB"])

# local weight flow (stays inside memory)
arrow(axB, 0.72, 0.55, 0.68, 0.55, MEM_C, "", lw=5.0)
axB.text(0.60, 0.875, "weights: 16.0 GB/token stay local (0 GB crosses)",
         ha="center", fontsize=9.0, color=MEM_C, fontweight="bold")
# KV crosses the coil interface: elbow routed below the boxes
axB.plot([0.84, 0.84], [0.42, 0.335], color=RED, lw=6.0,
         solid_capstyle="round")
axB.plot([0.215, 0.215], [0.335, 0.42], color=RED, lw=6.0,
         solid_capstyle="round")
arrow(axB, 0.84, 0.335, 0.215, 0.335, RED, "", lw=6.0)
axB.text(0.56, 0.355, "KV: 17.18 GB/token @ 39.2 Tb/s", ha="center",
         va="bottom", fontsize=9.4, color=RED, fontweight="bold")
# activations / logits (tiny)
arrow(axB, 0.44, 0.50, 0.38, 0.50, GRAY, "", lw=1.6)
axB.text(0.41, 0.46, "act ~MB", ha="center", va="top", fontsize=7.5,
         color=GRAY)

axB.text(0.5, 0.20,
         "interface time = 17.18 GB / 4.9 TB/s = 3.51 ms\n"
         "throughput = ~285 tok/s   (1.93x, MAC >= 4.3 TFLOPS)",
         ha="center", fontsize=10, color=RED, fontweight="bold",
         bbox=dict(fc="white", ec=RED, alpha=0.9, boxstyle="round,pad=0.45"))
axB.text(0.5, 0.045,
         "interface carries 52% of consumed bytes -> effective BW x1.93",
         ha="center", fontsize=9.5, color=GRAY, style="italic")

# ---------------- Panel C: amplification vs context ----------------
axC = fig.add_subplot(gs[1, :])
S = np.logspace(np.log10(2e3), np.log10(1.2e6), 400)
KV_GB = 131072.0 * S / 1e9          # 131072 B of KV per context token (32 layers)
W_GB = 16.0
A = 1.0 + W_GB / KV_GB
axC.semilogx(S, A, color=MAC_C, lw=3.0)
for s_mark, note, dx, ha in ((8e3, "8K: 15.9x", 10, "left"),
                             (32e3, "32K: 4.7x", 10, "left"),
                             (131072, "128K: 1.93x", 10, "left"),
                             (1e6, "1M: 1.12x", -10, "right")):
    a_mark = 1.0 + W_GB / (131072.0 * s_mark / 1e9)
    axC.plot([s_mark], [a_mark], "o", color=RED, ms=8, zorder=5)
    axC.annotate(note, (s_mark, a_mark), textcoords="offset points",
                 xytext=(dx, 10), ha=ha, fontsize=9.5, color=RED,
                 fontweight="bold")
axC.set_xlabel("context length S (tokens)", fontsize=10.5)
axC.set_ylabel("interface bandwidth amplification\n"
               "A = (weights + KV) / KV  =  1 + 16 GB / KV(S)", fontsize=10)
axC.set_title("C. Weight-offload amplification vs context length "
              "(B = 1, fp16, LLaMA-3.1-8B-class)",
              fontsize=11.5, fontweight="bold", loc="left")
axC.grid(True, which="both", alpha=0.3)
axC.set_ylim(0.9, 18)
axC.text(0.985, 0.97,
         "MAC removes the weight share of interface traffic;\n"
         "long-context KV pressure is covered by the capacity wall (local residency)",
         transform=axC.transAxes, ha="right", va="top", fontsize=9.0,
         color=GRAY, style="italic")

fig.savefig(OUT / "mac_gpu_operator_division_v0.png", dpi=220,
            bbox_inches="tight")
print("saved:", OUT / "mac_gpu_operator_division_v0.png")
