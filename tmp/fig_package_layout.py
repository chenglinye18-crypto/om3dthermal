"""Final package layout schematic after platform revision v2.

Conventional baseline: GPU 32x24, 4 HBM stacks 12.2x11.8 (two 12.4x24 groups,
7.2 mm center thermal-Si strip), 145.0 GB.
M3D: GPU 32x24, 106 orthogonal slabs @ 300 um pitch (31.8 mm), plane 22x5.5 mm
(slab design frozen), 497.9 GB; 1 mm y-strips on both sides:
arm A = mold fill (same as conventional), arm B = thermal Si bar (ablation).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))
from daimon_runtime import setup_plot  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

setup_plot()

OUT = Path("docs/research/figures")
OUT.mkdir(parents=True, exist_ok=True)

GPU_C = "#4C72B0"
MEM_C = "#55A868"
MAC_C = "#DD8452"
SI_C = "#999999"
RED = "#C44E52"
GRAY = "#666666"


def rect(ax, x, y, w, h, fc, ec=None, alpha=0.25, lw=1.4, zorder=2, ls="-"):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec or fc,
                           alpha=alpha, lw=lw, zorder=zorder, linestyle=ls))


def dim_h(ax, x0, x1, y, text, color=GRAY, fs=8.5):
    ax.annotate("", (x0, y), (x1, y),
                arrowprops=dict(arrowstyle="<->", color=color, lw=1.1))
    ax.text((x0 + x1) / 2, y, text, ha="center", va="bottom",
            fontsize=fs, color=color)


def dim_v(ax, x, y0, y1, text, color=GRAY, fs=8.5):
    ax.annotate("", (x, y0), (x, y1),
                arrowprops=dict(arrowstyle="<->", color=color, lw=1.1))
    ax.text(x, (y0 + y1) / 2, text, ha="left", va="center",
            fontsize=fs, color=color, rotation=90)


fig = plt.figure(figsize=(14.5, 9.6))
gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.22)

# ---------- A. conventional top view ----------
axA = fig.add_subplot(gs[0, 0])
axA.set_title("A. Conventional HBM-on-GPU (top view)\n"
              "4 stacks x 36.24 GB = 145.0 GB",
              fontsize=11.5, fontweight="bold", loc="left")
axA.set_xlim(-22, 22); axA.set_ylim(-18, 18)
axA.set_aspect("equal"); axA.axis("off")

rect(axA, -16, -12, 32, 24, GPU_C, alpha=0.15)          # GPU die
# HBM groups (12.4 x 24) and dram (12.2 x 23.8)
for sgn in (-1, 1):
    gx = sgn * 3.6 if sgn > 0 else sgn * 3.6 - 12.4
    rect(axA, gx, -12, 12.4, 24, MEM_C, alpha=0.13, lw=1.2, ls="--")
    rect(axA, gx + 0.1, -11.9, 12.2, 23.8, MEM_C, alpha=0.35)
    axA.plot([gx + 0.1, gx + 12.3], [0, 0], color=MEM_C, lw=1.0, alpha=0.8)
    axA.text(gx + 6.2, 6.0, "stack\n12.2x11.8\n36.24 GB", ha="center",
             va="center", fontsize=8.2, color="#1a5c2a")
    axA.text(gx + 6.2, -6.0, "stack\n12.2x11.8\n36.24 GB", ha="center",
             va="center", fontsize=8.2, color="#1a5c2a")
rect(axA, -3.6, -12, 7.2, 24, SI_C, alpha=0.30)          # center thermal Si
axA.text(0, 0, "thermal Si\n7.2x24", ha="center", va="center",
         fontsize=8.2, color="#444444")
axA.text(0, -13.3, "GPU die 32.0 x 24.0 mm (768 mm$^2$, GH100-class)",
         ha="center", fontsize=9.0, color=GPU_C, fontweight="bold")
dim_h(axA, -16, -3.6, 14.0, "12.4", fs=8)
dim_h(axA, -3.6, 3.6, 14.0, "7.2", fs=8)
dim_h(axA, 3.6, 16, 14.0, "12.4", fs=8)
dim_h(axA, -16, 16, 16.2, "32.0", color="#333333", fs=9)
dim_v(axA, 17.3, -12, 12, "24.0", color="#333333", fs=9)
axA.text(0, -16.8, "all 0.2 mm margins/seams preserved; "
         "12Hi per stack (HBM3E-36GB-class)",
         ha="center", fontsize=8.5, color=GRAY, style="italic")

# ---------- B. M3D top view ----------
axB = fig.add_subplot(gs[0, 1])
axB.set_title("B. IOM3D-HBM (top view)\n"
              "106 orthogonal slabs = 497.9 GB (3.4x baseline)",
              fontsize=11.5, fontweight="bold", loc="left")
axB.set_xlim(-22, 22); axB.set_ylim(-18, 18)
axB.set_aspect("equal"); axB.axis("off")

rect(axB, -16, -12, 32, 24, GPU_C, alpha=0.15)          # GPU die
rect(axB, -15.9, -11, 31.8, 22, MEM_C, alpha=0.22)      # M3D cube (22 deep)
# slab stripes (thickness along x, 300 um pitch): draw subset
for x in np.arange(-15.9, 15.9, 0.6):
    axB.plot([x, x], [-11, 11], color=MEM_C, lw=0.5, alpha=0.55)
# 1 mm y-strips on both sides: mold fill (arm A) or thermal Si bar (arm B)
for y0 in (-12, 11):
    rect(axB, -16, y0, 32, 1, SI_C, alpha=0.45, lw=1.0)
axB.text(-16.8, 11.5, "1 mm strip:\nmold (A) / Si bar (B)", ha="right",
         va="center", fontsize=7.2, color="#444444")
axB.text(-16.8, -11.5, "1 mm strip:\nmold (A) / Si bar (B)", ha="right",
         va="center", fontsize=7.2, color="#444444")
axB.text(0, 0, "M3D cube 31.8 x 22.0 mm\n106 slabs @ 300 um pitch\n"
         "x-slack 0.2 mm", ha="center", va="center",
         fontsize=9.0, color="#1a5c2a", fontweight="bold",
         bbox=dict(fc="white", ec=MEM_C, alpha=0.85, boxstyle="round,pad=0.35"))
axB.text(0, -13.3, "GPU die 32.0 x 24.0 mm (same as baseline)",
         ha="center", fontsize=9.0, color=GPU_C, fontweight="bold")
dim_h(axB, -15.9, 15.9, 14.0, "31.8", fs=8.5)
dim_h(axB, -16, 16, 16.2, "32.0", color="#333333", fs=9)
dim_v(axB, 19.4, -12, 12, "24.0", color="#333333", fs=9)
dim_v(axB, 16.9, -11, 11, "22.0", fs=8)
axB.text(0, -16.8, "slab design frozen (22x5.5 mm plane); capacity scales "
         "with slab count only: 98 -> 106",
         ha="center", fontsize=8.5, color=GRAY, style="italic")

# ---------- C. conventional cross-section ----------
axC = fig.add_subplot(gs[1, 0])
axC.set_title("C. Conventional cross-section (x-z)",
              fontsize=11.5, fontweight="bold", loc="left")
axC.set_xlim(-20, 20); axC.set_ylim(-2, 12)
axC.set_aspect("equal"); axC.axis("off")

rect(axC, -18, -1.2, 36, 1.2, SI_C, alpha=0.25)          # package
axC.text(0, -0.6, "package 65 x 65 mm", ha="center", va="center",
         fontsize=8.0, color="#444444")
rect(axC, -16, 0, 32, 1.6, GPU_C, alpha=0.45)            # GPU die
axC.text(0, 0.8, "GPU die", ha="center", va="center", fontsize=8.5,
         color="white", fontweight="bold")
for sgn in (-1, 1):                                       # 12Hi HBM stacks
    gx = sgn * 3.6 if sgn > 0 else sgn * 3.6 - 12.4
    rect(axC, gx + 0.1, 1.6, 12.2, 4.4, MEM_C, alpha=0.45)
    for i in range(1, 12):
        axC.plot([gx + 0.1, gx + 12.3], [1.6 + i * 0.3667] * 2,
                 color="white", lw=0.4, alpha=0.6)
    axC.text(gx + 6.2, 3.8, "12Hi DRAM\n+ base logic", ha="center",
             va="center", fontsize=8.0, color="#123f1d")
rect(axC, -3.6, 1.6, 7.2, 4.4, SI_C, alpha=0.45)          # thermal Si
axC.text(0, 3.8, "thermal Si", ha="center", va="center", fontsize=8.0,
         color="#333333")
rect(axC, -16, 6.0, 32, 0.5, "#C9A227", alpha=0.5)       # TIM
rect(axC, -17, 6.5, 34, 1.6, "#B0B7BF", alpha=0.6)       # lid
axC.text(0, 7.3, "TIM + lid", ha="center", va="center", fontsize=8.0,
         color="#333333")
dim_v(axC, 18.6, 1.6, 6.0, "stack ~0.6 mm", fs=7.5)

# ---------- D. M3D cross-section ----------
axD = fig.add_subplot(gs[1, 1])
axD.set_title("D. IOM3D cross-section (x-z): slabs stand 5.5 mm",
              fontsize=11.5, fontweight="bold", loc="left")
axD.set_xlim(-20, 20); axD.set_ylim(-2, 12)
axD.set_aspect("equal"); axD.axis("off")

rect(axD, -18, -1.2, 36, 1.2, SI_C, alpha=0.25)
axD.text(0, -0.6, "package 65 x 65 mm", ha="center", va="center",
         fontsize=8.0, color="#444444")
rect(axD, -16, 0, 32, 1.6, GPU_C, alpha=0.45)
axD.text(0, 0.8, "GPU die + FEOL-MAC (on die)", ha="center", va="center",
         fontsize=8.5, color="white", fontweight="bold")
for x in np.arange(-15.9, 15.9, 0.3):                    # 106 slabs
    rect(axD, x, 1.6, 0.21, 5.5, MEM_C, alpha=0.85, lw=0.0)
axD.text(0, 7.6, "106 vertical memory slabs\n300 um pitch, 5.5 mm tall\n"
         "IGZO 2T0C, 8 bitcell layers (M3D)", ha="center", va="center",
         fontsize=8.5, color="#1a5c2a",
         bbox=dict(fc="white", ec=MEM_C, alpha=0.85, boxstyle="round,pad=0.35"))
dim_h(axD, -15.9, 15.9, 10.3, "31.8", fs=8.5)
dim_v(axD, 17.0, 1.6, 7.1, "5.5", fs=8.5)

fig.savefig(OUT / "package_layout_v1.png", dpi=220, bbox_inches="tight")
print("saved:", OUT / "package_layout_v1.png")
