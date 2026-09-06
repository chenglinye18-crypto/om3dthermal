"""Unified GPU platform + host-link table for the IOM3D-HBM paper (rev v2).

Single CSV combining:
  A. GPU platform energy invariants (per-bit / per-FLOP unit costs)
  B. Host (GPU<->DDR) link options
  C. Measured idle-power anchors for P_static calibration

Sources:
- Decode/prefill power as fraction of TDP: arXiv:2604.10852v1 Fig. 8
  (A100/H100 decode ~45-60% TDP; MI300 ~80%; prefill 75-100%).
- Platform specs: vendor datasheets (catalogued in
  docs/research/HBM_benchmark_landscape_2026-08-25.md).
- Host links: NVIDIA GH200 whitepaper / Grace Hopper Architecture In-Depth
  (NVLink-C2C 900 GB/s total, 450 GB/s/dir); NVIDIA H100 PCIe PB-11133-001
  + DGX H200 system docs (PCIe Gen5 128 GB/s bidir); Tyan measured H2D
  56.2 GB/s -> eta 0.878; AMD EPYC 9654 DDR5 460.8 GB/s.
- Idle anchors: ai-gpu-energy-optimizer whitepaper (72+ HW tests):
  A100 SXM 65-69 W, H100 SXM 69.5 W, H200 SXM ~74 W idle floor
  (HBM3e ghost power +79-136 W post-load), B200 SXM 143-145 W;
  GitHub h200-gpu-benchmark-suite: H200 NVL idle ~121 W.

Derived invariants:
  e_compute = P_prefill / TFLOPS   [pJ/FLOP] (compute-bound)
  1 W / (1 TFLOP/s) = 1 pJ/FLOP

Decode section records exactly two model parameters, nothing else:
  static_power_W            measured idle floor (per-second billing)
  e_decode_dynamic_pJ/bit = (P_decode - P_static) / (BW_peak x 8)
                            (per-bit billing; board-level, includes HBM
                            dynamic energy -- when the E2E model books memory
                            energy separately (E4), subtract the memory
                            per-bit term from this parameter to avoid
                            double counting: baseline HBM 1.397 pJ/bit,
                            proposed M3D 0.855 pJ/bit)
  1 W / (1 TB/s) = 0.125 pJ/bit

Decisions recorded (2026-09-06):
- Scenario matched bandwidth capped at 39.2 Tb/s (98-slab equivalent);
  106-slab capability 42.4 Tb/s is design margin, acknowledged in prose.
- Host link: PCIe Gen5 64 GB/s/dir as main baseline (mainstream x86
  deployment); NVLink-C2C 450 GB/s/dir as robustness sensitivity.
- P_static = 74 W for ALL project rows (H200 SXM measured idle floor,
  same silicon as project GPU); e_decode_dynamic identical across
  H200 / baseline / proposed rows by construction. Legacy nominal
  100 W replaced in rev v2.
"""
import csv
from pathlib import Path

OUT = Path("docs/research/gpu_platform_table_2026-09-06.csv")
if OUT.exists():
    try:
        with OUT.open("a"):
            pass
    except PermissionError:
        OUT = OUT.with_name(OUT.stem + "_v2.csv")

PJBIT_PER_W_PER_TBS = 0.125
PJFLOP_PER_W_PER_TFLOPS = 1.0

header = [
    "category", "name",
    # GPU platform / energy invariant columns
    "memory_type", "capacity_GB", "peak_bandwidth_TB_per_s",
    "peak_compute_BF16_dense_TFLOPS", "TDP_W", "static_power_W",
    "decode_power_frac_TDP_min", "decode_power_frac_TDP_max",
    "e_decode_dynamic_pJ_per_bit_min", "e_decode_dynamic_pJ_per_bit_max",
    "prefill_power_frac_TDP_min", "prefill_power_frac_TDP_max",
    "compute_bound_power_W_min", "compute_bound_power_W_max",
    "e_compute_pJ_per_FLOP_min", "e_compute_pJ_per_FLOP_max",
    # host link columns
    "host_link", "host_link_per_direction_GBps",
    "host_link_bidirectional_GBps", "host_memory",
    "host_memory_bandwidth_GBps", "coherent_unified_memory",
    # common
    "notes", "provenance_status", "source",
]
N_COLS = len(header)
assert N_COLS == 27

# (name, mem, cap, BW_TBs, TFLOPS, TDP, dfrac, pfrac, static_W|None,
#  host_link_GBps|None, notes, provenance, source)
GPU_ROWS = [
    ("A100 SXM 80GB", "HBM2e", 80, 2.039, 312.0, 400,
     (0.45, 0.60), (0.75, 1.00), 67.0, 64.0, "",
     "MEASURED_REFERENCE (decode/prefill frac, idle), VENDOR_SPEC (platform)",
     "arXiv:2604.10852v1 Fig.8; NVIDIA A100 datasheet; "
     "idle floor 65-69 W from ai-gpu-energy-optimizer whitepaper"),
    ("H100 SXM", "HBM3", 80, 3.35, 989.5, 700,
     (0.45, 0.60), (0.75, 1.00), 69.5, 64.0, "",
     "MEASURED_REFERENCE (decode/prefill frac, idle), VENDOR_SPEC (platform)",
     "arXiv:2604.10852v1 Fig.8; NVIDIA H100 datasheet; "
     "idle 69-76 W from ai-gpu-energy-optimizer whitepaper"),
    ("H200 SXM", "HBM3e", 141, 4.80, 989.5, 700,
     (0.45, 0.60), (0.75, 1.00), 74.0, 64.0,
     "decode frac assumed = H100 (same GH100 silicon); idle floor ~74 W, "
     "post-load HBM3e ghost +79-136 W; H200 NVL measured idle ~121 W",
     "DERIVED (frac from H100); MEASURED_REFERENCE (idle); "
     "VENDOR_SPEC (platform)",
     "arXiv:2604.10852v1; NVIDIA H200 datasheet; "
     "ai-gpu-energy-optimizer whitepaper; h200-gpu-benchmark-suite"),
    ("MI300X", "HBM3", 192, 5.30, 1307.4, 750,
     (0.80, 0.80), (0.75, 1.00), None, 64.0,
     "no measured idle anchor found; dynamic split not computed",
     "MEASURED_REFERENCE (decode/prefill frac), VENDOR_SPEC (platform)",
     "arXiv:2604.10852v1 Fig.8; AMD MI300X datasheet"),
    ("IOM3D baseline GPU+HBM (rev v2, planned)", "HBM3e 12Hi x4",
     145.0, 4.80, 989.5, 700,
     (0.45, 0.60), (0.75, 1.00), 74.0, 64.0,
     "H200-anchored; 4x 36.24 GB stacks 11.8x12.2 mm; PCIe Gen5 host link; "
     "P_static = 74 W (H200 SXM measured idle floor); e_decode_dynamic "
     "identical to H200 by construction; board-level value includes HBM "
     "dynamic energy (1.397 pJ/bit, booked separately in E4); "
     "not yet implemented in configs",
     "PLANNED_REV_V2",
     "H200 datasheet + arXiv:2604.10852v1 frac + measured idle anchors; "
     "capacity from tmp/check_capacity_11p8x12p2.py"),
    ("IOM3D-HBM proposed (rev v2, planned)", "IGZO M3D 106 slabs",
     497.9, 4.80, 989.5, 700,
     (0.45, 0.60), (0.75, 1.00), 74.0, 64.0,
     "same H200 GPU silicon -> GPU-side e_decode_dynamic and P_static "
     "identical to baseline; M3D memory energy (0.855 pJ/bit) booked "
     "separately in E4, not subtracted here; scenario matched bw capped "
     "at 39.2 Tb/s (4.9 TB/s) -> u = 4.9/4.8 clamps at 1; 106-slab "
     "capability 42.4 Tb/s held as design margin; "
     "not yet implemented in configs",
     "PLANNED_REV_V2",
     "canonical case orthogonal_m3d_igzo slab IO + cap_bits_per_second; "
     "M3D read energy 0.855 pJ/bit from case model (E4 memory side)"),
]

# (name, link, per_dir, bidir, host_mem, host_bw, coherent, notes, prov, src)
LINK_ROWS = [
    ("PCIe Gen5 x16 to x86 host (MAIN BASELINE)",
     "PCIe Gen5 x16", 64.0, 128.0, "DDR5 (EPYC 9654 12ch)", 460.8, "no",
     "measured pinned H2D 56.2 GB/s -> eta 0.878 vs min(460.8, 64); "
     "mainstream DGX H200 deployment",
     "VENDOR_SPEC (link, DDR5); VENDOR_REPORTED_MEASURED (eta)",
     "NVIDIA H100 PCIe PB-11133-001; DGX H200 docs; Tyan HX FT65TB8050; "
     "AMD EPYC 9654 spec"),
    ("NVLink-C2C to Grace LPDDR5X (ROBUSTNESS SENSITIVITY)",
     "NVLink-C2C", 450.0, 900.0, "LPDDR5X up to 480 GB", 546.0,
     "yes (hardware-coherent, EGM)",
     "7x PCIe Gen5; vendor claim >5x lower energy per byte; even at "
     "450 GB/s still ~11x below HBM 4.9 TB/s so capacity story holds",
     "VENDOR_SPEC (NVIDIA whitepaper / dev blog)",
     "NVIDIA Grace Hopper Architecture In-Depth; GH200 whitepaper"),
    ("H200 NVL (2-GPU PCIe card variant)",
     "PCIe Gen5 x16", 64.0, 128.0, "host-dependent", None, "no",
     "600 W TDP variant; same host link class as main baseline",
     "VENDOR_SPEC",
     "NVIDIA H200 NVL specs (techpowerup/hmc-tech catalogued)"),
]

# (name, TDP, idle_W, notes, provenance, source)
IDLE_ROWS = [
    ("A100 SXM idle floor", 400, 67.0,
     "idle floor 65-69 W; post-load ghost power -> 146.66 W (HBM2e clock "
     "stays up)", "MEASURED_REFERENCE",
     "ai-gpu-energy-optimizer whitepaper (72+ HW tests)"),
    ("H100 SXM idle floor", 700, 69.5,
     "idle baseline 69-76 W; no ghost power on HBM3 part",
     "MEASURED_REFERENCE",
     "ai-gpu-energy-optimizer whitepaper (72+ HW tests)"),
    ("H200 SXM idle floor", 700, 74.0,
     "idle floor ~74 W; post-load ghost +79-136 W (HBM3e clock stays up); "
     "decode-relevant static likely between floor and ghost state",
     "MEASURED_REFERENCE",
     "ai-gpu-energy-optimizer whitepaper (72+ HW tests)"),
    ("H200 NVL idle", 600, 121.0,
     "measured idle ~121 W; 548 W under 8192^3 FP32 GEMM; idle/TDP = 20.2%",
     "MEASURED_REFERENCE",
     "GitHub h200-gpu-benchmark-suite (namanadep), production H200 NVL"),
    ("B200 SXM idle floor", 1000, 144.0,
     "idle floor 143-145 W; cold-boot ghost up to 574 W (HBM3e)",
     "MEASURED_REFERENCE",
     "ai-gpu-energy-optimizer whitepaper (72+ HW tests)"),
    ("P_static decision (project)", 700, 74.0,
     "74 W (H200 SXM measured idle floor) applied to ALL project rows; "
     "legacy 100 W nominal replaced in rev v2; sensitivity band 74-121 W "
     "covers floor-to-NVL spread if reviewers ask",
     "MODELING_CHOICE_WITHIN_MEASURED_RANGE",
     "range from rows above"),
]

rows_written = 0
with OUT.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(header)
    for (name, mem, cap, bw, tf, tdp, dfrac, pfrac, static_w,
         link_gb, notes, prov, src) in GPU_ROWS:
        p_pre = (tdp * pfrac[0], tdp * pfrac[1])
        e_cmp = tuple(p / tf * PJFLOP_PER_W_PER_TFLOPS for p in p_pre)
        if static_w is not None:
            e_dyn = (round((tdp * dfrac[0] - static_w) / bw
                           * PJBIT_PER_W_PER_TBS, 2),
                     round((tdp * dfrac[1] - static_w) / bw
                           * PJBIT_PER_W_PER_TBS, 2))
        else:
            e_dyn = ("", "")
        row = [
            "gpu_energy", name, mem, cap, bw, tf, tdp,
            static_w if static_w is not None else "",
            dfrac[0], dfrac[1], e_dyn[0], e_dyn[1],
            pfrac[0], pfrac[1], round(p_pre[0], 1), round(p_pre[1], 1),
            round(e_cmp[0], 3), round(e_cmp[1], 3),
            "PCIe Gen5 x16" if link_gb else "", link_gb or "",
            (link_gb * 2) if link_gb else "", "", "", "",
            notes, prov, src,
        ]
        assert len(row) == N_COLS, (name, len(row))
        w.writerow(row)
        rows_written += 1
        dyn_str = (f"  e_dyn {e_dyn[0]:5.2f}-{e_dyn[1]:5.2f} pJ/bit"
                   if static_w is not None else "")
        print(f"{name:44s} static {static_w} W{dyn_str}")
    for (name, link, per_dir, bidir, host_mem, host_bw, coh,
         notes, prov, src) in LINK_ROWS:
        row = (["host_link", name] + [""] * 16
               + [link, per_dir, bidir, host_mem, host_bw or "", coh,
                  notes, prov, src])
        assert len(row) == N_COLS, (name, len(row))
        w.writerow(row)
        rows_written += 1
        print(f"{name:44s} {per_dir} GB/s/dir ({link})")
    for (name, tdp, idle, notes, prov, src) in IDLE_ROWS:
        row = (["idle_power_anchor", name, "", "", "", "", tdp, idle]
               + [""] * 16 + [notes, prov, src])
        assert len(row) == N_COLS, (name, len(row))
        w.writerow(row)
        rows_written += 1
        print(f"{name:44s} idle {idle} W of {tdp} W TDP")

# Read-back validation: every row must have exactly N_COLS fields.
with OUT.open(newline="") as f:
    for i, r in enumerate(csv.reader(f)):
        assert len(r) == N_COLS, (i, len(r))

print(f"saved: {OUT}  ({rows_written} data rows, {N_COLS} cols, validated)")
