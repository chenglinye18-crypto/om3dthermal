DreamRAM
Upstream: https://github.com/harvard-acc/DreamRAM
Baseline branch: DATE2026
Pinned commit: c069ce14dfa85ce1983f3a1274a265d1e7b5494a

Purpose:
Reference implementation for Si DRAM/HBM access-energy modeling.
The upstream source is kept as an independent local Git repository and is not
modified by om3dthermal.

Policy:
- Do not copy DreamRAM implementation into om3dthermal.
- Do not modify upstream equations for calibration.
- om3dthermal-specific accounting must live in our own adapter/wrapper code.
