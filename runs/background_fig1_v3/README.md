# Compact Background Fig. 1

The left asset is a conceptual GPU–HBM data-movement schematic, not to scale. Repeated leftward arrows indicate weight/KV reads to the GPU; the smaller return arrow indicates KV/state writes, not weight modification. Arrow widths and block sizes encode no quantitative ratios. Autoregressive Decode and Growing KV Cache distinguish the schematic from a generic memory-wall diagram.

The right asset uses current formal benchmark definitions: Llama-3.1-8B and Qwen2.5-32B; cached H=126000; P=128; G=32; B={1,8,32}. Weights and final KV(H+P+G) come from runs/formal_long_context_v3/capacity_audit.csv. Decimal GB = 1e9 bytes. Formal capacity preflight additionally includes workspace/runtime state. The H200 HBM reference comes from the current resolved conventional-HBM backend (resolve_conventional_hbm_backend) and is checked against the audit capacity field, rather than hard-coded.

Both independent assets are 3.65 × 2.40 inches with Times New Roman, vector PDF/SVG text and matching palettes. No title, panel letter, caption, solution architecture, AI/roofline, performance or thermal numbers appear. PNGs are previews only. Formal benchmark data are byte-identical before/after generation. No benchmark, thermal solve or placement optimization is run. Existing figures and manuscript remain unchanged.

Generate in the om3dthermal Conda environment:
python scripts/plot_background_fig1_v3.py
