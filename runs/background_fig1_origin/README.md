# Fig. 1: editable Origin capacity chart

Created with Origin 2021 (COM version 9.8002). The OPJU contains a native stacked-column graph and its source worksheet (Book1, long name Capacity_Data); it is not an imported image. Both series remain grouped, with independent style editing. Numeric X positions encode the small inter-model gap; tick-indexed batch labels and native model labels supply the two categorical levels. Data, colors, spacing, axes, fonts, totals and capacity line are editable in Origin.

Source: ../formal_long_context_v3/capacity_audit.csv, H=126000, P=128, G=32, B=1/8/32, Llama-3.1-8B and Qwen2.5-32B. Weights come directly from weight_bytes, KV from final_KV_bytes, each divided by 1e9 (decimal GB). Total is their sum; workspace is not part of these two plotted components. Capacity comes directly from HBM_capacity_bytes: 144955146240 bytes = 144.95514624 GB. No values are inferred from rounded totals.

| Model | B | Weights GB | KV GB | Total GB |
|---|---:|---:|---:|---:|
| Llama-3.1-8B | 1 | 16.060522496 | 16.53604352 | 32.596566016 |
| Llama-3.1-8B | 8 | 16.060522496 | 132.28834816 | 148.348870656 |
| Llama-3.1-8B | 32 | 16.060522496 | 529.15339264 | 545.213915136 |
| Qwen2.5-32B | 1 | 65.527752704 | 33.07208704 | 98.599839744 |
| Qwen2.5-32B | 8 | 65.527752704 | 264.57669632 | 330.104449024 |
| Qwen2.5-32B | 32 | 65.527752704 | 1058.30678528 | 1123.834537984 |

Rebuild from repository root in PowerShell:

```powershell
./scripts/build_fig1_capacity_origin.ps1
```

Requires registered Origin.Application COM and PyMuPDF in the om3dthermal Conda environment. Origin directly generates OPJU, PDF and PNG. Origin 2021 predates SVG export (added in 2022b), so SVG is converted from the Origin vector PDF with PyMuPDF, preserving text rather than rasterizing. No matplotlib is used.

Validation: saved project reopened successfully; all six C/D/E worksheet values read back at full precision and agreed with capacity_data.csv; totals round to 32.6/148.3/545.2/98.6/330.1/1123.8. The capacity line is created at the canonical capacity, with Origin's native display-coordinate precision. PDF fonts are Times New Roman regular/bold. Exports visually inspected. Formal benchmark data modified = NO. No benchmark, thermal run or pytest performed.
