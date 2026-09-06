# Current DAC experiment entry points

Conventional HBM has one baseline: `configs/cases/conventional_hbm_2x1.yaml`.
This policy applies to paper tables, current figures, run examples, and formal
experiments. Orthogonal Si remains a technology ablation and M3D-IGZO is the
proposed architecture.

| Role | Configuration |
|---|---|
| Conventional baseline case | `configs/cases/conventional_hbm_2x1.yaml` |
| Conventional descriptor | `configs/architecture/conventional_hbm_2x1.yaml` |
| Formal conditional decode experiment | `configs/experiment/m3d_igzo_llama31_8b_decode_conditional_v0.yaml` |
| Capacity-aware serving experiment | `configs/experiment/capacity_aware_serving_v0.yaml` |

Use the canonical Windows Conda environment:

```powershell
conda activate om3dthermal
python -m om3dthermal experiment configs\experiment\m3d_igzo_llama31_8b_decode_conditional_v0.yaml
```

The formal experiment creates versioned result bundles under `results/`.
The earlier `runs/e2e_canonical/summary.json` NMP evaluation uses the old
operator assignment and does not establish results for the pending MAC/GPU
partition. Rerun after that model passes validation.

## Baseline identity

The two visible 11 x 22 mm thermal groups represent four physical HBM stack
equivalents in total, each with 12 DRAM dies. The merged group geometry and
capacity interpretation remain unchanged.

The baseline uses analytical workload memory power and a 300 W GPU input.
Do not substitute the older fixed 160 W memory-power fixture results for this
working point. Frozen numerical anchors and source records are listed in
[the thermal reference](thermal_results_overview.md).

## Retired results

The former multi-layout thermal tables and figures are excluded from current
DAC evaluation. Historical records reside in `docs/archive/`; configurations
under `configs/legacy/` remain only for compatibility tests and provenance.
A successful historical test does not make its result a current paper baseline.
