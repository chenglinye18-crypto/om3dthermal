# E2E bandwidth / energy ledger

`e2e_bandwidth_energy_ledger_2026-09-07.csv` is a **derived research and
provenance ledger**. It is generated, not authoritative. Runtime authority
remains the canonical YAML configuration and the existing power, bandwidth,
and transfer resolvers. Runtime evaluation code must never read this CSV.

Regenerate it from the repository root with:

```powershell
python scripts/generate_e2e_bandwidth_energy_ledger.py
```

Component rows are not all additive. `included_in_system_energy` marks the
normal system-accounting boundary, while `parent_aggregate` and
`double_counting_note` identify component rows already contained in an
aggregate. In particular, `M3D_TOTAL_READ` already includes MAT-local read,
global routing, MIV, FEOL routing, and the contactless interface; likewise,
`HOST_OFFLOAD_DYNAMIC_PATH` already includes DDR and PCIe dynamic energy.
`GPU_DECODE_DYNAMIC` is an independent GPU-only boundary and may be combined
with the selected memory read aggregate.

Blank bandwidth or energy cells mean “not applicable.” They are intentionally
not written as zero, because zero is a physical value.

The GPU decode nominal coefficient and peak bandwidth come from the canonical
platform YAML. Its min/max sensitivity endpoints remain non-runtime provenance
values and are copied programmatically from the existing GPU platform research
ledger; the generator verifies that ledger's nominal agrees with the runtime
YAML before writing this consolidated table.
