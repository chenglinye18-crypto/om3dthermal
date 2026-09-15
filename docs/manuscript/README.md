# IOM3D-HBM physical evidence revision

The editable manuscript is `iom3d_hbm_evidence.docx`; the complete compiled
paper is `iom3d_hbm_evidence.pdf`. This is a repository copy of the current
user-maintained Word source beside the repository, not a TeX conversion.
The original DOCX/PDF remain untouched. The original generated PDF had six
body pages and excluded the source's separate references page. This revision
fits body and references in six pages at the original body font size.

## Changes

* Introduction: complete orthogonal package + IGZO BEOL + Si FEOL architecture
  integration is the first contribution; DNS/CPA jointly exploit locality;
  physical/thermal evaluation supplies evidence, not a simulator novelty claim.
* III-B: finite per-die ports, nearest-group routing, regional striping,
  broadcast ingress, shared-port serialization and global ceilings.
* III-C: physical operator/partition/layer/group/region hierarchy; explicit
  service equations and shared-request aggregation; activity-based energy;
  parameter provenance and numerical/accounting consistency boundaries.
  The original unsupported 39.15-TB/s trace-average sentence is removed;
  nominal/internal ceiling and workload-achieved bandwidth are distinguished.
* IV-C: 263 original extracted words including Algorithm 1 become 109 words
  plus the retained objective/constraint equations. One resident pass and at
  most two tile rounds accurately describe the bounded implementation.
* V-B: five HBM-resident cases establish 3.0692x CPA/HBM and 2.9584x CPA/M3D;
  13 overflow cases and all-case geometric means are also reported. Three
  existing 8B/70B/405B points anchor the normalized figure in tokens/s and TPOT.
* Fig. 11: full-width physical Decode anatomy, one formal CPA checkpoint,
  reconstructed operator intervals and normalized resource service matrix.
  Fig. 1–10 artwork is preserved. Fig. 1's mismatched caption and Fig. X
  placeholders are corrected. Template keywords/funding/footer text removed.
* Thermal text: workload activity, per-die BEOL uniform mapping, GPU FEOL,
  3-D steady-state solve and shared package boundary conditions. An existing
  text-only typo (HBM24Hi 1.76 TB/s) is corrected to the canonical 1.96 TB/s,
  already shown in Fig. 5. Abstract NMP Tmax is rounded consistently to 81.41°C.
* Reference formatting: remove displayed HTML superscript tags and a forced
  references page break, preserve existing bibliographic sources and DOIs.

`manuscript_edits.json` contains exact before/after paragraph text and source
SHA-256. No workload, precision, hardware, DNS/CPA implementation, benchmark
result, or primary conclusion changes in this revision.

## Evidence audit

The repository and origin/main both started at
`202a7979c5a865d710709c5cda6b6c38c4558c6e`.
The reading included the entire Word manuscript and six-page source PDF,
formal configuration/candidates/capacity/thermal data, both old and corrected
GPU port diagnostics, attention-probe latency/traffic read from Git (the user
deleted the historical working files), frequency-sweep report and cache,
geometry sensitivity thermal limits, and the directly relevant physical,
placement, event-energy, routing, scheduler and thermal solver code.

The external service equation in `power/nmp_die_activity.py` is
max(total_bytes/global_cap, max_port(port_bytes/port_rate + route_startup)).
GROUP_DIRECT picks the nearest physical port (35 of 50 are reachable in the
current geometry); REGION_DIRECT stripes among assigned region ports;
REGION_BROADCAST_INGRESS admits one copy before intra-die multicast. Requests
sharing resources merge loads before maxima, including batched NMP service.
There is no unrestricted 318x50 independent GPU bandwidth claim.

The operator scheduler is a fixed dependency sequence, not a cycle-accurate
GPU simulator or an independently calibrated software synchronization model.
CPA evaluates complete physical **stage** equations, not a whole-workload
rerun for every candidate. Global optimality is not asserted. Thermal power
is workload-dependent but NMP intra-die component hotspots are not claimed:
the formal map is die-grouped BEOL uniform.

## Build

```powershell
conda run --no-capture-output -n om3dthermal python scripts/plot_physical_decode_execution_anatomy.py
conda run --no-capture-output -n om3dthermal python scripts/revise_physical_evidence_manuscript.py --source '<original manuscript.docx>'
./scripts/compile_evidence_manuscript.ps1
```

Word is used for native layout and PDF export. The added SVG is embedded as
a vector image. Original architecture artwork and equations are retained.
PDF pages are rendered using Poppler for visual inspection. The installed
documents skill does not contain its advertised `scripts/render_docx.py`;
native Word export plus Poppler supplies the equivalent render/inspect path.

Related tests: `tests/test_physical_decode_execution_anatomy.py`. They check
serial dependencies, nonadditive resource equations, exact matrix semantics,
subset aggregation and deterministic SVG/PDF export. No full pytest suite,
thermal benchmark, optimizer or workload sweep is run.

## Deferred

Workload redesign, quantization, capacity-matched/alternative baselines,
external SOTA PIM/NMP comparisons and software synchronization sensitivity
remain for a separate research task. No new mechanisms are introduced here.
