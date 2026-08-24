# Historical research records

Files in this directory are preserved for research-history and provenance
only. They are not the current project source of truth and must not be used to
override validated code, canonical configs, current audit reports, or
`docs/architecture/E2E_ARCHITECTURE.md`.

Contents:

- `PROJECT_STATUS_2026-08-18.md`: project snapshot before the workload-aware
  E2E chain and architecture refactor were completed.
- `handoffs/PROJECT_LEAD_HANDOFF_2026-08-18.md`: early research-lead handoff;
  several next-step recommendations are now superseded.
- `specs/LLM_DECODE_ACCOUNTING_B1_v0.md`: initial analytical draft.
- `specs/LLM_DECODE_ACCOUNTING_B1_R1.md`: first revision, superseded by the
  frozen B1-R2 semantics implemented in `workload/llm_decode.py` and locked by
  `tests/test_llm_decode.py`.

Current validated stage reports live under `docs/audit/`. Historical files
should remain unchanged; corrections belong in a new current document with
explicit provenance.
