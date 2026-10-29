# China 469 formation implementation plan

**Goal:** Implement three missing original-timing formulas as an auditable Tushare adaptation.

**Architecture:** Two registered KunQuant library families share a single factor set, with independent annual and quarterly publication calendars derived from first-announced statement metadata. A small date-alignment helper handles eligibility only; arithmetic belongs to KunQuant. Specs and evidence are in `docs/superpowers/specs/2026-10-29-china469-formation-design.md`.

- [ ] TDD formation-date helper and first-announcement selection on weekends, holidays, late reports, older eligible quarters and partial months.
- [ ] TDD quarter and annual graph arithmetic, masks and freeze, with independent direct numerical reference and truncation invariance.
- [ ] Register factor set, add input manifest hashes, metadata, attribution and documentation, preserve all existing factor names and outputs.
- [ ] Run focused tests, Ruff and full Python 3.11 pytest serially; independent quantitative review; draft PR stacked on JKP branch after verification.
