# JKP long seasonality implementation plan

**Goal:** Add the supported 11–15-year seasonal-return characteristics while preserving every existing `jkpx_001`–`jkpx_362` mapping.

**Architecture:** Append two names to the locked JKP406 output sequence. Reuse the existing KunQuant 21-trading-day seasonal operator and independently calculate an array reference. Preserve the 16–20-year omissions, report the local adaptation, and invalidate fingerprints through the existing implementation mechanism.

**Constraints:** Existing shared checkout has unrelated uncommitted work; edit only this isolated PR worktree. Tushare inputs and PIT semantics remain unchanged. Run costly tests serially on Python 3.11 using an ASCII TEMP; do not replace existing factor stores without checking impact. Previously produced JKP406 results refer to the earlier 362-column implementation.

- [x] Add failing regressions fixing old names/ordinals, new output metadata, >3780-row values, NaN/suspension masks and truncated-history parity.
- [x] Append `seas_11_15an` and `seas_11_15na` as `jkpx_363`–`jkpx_364`; update inventory, graph, independent reference and counts without reordering prior names.
- [x] Run focused tests, Ruff and full pytest; obtain independent quantitative review; document changed factor fingerprint and existing-store rebuild requirements.

Final Python 3.11 verification: Ruff passed, full pytest 296 passed, 1 skipped, 1 existing MLflow deprecation warning in 527.06 seconds. Independent quantitative review caught two issues first: the 3780-day per-stock availability gate, and non-annual missingness when an annual block is non-finite. Tests demonstrated both failures before the fixes. Non-annual uses the cached annual expression as its finite-value gate. A low-memory sample of two stocks found 702 finite values each per new signal, from 2023-11-24 through 2026-10-21; this does not establish full-universe coverage.
