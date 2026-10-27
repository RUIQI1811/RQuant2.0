# Remaining KunQuant migrations

> Execute the user's approved continuation of PR #18. The two remaining libraries have independent formula code and tests; use the dispatching-parallel-agents skill for those domains, then integrate and verify centrally.

**Goal:** Execute all 233 jq240 and 142 oap formulas in KunQuant, retaining stable columns, point-in-time semantics, and independent pandas parity references.

**Architecture:** Continue PR head b2cb021, using its two-stage statement/daily graphs, C++ series kernels, and library reference-input interface. Python may arrange data, quarters and announcement dates; it must not calculate production factor formulas. Prefer existing operators; add narrowly scoped library extensions for missing operations.

**Constraints:** Python 3.11 rquant environment only; KunQuant 0.1.11; preserve original reference formulas; preserve NaN/Inf handling, suspended trading-day windows, historical universe masks and financial announcement alignment. No changes to existing production data or environment installation. Keep library counts and names unchanged. Finish the user's existing PR #18 workflow only after verification; publish by fast-forward without rewriting others' commits.

## Tasks

- [x] Confirm existing KunQuant compilation works on this Windows host using temporary synthetic data.
- [x] jq240: preserve original in tests/panel_reference/jq240.py; add failing KunQuant backend and full numerical parity tests; implement financial graph, compressed technical calculations and universe-scoped cross-sectional operations in libraries/jq240.py and a separate extension if needed; run parity, prefix and existing regression tests.
- [x] oap: preserve original in tests/panel_reference/oap.py; add failing KunQuant backend and full numerical parity tests; implement statement and daily graphs, monthly blocks, grouped operations and regression kernels in libraries/oap.py and a separate extension if needed; run parity, prefix and existing regression tests.
- [x] Integration: update README, agent.md and affected registry/backend tests; inspect implementation fingerprints and graph/reference input plumbing; run focused tests, Ruff and full fast suite.
- [x] Review complete changes and report exact passing, failing and unverified coverage. Prepare PR description locally; do not push or modify PR without authorization.

## Review focus

- Odd stock counts and padded lanes must not enter cross-sectional calculations.
- Missing financial quarters and out-of-order announcements must not expose future data.
- Suspension gaps must preserve each library's original window and state semantics.
- Rank-deficient regressions and constant cross sections must retain reference missing-value behavior.
- Long-history signals require fixtures reaching their actual lookbacks, plus truncated-input tests.

## Execution notes

- Local main remains at 9f9be7a; isolated worktree is .claude/worktrees/codex-kunquant-remaining on codex/kunquant-remaining.
- Existing Conda launcher fails with Windows DLL relocation; direct rquant/python.exe works. Running from worktree/src imports this checkout without modifying environment installation or PYTHONPATH.
- Default sandbox TEMP fails access checks; project TEMP contains non-ASCII characters unsupported by KunQuant DLL loading. Use the authorized ASCII visualization workspace test-temp directory for test processes only.
- Windows short-path TEMP fixed linker crashes caused by paths exceeding the native linker's limit, but caused a CLI path-comparison test to see a short alias versus its resolved full name. Final broad verification uses the normal user TEMP and preloads this checkout's `rquant` package before changing to the project root. No environment installation was changed.
- jq240: 20 tests passed in 235.32 seconds, including all 233 columns, seven-stock padding, prefixes, suspensions, financial publication, universe perturbations, tiny moments, degenerate OLS and exact near-tie ranking.
- oap: 18 tests passed in 631.88 seconds, including all 142 columns on 2200 x 40 synthetic input, prefix recomputation, announcement timing and engine publication.
- OAP operator/fallback regressions: 11 tests passed; the long-history momentum test separately passed on 2650 x 3 inputs, including the full 120-month lag.
- Independent review found a shared-operator fingerprint omission; the focused test failed before adding the operator source to OAP's fingerprint and passed after the repair. No outstanding confirmed review defects.
- No full-history production data rebuild, live data synchronization, training, backtest or real-history performance benchmark was performed.
- Delivery follows the user's supplied continuation of the existing PR #18 workflow, including its final PR-description update. The remote head was rechecked at b2cb021 before publication; main and existing production stores remain outside this change.
- Final broad suite: 222 passed, 1 skipped, 1 third-party deprecation warning in 1393.77 seconds. The skip is the unrelated RD-Agent HDF5 export requiring optional PyTables, which is not installed. The separately added full-longest-lag test passes (1 passed in 22.74 seconds), covering the only test added after broad-suite collection.
- The two independent pandas references compare unchanged against the original b2cb021 production modules (`git diff --no-index` reports no differences).
- Final `python -m ruff check src tests` and staged/working whitespace checks pass. Linux CI is checked after the existing PR branch is updated.
