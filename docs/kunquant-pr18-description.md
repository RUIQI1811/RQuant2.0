# Compute every built-in factor library with KunQuant

All nine built-in formula libraries now execute through KunQuant. This continuation of PR #18 completes jq240 (233 factors) and oap (142 factors); Hquality and recent continue to select columns from their source stores. The 1439-column catalog, public names and ordering are unchanged.

Financial formulas run on fiscal-quarter graphs and publish from their first usable announcement dates. Python retains data loading, cleaning and date alignment. Daily formulas, suspension-aware technical windows, monthly sampling, industry operations and regressions run as KunQuant expressions or C++ kernels. The original pandas implementations remain independent references under `tests/panel_reference/`.

The migration preserves exact jq240 ranking, including near ties; handles constant and tiny-variance windows; supports rank-deficient cross-sectional and rolling regressions; and excludes nonmembers and padded stocks from fitting populations. OAP's tail index retains its full-month warmup. Shared C++ helpers can coexist when libraries are combined, and imported operators/helpers participate in implementation fingerprints. Large graphs may split C++ sources during compilation.

Validation includes all-column numerical parity and NaN masks, truncated-input recomputation, announcement timing, suspensions, universe changes, odd stock counts, degenerate regressions, and the full 120-month seasonal lag. On Windows, the complete fast suite passes 222 tests and skips the optional PyTables HDF5-export test; the supplementary 2650-day momentum test also passes. The focused jq240 and oap suites pass 20 and 18 tests respectively. Ruff and whitespace checks pass. The existing GitHub workflow runs the suite on Ubuntu.

The earlier GTJA migration's documented rank/ts_rank grid (40 significant binary bits, zero below 1e-12) and allowance of up to four GTJA113 constant-window rounding differences in its reference fixture remain unchanged.

Existing stores for migrated libraries need rebuilding under their new fingerprints, followed by dependent selections. Production factor stores have not been rebuilt, and full-history runtime or memory performance has not been benchmarked.
