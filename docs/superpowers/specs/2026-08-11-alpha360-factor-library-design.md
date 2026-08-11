# Qlib Alpha360 Factor Library Design

## Objective

Add Qlib's Alpha360 feature family to RQuant as a standalone, repository-owned KunQuant factor library named
`qlib_alpha360`. Preserve the existing `combined` factor set as the locked 259-column Alpha158 plus Alpha101
compatibility contract.

## Upstream Semantics

The implementation targets the `Alpha360DL.get_feature_config()` contract installed with the project's pinned
`pyqlib==0.9.7`. Alpha360 is 360 lagged, normalized market features rather than 360 independently designed alpha
formulas:

- `CLOSE59` through `CLOSE0`: lagged close divided by current close;
- `OPEN59` through `OPEN0`: lagged open divided by current close;
- `HIGH59` through `HIGH0`: lagged high divided by current close;
- `LOW59` through `LOW0`: lagged low divided by current close;
- `VWAP59` through `VWAP0`: lagged VWAP divided by current close;
- `VOLUME59` through `VOLUME0`: lagged volume divided by current volume plus `1e-12`.

Each group is ordered from lag 59 down to lag 0. The complete source order is therefore six groups of 60 columns.

## Public Contract

- Register one standalone factor set: `qlib_alpha360`.
- Expose canonical names `a360_001` through `a360_360` in exact Qlib source order.
- Retain the Qlib names such as `CLOSE59` and `VOLUME0` as catalog `source_name` metadata.
- Require only `open`, `high`, `low`, `close`, `volume`, and `vwap` canonical inputs.
- Execute through the existing KunQuant backend and generic factor engine.
- Keep `combined` exactly `qlib_alpha158` plus `wq_alpha101`, with 259 columns and unchanged ordering.
- Increase the unique registered catalog from 449 to 809 factors.
- Let CLI choices, artifact directories, loaders, evaluation, and walk-forward routing derive `qlib_alpha360`
  automatically from the registry instead of hard-coding the name in those consumers.

## Repository Ownership and Provenance

Add a focused `src/rquant/factors/libraries/alpha360.py` module containing the local expression construction,
catalog metadata, and provider class. Runtime construction must not import formulas from Qlib or depend on Qlib's
installed source. Qlib is used in tests only to lock source names and expressions against the pinned version.

Document that the local formula pattern is derived from Qlib Alpha360, record the pinned source/version boundary,
and add the applicable Qlib license notice without changing the existing KunQuant notice.

## Implementation Shape

The module will generate the six field groups from small declarative constants rather than repeat 360 expressions.
For each source feature it will create a `FactorSpec` with stable canonical name, source name, ordinal, formula text,
59-lag maximum history metadata where applicable, implementation path, and a new Alpha360 catalog version.

The library's `build()` method will construct KunQuant `BackRef` and division expressions in the locked source order.
It will validate that the generated names match its specs before returning the source-name-to-expression mapping.
The registry will instantiate the library before registering `combined`, so the standalone factor-set order becomes:
`qlib_alpha158`, `wq_alpha101`, `gtja191`, `qlib_alpha360`, `combined`.

## Error Handling and Compatibility

- Raise the existing dependency error if KunQuant is unavailable when building the graph.
- Raise the existing factor-contract error if the generated count or order differs from the locked 360-column
  catalog.
- Do not modify or rebuild existing factor artifacts as part of implementation.
- A later `factors build --factor-set qlib_alpha360` writes only `data/factors/qlib_alpha360/` through the existing
  staging and atomic replacement path.
- Existing `combined` manifests, models, and feature ordering remain compatible because that bundle is unchanged.

## Tests and Verification

Use test-driven development and cover these behaviors:

1. The registry exposes `qlib_alpha360` with the exact required-input contract while retaining `combined` unchanged.
2. The catalog exposes exactly `a360_001` through `a360_360`.
3. Source names and formula strings exactly match Qlib 0.9.7 `Alpha360DL.get_feature_config()`.
4. The total unique catalog and doctor health expectation are 809.
5. The generic KunQuant engine builds an Alpha360 graph with all 360 outputs.
6. A small deterministic runtime fixture verifies representative close, open, VWAP, and volume lag calculations,
   including lag 0 and lag 59.
7. Existing Alpha158, Alpha101, GTJA191, and 259-column `combined` tests remain green.
8. README, `agent.md`, third-party notices, and license files describe the new factor set and updated counts.

Run focused tests first, then the full fast test suite, Ruff, environment/doctor checks, and `git diff --check`.
Do not run a full-data Alpha360 build, factor evaluation, model training, or backtest without separate authorization.

## Success Criteria

The change is complete when `qlib_alpha360` is registry-derived across the CLI and generic engine, its 360-column
source and numerical contracts are covered by passing tests, the unique catalog count is 809, `combined` remains
259 columns, documentation and provenance are current, and no production factor directory has been created or
replaced.
