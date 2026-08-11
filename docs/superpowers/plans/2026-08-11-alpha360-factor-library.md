# Qlib Alpha360 Factor Library Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Qlib Alpha360 as a standalone, repository-owned 360-column KunQuant factor library without changing the 259-column `combined` contract.

**Architecture:** A focused `alpha360.py` provider generates Qlib's six field groups and 60 lags from declarative metadata. Registration remains the only integration point, so catalog, CLI, engine, artifacts, evaluation, and model routing stay generic; Qlib is consulted only in tests that lock the pinned upstream contract.

**Tech Stack:** Python 3.11, pyqlib 0.9.7, KunQuant 0.1.11, NumPy, unittest/pytest, Ruff.

## Global Constraints

- Register standalone `qlib_alpha360` with `a360_001` through `a360_360`.
- Preserve `combined` as exactly Alpha158 plus Alpha101, with 259 columns in its existing order.
- Match `Alpha360DL.get_feature_config()` from pinned `pyqlib==0.9.7` exactly.
- Keep the runtime implementation in this repository; never import Qlib formula builders at runtime.
- Preserve the user's existing uncommitted edits in `README.md`, `agent.md`, `src/rquant/cli.py`, and `tests/test_cli.py`.
- Do not create or replace `data/factors/qlib_alpha360/`, or run full-data evaluation, training, or backtests.
- Do not create Git commits unless the user separately authorizes them.

## File Map

- Create `src/rquant/factors/libraries/alpha360.py` for metadata, specs, and KunQuant expressions.
- Modify `src/rquant/factors/libraries/registry.py` to register the standalone library.
- Modify `tests/test_factor_libraries.py`, `tests/test_catalog.py`, and `tests/test_kunquant_smoke.py` for contracts and numerical behavior.
- Modify `src/rquant/cli.py` and `tests/test_cli.py` for the 809-factor doctor contract and catalog output.
- Modify `README.md`, `agent.md`, and `THIRD_PARTY_NOTICES.md`; create `licenses/Qlib-0.9.7-LICENSE.txt`.

---

### Task 1: Lock the catalog and registry contract

**Files:**

- Create: `src/rquant/factors/libraries/alpha360.py`
- Modify: `src/rquant/factors/libraries/registry.py`
- Modify: `tests/test_factor_libraries.py`
- Modify: `tests/test_catalog.py`

**Interfaces:**

- Consumes: `FactorLibrary`, `FactorSpec`, and `FactorLibraryRegistry.register_library()`.
- Produces: `ALPHA360_SOURCE_NAMES`, `ALPHA360_SOURCE_EXPRESSIONS`, and `QlibAlpha360Library`.

- [ ] **Step 1: Write failing registry and catalog tests**

Extend the default registry assertion and input contract in `tests/test_factor_libraries.py`:

```python
self.assertEqual(
    ("qlib_alpha158", "wq_alpha101", "gtja191", "qlib_alpha360", "combined"),
    registry.factor_sets(),
)
self.assertEqual(
    ("open", "high", "low", "close", "volume", "vwap"),
    registry.required_inputs("qlib_alpha360"),
)
```

Import `alpha360` beside the other KunQuant libraries, assert 360 source names, reject `KunQuant.predefined` imports, and require all Alpha360 implementation paths to start with `rquant.factors.libraries.`.

Add to `tests/test_catalog.py`:

```python
def test_alpha360_is_strictly_numbered(self) -> None:
    names = get_catalog().canonical_names("qlib_alpha360")
    self.assertEqual(tuple(f"a360_{value:03d}" for value in range(1, 361)), names)

def test_alpha360_source_contract_matches_locked_qlib(self) -> None:
    from qlib.contrib.data.loader import Alpha360DL

    expressions, source_names = Alpha360DL.get_feature_config()
    specs = get_catalog().select("qlib_alpha360")
    self.assertEqual(source_names, [spec.source_name for spec in specs])
    self.assertEqual(expressions, [spec.formula for spec in specs])

def test_alpha360_does_not_expand_combined(self) -> None:
    catalog = get_catalog()
    self.assertEqual(809, len(catalog.specs))
    self.assertEqual(259, len(catalog.select("combined")))
```

- [ ] **Step 2: Run RED verification**

Run `conda run -n rquant python -m pytest tests/test_factor_libraries.py tests/test_catalog.py -q`.

Expected: FAIL because `qlib_alpha360` is absent and the catalog still has 449 unique specs.

- [ ] **Step 3: Add the minimal catalog provider**

Create `src/rquant/factors/libraries/alpha360.py`; Task 1 intentionally leaves `build()` inherited so Task 2 starts RED:

```python
"""RQuant-owned Qlib Alpha360 feature definitions for KunQuant execution."""

from __future__ import annotations

from rquant.errors import FactorContractError
from rquant.factors.libraries.base import FactorLibrary, FactorSpec

CATALOG_VERSION = 1
_LAGS = tuple(range(59, -1, -1))
_GROUPS = (
    ("CLOSE", "close"),
    ("OPEN", "open"),
    ("HIGH", "high"),
    ("LOW", "low"),
    ("VWAP", "vwap"),
    ("VOLUME", "volume"),
)


def _source_formula(field: str, lag: int) -> str:
    field = field.lower()
    numerator = f"Ref(${field}, {lag})" if lag else f"${field}"
    denominator = "($volume+1e-12)" if field == "volume" else "$close"
    return f"{numerator}/{denominator}"


ALPHA360_SOURCE_NAMES = tuple(f"{label}{lag}" for label, _ in _GROUPS for lag in _LAGS)
ALPHA360_SOURCE_EXPRESSIONS = tuple(_source_formula(field, lag) for _, field in _GROUPS for lag in _LAGS)


class QlibAlpha360Library(FactorLibrary):
    family = "qlib_alpha360"
    required_inputs = ("open", "high", "low", "close", "volume", "vwap")

    @property
    def specs(self) -> tuple[FactorSpec, ...]:
        if len(ALPHA360_SOURCE_NAMES) != 360:
            raise FactorContractError("Locked Alpha360 catalog must contain exactly 360 features")
        return tuple(
            FactorSpec(
                canonical_name=f"a360_{ordinal:03d}",
                source_name=source_name,
                family=self.family,
                ordinal=ordinal,
                formula=formula,
                max_lookback=max(int(source_name.removeprefix(source_name.rstrip("0123456789"))), 1),
                implementation=f"rquant.factors.libraries.alpha360.QlibAlpha360Library.build[{source_name}]",
                catalog_version=CATALOG_VERSION,
            )
            for ordinal, (source_name, formula) in enumerate(
                zip(ALPHA360_SOURCE_NAMES, ALPHA360_SOURCE_EXPRESSIONS, strict=True), 1
            )
        )
```

Import and register `QlibAlpha360Library()` after GTJA191 in `_default_registry()`. Keep this line unchanged:

```python
registry.register_factor_set("combined", ("qlib_alpha158", "wq_alpha101"))
```

- [ ] **Step 4: Run GREEN verification and inspect scope**

Re-run the Step 2 command, then run:

```bash
git diff --check -- src/rquant/factors/libraries/alpha360.py src/rquant/factors/libraries/registry.py tests/test_factor_libraries.py tests/test_catalog.py
```

Expected: all selected tests pass, the unique catalog is 809, `combined` is 259, and the diff check exits 0.

### Task 2: Implement and verify the KunQuant graph

**Files:**

- Modify: `src/rquant/factors/libraries/alpha360.py`
- Modify: `tests/test_kunquant_smoke.py`

**Interfaces:**

- Consumes: Task 1's ordered metadata and `KunQuantFactorEngine`.
- Produces: `QlibAlpha360Library.build(inputs: Mapping[str, Any]) -> Mapping[str, Any]`.

- [ ] **Step 1: Write failing graph and numerical tests**

Extend `test_locked_graphs_build()` to build `qlib_alpha360` and assert more than 360 graph operations. Add a clang-gated test using 70 rows and three securities, then assert:

```python
self.assertEqual(tuple(f"a360_{ordinal:03d}" for ordinal in range(1, 361)), tuple(output))
self.assertTrue(np.isnan(output["a360_001"][:59]).all())
np.testing.assert_allclose(output["a360_001"][59:], close[:-59] / close[59:])
np.testing.assert_allclose(output["a360_060"], close / close)
np.testing.assert_allclose(output["a360_061"][59:], open_[:-59] / close[59:])
np.testing.assert_allclose(output["a360_120"], open_ / close)
np.testing.assert_allclose(output["a360_241"][59:], vwap[:-59] / close[59:])
np.testing.assert_allclose(output["a360_300"], vwap / close)
np.testing.assert_allclose(output["a360_301"][59:], volume[:-59] / (volume[59:] + 1e-12))
np.testing.assert_allclose(output["a360_360"], volume / (volume + 1e-12))
```

Use deterministic positive arrays so division by zero cannot obscure formula semantics.

- [ ] **Step 2: Run RED verification**

Run `conda run -n rquant python -m pytest tests/test_kunquant_smoke.py -q`.

Expected: FAIL with `qlib_alpha360 does not implement the KunQuant backend`.

- [ ] **Step 3: Implement the ordered graph builder**

Add `Mapping`, `Any`, and `DependencyError`, then implement:

```python
def build(self, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
    try:
        from KunQuant.ops import BackRef
    except ImportError as exc:
        raise DependencyError("KunQuant==0.1.11 is required to build Alpha360") from exc

    outputs: dict[str, Any] = {}
    for label, field in _GROUPS:
        denominator = inputs["volume"] + 1e-12 if field == "volume" else inputs["close"]
        for lag in _LAGS:
            numerator = BackRef(inputs[field], lag) if lag else inputs[field]
            outputs[f"{label}{lag}"] = numerator / denominator
    if tuple(outputs) != ALPHA360_SOURCE_NAMES:
        raise FactorContractError("Local Alpha360 output order differs from the locked catalog")
    return outputs
```

- [ ] **Step 4: Run GREEN verification**

Run:

```bash
conda run -n rquant python -m pytest tests/test_kunquant_smoke.py -q
conda run -n rquant python -m pytest tests/test_factor_libraries.py tests/test_catalog.py tests/test_kunquant_smoke.py -q
git diff --check -- src/rquant/factors/libraries/alpha360.py tests/test_kunquant_smoke.py
```

Expected: graph, compiled numerical, and all factor-library tests pass.

### Task 3: Update the CLI health contract

**Files:**

- Modify: `tests/test_cli.py`
- Modify: `src/rquant/cli.py`

**Interfaces:**

- Consumes: the registry-derived Alpha360 set and 809-spec catalog.
- Produces: doctor status `ok` at 809 and machine-readable Alpha360 catalog output.

- [ ] **Step 1: Write the failing doctor expectation and CLI behavior test**

Change only the existing doctor assertion from 449 to 809 and add:

```python
def test_alpha360_catalog_json_is_machine_readable(self) -> None:
    output = io.StringIO()
    with redirect_stdout(output):
        code = main(["factors", "catalog", "--factor-set", "qlib_alpha360", "--format", "json"])
    payload = json.loads(output.getvalue())
    self.assertEqual(0, code)
    self.assertEqual(360, len(payload["factors"]))
    self.assertEqual("a360_001", payload["factors"][0]["canonical_name"])
    self.assertEqual("a360_360", payload["factors"][-1]["canonical_name"])
```

Preserve all existing categorized-run-directory edits in this dirty test file.

- [ ] **Step 2: Run RED verification**

Run `conda run -n rquant python -m pytest tests/test_cli.py -q`.

Expected: doctor fails because production still treats 449 as healthy.

- [ ] **Step 3: Change `_doctor()` to require 809 unique specs**

Change only `len(get_catalog().specs) == 449` to `len(get_catalog().specs) == 809`. Preserve the user's `run_category` and report-resolution changes.

- [ ] **Step 4: Run GREEN verification**

Run:

```bash
conda run -n rquant python -m pytest tests/test_cli.py tests/test_catalog.py -q
git diff --check -- src/rquant/cli.py tests/test_cli.py
```

Expected: all selected tests pass and the diff check exits 0.

### Task 4: Document ownership, usage, and licensing

**Files:**

- Modify: `README.md`
- Modify: `agent.md`
- Modify: `THIRD_PARTY_NOTICES.md`
- Create: `licenses/Qlib-0.9.7-LICENSE.txt`

**Interfaces:**

- Consumes: verified names, counts, inputs, source checksum, and safe-build boundary.
- Produces: executable guidance and auditable Qlib MIT provenance.

- [ ] **Step 1: Update README without overwriting run-directory changes**

Document `qlib_alpha360`, `a360_001` through `a360_360`, its six normalized fields times 60 lags, and `alpha360.py`. Add standalone catalog/build/validate examples:

```bash
rquant factors catalog --factor-set qlib_alpha360
rquant factors build --factor-set qlib_alpha360
rquant factors validate --factor-set qlib_alpha360
```

State that `combined` excludes Alpha360 and remains 259 columns, and that a build atomically replaces the complete `data/factors/qlib_alpha360/` directory.

- [ ] **Step 2: Update agent guidance and locked counts**

Add Alpha360 to repository-owned KunQuant libraries and canonical-name contracts. Set the exact expected counts to:

```text
unique catalog=809
qlib_alpha158=158
wq_alpha101=101
gtja191=190
qlib_alpha360=360
combined=259
```

Add Alpha360 to read-only catalog checks and optional standalone build/validate examples while preserving the rule that expensive jobs need authorization.

- [ ] **Step 3: Add provenance and the complete MIT license**

Add a Qlib 0.9.7 section to `THIRD_PARTY_NOTICES.md` naming upstream `https://github.com/microsoft/qlib`, local `src/rquant/factors/libraries/alpha360.py`, installed source `qlib/contrib/data/loader.py`, and SHA-256:

```text
814b7f7ab3d418ae3c87ce352220080b239eba2670eac9e38376b794be4075cb
```

Explain that RQuant reproduces the Alpha360 expression pattern and adds local naming, metadata, fingerprinting, and KunQuant execution. Create `licenses/Qlib-0.9.7-LICENSE.txt` with the full MIT text headed `MIT License` and `Copyright (c) Microsoft Corporation`.

- [ ] **Step 4: Scan for stale contracts and formatting errors**

Run:

```bash
rg -n "449|qlib_alpha360|Alpha360|combined.*259|809" README.md agent.md THIRD_PARTY_NOTICES.md src tests
git diff --check -- README.md agent.md THIRD_PARTY_NOTICES.md licenses/Qlib-0.9.7-LICENSE.txt
```

Expected: active catalog expectations are 809, `combined` is never expanded, Alpha360 provenance is present, and diff validation exits 0.

### Task 5: Run fresh complete verification

**Files:**

- Verify only; no new files.

**Interfaces:**

- Consumes: Tasks 1-4.
- Produces: environment, regression, lint, doctor, count, and scope evidence.

- [ ] **Step 1: Verify environment and imports**

Run:

```bash
conda run -n rquant python -c "import sys, rquant, qlib; print(sys.executable); print(rquant.__file__); print(qlib.__version__)"
```

Expected: the interpreter is under `/opt/miniconda3/envs/rquant`, `rquant` imports from this checkout, and Qlib is 0.9.7.

- [ ] **Step 2: Run focused and full tests**

Run:

```bash
conda run -n rquant python -m pytest tests/test_factor_libraries.py tests/test_catalog.py tests/test_kunquant_smoke.py tests/test_cli.py -q
conda run -n rquant python -m pytest -q
```

Expected: both commands pass with no failures or warnings.

- [ ] **Step 3: Run static, doctor, and catalog verification**

Run:

```bash
conda run -n rquant python -m ruff check src tests
git diff --check
conda run -n rquant rquant doctor --skip-permission-check
conda run -n rquant python -c "from rquant.factors.catalog import get_catalog; c=get_catalog(); print(len(c.specs)); print({s: len(c.select(s)) for s in c.factor_sets()})"
```

Expected: Ruff succeeds, diff validation exits 0, doctor is `ok`, and the counts are `809`, `158`, `101`, `190`, `360`, and `259` for the named contracts.

- [ ] **Step 4: Confirm no production artifact was built**

Run:

```bash
git status --short --branch
git diff --stat
test ! -e data/factors/qlib_alpha360
```

Expected: only user changes plus planned Alpha360 work are present, and no `data/factors/qlib_alpha360` directory exists. Report that full-data build, evaluation, training, and backtest were not run. Do not commit without separate authorization.
