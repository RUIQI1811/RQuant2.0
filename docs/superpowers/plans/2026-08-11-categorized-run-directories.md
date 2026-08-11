# Categorized Run Directories Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route new factor-evaluation and walk-forward runs into separate category directories while keeping `rquant report` able to read legacy walk-forward directories.

**Architecture:** Add parser-owned run-category metadata consumed by the existing shared `_run` wrapper, so only the two selected commands change their write roots. Isolate report lookup in a deterministic helper that checks the new walk-forward category first and the legacy root second.

**Tech Stack:** Python 3.11, `argparse`, `pathlib`, `unittest`, pytest, Ruff.

## Global Constraints

- The configured `project.runs_root` remains authoritative; `runs` is only the current default.
- New factor-evaluation runs write only to `<runs_root>/factors-evaluate/<run-id>/`.
- New walk-forward runs write only to `<runs_root>/walk-forward/<run-id>/`.
- Other commands retain `<runs_root>/<run-id>/`.
- Report lookup is new path first, legacy path second, without scanning unrelated directories.
- Existing run directories and existing uncommitted work are not moved, renamed, modified, deleted, or overwritten.
- No real factor evaluation, model training, portfolio backtest, Git commit, push, or pull request is part of this implementation.

---

### Task 1: Route categorized runs and resolve reports compatibly

**Files:**
- Modify: `tests/test_cli.py`
- Modify: `src/rquant/cli.py`

**Interfaces:**
- Consumes: parsed `argparse.Namespace`, `ProjectPaths.runs`, and the existing `RunContext(runs_root, command, config)` constructor.
- Produces: optional `args.run_category: str`, `_resolve_report_run_directory(runs_root: str | Path, run_id: str) -> Path`, and categorized `run_directory` values in the existing CLI JSON response.

- [ ] **Step 1: Add failing parser and run-root tests**

Add imports for `argparse`, `TemporaryDirectory`, `Path`, `ProjectPaths`, and `_run`. Add these tests:

```python
def test_only_factor_evaluate_and_walk_forward_have_run_categories(self) -> None:
    parser = build_parser()
    factor_args = parser.parse_args(["factors", "evaluate", "--factor-set", "combined", "--horizon", "20d"])
    walk_args = parser.parse_args(
        ["walk-forward", "--model", "lgb", "--horizon", "1d", "--factor-set", "combined"]
    )
    doctor_args = parser.parse_args(["doctor", "--skip-permission-check"])

    self.assertEqual("factors-evaluate", factor_args.run_category)
    self.assertEqual("walk-forward", walk_args.run_category)
    self.assertFalse(hasattr(doctor_args, "run_category"))

def test_run_uses_optional_category_beneath_configured_runs_root(self) -> None:
    with TemporaryDirectory() as temporary:
        paths = ProjectPaths.from_root(temporary, runs_root="custom-runs")
        args = argparse.Namespace(run_category="walk-forward")
        output = io.StringIO()
        with (
            patch("rquant.cli._config", return_value=({}, paths)),
            patch("rquant.cli._dependency_versions", return_value={}),
            redirect_stdout(output),
        ):
            code = _run(args, lambda run, config, project_paths: {"ok": True})

        payload = json.loads(output.getvalue())
        run_directory = Path(payload["run_directory"])
        self.assertEqual(0, code)
        self.assertEqual(paths.runs / "walk-forward", run_directory.parent)
        self.assertTrue((run_directory / "run.json").is_file())
```

- [ ] **Step 2: Run the new parser and run-root tests to verify RED**

Run:

```bash
/opt/miniconda3/envs/rquant/bin/python -m pytest \
  tests/test_cli.py::CliTests::test_only_factor_evaluate_and_walk_forward_have_run_categories \
  tests/test_cli.py::CliTests::test_run_uses_optional_category_beneath_configured_runs_root -q
```

Expected: FAIL because the parsed namespaces have no `run_category` and `_run` still passes `paths.runs` directly to `RunContext`.

- [ ] **Step 3: Implement minimal categorized run routing**

In `build_parser`, attach the categories only to the selected commands:

```python
factor_evaluate.set_defaults(handler=_factor_evaluate, run_category="factors-evaluate")
walk.set_defaults(handler=_walk_forward, run_category="walk-forward")
```

In `_run`, select the root without changing `RunContext` or manifest semantics:

```python
run_category = getattr(args, "run_category", None)
runs_root = paths.runs / run_category if run_category else paths.runs
run = RunContext(runs_root, list(sys.argv if sys.argv else ["rquant"]), config)
```

- [ ] **Step 4: Run the parser and run-root tests to verify GREEN**

Run the Step 2 command again.

Expected: 2 passed.

- [ ] **Step 5: Add failing report-resolution tests**

Import `DataContractError` and `_resolve_report_run_directory`, then add:

```python
def test_report_run_resolution_prefers_categorized_directory(self) -> None:
    with TemporaryDirectory() as temporary:
        runs_root = Path(temporary)
        categorized = runs_root / "walk-forward" / "same-id"
        legacy = runs_root / "same-id"
        categorized.mkdir(parents=True)
        legacy.mkdir()

        self.assertEqual(categorized, _resolve_report_run_directory(runs_root, "same-id"))

def test_report_run_resolution_falls_back_to_legacy_directory(self) -> None:
    with TemporaryDirectory() as temporary:
        runs_root = Path(temporary)
        legacy = runs_root / "legacy-id"
        legacy.mkdir()

        self.assertEqual(legacy, _resolve_report_run_directory(runs_root, "legacy-id"))

def test_report_run_resolution_rejects_missing_run_without_creating_paths(self) -> None:
    with TemporaryDirectory() as temporary:
        runs_root = Path(temporary)
        expected = runs_root / "walk-forward" / "missing-id"

        with self.assertRaisesRegex(DataContractError, str(expected)):
            _resolve_report_run_directory(runs_root, "missing-id")

        self.assertFalse(expected.exists())
```

- [ ] **Step 6: Run report-resolution tests to verify RED**

Run:

```bash
/opt/miniconda3/envs/rquant/bin/python -m pytest tests/test_cli.py -k report_run_resolution -q
```

Expected: test collection fails because `_resolve_report_run_directory` does not exist.

- [ ] **Step 7: Implement minimal report resolver and use it in `_report`**

Add this focused helper near `_report`:

```python
def _resolve_report_run_directory(runs_root: str | Path, run_id: str) -> Path:
    root = Path(runs_root)
    categorized = root / "walk-forward" / run_id
    if categorized.is_dir():
        return categorized
    legacy = root / run_id
    if legacy.is_dir():
        return legacy
    raise DataContractError(f"Run does not exist: {categorized}")
```

Replace the current inline `paths.runs / args.run_id` lookup and existence check with:

```python
run_directory = _resolve_report_run_directory(paths.runs, args.run_id)
```

- [ ] **Step 8: Run the report-resolution and complete CLI tests to verify GREEN**

Run:

```bash
/opt/miniconda3/envs/rquant/bin/python -m pytest tests/test_cli.py -q
```

Expected: all `tests/test_cli.py` tests pass with no warnings.

- [ ] **Step 9: Review the focused source and test diff**

Run:

```bash
git diff --check -- src/rquant/cli.py tests/test_cli.py
git diff -- src/rquant/cli.py tests/test_cli.py
```

Confirm that no factor calculations, model logic, manifest schema, or unrelated user changes were modified.

---

### Task 2: Update run-path documentation and verify the repository

**Files:**
- Modify: `README.md`
- Modify: `agent.md`

**Interfaces:**
- Consumes: the categorized directory and legacy report contracts implemented in Task 1.
- Produces: executable operator guidance using `runs/factors-evaluate/<run-id>/` and `runs/walk-forward/<run-id>/`, plus an explicit legacy report note.

- [ ] **Step 1: Update README factor-evaluation paths**

Change the factor-evaluation output description from `runs/<run-id>/` to:

```text
runs/factors-evaluate/<run-id>/
```

Keep all existing filenames unchanged.

- [ ] **Step 2: Update README walk-forward and report paths**

Change the walk-forward tree and its inspection commands to
`runs/walk-forward/<run-id>/`. Change report output and inspection examples to
the same directory, then state that `rquant report <run-id>` first checks the
new path and falls back to legacy `runs/<run-id>/`.

- [ ] **Step 3: Clarify the general run-manifest section**

Document the three layouts explicitly:

```text
runs/factors-evaluate/<run-id>/run.json
runs/walk-forward/<run-id>/run.json
runs/<run-id>/run.json                    # other commands and legacy runs
```

Update recent-run examples so they include the two category levels rather than
assuming every run is a direct child of `runs`.

- [ ] **Step 4: Update agent guide contracts**

In `agent.md`, replace the single flat run-directory claim with the two new
categorized paths plus the unchanged legacy/other-command layout. Update the
long-run audit guidance to tell future agents to inspect the actual
`run_directory` returned by stdout and preserve report fallback compatibility.

- [ ] **Step 5: Run documentation consistency and diff checks**

Run:

```bash
rg -n "runs/<run-id>|runs/RUN_ID|runs/\*/" README.md agent.md
git diff --check -- README.md agent.md
git diff -- README.md agent.md
```

Inspect every remaining flat-path reference and retain it only when it clearly
describes other commands or legacy layout.

- [ ] **Step 6: Verify interpreter and package source**

Run:

```bash
/opt/miniconda3/envs/rquant/bin/python -c \
  "import os, sys, rquant; print(os.environ.get('CONDA_DEFAULT_ENV')); print(sys.executable); print(rquant.__file__)"
```

Expected: Python comes from `/opt/miniconda3/envs/rquant`, and `rquant.__file__`
points into this checkout's `src/rquant/`.

- [ ] **Step 7: Run targeted and full automated verification**

Run:

```bash
/opt/miniconda3/envs/rquant/bin/python -m pytest tests/test_cli.py tests/test_runs.py -q
/opt/miniconda3/envs/rquant/bin/python -m pytest -q
/opt/miniconda3/envs/rquant/bin/python -m ruff check src tests
```

Expected: all targeted tests, all fast tests, and Ruff pass without errors.

- [ ] **Step 8: Perform final scope and worktree verification**

Run:

```bash
git diff --check
git status --short
git diff --stat
```

Confirm the pre-existing modifications to `README.md`, `agent.md`,
`src/rquant/factors/evaluation.py`, and `tests/test_factor_evaluation.py` remain
preserved, and distinguish this task's edits in the final handoff. Do not run
real factor evaluation, walk-forward training, or report backtesting.
