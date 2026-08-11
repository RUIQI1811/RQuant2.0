# Categorized Run Directories Design

## Goal

Separate new factor-evaluation and walk-forward artifacts beneath the configured
`runs` root while preserving read access to existing walk-forward runs.

## Directory contract

New `rquant factors evaluate` runs write all of their existing per-run artifacts
to:

```text
runs/factors-evaluate/<run-id>/
```

New `rquant walk-forward` runs write all of their existing per-run artifacts to:

```text
runs/walk-forward/<run-id>/
```

The configured `project.runs_root` remains authoritative; `runs` above is only
the current default. Run IDs and the contents of each run directory do not
change. Other commands retain their current `runs/<run-id>/` layout.

## Report lookup and compatibility

`rquant report <run-id>` first resolves
`<runs_root>/walk-forward/<run-id>`. If that directory does not exist, it falls
back to the legacy `<runs_root>/<run-id>` path. It does not scan unrelated
directories and never treats a factor-evaluation directory as report input.

Existing run directories are not moved, renamed, modified, or deleted. New
factor-evaluation and walk-forward commands never write to the legacy layout.

## Implementation boundary

The shared CLI run wrapper accepts an optional category and passes the selected
category directory to `RunContext`. Only the factor-evaluation and walk-forward
handlers supply categories. Report path resolution is kept in a small helper so
its new-first, legacy-second behavior can be tested independently.

No configuration keys, run-manifest fields, factor calculations, model logic,
or output filenames change.

## Errors and observability

The CLI JSON response continues to return both `run_id` and the actual
`run_directory`, which now exposes the categorized path. If neither report path
exists, the error identifies the attempted new path and does not create any
directory.

## Tests and documentation

Focused tests verify that:

- factor evaluation routes new runs to `factors-evaluate/<run-id>`;
- walk-forward routes new runs to `walk-forward/<run-id>`;
- other commands remain directly beneath the configured runs root;
- report lookup prefers the categorized path;
- report lookup falls back to an existing legacy path;
- a missing report run still raises the existing data-contract error.

README examples and the project agent guide are updated where they describe the
affected output and report paths. Verification uses the named Conda environment
`rquant`, targeted CLI/run tests, the full fast test suite, and Ruff. It does not
start real factor evaluation, model training, or backtesting.
