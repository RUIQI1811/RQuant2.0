"""Prepare an RD-Agent working directory from an ``rquant rdagent export`` bundle.

Run this inside the RD-Agent conda environment on Linux/WSL, not in the rquant environment::

    python rdagent_setup.py --bundle /mnt/d/.../data/rdagent --workdir ~/rdagent-work

It (1) converts the exported Parquet into the ``daily_pv.h5`` folders RD-Agent reads from its working directory,
(2) patches RD-Agent's Qlib factor templates to RQuant's trading conventions, (3) patches RD-Agent to batch embedding
requests and to screen new factors against RQuant's factor libraries (``rdagent_novelty.py``), (4) writes ``.env``
from the bundle's template when none exists, and (5) checks the Qlib provider RD-Agent backtests against. It only
needs the standard library, pandas and PyTables, all of which RD-Agent installs.

After filling in the API key, check it with ``python rdagent_setup.py --workdir ~/rdagent-work --check-llm``;
``rdagent health_check`` only understands OpenAI and DeepSeek keys and crashes on the default Qwen settings.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

# RD-Agent 1.0.0's label; RQuant's runs from the next open, when it trades, to the open ``label_days`` later.
ORIGINAL_LABEL = '["Ref($close, -2)/Ref($close, -1) - 1"]'
# Factors that predict ten days ahead are found with far less noise than one-day ones (CogAlpha, AlphaForge), and
# RQuant's own walk-forward still judges them on its one-day setup.
DEFAULT_LABEL_DAYS = 10
# (RD-Agent 1.0.0 original, RQuant replacement). Costs are commission 0.03% each side plus 0.1% stamp tax on sells,
# which applied throughout the default mining period (it halved only on 2023-08-28).
TEMPLATE_REPLACEMENTS = (
    ("deal_price: close", "deal_price: open"),
    ("open_cost: 0.0005", "open_cost: 0.0003"),
    ("close_cost: 0.0015", "close_cost: 0.0013"),
)
TEMPLATE_FILES = ("conf_baseline.yaml", "conf_combined_factors.yaml", "conf_combined_factors_sota_model.yaml")
DATA_FOLDERS = {
    "factor_source_data": "git_ignore_folder/factor_implementation_source_data",
    "factor_source_data_debug": "git_ignore_folder/factor_implementation_source_data_debug",
}


def label_expression(days: int) -> str:
    if days < 1:
        raise ValueError("label_days must be positive")
    return f'["Ref($open, -{days + 1})/Ref($open, -1) - 1"]'


def patch_template_text(text: str, threads: int, label_days: int = DEFAULT_LABEL_DAYS) -> str:
    """Apply RQuant's conventions to RD-Agent's pristine template. Fails if the template no longer matches."""
    for original, replacement in ((ORIGINAL_LABEL, label_expression(label_days)), *TEMPLATE_REPLACEMENTS):
        if original in text:
            text = text.replace(original, replacement)
        elif replacement not in text:
            raise ValueError(f"Template contains neither {original!r} nor {replacement!r}")
    lines = text.splitlines(keepends=True)
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("num_threads:"):
            lines[index] = line[: len(line) - len(line.lstrip())] + f"num_threads: {threads}" + _newline(line)
    return "".join(lines)


def patch_templates(template_root: Path, threads: int, label_days: int = DEFAULT_LABEL_DAYS) -> list[str]:
    patched = []
    for name in TEMPLATE_FILES:
        path = template_root / name
        if not path.exists():
            raise FileNotFoundError(f"RD-Agent template is missing: {path}")
        backup = path.with_name(path.name + ".orig")
        if not backup.exists():
            shutil.copyfile(path, backup)
        # Always patch from the pristine copy so a rerun with another thread count does not compound.
        text = patch_template_text(backup.read_text(encoding="utf-8"), threads, label_days)
        path.write_text(text, encoding="utf-8")
        patched.append(str(path))
    return patched


# RD-Agent 1.0.0's LiteLLM backend sends every text in one embedding request and ignores embedding_max_str_num
# (only its deprecated OpenAI backend batches). DashScope rejects more than 10 texts per request, which crashes the
# loop once its knowledge base outgrows 10 entries, around the fifth loop.
EMBEDDING_ORIGINAL = """        response = embedding(
            model=model_name,
            input=input_content_list,
        )
        response_list = [data["embedding"] for data in response.data]
        return response_list
"""
EMBEDDING_BATCHED = """        # RQuant patch: batch by embedding_max_str_num, as RD-Agent's deprecated backend does.
        response_list = []
        step = LITELLM_SETTINGS.embedding_max_str_num
        for start in range(0, len(input_content_list), step):
            response = embedding(
                model=model_name,
                input=input_content_list[start : start + step],
            )
            response_list.extend(data["embedding"] for data in response.data)
        return response_list
"""


def patch_embedding_batches(backend: Path) -> None:
    """Make the LiteLLM backend batch embedding requests. Idempotent; fails if the backend no longer matches."""
    _patch_source(backend, EMBEDDING_ORIGINAL, EMBEDDING_BATCHED)


# RD-Agent 1.0.0 drops a new factor only when its correlation with an accepted one reaches 0.99, never compares with
# known libraries, and then drops every stock-day on which any factor is missing, so one sparse factor silently
# removed 82% of the rows in a run. Qlib's loader joins these factors onto Alpha158 and LightGBM handles gaps, so
# the patch keeps missing values and screens with rquant_novelty.screen_new_factors instead.
DEDUPLICATION_ORIGINAL = """            if SOTA_factor is not None and not SOTA_factor.empty:
                new_factors = self.deduplicate_new_factors(SOTA_factor, new_factors)
                if new_factors.empty:
                    raise FactorEmptyError(
                        "The factors generated in this round are highly similar to the previous factors. Please change the direction for creating new factors."
                    )
                combined_factors = pd.concat([SOTA_factor, new_factors], axis=1).dropna()
            else:
                combined_factors = new_factors
"""  # noqa: E501
DEDUPLICATION_SCREENED = """            # RQuant patch: screen against RQuant's factor libraries and the accepted factors, keep missing values.
            from rdagent.scenarios.qlib.developer.rquant_novelty import screen_new_factors

            codes = {
                task.factor_name: workspace.file_dict.get("factor.py", "")
                for task, workspace in zip(exp.sub_tasks, exp.sub_workspace_list)
                if workspace is not None
            }
            new_factors, rejected = screen_new_factors(new_factors, SOTA_factor, codes=codes)
            if rejected:
                logger.info("RQuant novelty screen rejected: " + "; ".join(rejected))
            if new_factors.empty:
                raise FactorEmptyError(
                    "Every factor in this round was rejected by the novelty screen: "
                    + "; ".join(rejected)
                    + ". Propose factors based on a different economic mechanism, not variants of existing factors."
                )
            if SOTA_factor is not None and not SOTA_factor.empty:
                combined_factors = pd.concat([SOTA_factor, new_factors], axis=1)
            else:
                combined_factors = new_factors
"""  # noqa: E501
NOVELTY_MODULE_NAME = "rquant_novelty.py"


def patch_factor_runner(runner: Path, novelty_source: Path) -> None:
    """Install the novelty screen next to RD-Agent's factor runner and call it there."""
    module = runner.with_name(NOVELTY_MODULE_NAME)
    text = novelty_source.read_text(encoding="utf-8")
    if not module.exists() or module.read_text(encoding="utf-8") != text:
        module.write_text(text, encoding="utf-8")
    _patch_source(runner, DEDUPLICATION_ORIGINAL, DEDUPLICATION_SCREENED)


def _patch_source(path: Path, original: str, replacement: str) -> None:
    """Replace ``original`` in RD-Agent's pristine copy (kept as ``.orig``); writes only when the result changes."""
    backup = path.with_name(path.name + ".orig")
    if not backup.exists():
        shutil.copyfile(path, backup)
    text = backup.read_text(encoding="utf-8")
    if original not in text:
        raise ValueError(f"RD-Agent source no longer matches the RQuant patch: {path}")
    patched = text.replace(original, replacement)
    if path.read_text(encoding="utf-8") != patched:
        path.write_text(patched, encoding="utf-8")


# RD-Agent caches backtest results under <workdir>/pickle_cache keyed only by the factor tasks, not by the templates,
# so after a label change the Alpha158 baseline (which has no tasks) silently came back from the old label's cache.
CACHE_FOLDER = "pickle_cache"
SETUP_RECORD = ".rquant_setup.json"


def retire_stale_cache(workdir: Path, settings: dict[str, object]) -> Path | None:
    """Move RD-Agent's result cache aside when the backtest settings changed since the last setup."""
    record = workdir / SETUP_RECORD
    previous = json.loads(record.read_text(encoding="utf-8")) if record.exists() else None
    cache = workdir / CACHE_FOLDER
    moved = None
    if previous != settings and cache.exists():
        from datetime import datetime, timezone

        # timezone.utc, not datetime.UTC: RD-Agent's env runs Python 3.10.
        moved = cache.with_name(f"{CACHE_FOLDER}.stale-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}")  # noqa: UP017
        cache.rename(moved)
    record.write_text(json.dumps(settings, indent=2, sort_keys=True), encoding="utf-8")
    return moved


def convert_data(bundle: Path, workdir: Path, readme: Path) -> dict[str, int]:
    import pandas as pd

    rows = {}
    for source_name, target_name in DATA_FOLDERS.items():
        frame = pd.read_parquet(bundle / source_name / "daily_pv.parquet")
        target = workdir / target_name
        target.mkdir(parents=True, exist_ok=True)
        frame.to_hdf(target / "daily_pv.h5", key="data")
        shutil.copyfile(readme, target / "README.md")
        rows[target_name] = len(frame)
    return rows


def write_env(source: Path, target: Path) -> None:
    """Copy the .env template with Unix line endings; a CR left by a Windows export would end up in the API key."""
    text = source.read_text(encoding="utf-8").replace("\r\n", "\n")
    target.write_text(text, encoding="utf-8", newline="\n")
    target.chmod(0o600)


def check_qlib(qlib_dir: Path) -> list[str]:
    required = (qlib_dir / "calendars" / "day.txt", qlib_dir / "instruments" / "csi300.txt", qlib_dir / "features")
    problems = [f"missing {path}" for path in required if not path.exists()]
    if (qlib_dir / "features").exists() and not (qlib_dir / "features" / "sh000300").exists():
        problems.append("benchmark SH000300 is missing from features/")
    return problems


def check_llm(workdir: Path) -> int:
    """Send one JSON-mode chat and one embedding request through RD-Agent's own LLM backend, as its loop does."""
    import os

    env_path = workdir / ".env"
    if not env_path.exists():
        print(f"error: {env_path} does not exist; run the setup first", file=sys.stderr)
        return 2
    os.chdir(workdir)
    from dotenv import load_dotenv

    load_dotenv(env_path, override=True)
    # RD-Agent retries a failed call ten times by default; one is enough to tell a bad key from a good one.
    os.environ.setdefault("MAX_RETRY", "1")
    from rdagent.oai.llm_utils import APIBackend

    backend = APIBackend()
    try:
        reply = backend.build_messages_and_create_chat_completion(
            user_prompt='Reply with the JSON object {"ok": true}.',
            system_prompt="You reply with JSON only.",
            json_mode=True,
        )
        vector = backend.create_embedding("hello")
    except Exception as exc:  # noqa: BLE001 - any provider failure means the settings do not work
        print(f"error: LLM check failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"chat OK ({os.environ.get('CHAT_MODEL')}): {reply.strip()[:80]}")
    print(f"embedding OK ({os.environ.get('EMBEDDING_MODEL')}): {len(vector)} dimensions")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bundle", type=Path, help="Directory written by `rquant rdagent export`")
    parser.add_argument("--workdir", required=True, type=Path, help="Directory you will run `rdagent` from")
    parser.add_argument(
        "--qlib-dir",
        type=Path,
        default=Path("~/.qlib/qlib_data/cn_data"),
        help="Copy of RQuant's data/qlib; RD-Agent's templates hard-code this location",
    )
    parser.add_argument("--threads", type=int, default=4, help="LightGBM threads per backtest (default 4)")
    parser.add_argument(
        "--label-days",
        type=int,
        default=DEFAULT_LABEL_DAYS,
        help=f"Return horizon RD-Agent's LightGBM learns, open to open (default {DEFAULT_LABEL_DAYS})",
    )
    parser.add_argument(
        "--check-llm", action="store_true", help="Only test the chat and embedding models configured in <workdir>/.env"
    )
    args = parser.parse_args(argv)

    workdir = args.workdir.expanduser().resolve()
    if args.check_llm:
        return check_llm(workdir)
    if args.bundle is None:
        parser.error("--bundle is required unless --check-llm is given")
    bundle = args.bundle.expanduser().resolve()
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        print(f"error: export bundle is incomplete: {bundle}", file=sys.stderr)
        return 2

    try:
        from rdagent.scenarios.qlib.experiment import utils as qlib_experiment_utils
    except ImportError:
        print("error: run this inside the RD-Agent conda env (pip install rdagent)", file=sys.stderr)
        return 2
    import importlib.metadata
    import importlib.util

    installed = importlib.metadata.version("rdagent")
    if installed != manifest["rdagent_version"]:
        print(
            f"error: rdagent {installed} is installed; the bundle targets {manifest['rdagent_version']}. "
            f"Run: pip install rdagent=={manifest['rdagent_version']}",
            file=sys.stderr,
        )
        return 2

    if importlib.util.find_spec("litellm.llms.dashscope.embed") is None:
        # The default .env embeds through LiteLLM's native DashScope provider, which older LiteLLM lacks.
        print("error: this LiteLLM has no DashScope embedding provider. Run: pip install -U litellm", file=sys.stderr)
        return 2

    experiment_root = Path(qlib_experiment_utils.__file__).parent
    patched = patch_templates(experiment_root / "factor_template", args.threads, args.label_days)
    litellm_backend = experiment_root.parents[2] / "oai" / "backend" / "litellm.py"
    patch_embedding_batches(litellm_backend)
    patched.append(str(litellm_backend))
    factor_runner = experiment_root.parent / "developer" / "factor_runner.py"
    patch_factor_runner(factor_runner, bundle / "rdagent_novelty.py")
    patched.append(str(factor_runner))
    import hashlib

    settings = {
        "label": label_expression(args.label_days),
        "threads": args.threads,
        "templates": [list(pair) for pair in TEMPLATE_REPLACEMENTS],
        "novelty": hashlib.sha256((bundle / "rdagent_novelty.py").read_bytes()).hexdigest(),
        "bundle": manifest.get("fingerprint"),
    }
    retired = retire_stale_cache(workdir, settings)
    if retired is not None:
        print(f"backtest settings changed: moved RD-Agent's result cache to {retired} (delete it when done)")
    reference = bundle / "reference_factors.parquet"
    if reference.exists():
        target = workdir / "git_ignore_folder" / "rquant_reference_factors.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(reference, target)
    # The bundle's README describes RQuant's columns; bundles exported before it existed fall back to RD-Agent's.
    readme = bundle / "README.md"
    if not readme.exists():
        readme = experiment_root / "factor_data_template" / "README.md"
    rows = convert_data(bundle, workdir, readme)

    env_path = workdir / ".env"
    if env_path.exists():
        env_note = f"kept existing {env_path}; compare it with {bundle / 'rdagent.env'}"
    else:
        write_env(bundle / "rdagent.env", env_path)
        env_note = f"wrote {env_path}; fill in DASHSCOPE_API_KEY"

    problems = check_qlib(args.qlib_dir.expanduser())
    print(f"rdagent {installed}: {experiment_root}")
    print("patched templates:\n  " + "\n  ".join(patched))
    for target, count in rows.items():
        print(f"{target}: {count} rows")
    print(env_note)
    if problems:
        print(f"Qlib provider at {args.qlib_dir} is not ready: " + "; ".join(problems), file=sys.stderr)
        return 1
    print(f"Qlib provider OK: {args.qlib_dir}")
    print(f"Next: fill in the API key in {env_path}, then check it with")
    print(f"  python {Path(__file__).resolve()} --workdir {workdir} --check-llm")
    print(f"and start mining with: cd {workdir} && rdagent fin_factor --loop-n 20")
    return 0


def _newline(line: str) -> str:
    return line[len(line.rstrip("\r\n")) :]


if __name__ == "__main__":
    raise SystemExit(main())
