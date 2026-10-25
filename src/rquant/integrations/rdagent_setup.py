"""Prepare an RD-Agent working directory from an ``rquant rdagent export`` bundle.

Run this inside the RD-Agent conda environment on Linux/WSL, not in the rquant environment::

    python rdagent_setup.py --bundle /mnt/d/.../data/rdagent --workdir ~/rdagent-work

It (1) converts the exported Parquet into the ``daily_pv.h5`` folders RD-Agent reads from its working directory,
(2) patches RD-Agent's Qlib factor templates to RQuant's trading conventions, (3) writes ``.env`` from the bundle's
template when none exists, and (4) checks the Qlib provider RD-Agent backtests against. It only needs the
standard library, pandas and PyTables, all of which RD-Agent installs.

After filling in the API key, check it with ``python rdagent_setup.py --workdir ~/rdagent-work --check-llm``;
``rdagent health_check`` only understands OpenAI and DeepSeek keys and crashes on the default Qwen settings.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

# (RD-Agent 1.0.0 original, RQuant replacement). RQuant buys and sells at the next open, so the label runs from the
# next open to the open after it; costs are commission 0.03% each side plus 0.1% stamp tax on sells, which applied
# throughout the default mining period (it halved only on 2023-08-28).
TEMPLATE_REPLACEMENTS = (
    ('["Ref($close, -2)/Ref($close, -1) - 1"]', '["Ref($open, -2)/Ref($open, -1) - 1"]'),
    ("deal_price: close", "deal_price: open"),
    ("open_cost: 0.0005", "open_cost: 0.0003"),
    ("close_cost: 0.0015", "close_cost: 0.0013"),
)
TEMPLATE_FILES = ("conf_baseline.yaml", "conf_combined_factors.yaml", "conf_combined_factors_sota_model.yaml")
DATA_FOLDERS = {
    "factor_source_data": "git_ignore_folder/factor_implementation_source_data",
    "factor_source_data_debug": "git_ignore_folder/factor_implementation_source_data_debug",
}


def patch_template_text(text: str, threads: int) -> str:
    """Apply RQuant's conventions to one template. Idempotent; fails if a template no longer matches."""
    for original, replacement in TEMPLATE_REPLACEMENTS:
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


def patch_templates(template_root: Path, threads: int) -> list[str]:
    patched = []
    for name in TEMPLATE_FILES:
        path = template_root / name
        if not path.exists():
            raise FileNotFoundError(f"RD-Agent template is missing: {path}")
        backup = path.with_name(path.name + ".orig")
        if not backup.exists():
            shutil.copyfile(path, backup)
        # Always patch from the pristine copy so a rerun with another thread count does not compound.
        path.write_text(patch_template_text(backup.read_text(encoding="utf-8"), threads), encoding="utf-8")
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
    backup = backend.with_name(backend.name + ".orig")
    if not backup.exists():
        shutil.copyfile(backend, backup)
    text = backup.read_text(encoding="utf-8")
    if EMBEDDING_ORIGINAL not in text:
        raise ValueError(f"RD-Agent LiteLLM backend no longer matches the embedding patch: {backend}")
    backend.write_text(text.replace(EMBEDDING_ORIGINAL, EMBEDDING_BATCHED), encoding="utf-8")


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
    patched = patch_templates(experiment_root / "factor_template", args.threads)
    litellm_backend = experiment_root.parents[2] / "oai" / "backend" / "litellm.py"
    patch_embedding_batches(litellm_backend)
    patched.append(str(litellm_backend))
    rows = convert_data(bundle, workdir, experiment_root / "factor_data_template" / "README.md")

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
