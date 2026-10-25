from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

from rquant.errors import DataContractError, DependencyError
from rquant.io import atomic_write_json, sha256_file, stable_hash
from rquant.qlib_ext.runtime import qlib_init_kwargs

# RD-Agent ships its Qlib factor templates inside the package, so the setup script patches one exact release.
RDAGENT_VERSION = "1.0.0"
# The columns of RD-Agent's own daily_pv.h5 (factor_data_template/generate.py): adjusted prices and the adjustment.
PV_FIELDS = ("$open", "$close", "$high", "$low", "$volume", "$factor")
SETUP_SCRIPT = Path(__file__).with_name("rdagent_setup.py")


@dataclass(frozen=True)
class MiningWindow:
    """RD-Agent's fixed train/valid/test split, all of it inside the mining period.

    RD-Agent keeps a factor only when its backtest on ``test`` beats the current best, so ``test`` is used for
    selection. Everything after ``test_end`` must stay unseen and is left to RQuant's walk-forward.
    """

    train_start: date
    train_end: date
    valid_start: date
    valid_end: date
    test_start: date
    test_end: date

    def __post_init__(self) -> None:
        ordered = (self.train_start, self.train_end, self.valid_start, self.valid_end, self.test_start, self.test_end)
        if any(left >= right for left, right in zip(ordered, ordered[1:], strict=False)):
            raise ValueError(f"RD-Agent segments must be strictly increasing: {ordered}")

    def env(self) -> dict[str, str]:
        return {f"QLIB_FACTOR_{name.upper()}": value.isoformat() for name, value in asdict(self).items()}


def default_mining_window(formal_start: date, mining_end: date) -> MiningWindow:
    """Test on the last three calendar years, validate on the two before, train on everything earlier."""
    test_start = date(mining_end.year - 2, 1, 1)
    valid_start = date(mining_end.year - 4, 1, 1)
    train_end = date(valid_start.year - 1, 12, 31)
    if train_end.year - formal_start.year + 1 < 3:
        raise ValueError(f"Mining end {mining_end} leaves fewer than three training years after {formal_start}")
    return MiningWindow(
        train_start=formal_start,
        train_end=train_end,
        valid_start=valid_start,
        valid_end=date(test_start.year - 1, 12, 31),
        test_start=test_start,
        test_end=mining_end,
    )


def env_template(window: MiningWindow) -> str:
    lines = [
        "# RD-Agent settings for RQuant. Fill in the API key; never commit this file.",
        "# Qwen on Alibaba Cloud Model Studio (DashScope): one key serves both chat and embedding.",
        "# qwen-plus has a 128k context; qwen-max (30k) is too short for RD-Agent's prompts, qwen3-max also works.",
        "CHAT_MODEL=dashscope/qwen-plus",
        "EMBEDDING_MODEL=dashscope/text-embedding-v4",
        "DASHSCOPE_API_KEY=",
        "# DashScope embeds at most 10 texts per request; rdagent_setup.py patches RD-Agent to batch by this.",
        "EMBEDDING_MAX_STR_NUM=10",
        "",
        "# Alternative: DeepSeek chat with SiliconFlow embedding (comment out the Qwen lines above).",
        "# CHAT_MODEL=deepseek/deepseek-chat",
        "# DEEPSEEK_API_KEY=",
        "# EMBEDDING_MODEL=litellm_proxy/BAAI/bge-m3",
        "# LITELLM_PROXY_API_KEY=",
        "# LITELLM_PROXY_API_BASE=https://api.siliconflow.cn/v1",
        "",
        "# Run Qlib backtests in a local conda env (rdagent4qlib) instead of Docker.",
        "MODEL_CoSTEER_ENV_TYPE=conda",
        "# rdagent4qlib installs MLflow 3.x, which refuses qlib's ./mlruns file store without this.",
        "MLFLOW_ALLOW_FILE_STORE=true",
        "",
        "# Mining split. Everything after QLIB_FACTOR_TEST_END is held out for RQuant walk-forward.",
        *(f"{key}={value}" for key, value in window.env().items()),
        "",
    ]
    return "\n".join(lines)


class RDAgentExporter:
    """Write the inputs RD-Agent's fin_factor loop needs, cut off at the end of the mining period."""

    def __init__(self, qlib_root: str | Path, output_root: str | Path, universe: str = "csi300") -> None:
        self.qlib_root = Path(qlib_root)
        self.output_root = Path(output_root)
        self.universe = universe

    def export(self, window: MiningWindow, *, debug_instruments: int = 100) -> dict[str, Any]:
        if debug_instruments < 1:
            raise ValueError("debug_instruments must be positive")
        qlib_manifest = self.qlib_root / "manifest.json"
        if not qlib_manifest.exists():
            raise DataContractError(f"Qlib manifest does not exist: {qlib_manifest}")
        try:
            import qlib
            from qlib.constant import REG_CN
        except ImportError as exc:
            raise DependencyError("pyqlib==0.9.7 is required to export RD-Agent data") from exc

        qlib.init(provider_uri=str(self.qlib_root), region=REG_CN, **qlib_init_kwargs())
        from qlib.data import D

        calendar = D.calendar(freq="day")
        if len(calendar) == 0 or calendar[-1].date() < window.test_end:
            raise DataContractError(f"Qlib calendar does not reach the mining end {window.test_end}")
        first_day = calendar[0]
        if first_day.date() > window.train_start:
            raise DataContractError(f"Qlib calendar starts {first_day.date()}, after train_start {window.train_start}")

        # Every stock that was a point-in-time member at any time up to the mining end, with its full history so
        # rolling windows have warm-up. Rows after the mining end are never written.
        members = D.list_instruments(
            D.instruments(market=self.universe),
            start_time=first_day,
            end_time=window.test_end.isoformat(),
            freq="day",
            as_list=True,
        )
        if not members:
            raise DataContractError(f"No {self.universe} members before {window.test_end}")
        full = self._features(D, sorted(members), first_day, window.test_end)

        debug_start = date(window.test_end.year - 1, 1, 1)
        recent = D.list_instruments(
            D.instruments(market=self.universe),
            start_time=debug_start.isoformat(),
            end_time=window.test_end.isoformat(),
            freq="day",
            as_list=True,
        )
        debug_members = sorted(recent)[:debug_instruments]
        debug = full[
            (full.index.get_level_values("datetime") >= debug_start.isoformat())
            & full.index.get_level_values("instrument").isin(debug_members)
        ]

        if self.output_root.exists():
            shutil.rmtree(self.output_root)
        self.output_root.mkdir(parents=True)
        full_path = self.output_root / "factor_source_data" / "daily_pv.parquet"
        debug_path = self.output_root / "factor_source_data_debug" / "daily_pv.parquet"
        for frame, path in ((full, full_path), (debug, debug_path)):
            path.parent.mkdir(parents=True)
            frame.to_parquet(path)
        # RD-Agent reads this file on Linux; Windows line endings would leave a CR at the end of every value.
        (self.output_root / "rdagent.env").write_text(env_template(window), encoding="utf-8", newline="\n")
        shutil.copyfile(SETUP_SCRIPT, self.output_root / SETUP_SCRIPT.name)

        with qlib_manifest.open("r", encoding="utf-8") as handle:
            qlib_fingerprint = json.load(handle).get("fingerprint")
        manifest: dict[str, Any] = {
            "status": "complete",
            "rdagent_version": RDAGENT_VERSION,
            "universe": self.universe,
            "window": {name: value.isoformat() for name, value in asdict(window).items()},
            "qlib_fingerprint": qlib_fingerprint,
            "fields": list(PV_FIELDS),
            "full": _summary(full, full_path),
            "debug": _summary(debug, debug_path),
        }
        manifest["fingerprint"] = stable_hash(manifest)
        atomic_write_json(self.output_root / "manifest.json", manifest)
        return manifest

    @staticmethod
    def _features(D: Any, instruments: list[str], start: Any, end: date) -> Any:
        frame = D.features(instruments, list(PV_FIELDS), start_time=start, end_time=end.isoformat(), freq="day")
        # RD-Agent factor code expects <datetime, instrument>, the reverse of D.features.
        frame = frame.swaplevel().sort_index()
        frame.index = frame.index.set_names(["datetime", "instrument"])
        return frame[list(PV_FIELDS)].astype("float32")


def _summary(frame: Any, path: Path) -> dict[str, Any]:
    dates = frame.index.get_level_values("datetime")
    return {
        "rows": int(len(frame)),
        "instruments": int(frame.index.get_level_values("instrument").nunique()),
        "start": str(dates.min().date()) if len(frame) else None,
        "end": str(dates.max().date()) if len(frame) else None,
        "sha256": sha256_file(path),
    }
