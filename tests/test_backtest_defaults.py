from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from rquant.cli import main
from rquant.config import load_config, resolve_paths


@pytest.fixture
def default_config():
    return load_config(Path(__file__).resolve().parents[1] / "config/default.yaml")


def test_default_universe_uses_separate_shanghai_main_store(default_config, tmp_path):
    assert default_config["data"]["universe"] == "sh_main"
    assert resolve_paths(default_config, tmp_path).factors == tmp_path / "data/factors-sh_main"


def invoke_report(monkeypatch, tmp_path, config, horizon, overrides=(), universe="sh_main", expected_code=0):
    paths = resolve_paths(config, tmp_path)
    paths.qlib.mkdir(parents=True)
    (paths.qlib / "manifest.json").write_text(
        json.dumps({"status": "complete", "fingerprint": "test-quotes"}), encoding="utf-8"
    )
    run = paths.runs / "walk-forward" / "test-training"
    run.mkdir(parents=True)
    (run / "predictions.parquet").write_bytes(b"fixture: runner is mocked")
    metadata = {"universe": universe}
    if horizon is not None:
        metadata["horizon"] = horizon
    (run / "walk_forward.json").write_text(json.dumps(metadata), encoding="utf-8")
    captured = []

    class FakeRunner:
        def __init__(self, **kwargs):
            captured.append(kwargs["config"])

        def run(self):
            return {"status": "complete"}

    monkeypatch.setattr("rquant.cli._config", lambda args: (config, paths))
    monkeypatch.setattr("rquant.cli._dependency_versions", lambda: {})
    monkeypatch.setattr("rquant.cli.PortfolioBacktestRunner", FakeRunner)
    monkeypatch.setattr("rquant.cli.build_report", lambda directory: {"status": "complete"})
    assert main(["report", "test-training", *overrides]) == expected_code
    manifests = list((paths.runs / "report").glob("*/run.json"))
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    if expected_code:
        assert manifest["status"] == "failed"
        assert not captured
        return None
    assert manifest["inputs"]["portfolio"] == asdict(captured[0])
    return captured[0]


@pytest.mark.parametrize("horizon,drop,days", [("1d", 5, 1), ("5d", 50, 5)])
def test_report_selects_default_by_training_horizon(monkeypatch, tmp_path, default_config, horizon, drop, days):
    portfolio = invoke_report(monkeypatch, tmp_path, default_config, horizon)
    assert (portfolio.topk, portfolio.n_drop, portfolio.rebalance_days) == (50, drop, days)
    assert portfolio.universe == "sh_main"


def test_cli_overrides_horizon_profile_and_preserves_training_universe(monkeypatch, tmp_path, default_config):
    portfolio = invoke_report(
        monkeypatch, tmp_path, default_config, "5d",
        ("--topk", "30", "--n-drop", "3", "--rebalance-days", "2"), universe="csi300",
    )
    assert (portfolio.topk, portfolio.n_drop, portfolio.rebalance_days) == (30, 3, 2)
    assert portfolio.universe == "csi300"


@pytest.mark.parametrize("horizon", [None, "1d", "5d"])
def test_legacy_config_without_profiles_keeps_explicit_settings(monkeypatch, tmp_path, horizon):
    config = {"data": {"universe": "sh_main"}, "backtest": {"topk": 20, "n_drop": 2, "rebalance_days": 3}}
    portfolio = invoke_report(monkeypatch, tmp_path, config, horizon)
    assert (portfolio.topk, portfolio.n_drop, portfolio.rebalance_days) == (20, 2, 3)


@pytest.mark.parametrize("option", ["--topk", "--n-drop", "--rebalance-days"])
def test_explicit_zero_is_rejected_not_replaced_by_defaults(monkeypatch, tmp_path, default_config, option):
    invoke_report(monkeypatch, tmp_path, default_config, "1d", (option, "0"), expected_code=2)
