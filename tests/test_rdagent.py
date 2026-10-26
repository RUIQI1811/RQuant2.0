from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

from rquant.integrations.rdagent import (
    PV_FIELDS,
    MiningWindow,
    RDAgentExporter,
    default_mining_window,
    env_template,
)
from rquant.integrations.rdagent_setup import (
    DEDUPLICATION_ORIGINAL,
    EMBEDDING_ORIGINAL,
    check_qlib,
    convert_data,
    label_expression,
    patch_embedding_batches,
    patch_factor_runner,
    patch_template_text,
    patch_templates,
    write_env,
)

HAS_QLIB = all(importlib.util.find_spec(name) is not None for name in ("pandas", "pyarrow", "qlib"))
HAS_TABLES = HAS_QLIB and importlib.util.find_spec("tables") is not None

# The lines of RD-Agent 1.0.0's factor templates that the setup script rewrites.
TEMPLATE_EXCERPT = """\
                        label:
                            - ["Ref($close, -2)/Ref($close, -1) - 1"]
                            - ["LABEL0"]
        exchange_kwargs:
            limit_threshold: 0.095
            deal_price: close
            open_cost: 0.0005
            close_cost: 0.0015
            min_cost: 5
            num_threads: 20
"""


class MiningWindowTests(unittest.TestCase):
    def test_default_window_tests_last_three_years_and_validates_two(self) -> None:
        window = default_mining_window(date(2010, 1, 1), date(2021, 12, 31))
        self.assertEqual(
            MiningWindow(
                train_start=date(2010, 1, 1),
                train_end=date(2016, 12, 31),
                valid_start=date(2017, 1, 1),
                valid_end=date(2018, 12, 31),
                test_start=date(2019, 1, 1),
                test_end=date(2021, 12, 31),
            ),
            window,
        )
        self.assertEqual("2021-12-31", window.env()["QLIB_FACTOR_TEST_END"])

    def test_default_window_needs_three_training_years(self) -> None:
        with self.assertRaises(ValueError):
            default_mining_window(date(2010, 1, 1), date(2016, 12, 31))

    def test_segments_must_increase(self) -> None:
        with self.assertRaises(ValueError):
            MiningWindow(*(date(2020, 1, day) for day in (1, 2, 3, 3, 4, 5)))

    def test_env_template_has_split_and_blank_keys(self) -> None:
        text = env_template(default_mining_window(date(2010, 1, 1), date(2021, 12, 31)))
        lines = text.splitlines()
        self.assertIn("QLIB_FACTOR_TEST_START=2019-01-01", lines)
        self.assertIn("MODEL_CoSTEER_ENV_TYPE=conda", lines)
        self.assertIn("MLFLOW_ALLOW_FILE_STORE=true", lines)
        self.assertIn("CHAT_MODEL=dashscope/qwen-plus", lines)
        self.assertIn("EMBEDDING_MODEL=dashscope/text-embedding-v4", lines)
        self.assertIn("DASHSCOPE_API_KEY=", lines)
        self.assertIn("EMBEDDING_MAX_STR_NUM=10", lines)


class TemplatePatchTests(unittest.TestCase):
    def test_patch_switches_to_next_open_conventions(self) -> None:
        patched = patch_template_text(TEMPLATE_EXCERPT, threads=4)
        self.assertIn('["Ref($open, -11)/Ref($open, -1) - 1"]', patched)
        self.assertIn("deal_price: open", patched)
        self.assertIn("open_cost: 0.0003", patched)
        self.assertIn("close_cost: 0.0013", patched)
        self.assertIn("            num_threads: 4\n", patched)
        self.assertNotIn("$close", patched)
        self.assertEqual(patched, patch_template_text(patched, threads=4))

    def test_patch_rejects_unrecognized_template(self) -> None:
        with self.assertRaises(ValueError):
            patch_template_text(TEMPLATE_EXCERPT.replace("deal_price: close", "deal_price: vwap"), threads=4)

    def test_rerun_patches_from_pristine_backup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("conf_baseline.yaml", "conf_combined_factors.yaml", "conf_combined_factors_sota_model.yaml"):
                (root / name).write_text(TEMPLATE_EXCERPT, encoding="utf-8")
            patch_templates(root, threads=4)
            patch_templates(root, threads=8)
            text = (root / "conf_baseline.yaml").read_text(encoding="utf-8")
            self.assertIn("num_threads: 8", text)
            self.assertEqual(TEMPLATE_EXCERPT, (root / "conf_baseline.yaml.orig").read_text(encoding="utf-8"))


class EmbeddingPatchTests(unittest.TestCase):
    def test_embedding_requests_are_batched_and_patch_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            backend = Path(temporary) / "litellm.py"
            original = "def _create_embedding_inner_function(self, input_content_list):\n" + EMBEDDING_ORIGINAL
            backend.write_text(original, encoding="utf-8")
            patch_embedding_batches(backend)
            patch_embedding_batches(backend)
            text = backend.read_text(encoding="utf-8")
            self.assertEqual(1, text.count("for start in range(0, len(input_content_list), step):"))
            self.assertIn("input=input_content_list[start : start + step],", text)
            self.assertEqual(original, (Path(temporary) / "litellm.py.orig").read_text(encoding="utf-8"))

    def test_unrecognized_backend_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            backend = Path(temporary) / "litellm.py"
            backend.write_text("def something_else():\n    pass\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                patch_embedding_batches(backend)


class FactorRunnerPatchTests(unittest.TestCase):
    def test_runner_calls_the_novelty_screen_and_keeps_missing_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runner = Path(temporary) / "factor_runner.py"
            runner.write_text("def develop(self, exp):\n" + DEDUPLICATION_ORIGINAL, encoding="utf-8")
            source = Path(temporary) / "rdagent_novelty.py"
            source.write_text("def screen_new_factors(new, accepted=None):\n    return new, []\n", encoding="utf-8")
            patch_factor_runner(runner, source)
            patch_factor_runner(runner, source)
            text = runner.read_text(encoding="utf-8")
            self.assertEqual(1, text.count("screen_new_factors(new_factors, SOTA_factor)"))
            self.assertNotIn(".dropna()", text)
            self.assertEqual(source.read_text(encoding="utf-8"), (Path(temporary) / "rquant_novelty.py").read_text())

    def test_label_runs_open_to_open_over_the_horizon(self) -> None:
        self.assertEqual('["Ref($open, -11)/Ref($open, -1) - 1"]', label_expression(10))
        self.assertEqual('["Ref($open, -2)/Ref($open, -1) - 1"]', label_expression(1))
        self.assertIn(label_expression(5), patch_template_text(TEMPLATE_EXCERPT, threads=4, label_days=5))


@unittest.skipUnless(HAS_QLIB, "pandas is required")
class NoveltyScreenTests(unittest.TestCase):
    def setUp(self) -> None:
        import numpy as np
        import pandas as pd

        self.pd = pd
        rng = np.random.default_rng(0)
        dates = pd.bdate_range("2020-01-01", periods=30)
        instruments = [f"SH6000{number:02d}" for number in range(40)]
        self.index = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])
        size = len(self.index)
        self.base = rng.normal(size=size)
        self.reference = pd.DataFrame({"qlib_alpha158.STD5": self.base}, index=self.index)
        sparse = rng.normal(size=size)
        sparse[: size // 2] = np.nan
        self.new = pd.DataFrame(
            {
                "near_copy": -self.base + rng.normal(scale=0.1, size=size),
                "fresh": rng.normal(size=size),
                "sparse": sparse,
            },
            index=self.index,
        )

    def test_rejects_copies_of_reference_factors_and_sparse_factors(self) -> None:
        from rquant.integrations.rdagent_novelty import screen_new_factors

        kept, rejected = screen_new_factors(self.new, None, self.reference)
        self.assertEqual(["fresh"], list(kept.columns))
        self.assertEqual(2, len(rejected))
        self.assertIn("qlib_alpha158.STD5", rejected[0])
        self.assertTrue(rejected[0].startswith("near_copy: rank correlation -0.9"))
        self.assertIn("50%", rejected[1])

    def test_compares_with_accepted_factors_without_reference(self) -> None:
        from rquant.integrations.rdagent_novelty import screen_new_factors

        accepted = self.pd.DataFrame({"momentum_10d": self.base}, index=self.index)
        kept, rejected = screen_new_factors(self.new[["near_copy", "fresh"]], accepted, None)
        self.assertEqual(["fresh"], list(kept.columns))
        self.assertIn("accepted.momentum_10d", rejected[0])

    def test_rank_correlation_ignores_level_and_scale(self) -> None:
        from rquant.integrations.rdagent_novelty import rank_correlation

        left = self.pd.DataFrame({"a": self.base}, index=self.index)
        right = self.pd.DataFrame({"b": self.base**3 * 10 + 5}, index=self.index)
        self.assertAlmostEqual(1.0, float(rank_correlation(left, right).loc["a", "b"]), places=6)


class WriteEnvTests(unittest.TestCase):
    def test_windows_line_endings_do_not_reach_the_key(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "rdagent.env"
            source.write_bytes(b"CHAT_MODEL=dashscope/qwen-plus\r\nDASHSCOPE_API_KEY=\r\n")
            target = Path(temporary) / ".env"
            write_env(source, target)
            self.assertEqual(b"CHAT_MODEL=dashscope/qwen-plus\nDASHSCOPE_API_KEY=\n", target.read_bytes())
            self.assertEqual(0o600, target.stat().st_mode & 0o777)


@unittest.skipUnless(HAS_QLIB, "pandas, pyarrow and pyqlib are required")
class ExportTests(unittest.TestCase):
    def setUp(self) -> None:
        import numpy as np
        import pandas as pd

        self.pd = pd
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.qlib = self.root / "qlib"
        calendar = pd.bdate_range("2020-01-01", "2020-03-31")
        (self.qlib / "calendars").mkdir(parents=True)
        (self.qlib / "calendars" / "day.txt").write_text(
            "".join(f"{day:%Y-%m-%d}\n" for day in calendar), encoding="utf-8"
        )
        # SH600000 is a member throughout, SZ000001 leaves in February, SZ000002 joins after the mining end.
        members = {
            "SH600000": ("2020-01-01", "2020-03-31"),
            "SZ000001": ("2020-01-01", "2020-02-14"),
            "SZ000002": ("2020-03-16", "2020-03-31"),
        }
        (self.qlib / "instruments").mkdir()
        (self.qlib / "instruments" / "csi300.txt").write_text(
            "".join(f"{name}\t{start}\t{end}\n" for name, (start, end) in members.items()), encoding="utf-8"
        )
        for offset, name in enumerate(members):
            directory = self.qlib / "features" / name.lower()
            directory.mkdir(parents=True)
            for field in ("open", "close", "high", "low", "volume", "amount", "vwap", "factor"):
                values = np.arange(len(calendar), dtype=np.float32) + offset * 100
                np.concatenate((np.asarray([0], dtype=np.float32), values)).tofile(directory / f"{field}.day.bin")
        (self.qlib / "manifest.json").write_text(json.dumps({"fingerprint": "qlib-v1"}), encoding="utf-8")
        self.factors = self.root / "factors"
        store = self.factors / "qlib_alpha158"
        (store / "year=2020").mkdir(parents=True)
        rows = pd.DataFrame(
            [(day, name) for day in calendar for name in ("SH600000", "SZ000001")], columns=["datetime", "instrument"]
        )
        rows["KMID"] = np.arange(len(rows), dtype=np.float32)
        rows.to_parquet(store / "year=2020" / "factors.parquet", index=False)
        (store / "manifest.json").write_text(
            json.dumps({"status": "complete", "fingerprint": "alpha158-v1"}), encoding="utf-8"
        )
        self.window = MiningWindow(
            train_start=date(2020, 1, 1),
            train_end=date(2020, 1, 31),
            valid_start=date(2020, 2, 3),
            valid_end=date(2020, 2, 14),
            test_start=date(2020, 2, 17),
            test_end=date(2020, 3, 13),
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_export_cuts_at_mining_end_and_keeps_point_in_time_members(self) -> None:
        output = self.root / "bundle"
        exporter = RDAgentExporter(self.qlib, output, factor_root=self.factors, reference_sets=("qlib_alpha158",))
        manifest = exporter.export(self.window, debug_instruments=1, reference_dates=10)

        frame = self.pd.read_parquet(output / "factor_source_data" / "daily_pv.parquet")
        self.assertEqual(["datetime", "instrument"], list(frame.index.names))
        self.assertEqual(list(PV_FIELDS), list(frame.columns))
        self.assertEqual(["SH600000", "SZ000001"], sorted(frame.index.get_level_values("instrument").unique()))
        self.assertEqual(self.pd.Timestamp("2020-03-13"), frame.index.get_level_values("datetime").max())
        # A member's history continues after it leaves the index so its factors stay computable.
        self.assertIn((self.pd.Timestamp("2020-03-02"), "SZ000001"), frame.index)
        self.assertEqual("float32", str(frame.dtypes.iloc[0]))

        debug = self.pd.read_parquet(output / "factor_source_data_debug" / "daily_pv.parquet")
        self.assertEqual(["SH600000"], list(debug.index.get_level_values("instrument").unique()))

        self.assertEqual("complete", manifest["status"])
        self.assertEqual("qlib-v1", manifest["qlib_fingerprint"])
        self.assertEqual(2, manifest["full"]["instruments"])
        self.assertEqual("2020-03-13", manifest["full"]["end"])
        self.assertTrue((output / "rdagent_setup.py").exists())
        self.assertNotIn("$factor", list(frame.columns))
        readme = (output / "README.md").read_text(encoding="utf-8")
        for field in PV_FIELDS:
            self.assertIn(f"{field}:", readme)
        self.assertNotIn(b"\r", (output / "rdagent.env").read_bytes())
        env_lines = (output / "rdagent.env").read_text(encoding="utf-8").splitlines()
        self.assertIn("QLIB_FACTOR_TEST_END=2020-03-13", env_lines)
        self.assertEqual(manifest, json.loads((output / "manifest.json").read_text(encoding="utf-8")))
        reference = self.pd.read_parquet(output / "reference_factors.parquet")
        self.assertEqual(["qlib_alpha158.KMID"], list(reference.columns))
        reference_days = reference.index.get_level_values("datetime").unique()
        self.assertTrue(10 <= len(reference_days) <= 20)
        self.assertLessEqual(reference_days.max(), self.pd.Timestamp("2020-03-13"))
        self.assertEqual({"fingerprint": "alpha158-v1", "factors": 1}, manifest["reference"]["sets"]["qlib_alpha158"])
        self.assertIn("compared with 1 existing factors (Qlib Alpha158)", readme)
        self.assertTrue((output / "rdagent_novelty.py").exists())

    def test_export_rejects_mining_end_beyond_data(self) -> None:
        window = MiningWindow(
            train_start=date(2020, 1, 1),
            train_end=date(2020, 1, 31),
            valid_start=date(2020, 2, 3),
            valid_end=date(2020, 2, 14),
            test_start=date(2020, 2, 17),
            test_end=date(2020, 6, 30),
        )
        from rquant.errors import DataContractError

        with self.assertRaises(DataContractError):
            RDAgentExporter(self.qlib, self.root / "bundle").export(window)

    @unittest.skipUnless(HAS_TABLES, "PyTables is required to write HDF5")
    def test_setup_converts_bundle_to_rdagent_hdf5(self) -> None:
        output = self.root / "bundle"
        RDAgentExporter(self.qlib, output).export(self.window, debug_instruments=1)
        workdir = self.root / "work"

        rows = convert_data(output, workdir, output / "README.md")

        target = workdir / "git_ignore_folder" / "factor_implementation_source_data"
        restored = self.pd.read_hdf(target / "daily_pv.h5", key="data")
        self.assertEqual(rows["git_ignore_folder/factor_implementation_source_data"], len(restored))
        self.assertIn("$amount:", (target / "README.md").read_text(encoding="utf-8"))
        self.assertTrue(
            (workdir / "git_ignore_folder" / "factor_implementation_source_data_debug" / "daily_pv.h5").exists()
        )
        self.assertEqual(["benchmark SH000300 is missing from features/"], check_qlib(self.qlib))


if __name__ == "__main__":
    unittest.main()
