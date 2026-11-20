import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analysis" / "blend_runs.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("blend_runs", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _predictions(scores: dict[tuple[str, str], float]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"datetime": pd.Timestamp(day), "instrument": code, "score": score} for (day, code), score in scores.items()]
    )


class BlendRunsTests(unittest.TestCase):
    # Break caught: a blend dominated by the run with the widest score scale, or one that keeps stocks only some runs
    # scored.
    def test_blend_averages_daily_zscores_on_the_shared_rows(self) -> None:
        blend = _load_script().blend
        first = _predictions(
            {("2024-01-02", "A"): 1.0, ("2024-01-02", "B"): 2.0, ("2024-01-02", "C"): 3.0, ("2024-01-03", "A"): 5.0,
             ("2024-01-03", "B"): 5.0}
        )
        # Ten times the scale and the opposite order on the first day; D is scored by this run only.
        second = _predictions(
            {("2024-01-02", "A"): 30.0, ("2024-01-02", "B"): 20.0, ("2024-01-02", "C"): 10.0, ("2024-01-02", "D"): 99.0,
             ("2024-01-03", "A"): 1.0, ("2024-01-03", "B"): 3.0}
        )
        result = blend([first, second]).set_index(["datetime", "instrument"])["score"]

        self.assertEqual(5, len(result))
        self.assertNotIn("D", result.index.get_level_values("instrument"))
        # Day one: the two runs' z-scores cancel exactly.
        np.testing.assert_allclose(result.loc[pd.Timestamp("2024-01-02")].to_numpy(), 0.0, atol=1e-12)
        # Day two: the first run scores A and B alike (z = 0); the second run's z-scores are -/+ 1/sqrt(2).
        day_two = result.loc[pd.Timestamp("2024-01-03")]
        self.assertAlmostEqual(-0.5 / np.sqrt(2), day_two["A"])
        self.assertAlmostEqual(0.5 / np.sqrt(2), day_two["B"])

    def test_main_writes_a_run_that_report_can_read(self) -> None:
        module = _load_script()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for seed, sign in ((42, 1.0), (43, 2.0)):
                run = root / "walk-forward" / f"seed{seed}"
                run.mkdir(parents=True)
                _predictions({("2024-01-02", "A"): sign, ("2024-01-02", "B"): -sign}).to_parquet(
                    run / "predictions.parquet", index=False
                )
                manifest = {"horizon": "1d", "factor_set": "jq240", "universe": "sh_main", "seed": seed, "windows": []}
                (run / "walk_forward.json").write_text(json.dumps(manifest), encoding="utf-8")
            module.main([str(root), "blend", "seed42", "seed43"])

            output = root / "walk-forward" / "blend"
            predictions = pd.read_parquet(output / "predictions.parquet")
            self.assertEqual(["datetime", "instrument", "score"], list(predictions.columns))
            self.assertGreater(predictions.set_index("instrument").loc["A", "score"], 0.0)
            manifest = json.loads((output / "walk_forward.json").read_text(encoding="utf-8"))
            self.assertEqual(("1d", "sh_main"), (manifest["horizon"], manifest["universe"]))
            self.assertEqual([42, 43], manifest["blend"]["seeds"])
            self.assertNotIn("windows", manifest)
            with self.assertRaises(SystemExit):
                module.main([str(root), "blend", "seed42", "seed43"])  # never overwrites a run


if __name__ == "__main__":
    unittest.main()
