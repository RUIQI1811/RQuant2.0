from __future__ import annotations

import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(importlib.util.find_spec("KunQuant") is not None, "KunQuant is required")
class KunQuantSmokeTests(unittest.TestCase):
    def test_locked_graphs_build(self) -> None:
        from rquant.factors.engine import FactorBuildConfig, KunQuantFactorEngine

        with tempfile.TemporaryDirectory() as temporary:
            alpha158, _ = KunQuantFactorEngine(temporary, FactorBuildConfig(factor_set="qlib_alpha158")).build_graph()
            alpha101, _ = KunQuantFactorEngine(temporary, FactorBuildConfig(factor_set="wq_alpha101")).build_graph()
        self.assertGreater(len(alpha158.ops), 158)
        self.assertGreater(len(alpha101.ops), 101)

    @unittest.skipUnless(shutil.which("clang++"), "clang++ is required for KunQuant compilation")
    def test_combined_graph_compiles_and_runs_all_259_outputs(self) -> None:
        import numpy as np

        from rquant.factors.catalog import get_catalog
        from rquant.factors.engine import INPUT_FIELDS, FactorBuildConfig, KunQuantFactorEngine

        # Apple Silicon requires the stock axis to align to two float64 lanes. Use an odd
        # stock count to verify that the engine pads for KunQuant and trims its outputs.
        rows, columns = 320, 13
        rng = np.random.default_rng(42)
        close = 100.0 + np.cumsum(rng.normal(0.0, 0.5, size=(rows, columns)), axis=0)
        volume = rng.uniform(1_000.0, 10_000.0, size=(rows, columns))
        groups = np.broadcast_to(np.arange(columns) % 3, (rows, columns)).astype(np.float64)
        inputs = {
            "open": close + rng.normal(0.0, 0.2, size=(rows, columns)),
            "high": close + rng.uniform(0.2, 1.0, size=(rows, columns)),
            "low": close - rng.uniform(0.2, 1.0, size=(rows, columns)),
            "close": close,
            "volume": volume,
            "amount": close * volume,
            "vwap": close + rng.normal(0.0, 0.1, size=(rows, columns)),
            "cap": close * rng.uniform(1e7, 2e7, size=(rows, columns)),
            "sector": groups,
            "industry": groups,
            "subindustry": groups,
        }
        self.assertEqual(set(INPUT_FIELDS), set(inputs))
        with tempfile.TemporaryDirectory() as temporary:
            engine = KunQuantFactorEngine(temporary, FactorBuildConfig(factor_set="combined", workers=2))
            module = engine.compile()
            output = engine.run(inputs, module=module)
            padded_columns = columns + (-columns) % int(module.blocking_len)
            output_store = np.memmap(
                Path(temporary) / "outputs.dat",
                mode="w+",
                dtype=np.float64,
                shape=(259, rows, padded_columns),
            )
            output_buffers = {
                name: output_store[index] for index, name in enumerate(get_catalog().canonical_names("combined"))
            }
            mapped_output = engine.run(
                inputs,
                module=module,
                output_buffers=output_buffers,
                trim_outputs=False,
            )
            for name in output:
                np.testing.assert_allclose(
                    mapped_output[name][:, :columns], output[name], rtol=0.0, atol=0.0, equal_nan=True
                )
            output_buffers.clear()
            mapped_output.clear()
            output_store._mmap.close()
        expected = get_catalog().canonical_names("combined")
        self.assertEqual(expected, tuple(output))
        self.assertEqual((rows, columns), output["a158_001"].shape)
        self.assertEqual((rows, columns), output["a101_101"].shape)
        self.assertGreater(np.isfinite(output["a158_001"]).sum(), 0)
        self.assertGreater(np.isfinite(output["a101_101"]).sum(), 0)
        for name in ("a101_068", "a101_086"):
            finite = output[name][np.isfinite(output[name])]
            self.assertGreater(np.unique(finite).size, 1, f"{name} must not collapse to a constant")


if __name__ == "__main__":
    unittest.main()
