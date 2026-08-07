from __future__ import annotations

import importlib.util
import inspect
import shutil
import tempfile
import unittest

from rquant.errors import FactorContractError
from rquant.factors.catalog import catalog_from_registry
from rquant.factors.engine import FactorBuildConfig, KunQuantFactorEngine
from rquant.factors.libraries.custom import CustomFactor, CustomFactorLibrary
from rquant.factors.libraries.registry import FactorLibraryRegistry, get_library_registry


def price_spread(inputs):
    return inputs["close"] - inputs["open"]


def price_range(inputs):
    return inputs["high"] - inputs["low"]


def custom_price_library() -> CustomFactorLibrary:
    return CustomFactorLibrary(
        family="custom_price",
        required_inputs=("open", "high", "low", "close"),
        factors=(
            CustomFactor("custom_price_001", "price_spread", "close - open", 1, price_spread),
            CustomFactor("custom_price_002", "price_range", "high - low", 1, price_range),
        ),
    )


class FactorLibraryRegistryTests(unittest.TestCase):
    def test_default_registry_keeps_locked_factor_sets_and_input_contracts(self) -> None:
        registry = get_library_registry()

        self.assertEqual(("qlib_alpha158", "wq_alpha101", "gtja191", "combined"), registry.factor_sets())
        self.assertEqual(
            ("open", "high", "low", "close", "volume", "amount", "vwap"),
            registry.required_inputs("qlib_alpha158"),
        )
        self.assertEqual(11, len(registry.required_inputs("wq_alpha101")))
        self.assertEqual(
            ("open", "high", "low", "close", "volume", "amount", "vwap"),
            registry.required_inputs("gtja191"),
        )

    def test_alpha_formulas_are_complete_and_repository_owned(self) -> None:
        from rquant.factors.extensions import kunquant as kunquant_extensions
        from rquant.factors.libraries import alpha101, alpha158

        builtin = set(alpha101._BUILTIN_ALPHA101)
        extended = {f"alpha{ordinal:03d}" for ordinal in alpha101.ALPHA101_UPSTREAM_MISSING}
        self.assertFalse(builtin & extended)
        self.assertEqual({f"alpha{ordinal:03d}" for ordinal in range(1, 102)}, builtin | extended)
        self.assertEqual(158, len(alpha158.ALPHA158_SOURCE_NAMES))

        for module in (alpha101, alpha158):
            source = inspect.getsource(module)
            self.assertNotIn("from KunQuant.predefined", source)
            self.assertNotIn("import KunQuant.predefined", source)

        self.assertIsNone(importlib.util.find_spec("rquant.factors.operators"))
        extension_source = inspect.getsource(kunquant_extensions)
        self.assertIn("class GroupNeutralize", extension_source)
        self.assertNotIn("class OpBase", extension_source)

        registry = get_library_registry()
        for factor_set in ("qlib_alpha158", "wq_alpha101"):
            for library in registry.select(factor_set):
                for spec in library.specs:
                    self.assertTrue(
                        spec.implementation.startswith("rquant.factors.libraries."),
                        spec.implementation,
                    )

    def test_custom_library_registers_without_changing_engine_code(self) -> None:
        registry = FactorLibraryRegistry()
        registry.register_library(custom_price_library())
        catalog = catalog_from_registry(registry)

        self.assertEqual(("custom_price",), registry.factor_sets())
        self.assertEqual(("custom_price_001", "custom_price_002"), catalog.canonical_names("custom_price"))
        with tempfile.TemporaryDirectory() as temporary:
            engine = KunQuantFactorEngine(
                temporary,
                FactorBuildConfig(factor_set="custom_price"),
                registry=registry,
            )
            graph, _ = engine.build_graph()

        self.assertEqual(("open", "high", "low", "close"), engine.input_fields)
        self.assertGreater(len(graph.ops), 2)

    def test_registry_rejects_unknown_inputs_and_duplicate_libraries(self) -> None:
        registry = FactorLibraryRegistry()
        library = custom_price_library()
        registry.register_library(library)

        with self.assertRaisesRegex(FactorContractError, "already registered"):
            registry.register_library(library)
        invalid = CustomFactorLibrary(
            family="invalid",
            required_inputs=("future_secret",),
            factors=(CustomFactor("invalid_001", "invalid", "invalid", 1, price_spread),),
        )
        with self.assertRaisesRegex(FactorContractError, "unknown inputs"):
            registry.register_library(invalid)

    @unittest.skipUnless(
        importlib.util.find_spec("KunQuant") is not None and shutil.which("clang++"),
        "KunQuant and clang++ are required",
    )
    def test_custom_library_compiles_and_runs_through_generic_engine(self) -> None:
        import numpy as np

        registry = FactorLibraryRegistry()
        registry.register_library(custom_price_library())
        inputs = {
            "open": np.array([[10.0, 20.0, 30.0], [11.0, 19.0, 32.0]]),
            "high": np.array([[13.0, 24.0, 35.0], [14.0, 23.0, 36.0]]),
            "low": np.array([[8.0, 18.0, 27.0], [9.0, 17.0, 28.0]]),
            "close": np.array([[12.0, 21.0, 33.0], [13.0, 22.0, 31.0]]),
        }
        with tempfile.TemporaryDirectory() as temporary:
            engine = KunQuantFactorEngine(
                temporary,
                FactorBuildConfig(factor_set="custom_price", workers=1),
                registry=registry,
            )
            output = engine.run(inputs)

        self.assertEqual(("custom_price_001", "custom_price_002"), tuple(output))
        np.testing.assert_allclose(output["custom_price_001"], inputs["close"] - inputs["open"])
        np.testing.assert_allclose(output["custom_price_002"], inputs["high"] - inputs["low"])


if __name__ == "__main__":
    unittest.main()
