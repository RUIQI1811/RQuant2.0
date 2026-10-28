from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from rquant.errors import DataContractError, FactorContractError
from rquant.factors.catalog import catalog_from_registry, get_catalog
from rquant.factors.engine import FactorBuildConfig, SelectionFactorEngine, create_factor_engine
from rquant.factors.libraries.base import FactorSelection, FactorSpec, PanelFactorLibrary, SelectedFactor
from rquant.factors.libraries.custom import CustomFactor, CustomFactorLibrary
from rquant.factors.libraries.hquality import HQUALITY_SELECTION
from rquant.factors.libraries.recent import RECENT_SELECTION
from rquant.factors.libraries.recent_sh_main import RECENT_SH_MAIN_SELECTION
from rquant.factors.libraries.registry import FactorLibraryRegistry, get_library_registry
from rquant.factors.selection import (
    StabilityCriteria,
    factor_coverage,
    mean_rank_correlation,
    prune_correlated,
    score_factor_stability,
)
from rquant.io import atomic_write_json


def _panel_library(family: str, prefix: str, transforms: dict[str, object]) -> PanelFactorLibrary:
    class Library(PanelFactorLibrary):
        required_inputs = ("close",)

        @property
        def specs(self) -> tuple[FactorSpec, ...]:
            return tuple(
                FactorSpec(
                    canonical_name=f"{prefix}_{ordinal:03d}",
                    source_name=source,
                    family=family,
                    ordinal=ordinal,
                    formula=source,
                    max_lookback=1,
                    implementation="test",
                    catalog_version=1,
                )
                for ordinal, source in enumerate(transforms, 1)
            )

        def calculate(self, inputs):  # type: ignore[no-untyped-def]
            return {source: transform(inputs["close"]) for source, transform in transforms.items()}

    Library.family = family
    return Library()


def _selection(members: tuple[SelectedFactor, ...], sources: tuple[str, ...] = ("left", "right")) -> FactorSelection:
    return FactorSelection(
        name="picked", source_factor_sets=sources, members=members, criteria={"rule": "test"}, provenance={}
    )


def _registry(selection: FactorSelection | None = None) -> FactorLibraryRegistry:
    registry = FactorLibraryRegistry()
    registry.register_library(_panel_library("left", "lf", {"PLUS": lambda c: c + 1.0, "TWICE": lambda c: c * 2.0}))
    registry.register_library(_panel_library("right", "rt", {"NEG": lambda c: -c}))
    if selection is not None:
        registry.register_factor_selection(selection)
    return registry


def _write_canonical(root: Path) -> Path:
    dates = pd.bdate_range("2020-12-28", periods=10)
    bars = pd.DataFrame(
        [
            {"datetime": date, "instrument": instrument, "close": float(index + offset), "in_csi300": True}
            for index, date in enumerate(dates)
            for instrument, offset in (("SH600000", 1), ("SZ000001", 100))
        ]
    )
    canonical = root / "canonical"
    (canonical / "bars").mkdir(parents=True)
    (canonical / "reference").mkdir(parents=True)
    bars.to_parquet(canonical / "bars" / "part.parquet", index=False)
    atomic_write_json(canonical / "manifest.json", {"status": "complete", "fingerprint": "canonical-test"})
    pd.DataFrame({"ts_code": "000300.SH", "datetime": dates, "open": 1.0, "close": 1.0}).to_parquet(
        canonical / "reference" / "index_daily.parquet", index=False
    )
    return canonical


class FactorSelectionRegistryTests(unittest.TestCase):
    # Break caught: a selection that silently publishes whole families instead of the chosen members in order.
    def test_selection_catalog_lists_members_in_declared_order(self) -> None:
        selection = _selection((SelectedFactor("rt_001", -1), SelectedFactor("lf_002", 1)))
        registry = _registry(selection)
        catalog = catalog_from_registry(registry)

        self.assertEqual(("rt_001", "lf_002"), catalog.canonical_names("picked"))
        self.assertEqual(("right", "left"), tuple(library.family for library in registry.select("picked")))
        self.assertEqual("left", registry.source_factor_set(selection, "lf_002"))
        self.assertIsInstance(create_factor_engine(".", FactorBuildConfig(factor_set="picked"), registry=registry),
                              SelectionFactorEngine)

    def test_selection_rejects_unknown_duplicate_or_unsigned_members(self) -> None:
        for members in (
            (SelectedFactor("missing_001", 1),),
            (SelectedFactor("lf_001", 1), SelectedFactor("lf_001", -1)),
            (SelectedFactor("lf_001", 0),),
        ):
            with self.subTest(members=members), self.assertRaises(FactorContractError):
                _registry(_selection(members))
        with self.assertRaises(FactorContractError):
            _registry(_selection((SelectedFactor("rt_001", 1),), sources=("left",)))

    # Break caught: the engine refusing a selection only because its members come from different backends.
    def test_selection_may_mix_kunquant_and_panel_libraries(self) -> None:
        registry = _registry()
        registry.register_library(
            CustomFactorLibrary(
                family="custom_price",
                required_inputs=("open", "close"),
                factors=(CustomFactor("cp_001", "spread", "close - open", 1, lambda inputs: inputs["close"]),),
            )
        )
        registry.register_factor_selection(
            _selection((SelectedFactor("cp_001", 1), SelectedFactor("lf_001", 1)), sources=("custom_price", "left"))
        )
        self.assertEqual({"kunquant", "panel"}, {library.backend for library in registry.select("picked")})
        self.assertEqual(("open", "close"), registry.required_inputs("picked"))

    def test_default_registry_publishes_hquality_from_built_source_sets(self) -> None:
        registry = get_library_registry()
        catalog = get_catalog()
        selection = registry.selection("Hquality")

        self.assertIs(HQUALITY_SELECTION, selection)
        self.assertEqual(HQUALITY_SELECTION.canonical_names, catalog.canonical_names("Hquality"))
        self.assertEqual(1696, len(catalog.specs))
        for name in selection.canonical_names:
            self.assertIn(registry.source_factor_set(selection, name), ("qlib_alpha158", "wq_alpha101", "gtja191"))
        self.assertNotIn("gtja_030", selection.canonical_names)

    # Break caught: a recent member whose only source is a factor set the selection does not declare.
    def test_default_registry_publishes_recent_from_all_library_factor_sets(self) -> None:
        registry = get_library_registry()
        catalog = get_catalog()
        selection = registry.selection("recent")

        self.assertIs(RECENT_SELECTION, selection)
        self.assertEqual(RECENT_SELECTION.canonical_names, catalog.canonical_names("recent"))
        self.assertEqual(117, len(selection.canonical_names))
        sources = {registry.source_factor_set(selection, name) for name in selection.canonical_names}
        self.assertEqual(set(selection.source_factor_sets), sources)
        self.assertEqual({"kunquant"}, {library.backend for library in registry.select("recent")})

    # Break caught: a recent_sh_main member from a library the selection does not list as a source.
    def test_default_registry_publishes_recent_sh_main_from_all_twelve_libraries(self) -> None:
        registry = get_library_registry()
        catalog = get_catalog()
        selection = registry.selection("recent_sh_main")

        self.assertIs(RECENT_SH_MAIN_SELECTION, selection)
        self.assertEqual(RECENT_SH_MAIN_SELECTION.canonical_names, catalog.canonical_names("recent_sh_main"))
        self.assertEqual(291, len(selection.canonical_names))
        self.assertEqual(12, len(selection.source_factor_sets))
        sources = {registry.source_factor_set(selection, name) for name in selection.canonical_names}
        self.assertEqual(set(selection.source_factor_sets), sources)
        self.assertEqual("sh_main", selection.provenance["universe"])


class SelectionFactorEngineTests(unittest.TestCase):
    def _build_sources(self, root: Path, registry: FactorLibraryRegistry, canonical: Path) -> None:
        for factor_set in ("left", "right"):
            create_factor_engine(root / "cache", FactorBuildConfig(factor_set=factor_set), registry=registry)\
                .build_from_canonical(canonical, root / "factors")

    # Break caught: a selection build that reorders, recomputes, or sign-flips the copied source columns.
    def test_build_copies_selected_columns_unchanged_and_records_sources(self) -> None:
        selection = _selection((SelectedFactor("rt_001", -1), SelectedFactor("lf_002", 1)))
        registry = _registry(selection)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            canonical = _write_canonical(root)
            self._build_sources(root, registry, canonical)
            engine = create_factor_engine(root / "cache", FactorBuildConfig(factor_set="picked"), registry=registry)
            manifest = engine.build_from_canonical(canonical, root / "factors")
            frame = pd.concat(
                pd.read_parquet(path) for path in sorted((root / "factors" / "picked").glob("year=*/factors.parquet"))
            )
            left = pd.concat(
                pd.read_parquet(path) for path in sorted((root / "factors" / "left").glob("year=*/factors.parquet"))
            )
            stored = json.loads((root / "factors" / "picked" / "manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(["datetime", "instrument", "rt_001", "lf_002"], list(frame.columns))
        np.testing.assert_allclose(left["lf_002"].to_numpy(), frame["lf_002"].to_numpy())
        np.testing.assert_allclose(-(left["lf_002"].to_numpy() / 2.0), frame["rt_001"].to_numpy())
        self.assertEqual({"2020", "2021"}, set(manifest["partitions"]))
        self.assertEqual(20, manifest["rows"])
        self.assertEqual({"left": ["lf_002"], "right": ["rt_001"]},
                         {source: value["columns"] for source, value in stored["source_factor_sets"].items()})
        self.assertEqual(-1, stored["selection"]["members"][0]["direction"])
        self.assertEqual(engine.compilation_fingerprint(), stored["compilation_fingerprint"])

    # Break caught: reusing a source store whose recorded engine fingerprint no longer matches the current code.
    def test_build_refuses_stale_or_missing_source_stores(self) -> None:
        selection = _selection((SelectedFactor("rt_001", 1), SelectedFactor("lf_001", 1)))
        registry = _registry(selection)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            canonical = _write_canonical(root)
            engine = create_factor_engine(root / "cache", FactorBuildConfig(factor_set="picked"), registry=registry)
            with self.assertRaisesRegex(DataContractError, "not built"):
                engine.build_from_canonical(canonical, root / "factors")

            self._build_sources(root, registry, canonical)
            manifest_path = root / "factors" / "right" / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["compilation_fingerprint"] = "stale"
            from rquant.io import stable_hash

            manifest.pop("fingerprint")
            manifest["fingerprint"] = stable_hash(manifest)
            atomic_write_json(manifest_path, manifest)
            with self.assertRaisesRegex(DataContractError, "stale factor code"):
                engine.build_from_canonical(canonical, root / "factors")
            self.assertFalse((root / "factors" / "picked").exists())


class FactorStabilityScoreTests(unittest.TestCase):
    def _frames(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        years = range(2010, 2018)
        annual = []
        for year in years:
            # "steady" predicts low returns every year; "flaky" is positive overall but only in half of the years.
            annual.append({"factor": "steady", "year": year, "return_days": 240, "rank_ic_mean": -0.03,
                           "long_short_return": -0.2})
            sign = 1.0 if year % 2 else -0.5
            annual.append({"factor": "flaky", "year": year, "return_days": 240, "rank_ic_mean": 0.03 * sign,
                           "long_short_return": 0.1 * sign})
        # A short final year must not count, even though it would break "steady"'s record.
        annual.append({"factor": "steady", "year": 2018, "return_days": 30, "rank_ic_mean": 0.05,
                       "long_short_return": 0.3})
        summary = pd.DataFrame(
            [
                {"factor": "steady", "rank_ic_mean": -0.03, "rank_icir": -0.2, "rank_ic_days": 1920,
                 "long_short_return": -1.0},
                {"factor": "flaky", "rank_ic_mean": 0.01, "rank_icir": 0.05, "rank_ic_days": 1920,
                 "long_short_return": 0.2},
            ]
        )
        return summary, pd.DataFrame(annual)

    # Break caught: dropping reliably negative factors instead of keeping them with an inverted direction.
    def test_inverted_factor_is_selected_with_negative_direction(self) -> None:
        summary, annual = self._frames()
        coverage = pd.Series({"steady": 0.99, "flaky": 0.99})
        scored = score_factor_stability(summary, annual, coverage, StabilityCriteria()).set_index("factor")

        self.assertTrue(scored.loc["steady", "selected"])
        self.assertEqual(-1, scored.loc["steady", "direction"])
        self.assertEqual(8, scored.loc["steady", "counted_years"])
        self.assertAlmostEqual(1.0, scored.loc["steady", "oriented_long_short_return"])
        self.assertFalse(scored.loc["flaky", "selected"])
        self.assertIn("rank_ic_years", scored.loc["flaky", "failed_checks"])

    # Break caught: a t-statistic built from the per-day ICIR alone, which ignores how many days back the mean.
    def test_rank_ic_t_stat_scales_daily_icir_by_square_root_of_days(self) -> None:
        summary, annual = self._frames()
        coverage = pd.Series({"steady": 0.99, "flaky": 0.99})
        scored = score_factor_stability(
            summary, annual, coverage, StabilityCriteria(min_abs_rank_ic_t_stat=8.0)
        ).set_index("factor")
        self.assertAlmostEqual(-0.2 * np.sqrt(1920), scored.loc["steady", "rank_ic_t_stat"])
        self.assertTrue(scored.loc["steady", "selected"])

        short = summary.assign(rank_ic_days=100)
        scored = score_factor_stability(
            short, annual, coverage, StabilityCriteria(min_abs_rank_ic_t_stat=8.0)
        ).set_index("factor")
        self.assertFalse(scored.loc["steady", "selected"])
        self.assertEqual("rank_ic_t_stat", scored.loc["steady", "failed_checks"])

    def test_low_coverage_blocks_selection(self) -> None:
        summary, annual = self._frames()
        scored = score_factor_stability(
            summary, annual, pd.Series({"steady": 0.5, "flaky": 0.99}), StabilityCriteria()
        ).set_index("factor")
        self.assertFalse(scored.loc["steady", "selected"])
        self.assertEqual("coverage", scored.loc["steady", "failed_checks"])

    def test_coverage_uses_best_covered_factor_per_date_as_denominator(self) -> None:
        daily = pd.DataFrame(
            {
                "datetime": ["2020-01-02", "2020-01-02", "2020-01-03", "2020-01-03"],
                "factor": ["full", "sparse", "full", "sparse"],
                "observations": [300, 150, 100, 100],
            }
        )
        coverage = factor_coverage(daily)
        self.assertAlmostEqual(1.0, coverage["full"])
        self.assertAlmostEqual(250 / 400, coverage["sparse"])


class FactorDeduplicationTests(unittest.TestCase):
    def _frame(self) -> pd.DataFrame:
        rng = np.random.default_rng(7)
        rows = []
        for date in pd.bdate_range("2021-01-04", periods=5):
            base = rng.normal(size=30)
            other = rng.normal(size=30)
            for value, independent in zip(base, other, strict=True):
                # "mirror" is a monotone transform of "base" with the opposite sign, "noisy" is unrelated.
                rows.append({"datetime": date, "base": value, "mirror": -np.exp(value), "noisy": independent})
        return pd.DataFrame(rows)

    # Break caught: correlation that ignores rank equivalence, or a pruner that keeps an inverted duplicate.
    def test_monotone_and_sign_flipped_duplicates_are_pruned_in_strength_order(self) -> None:
        correlation = mean_rank_correlation(self._frame(), ["base", "mirror", "noisy"])
        self.assertAlmostEqual(-1.0, correlation.loc["base", "mirror"])
        self.assertLess(abs(correlation.loc["base", "noisy"]), 0.7)

        strength = pd.Series({"base": 0.02, "mirror": 0.03, "noisy": 0.01})
        kept, dropped = prune_correlated(strength, correlation, max_abs_correlation=0.7)
        self.assertEqual(["mirror", "noisy"], kept)
        self.assertEqual({"base": "mirror"}, dropped)

    def test_dates_below_the_cross_section_floor_do_not_count(self) -> None:
        frame = self._frame()
        correlation = mean_rank_correlation(frame, ["base", "noisy"], min_cross_section=31)
        self.assertTrue(np.isnan(correlation.loc["base", "noisy"]))

    def test_locked_hquality_has_no_pair_above_its_correlation_limit(self) -> None:
        dropped = HQUALITY_SELECTION.provenance["deduplicated"]
        kept = set(HQUALITY_SELECTION.canonical_names)
        self.assertEqual(137, len(kept))
        self.assertFalse(kept & set(dropped))
        self.assertTrue(set(dropped.values()) <= kept)
        self.assertEqual(0.8, HQUALITY_SELECTION.criteria["max_abs_rank_correlation"])

    def test_locked_recent_has_no_pair_above_its_correlation_limit(self) -> None:
        dropped = RECENT_SELECTION.provenance["deduplicated"]
        kept = set(RECENT_SELECTION.canonical_names)
        self.assertEqual(117, len(kept))
        self.assertFalse(kept & set(dropped))
        self.assertTrue(set(dropped.values()) <= kept)
        self.assertEqual(0.8, RECENT_SELECTION.criteria["max_abs_rank_correlation"])

    def test_locked_recent_sh_main_has_no_pair_above_its_correlation_limit(self) -> None:
        dropped = RECENT_SH_MAIN_SELECTION.provenance["deduplicated"]
        kept = set(RECENT_SH_MAIN_SELECTION.canonical_names)
        self.assertEqual(291, len(kept))
        self.assertFalse(kept & set(dropped))
        self.assertTrue(set(dropped.values()) <= kept)
        self.assertEqual(0.7, RECENT_SH_MAIN_SELECTION.criteria["min_positive_rank_ic_year_ratio"])


if __name__ == "__main__":
    unittest.main()
