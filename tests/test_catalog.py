from __future__ import annotations

import unittest

from rquant.factors.catalog import ALPHA101_UPSTREAM_MISSING, get_catalog
from rquant.factors.engine import validate_factor_frame


class FakeFrame:
    def __init__(self, columns: list[str], rows: int = 3) -> None:
        self.columns = columns
        self._rows = rows

    def __len__(self) -> int:
        return self._rows


class FactorCatalogTests(unittest.TestCase):
    # Break caught: Alpha360 specs that omit, duplicate, or reorder canonical output positions.
    def test_alpha360_is_strictly_numbered(self) -> None:
        names = get_catalog().canonical_names("qlib_alpha360")
        self.assertEqual(tuple(f"a360_{value:03d}" for value in range(1, 361)), names)

    # Break caught: RQuant formula metadata diverging from the locked pyqlib 0.9.7 Alpha360DL source contract.
    def test_alpha360_source_contract_matches_locked_qlib(self) -> None:
        from qlib.contrib.data.loader import Alpha360DL

        expressions, source_names = Alpha360DL.get_feature_config()
        specs = get_catalog().select("qlib_alpha360")
        self.assertEqual(source_names, [spec.source_name for spec in specs])
        self.assertEqual(expressions, [spec.formula for spec in specs])

    # Break caught: registering standalone Alpha360 accidentally changes the locked combined Alpha158-plus-Alpha101 bundle.
    def test_alpha360_does_not_expand_combined(self) -> None:
        catalog = get_catalog()
        self.assertEqual(809, len(catalog.specs))
        self.assertEqual(259, len(catalog.select("combined")))

    # Break caught: Alpha158 specs that omit, duplicate, or reorder canonical output positions.
    def test_alpha158_is_strictly_numbered(self) -> None:
        names = get_catalog().canonical_names("qlib_alpha158")
        self.assertEqual(158, len(names))
        self.assertEqual(tuple(f"a158_{value:03d}" for value in range(1, 159)), names)

    def test_alpha158_source_order_matches_locked_qlib(self) -> None:
        from qlib.contrib.data.handler import Alpha158

        _, source_names = Alpha158.__new__(Alpha158).get_feature_config()
        catalog_names = [spec.source_name for spec in get_catalog().select("qlib_alpha158")]
        self.assertEqual(source_names, catalog_names)

    def test_alpha101_is_strictly_numbered(self) -> None:
        names = get_catalog().canonical_names("wq_alpha101")
        self.assertEqual(tuple(f"a101_{value:03d}" for value in range(1, 102)), names)
        self.assertEqual(19, len(ALPHA101_UPSTREAM_MISSING))

    def test_combined_order_is_locked(self) -> None:
        names = get_catalog().canonical_names("combined")
        self.assertEqual(259, len(names))
        self.assertEqual("a158_158", names[157])
        self.assertEqual("a101_001", names[158])
        self.assertEqual("a101_101", names[-1])

    def test_factor_frame_accepts_only_canonical_order(self) -> None:
        expected = list(get_catalog().canonical_names("qlib_alpha158"))
        result = validate_factor_frame(FakeFrame(["datetime", "instrument", *expected]), "qlib_alpha158")
        self.assertEqual(158, result["columns"])
        with self.assertRaisesRegex(Exception, "out of order"):
            validate_factor_frame(FakeFrame(["datetime", "instrument", *reversed(expected)]), "qlib_alpha158")


if __name__ == "__main__":
    unittest.main()
