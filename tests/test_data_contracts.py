from __future__ import annotations

import unittest

from rquant.data.canonical import ts_code_to_qlib
from rquant.errors import DataContractError
from rquant.qlib_ext.loader import LABEL_EXPRESSIONS


class DataContractTests(unittest.TestCase):
    def test_tushare_code_conversion(self) -> None:
        self.assertEqual("SH600000", ts_code_to_qlib("600000.SH"))
        self.assertEqual("SZ000001", ts_code_to_qlib("000001.SZ"))
        self.assertEqual("BJ430047", ts_code_to_qlib("430047.BJ"))
        with self.assertRaises(DataContractError):
            ts_code_to_qlib("AAPL.US")

    def test_next_open_labels_are_locked(self) -> None:
        self.assertEqual("Ref($open, -2) / Ref($open, -1) - 1", LABEL_EXPRESSIONS["1d"])
        self.assertEqual("Ref($open, -6) / Ref($open, -1) - 1", LABEL_EXPRESSIONS["5d"])
        self.assertEqual("Ref($open, -21) / Ref($open, -1) - 1", LABEL_EXPRESSIONS["20d"])


if __name__ == "__main__":
    unittest.main()
