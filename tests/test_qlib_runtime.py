from __future__ import annotations

import unittest
from unittest.mock import patch

from rquant.qlib_ext.runtime import WINDOWS_MAX_QLIB_KERNELS, qlib_init_kwargs


class QlibRuntimeTests(unittest.TestCase):
    def test_windows_caps_the_expression_worker_pool(self) -> None:
        with patch("rquant.qlib_ext.runtime.sys.platform", "win32"), patch(
            "rquant.qlib_ext.runtime.os.cpu_count", return_value=24
        ):
            self.assertEqual({"kernels": WINDOWS_MAX_QLIB_KERNELS}, qlib_init_kwargs())
        with patch("rquant.qlib_ext.runtime.sys.platform", "win32"), patch(
            "rquant.qlib_ext.runtime.os.cpu_count", return_value=2
        ):
            self.assertEqual({"kernels": 2}, qlib_init_kwargs())

    def test_other_platforms_keep_qlib_defaults(self) -> None:
        with patch("rquant.qlib_ext.runtime.sys.platform", "darwin"):
            self.assertEqual({}, qlib_init_kwargs())


if __name__ == "__main__":
    unittest.main()
