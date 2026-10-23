from __future__ import annotations

import os
import sys
from typing import Any

# Each Windows worker is a spawned interpreter that re-imports Qlib, SciPy and Pandas; Qlib's default of one worker
# per core exhausts the commit limit on a 16 GB machine before any data is loaded.
WINDOWS_MAX_QLIB_KERNELS = 4


def qlib_init_kwargs() -> dict[str, Any]:
    """Extra ``qlib.init`` arguments that keep its expression-loading pool within the host's memory."""
    if sys.platform != "win32":
        return {}
    return {"kernels": max(1, min(os.cpu_count() or 1, WINDOWS_MAX_QLIB_KERNELS))}
