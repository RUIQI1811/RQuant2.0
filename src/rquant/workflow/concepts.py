"""HIST's predefined concepts from Shenwan industries.

Qlib's HIST model groups stocks by a stock-to-concept matrix (``stock2concept``, stocks x concepts) and maps each
instrument to its row with ``stock_index``. Qlib's benchmark downloads a CSI 300 concept matrix it prepared itself;
RQuant builds one from the point-in-time Shenwan level-2 industries of the canonical bars instead: a stock belongs to
every level-2 industry it was a member of on or before the last prediction date (stocks move between industries
rarely). Stocks never classified share one extra concept, so no row of the matrix is empty.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from rquant.errors import DataContractError


def write_industry_concepts(bars_root: Path, directory: Path, through: date) -> tuple[Path, Path]:
    """Write ``stock2concept.npy`` and ``stock_index.npy`` for every instrument of the canonical bars into
    ``directory`` and return their paths."""
    import numpy as np
    import pandas as pd

    partitions = sorted(
        path
        for path in Path(bars_root).glob("year=*/bars.parquet")
        if int(path.parent.name.split("=")[1]) <= through.year
    )
    if not partitions:
        raise DataContractError(f"No canonical bars under {bars_root}")
    frames = []
    for path in partitions:
        frame = pd.read_parquet(path, columns=["datetime", "instrument", "industry"])
        frame = frame[pd.to_datetime(frame["datetime"]) <= pd.Timestamp(through)]
        frames.append(frame[["instrument", "industry"]].drop_duplicates())
    memberships = pd.concat(frames, ignore_index=True).drop_duplicates()
    instruments = sorted(memberships["instrument"].unique())
    classified = memberships.dropna(subset=["industry"])
    industries = sorted(classified["industry"].unique())
    rows = {instrument: position for position, instrument in enumerate(instruments)}
    columns = {industry: position for position, industry in enumerate(industries)}

    matrix = np.zeros((len(instruments), len(industries) + 1), dtype=np.float32)
    matrix[
        classified["instrument"].map(rows).to_numpy(), classified["industry"].map(columns).to_numpy()
    ] = 1.0
    matrix[matrix[:, : len(industries)].sum(axis=1) == 0, len(industries)] = 1.0

    directory.mkdir(parents=True, exist_ok=True)
    concept_path = directory / "stock2concept.npy"
    index_path = directory / "stock_index.npy"
    np.save(concept_path, matrix)
    np.save(index_path, rows, allow_pickle=True)
    return concept_path, index_path
