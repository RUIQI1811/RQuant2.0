"""Locked membership of the Hquality factor set.

Hquality keeps the factors whose single-factor evaluation is stable across calendar years. It was selected with
``rquant.factors.selection.score_factor_stability`` and ``StabilityCriteria()`` from the 1d evaluations listed in
``provenance`` (``rquant factors evaluate --horizon 1d --start 2010-01-01 --end 2026-10-19`` on ``combined``,
``gtja191`` and ``qlib_alpha360``):

- direction is the sign of the full-period mean daily Rank IC; every yearly statistic is oriented by it;
- years with at least ``min_year_days`` evaluated days are counted (2010-2026; the last is partial);
- the oriented annual Rank IC is positive in at least ``min_positive_rank_ic_year_ratio`` of counted years;
- the oriented annual top-minus-bottom quintile return is positive in at least
  ``min_positive_long_short_year_ratio`` of counted years;
- the absolute full-period Rank IC is at least ``min_abs_rank_ic``;
- coverage (published observations over the best-covered factor's, pooled over dates) is at least ``min_coverage``;
- duplicates are then pruned with ``prune_correlated``: the 43 factors passing the checks above are ordered by absolute
  Rank IC, and a factor is dropped when the absolute mean daily cross-sectional Spearman correlation
  (``mean_rank_correlation`` over the same evaluation dates) with an already kept factor reaches
  ``max_abs_rank_correlation``. ``provenance["deduplicated"]`` maps each dropped factor to the kept one it duplicates.

Members keep their source canonical names and values. ``direction == -1`` marks a factor used inverted; the stored
column is not negated. Changing membership changes the Hquality catalog and engine fingerprints, so a stale
``data/factors/Hquality`` store is refused rather than reused.
"""

from __future__ import annotations

from rquant.factors.libraries.base import FactorSelection, SelectedFactor

HQUALITY_MEMBERS = (
    SelectedFactor("a158_034", -1),  # RankIC -0.0134, IC+ 16/17, LS+ 16/17, cov 0.999
    SelectedFactor("a158_035", -1),  # RankIC -0.0104, IC+ 15/17, LS+ 15/17, cov 1.000
    SelectedFactor("a158_079", 1),  # RankIC +0.0203, IC+ 14/17, LS+ 14/17, cov 0.989
    SelectedFactor("a158_092", -1),  # RankIC -0.0103, IC+ 15/17, LS+ 13/17, cov 1.000
    SelectedFactor("a158_097", -1),  # RankIC -0.0165, IC+ 17/17, LS+ 13/17, cov 1.000
    SelectedFactor("a158_098", -1),  # RankIC -0.0128, IC+ 16/17, LS+ 13/17, cov 1.000
    SelectedFactor("a158_109", -1),  # RankIC -0.0245, IC+ 14/17, LS+ 14/17, cov 1.000
    SelectedFactor("a101_001", -1),  # RankIC -0.0201, IC+ 16/17, LS+ 13/17, cov 0.938
    SelectedFactor("a101_005", 1),  # RankIC +0.0290, IC+ 15/17, LS+ 15/17, cov 0.974
    SelectedFactor("a101_008", 1),  # RankIC +0.0233, IC+ 15/17, LS+ 15/17, cov 0.966
    SelectedFactor("a101_011", 1),  # RankIC +0.0115, IC+ 13/17, LS+ 15/17, cov 0.991
    SelectedFactor("a101_021", 1),  # RankIC +0.0118, IC+ 16/17, LS+ 14/17, cov 1.000
    SelectedFactor("a101_047", 1),  # RankIC +0.0148, IC+ 15/17, LS+ 13/17, cov 1.000
    SelectedFactor("a101_056", 1),  # RankIC +0.0208, IC+ 17/17, LS+ 15/17, cov 0.972
    SelectedFactor("a101_080", 1),  # RankIC +0.0146, IC+ 16/17, LS+ 15/17, cov 0.942
    SelectedFactor("a101_083", 1),  # RankIC +0.0375, IC+ 17/17, LS+ 17/17, cov 0.983
    SelectedFactor("gtja_001", 1),  # RankIC +0.0220, IC+ 17/17, LS+ 14/17, cov 0.983
    SelectedFactor("gtja_003", -1),  # RankIC -0.0316, IC+ 15/17, LS+ 15/17, cov 0.983
    SelectedFactor("gtja_005", 1),  # RankIC +0.0168, IC+ 17/17, LS+ 14/17, cov 0.972
    SelectedFactor("gtja_011", -1),  # RankIC -0.0218, IC+ 15/17, LS+ 14/17, cov 0.984
    SelectedFactor("gtja_017", 1),  # RankIC +0.0285, IC+ 15/17, LS+ 13/17, cov 0.961
    SelectedFactor("gtja_022", -1),  # RankIC -0.0223, IC+ 14/17, LS+ 13/17, cov 0.977
    SelectedFactor("gtja_029", -1),  # RankIC -0.0273, IC+ 16/17, LS+ 13/17, cov 0.994
    SelectedFactor("gtja_054", 1),  # RankIC +0.0162, IC+ 16/17, LS+ 16/17, cov 0.974
    SelectedFactor("gtja_111", 1),  # RankIC +0.0202, IC+ 17/17, LS+ 16/17, cov 0.999
    SelectedFactor("gtja_146", -1),  # RankIC -0.0137, IC+ 14/17, LS+ 14/17, cov 0.945
    SelectedFactor("gtja_156", 1),  # RankIC +0.0134, IC+ 16/17, LS+ 13/17, cov 0.980
)

HQUALITY_SELECTION = FactorSelection(
    name="Hquality",
    source_factor_sets=("combined", "gtja191", "qlib_alpha360"),
    members=HQUALITY_MEMBERS,
    criteria={
        "min_year_days": 120,
        "min_positive_rank_ic_year_ratio": 0.75,
        "min_positive_long_short_year_ratio": 0.75,
        "min_abs_rank_ic": 0.01,
        "min_coverage": 0.9,
        "max_abs_rank_correlation": 0.7,
    },
    provenance={
        "horizon": "1d",
        "evaluation_start": "2010-01-04",
        "evaluation_end": "2026-10-19",
        "qlib_fingerprint": "4c48525d91abd4f4081a48179015fd9a9de899fed0158ef0b4db3a9880c009b8",
        "deduplicated": {
            "gtja_114": "a101_083",
            "a360_054": "gtja_003",
            "gtja_020": "gtja_003",
            "gtja_124": "a101_083",
            "a360_294": "gtja_003",
            "a101_084": "gtja_017",
            "a158_013": "a101_083",
            "a360_300": "a101_083",
            "a158_099": "a158_109",
            "gtja_043": "gtja_029",
            "gtja_037": "a101_008",
            "a101_057": "a101_083",
            "a101_018": "gtja_054",
            "a101_060": "gtja_111",
            "gtja_007": "a101_011",
            "gtja_004": "a101_021",
        },
        "evaluations": (
            {
                "factor_set": "combined",
                "run_id": "20261023T071832Z-7689aca5",
                "factor_fingerprint": "2d170b5b3d5731172de2d0e900455e1ea1e7f7f13f1d60da9e6e08f86835cc5f",
            },
            {
                "factor_set": "gtja191",
                "run_id": "20261023T003652Z-4a487c0e",
                "factor_fingerprint": "bc9d664f354f5cf78265f528eca208e5b85e69963cdaec2e2989df19a6a459d5",
            },
            {
                "factor_set": "qlib_alpha360",
                "run_id": "20261023T072300Z-e778fcf4",
                "factor_fingerprint": "d6cff8fc9b2f68f6cf605fd45a4a6ec3fd05859e81e310939b44a7e162200511",
            },
        ),
    },
)
