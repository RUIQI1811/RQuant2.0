"""Locked membership of the Hquality factor set.

Hquality keeps factors whose single-factor Rank IC is stable across calendar years, then removes duplicates. It was
selected with ``rquant.factors.selection`` from the 1d evaluations listed in ``provenance``
(``rquant factors evaluate --horizon 1d --start 2010-01-01 --end 2026-10-19`` on ``combined``, ``gtja191`` and
``qlib_alpha360``):

- direction is the sign of the full-period mean daily Rank IC; every yearly statistic is oriented by it;
- years with at least ``min_year_days`` evaluated days are counted (2010-2026; the last is partial);
- the oriented annual Rank IC is positive in at least ``min_positive_rank_ic_year_ratio`` of counted years;
- the yearly long-short return is not required (``min_positive_long_short_year_ratio`` is 0): requiring it in 75% of
  years left 43 factors whose LGB backtest returned 10.1% a year against 18.7% for ``combined``;
- the absolute full-period Rank IC is at least ``min_abs_rank_ic``;
- coverage (published observations over the best-covered factor's, pooled over dates) is at least ``min_coverage``;
- duplicates are then pruned with ``prune_correlated``: the 613 factors passing the checks above are
  ordered by absolute Rank IC, and a factor is dropped when its absolute mean daily cross-sectional Spearman
  correlation (``mean_rank_correlation`` on every fifth trading date of the evaluation period) with an already kept
  factor reaches ``max_abs_rank_correlation``. ``DEDUPLICATED`` maps each dropped factor to the kept one it duplicates.

Members keep their source canonical names and values. ``direction == -1`` marks a factor used inverted; the stored
column is not negated. Changing membership changes the Hquality catalog and engine fingerprints, so a stale
``data/factors/Hquality`` store is refused rather than reused.
"""

from __future__ import annotations

from rquant.factors.libraries.base import FactorSelection, SelectedFactor

HQUALITY_MEMBERS = (
    SelectedFactor("a158_001", -1),  # RankIC -0.0200, IC+ 14/17, cov 1.000
    SelectedFactor("a158_002", -1),  # RankIC -0.0221, IC+ 16/17, cov 1.000
    SelectedFactor("a158_006", -1),  # RankIC -0.0140, IC+ 16/17, cov 1.000
    SelectedFactor("a158_009", -1),  # RankIC -0.0108, IC+ 14/17, cov 1.000
    SelectedFactor("a158_012", 1),  # RankIC +0.0252, IC+ 15/17, cov 1.000
    SelectedFactor("a158_024", -1),  # RankIC -0.0177, IC+ 15/17, cov 0.989
    SelectedFactor("a158_025", -1),  # RankIC -0.0188, IC+ 16/17, cov 0.974
    SelectedFactor("a158_026", -1),  # RankIC -0.0161, IC+ 14/17, cov 0.948
    SelectedFactor("a158_034", -1),  # RankIC -0.0134, IC+ 16/17, cov 0.999
    SelectedFactor("a158_035", -1),  # RankIC -0.0104, IC+ 15/17, cov 1.000
    SelectedFactor("a158_040", -1),  # RankIC -0.0130, IC+ 14/17, cov 1.000
    SelectedFactor("a158_041", -1),  # RankIC -0.0210, IC+ 15/17, cov 1.000
    SelectedFactor("a158_042", -1),  # RankIC -0.0197, IC+ 14/17, cov 1.000
    SelectedFactor("a158_043", -1),  # RankIC -0.0210, IC+ 16/17, cov 1.000
    SelectedFactor("a158_044", 1),  # RankIC +0.0103, IC+ 12/17, cov 0.989
    SelectedFactor("a158_045", 1),  # RankIC +0.0116, IC+ 13/17, cov 0.974
    SelectedFactor("a158_049", 1),  # RankIC +0.0296, IC+ 17/17, cov 0.989
    SelectedFactor("a158_051", 1),  # RankIC +0.0219, IC+ 14/17, cov 0.948
    SelectedFactor("a158_054", 1),  # RankIC +0.0230, IC+ 14/17, cov 0.989
    SelectedFactor("a158_056", 1),  # RankIC +0.0227, IC+ 15/17, cov 0.948
    SelectedFactor("a158_058", 1),  # RankIC +0.0168, IC+ 14/17, cov 0.859
    SelectedFactor("a158_063", 1),  # RankIC +0.0206, IC+ 16/17, cov 0.859
    SelectedFactor("a158_072", -1),  # RankIC -0.0141, IC+ 13/17, cov 1.000
    SelectedFactor("a158_076", -1),  # RankIC -0.0111, IC+ 14/17, cov 0.948
    SelectedFactor("a158_077", -1),  # RankIC -0.0106, IC+ 14/17, cov 0.923
    SelectedFactor("a158_079", 1),  # RankIC +0.0203, IC+ 14/17, cov 0.989
    SelectedFactor("a158_084", -1),  # RankIC -0.0223, IC+ 14/17, cov 0.989
    SelectedFactor("a158_085", -1),  # RankIC -0.0133, IC+ 12/17, cov 0.974
    SelectedFactor("a158_090", -1),  # RankIC -0.0109, IC+ 14/17, cov 1.000
    SelectedFactor("a158_091", -1),  # RankIC -0.0117, IC+ 15/17, cov 1.000
    SelectedFactor("a158_092", -1),  # RankIC -0.0103, IC+ 15/17, cov 1.000
    SelectedFactor("a158_094", -1),  # RankIC -0.0217, IC+ 17/17, cov 1.000
    SelectedFactor("a158_095", -1),  # RankIC -0.0167, IC+ 17/17, cov 1.000
    SelectedFactor("a158_096", -1),  # RankIC -0.0161, IC+ 17/17, cov 1.000
    SelectedFactor("a158_097", -1),  # RankIC -0.0165, IC+ 17/17, cov 1.000
    SelectedFactor("a158_098", -1),  # RankIC -0.0128, IC+ 16/17, cov 1.000
    SelectedFactor("a158_105", 1),  # RankIC +0.0131, IC+ 14/17, cov 1.000
    SelectedFactor("a158_109", -1),  # RankIC -0.0245, IC+ 14/17, cov 1.000
    SelectedFactor("a158_133", 1),  # RankIC +0.0173, IC+ 16/17, cov 0.859
    SelectedFactor("a158_144", -1),  # RankIC -0.0178, IC+ 15/17, cov 0.986
    SelectedFactor("a158_145", -1),  # RankIC -0.0125, IC+ 15/17, cov 0.972
    SelectedFactor("a158_146", -1),  # RankIC -0.0135, IC+ 14/17, cov 0.945
    SelectedFactor("a158_147", -1),  # RankIC -0.0145, IC+ 17/17, cov 0.920
    SelectedFactor("a158_148", -1),  # RankIC -0.0159, IC+ 17/17, cov 0.857
    SelectedFactor("a101_001", -1),  # RankIC -0.0201, IC+ 16/17, cov 0.938
    SelectedFactor("a101_002", 1),  # RankIC +0.0136, IC+ 16/17, cov 1.000
    SelectedFactor("a101_004", 1),  # RankIC +0.0261, IC+ 15/17, cov 0.977
    SelectedFactor("a101_005", 1),  # RankIC +0.0290, IC+ 15/17, cov 0.974
    SelectedFactor("a101_008", 1),  # RankIC +0.0233, IC+ 15/17, cov 0.966
    SelectedFactor("a101_011", 1),  # RankIC +0.0115, IC+ 13/17, cov 0.991
    SelectedFactor("a101_021", 1),  # RankIC +0.0118, IC+ 16/17, cov 1.000
    SelectedFactor("a101_024", 1),  # RankIC +0.0261, IC+ 17/17, cov 0.996
    SelectedFactor("a101_025", 1),  # RankIC +0.0220, IC+ 16/17, cov 0.948
    SelectedFactor("a101_030", 1),  # RankIC +0.0148, IC+ 15/17, cov 0.991
    SelectedFactor("a101_034", 1),  # RankIC +0.0129, IC+ 14/17, cov 0.997
    SelectedFactor("a101_038", 1),  # RankIC +0.0246, IC+ 16/17, cov 0.974
    SelectedFactor("a101_040", 1),  # RankIC +0.0199, IC+ 16/17, cov 0.974
    SelectedFactor("a101_042", 1),  # RankIC +0.0273, IC+ 16/17, cov 1.000
    SelectedFactor("a101_046", 1),  # RankIC +0.0132, IC+ 13/17, cov 0.998
    SelectedFactor("a101_047", 1),  # RankIC +0.0148, IC+ 15/17, cov 1.000
    SelectedFactor("a101_051", 1),  # RankIC +0.0136, IC+ 12/17, cov 0.997
    SelectedFactor("a101_056", 1),  # RankIC +0.0208, IC+ 17/17, cov 0.972
    SelectedFactor("a101_060", 1),  # RankIC +0.0131, IC+ 15/17, cov 0.974
    SelectedFactor("a101_066", 1),  # RankIC +0.0199, IC+ 14/17, cov 0.972
    SelectedFactor("a101_069", 1),  # RankIC +0.0148, IC+ 16/17, cov 0.911
    SelectedFactor("a101_070", 1),  # RankIC +0.0123, IC+ 14/17, cov 0.801
    SelectedFactor("a101_073", 1),  # RankIC +0.0150, IC+ 16/17, cov 0.945
    SelectedFactor("a101_080", 1),  # RankIC +0.0146, IC+ 16/17, cov 0.942
    SelectedFactor("a101_082", 1),  # RankIC +0.0114, IC+ 13/17, cov 0.903
    SelectedFactor("a101_083", 1),  # RankIC +0.0375, IC+ 17/17, cov 0.983
    SelectedFactor("gtja_001", 1),  # RankIC +0.0220, IC+ 17/17, cov 0.983
    SelectedFactor("gtja_003", -1),  # RankIC -0.0316, IC+ 15/17, cov 0.983
    SelectedFactor("gtja_005", 1),  # RankIC +0.0168, IC+ 17/17, cov 0.972
    SelectedFactor("gtja_006", 1),  # RankIC +0.0219, IC+ 14/17, cov 0.995
    SelectedFactor("gtja_008", 1),  # RankIC +0.0249, IC+ 14/17, cov 0.995
    SelectedFactor("gtja_009", -1),  # RankIC -0.0262, IC+ 14/17, cov 0.997
    SelectedFactor("gtja_011", -1),  # RankIC -0.0218, IC+ 15/17, cov 0.984
    SelectedFactor("gtja_012", 1),  # RankIC +0.0219, IC+ 15/17, cov 0.974
    SelectedFactor("gtja_016", 1),  # RankIC +0.0135, IC+ 17/17, cov 0.977
    SelectedFactor("gtja_017", 1),  # RankIC +0.0285, IC+ 15/17, cov 0.961
    SelectedFactor("gtja_022", -1),  # RankIC -0.0223, IC+ 14/17, cov 0.977
    SelectedFactor("gtja_023", -1),  # RankIC -0.0143, IC+ 15/17, cov 0.948
    SelectedFactor("gtja_024", -1),  # RankIC -0.0212, IC+ 15/17, cov 0.994
    SelectedFactor("gtja_028", -1),  # RankIC -0.0204, IC+ 14/17, cov 0.977
    SelectedFactor("gtja_029", -1),  # RankIC -0.0273, IC+ 16/17, cov 0.994
    SelectedFactor("gtja_032", 1),  # RankIC +0.0114, IC+ 16/17, cov 0.989
    SelectedFactor("gtja_035", 1),  # RankIC +0.0138, IC+ 14/17, cov 0.940
    SelectedFactor("gtja_036", -1),  # RankIC -0.0126, IC+ 17/17, cov 0.983
    SelectedFactor("gtja_038", 1),  # RankIC +0.0217, IC+ 16/17, cov 0.948
    SelectedFactor("gtja_040", -1),  # RankIC -0.0178, IC+ 17/17, cov 0.930
    SelectedFactor("gtja_041", 1),  # RankIC +0.0254, IC+ 16/17, cov 0.980
    SelectedFactor("gtja_043", -1),  # RankIC -0.0238, IC+ 17/17, cov 0.983
    SelectedFactor("gtja_047", 1),  # RankIC +0.0115, IC+ 12/17, cov 0.986
    SelectedFactor("gtja_048", 1),  # RankIC +0.0225, IC+ 16/17, cov 0.948
    SelectedFactor("gtja_049", 1),  # RankIC +0.0133, IC+ 14/17, cov 0.966
    SelectedFactor("gtja_052", -1),  # RankIC -0.0149, IC+ 16/17, cov 0.930
    SelectedFactor("gtja_053", -1),  # RankIC -0.0150, IC+ 15/17, cov 0.966
    SelectedFactor("gtja_054", 1),  # RankIC +0.0162, IC+ 16/17, cov 0.974
    SelectedFactor("gtja_056", 1),  # RankIC +0.0111, IC+ 13/17, cov 0.841
    SelectedFactor("gtja_058", -1),  # RankIC -0.0126, IC+ 16/17, cov 0.945
    SelectedFactor("gtja_059", -1),  # RankIC -0.0167, IC+ 16/17, cov 0.945
    SelectedFactor("gtja_060", -1),  # RankIC -0.0117, IC+ 14/17, cov 0.942
    SelectedFactor("gtja_062", 1),  # RankIC +0.0115, IC+ 16/17, cov 0.989
    SelectedFactor("gtja_064", 1),  # RankIC +0.0114, IC+ 17/17, cov 0.813
    SelectedFactor("gtja_068", -1),  # RankIC -0.0201, IC+ 14/17, cov 0.997
    SelectedFactor("gtja_069", -1),  # RankIC -0.0130, IC+ 16/17, cov 0.945
    SelectedFactor("gtja_074", -1),  # RankIC -0.0121, IC+ 17/17, cov 0.850
    SelectedFactor("gtja_083", 1),  # RankIC +0.0120, IC+ 16/17, cov 0.989
    SelectedFactor("gtja_084", -1),  # RankIC -0.0166, IC+ 17/17, cov 0.945
    SelectedFactor("gtja_089", -1),  # RankIC -0.0186, IC+ 15/17, cov 1.000
    SelectedFactor("gtja_091", 1),  # RankIC +0.0123, IC+ 13/17, cov 0.891
    SelectedFactor("gtja_094", -1),  # RankIC -0.0164, IC+ 15/17, cov 0.920
    SelectedFactor("gtja_099", 1),  # RankIC +0.0134, IC+ 16/17, cov 0.989
    SelectedFactor("gtja_102", -1),  # RankIC -0.0122, IC+ 13/17, cov 0.997
    SelectedFactor("gtja_111", 1),  # RankIC +0.0202, IC+ 17/17, cov 0.999
    SelectedFactor("gtja_118", -1),  # RankIC -0.0135, IC+ 15/17, cov 0.948
    SelectedFactor("gtja_121", 1),  # RankIC +0.0217, IC+ 14/17, cov 0.825
    SelectedFactor("gtja_124", -1),  # RankIC -0.0291, IC+ 17/17, cov 0.920
    SelectedFactor("gtja_128", -1),  # RankIC -0.0136, IC+ 16/17, cov 0.961
    SelectedFactor("gtja_131", -1),  # RankIC -0.0111, IC+ 13/17, cov 0.819
    SelectedFactor("gtja_134", -1),  # RankIC -0.0155, IC+ 17/17, cov 0.992
    SelectedFactor("gtja_137", -1),  # RankIC -0.0252, IC+ 16/17, cov 0.997
    SelectedFactor("gtja_140", 1),  # RankIC +0.0237, IC+ 16/17, cov 0.804
    SelectedFactor("gtja_146", -1),  # RankIC -0.0137, IC+ 14/17, cov 0.945
    SelectedFactor("gtja_149", -1),  # RankIC -0.0120, IC+ 12/16, cov 0.915
    SelectedFactor("gtja_150", -1),  # RankIC -0.0173, IC+ 13/17, cov 1.000
    SelectedFactor("gtja_156", 1),  # RankIC +0.0134, IC+ 16/17, cov 0.980
    SelectedFactor("gtja_157", 1),  # RankIC +0.0158, IC+ 16/17, cov 0.969
    SelectedFactor("gtja_159", -1),  # RankIC -0.0252, IC+ 15/17, cov 0.935
    SelectedFactor("gtja_162", -1),  # RankIC -0.0114, IC+ 12/17, cov 0.966
    SelectedFactor("gtja_167", -1),  # RankIC -0.0180, IC+ 14/17, cov 0.966
    SelectedFactor("gtja_170", 1),  # RankIC +0.0241, IC+ 14/17, cov 0.948
    SelectedFactor("gtja_176", -1),  # RankIC -0.0149, IC+ 17/17, cov 0.955
    SelectedFactor("gtja_178", -1),  # RankIC -0.0227, IC+ 16/17, cov 0.997
    SelectedFactor("gtja_185", 1),  # RankIC +0.0114, IC+ 15/17, cov 1.000
    SelectedFactor("a360_179", 1),  # RankIC +0.0116, IC+ 13/17, cov 0.997
    SelectedFactor("a360_186", 1),  # RankIC +0.0127, IC+ 13/17, cov 0.985
    SelectedFactor("a360_220", 1),  # RankIC +0.0190, IC+ 15/17, cov 0.990
    SelectedFactor("a360_232", 1),  # RankIC +0.0242, IC+ 16/17, cov 0.993
    SelectedFactor("a360_235", 1),  # RankIC +0.0330, IC+ 15/17, cov 0.994
    SelectedFactor("a360_238", 1),  # RankIC +0.0297, IC+ 17/17, cov 0.996
    SelectedFactor("a360_239", 1),  # RankIC +0.0273, IC+ 15/17, cov 0.997
    SelectedFactor("a360_301", 1),  # RankIC +0.0151, IC+ 17/17, cov 0.985
    SelectedFactor("a360_304", 1),  # RankIC +0.0153, IC+ 17/17, cov 0.985
    SelectedFactor("a360_306", 1),  # RankIC +0.0185, IC+ 17/17, cov 0.985
    SelectedFactor("a360_308", 1),  # RankIC +0.0159, IC+ 17/17, cov 0.986
    SelectedFactor("a360_310", 1),  # RankIC +0.0181, IC+ 17/17, cov 0.986
    SelectedFactor("a360_313", 1),  # RankIC +0.0168, IC+ 17/17, cov 0.986
    SelectedFactor("a360_315", 1),  # RankIC +0.0159, IC+ 17/17, cov 0.986
    SelectedFactor("a360_318", 1),  # RankIC +0.0166, IC+ 17/17, cov 0.987
    SelectedFactor("a360_320", 1),  # RankIC +0.0156, IC+ 17/17, cov 0.987
    SelectedFactor("a360_322", 1),  # RankIC +0.0140, IC+ 15/17, cov 0.987
    SelectedFactor("a360_324", 1),  # RankIC +0.0153, IC+ 17/17, cov 0.988
    SelectedFactor("a360_326", 1),  # RankIC +0.0167, IC+ 17/17, cov 0.988
    SelectedFactor("a360_328", 1),  # RankIC +0.0151, IC+ 16/17, cov 0.988
    SelectedFactor("a360_329", 1),  # RankIC +0.0162, IC+ 17/17, cov 0.988
    SelectedFactor("a360_331", 1),  # RankIC +0.0138, IC+ 17/17, cov 0.989
    SelectedFactor("a360_332", 1),  # RankIC +0.0131, IC+ 15/17, cov 0.989
    SelectedFactor("a360_333", 1),  # RankIC +0.0146, IC+ 17/17, cov 0.989
    SelectedFactor("a360_334", 1),  # RankIC +0.0157, IC+ 16/17, cov 0.989
    SelectedFactor("a360_335", 1),  # RankIC +0.0150, IC+ 15/17, cov 0.989
    SelectedFactor("a360_336", 1),  # RankIC +0.0130, IC+ 15/17, cov 0.989
    SelectedFactor("a360_337", 1),  # RankIC +0.0133, IC+ 15/17, cov 0.990
    SelectedFactor("a360_338", 1),  # RankIC +0.0134, IC+ 14/17, cov 0.990
    SelectedFactor("a360_339", 1),  # RankIC +0.0131, IC+ 15/17, cov 0.990
    SelectedFactor("a360_341", 1),  # RankIC +0.0124, IC+ 14/17, cov 0.990
    SelectedFactor("a360_342", 1),  # RankIC +0.0125, IC+ 15/17, cov 0.990
    SelectedFactor("a360_343", 1),  # RankIC +0.0126, IC+ 16/17, cov 0.991
    SelectedFactor("a360_344", 1),  # RankIC +0.0116, IC+ 15/17, cov 0.991
    SelectedFactor("a360_345", 1),  # RankIC +0.0119, IC+ 15/17, cov 0.991
    SelectedFactor("a360_346", 1),  # RankIC +0.0120, IC+ 15/17, cov 0.991
    SelectedFactor("a360_347", 1),  # RankIC +0.0135, IC+ 16/17, cov 0.992
    SelectedFactor("a360_348", 1),  # RankIC +0.0113, IC+ 13/17, cov 0.992
    SelectedFactor("a360_349", 1),  # RankIC +0.0112, IC+ 13/17, cov 0.992
    SelectedFactor("a360_352", 1),  # RankIC +0.0105, IC+ 15/17, cov 0.993
    SelectedFactor("a360_353", 1),  # RankIC +0.0122, IC+ 13/17, cov 0.993
    SelectedFactor("a360_354", 1),  # RankIC +0.0139, IC+ 16/17, cov 0.994
)

DEDUPLICATED = {
        "a101_009": "a101_025",
        "a101_013": "gtja_099",
        "a101_015": "gtja_032",
        "a101_016": "gtja_083",
        "a101_018": "gtja_054",
        "a101_023": "gtja_038",
        "a101_026": "gtja_005",
        "a101_029": "gtja_157",
        "a101_033": "a158_001",
        "a101_044": "gtja_062",
        "a101_049": "a101_025",
        "a101_050": "gtja_016",
        "a101_055": "gtja_176",
        "a101_057": "gtja_124",
        "a101_084": "gtja_017",
        "a101_088": "gtja_140",
        "a101_094": "gtja_121",
        "a101_101": "a158_001",
        "a158_003": "a158_001",
        "a158_008": "gtja_124",
        "a158_010": "a158_001",
        "a158_013": "gtja_124",
        "a158_014": "a360_235",
        "a158_015": "a360_232",
        "a158_016": "a360_220",
        "a158_017": "a158_063",
        "a158_019": "a360_238",
        "a158_020": "a360_235",
        "a158_021": "a158_056",
        "a158_022": "a158_056",
        "a158_023": "a158_063",
        "a158_027": "a158_026",
        "a158_029": "gtja_008",
        "a158_030": "a360_232",
        "a158_031": "a360_220",
        "a158_046": "a158_056",
        "a158_050": "a158_049",
        "a158_052": "a158_051",
        "a158_053": "a158_063",
        "a158_055": "a360_235",
        "a158_057": "a158_056",
        "a158_059": "a360_238",
        "a158_060": "a360_235",
        "a158_061": "a360_232",
        "a158_062": "a158_051",
        "a158_064": "a158_054",
        "a158_065": "gtja_028",
        "a158_066": "a158_056",
        "a158_067": "a158_072",
        "a158_068": "a158_063",
        "a158_069": "a158_054",
        "a158_070": "gtja_028",
        "a158_071": "a158_056",
        "a158_073": "a158_063",
        "a158_074": "a158_084",
        "a158_099": "a158_109",
        "a158_100": "gtja_053",
        "a158_101": "gtja_058",
        "a158_104": "a158_109",
        "a158_110": "gtja_053",
        "a158_111": "gtja_058",
        "a158_114": "a360_235",
        "a158_115": "a360_232",
        "a158_116": "a360_220",
        "a158_117": "a158_058",
        "a158_119": "a360_235",
        "a158_120": "a360_232",
        "a158_121": "a360_220",
        "a158_122": "a158_058",
        "a158_124": "a360_235",
        "a158_125": "a360_232",
        "a158_126": "a360_220",
        "a158_127": "a158_058",
        "a158_130": "gtja_102",
        "a158_131": "gtja_102",
        "a158_132": "a158_133",
        "a158_149": "a158_144",
        "a158_150": "a158_145",
        "a158_151": "a158_146",
        "a158_152": "a158_147",
        "a158_153": "a158_148",
        "a158_154": "a158_144",
        "a158_155": "a158_145",
        "a158_156": "a158_146",
        "a158_157": "a158_147",
        "a158_158": "a158_148",
        "a360_002": "a360_186",
        "a360_003": "a360_186",
        "a360_004": "a360_186",
        "a360_005": "a360_186",
        "a360_006": "a360_186",
        "a360_007": "a360_186",
        "a360_008": "a360_186",
        "a360_009": "a158_063",
        "a360_010": "a158_063",
        "a360_011": "a158_063",
        "a360_012": "a158_063",
        "a360_013": "a158_063",
        "a360_014": "a158_063",
        "a360_015": "a158_063",
        "a360_016": "a158_058",
        "a360_017": "a158_058",
        "a360_018": "a158_058",
        "a360_019": "a158_058",
        "a360_020": "a158_058",
        "a360_021": "a158_058",
        "a360_022": "a158_058",
        "a360_023": "a158_058",
        "a360_024": "a158_058",
        "a360_025": "a158_058",
        "a360_026": "a158_058",
        "a360_027": "a158_058",
        "a360_028": "a158_058",
        "a360_029": "a158_058",
        "a360_030": "a158_063",
        "a360_031": "a158_058",
        "a360_032": "a158_058",
        "a360_033": "a360_220",
        "a360_034": "a360_220",
        "a360_035": "a360_220",
        "a360_036": "a360_220",
        "a360_037": "a360_220",
        "a360_038": "a360_220",
        "a360_039": "a360_220",
        "a360_040": "a360_220",
        "a360_041": "a360_220",
        "a360_042": "a360_220",
        "a360_043": "a360_220",
        "a360_044": "a360_220",
        "a360_045": "a158_056",
        "a360_046": "a158_056",
        "a360_047": "a158_056",
        "a360_048": "a158_056",
        "a360_049": "a158_056",
        "a360_050": "a360_232",
        "a360_051": "a360_232",
        "a360_052": "a360_232",
        "a360_053": "a360_235",
        "a360_054": "a360_235",
        "a360_055": "a360_235",
        "a360_056": "gtja_008",
        "a360_057": "a360_238",
        "a360_058": "a360_238",
        "a360_059": "a158_001",
        "a360_062": "a360_186",
        "a360_063": "a360_186",
        "a360_064": "a360_186",
        "a360_065": "a360_186",
        "a360_066": "a360_186",
        "a360_067": "a360_186",
        "a360_068": "a360_186",
        "a360_069": "a360_186",
        "a360_070": "a158_063",
        "a360_071": "a158_063",
        "a360_072": "a158_063",
        "a360_073": "a158_063",
        "a360_074": "a158_063",
        "a360_075": "a158_063",
        "a360_076": "a158_063",
        "a360_077": "a158_058",
        "a360_078": "a158_058",
        "a360_079": "a158_058",
        "a360_080": "a158_058",
        "a360_081": "a158_058",
        "a360_082": "a158_058",
        "a360_083": "a158_058",
        "a360_084": "a158_058",
        "a360_085": "a158_058",
        "a360_086": "a158_058",
        "a360_087": "a158_058",
        "a360_088": "a158_058",
        "a360_089": "a158_058",
        "a360_090": "a158_063",
        "a360_091": "a158_063",
        "a360_092": "a158_058",
        "a360_093": "a360_220",
        "a360_094": "a360_220",
        "a360_095": "a360_220",
        "a360_096": "a360_220",
        "a360_097": "a360_220",
        "a360_098": "a360_220",
        "a360_099": "a360_220",
        "a360_100": "a360_220",
        "a360_101": "a360_220",
        "a360_102": "a360_220",
        "a360_103": "a360_220",
        "a360_104": "a360_220",
        "a360_105": "a360_220",
        "a360_106": "a158_056",
        "a360_107": "a158_056",
        "a360_108": "a158_056",
        "a360_109": "a158_056",
        "a360_110": "a360_232",
        "a360_111": "a360_232",
        "a360_112": "a360_232",
        "a360_113": "a360_232",
        "a360_114": "a360_235",
        "a360_115": "a360_235",
        "a360_116": "a360_235",
        "a360_117": "gtja_008",
        "a360_118": "a360_238",
        "a360_119": "gtja_137",
        "a360_120": "a158_001",
        "a360_125": "a360_186",
        "a360_126": "a360_186",
        "a360_127": "a360_186",
        "a360_128": "a360_186",
        "a360_129": "a360_186",
        "a360_130": "a360_186",
        "a360_131": "a360_186",
        "a360_132": "a360_186",
        "a360_133": "a158_058",
        "a360_134": "a158_058",
        "a360_135": "a158_058",
        "a360_136": "a158_058",
        "a360_137": "a158_058",
        "a360_138": "a158_058",
        "a360_139": "a158_058",
        "a360_140": "a158_058",
        "a360_141": "a158_058",
        "a360_142": "a158_058",
        "a360_143": "a158_058",
        "a360_144": "a158_058",
        "a360_145": "a158_058",
        "a360_146": "a158_058",
        "a360_147": "a158_058",
        "a360_148": "a158_058",
        "a360_149": "a158_058",
        "a360_150": "a158_058",
        "a360_151": "a158_058",
        "a360_152": "a158_058",
        "a360_153": "a360_220",
        "a360_154": "a360_220",
        "a360_155": "a360_220",
        "a360_156": "a360_220",
        "a360_157": "a360_220",
        "a360_158": "a360_220",
        "a360_159": "a360_220",
        "a360_160": "a360_220",
        "a360_161": "a360_220",
        "a360_162": "a360_220",
        "a360_163": "a360_220",
        "a360_164": "a360_220",
        "a360_165": "a158_056",
        "a360_166": "a158_056",
        "a360_167": "a158_056",
        "a360_168": "a158_056",
        "a360_169": "a158_056",
        "a360_170": "a360_232",
        "a360_171": "a360_232",
        "a360_172": "a360_232",
        "a360_173": "a360_232",
        "a360_174": "a360_235",
        "a360_175": "a360_235",
        "a360_176": "gtja_008",
        "a360_177": "a158_054",
        "a360_178": "a158_054",
        "a360_181": "a360_186",
        "a360_182": "a360_186",
        "a360_183": "a360_186",
        "a360_184": "a360_186",
        "a360_185": "a360_186",
        "a360_187": "a158_063",
        "a360_188": "a158_063",
        "a360_189": "a158_063",
        "a360_190": "a158_063",
        "a360_191": "a158_063",
        "a360_192": "a158_063",
        "a360_193": "a158_063",
        "a360_194": "a158_063",
        "a360_195": "a158_063",
        "a360_196": "a158_063",
        "a360_197": "a158_063",
        "a360_198": "a158_063",
        "a360_199": "a158_063",
        "a360_200": "a158_063",
        "a360_201": "a158_063",
        "a360_202": "a158_063",
        "a360_203": "a158_063",
        "a360_204": "a158_063",
        "a360_205": "a158_063",
        "a360_206": "a158_063",
        "a360_207": "a158_063",
        "a360_208": "a158_063",
        "a360_209": "a158_063",
        "a360_210": "a158_063",
        "a360_211": "a158_063",
        "a360_212": "a158_063",
        "a360_213": "a360_220",
        "a360_214": "a360_220",
        "a360_215": "a360_220",
        "a360_216": "a360_220",
        "a360_217": "a360_220",
        "a360_218": "a360_220",
        "a360_219": "a360_220",
        "a360_221": "a360_220",
        "a360_222": "a360_220",
        "a360_223": "a360_220",
        "a360_224": "a360_220",
        "a360_225": "a360_220",
        "a360_226": "a158_056",
        "a360_227": "a158_056",
        "a360_228": "a158_056",
        "a360_229": "a360_232",
        "a360_230": "a360_232",
        "a360_231": "a360_232",
        "a360_233": "a360_235",
        "a360_234": "a360_235",
        "a360_236": "a360_235",
        "a360_237": "a360_238",
        "a360_240": "a158_012",
        "a360_242": "a360_186",
        "a360_243": "a360_186",
        "a360_244": "a360_186",
        "a360_245": "a360_186",
        "a360_246": "a360_186",
        "a360_247": "a360_186",
        "a360_248": "a360_186",
        "a360_249": "a360_186",
        "a360_250": "a158_063",
        "a360_251": "a158_063",
        "a360_252": "a158_063",
        "a360_253": "a158_063",
        "a360_254": "a158_063",
        "a360_255": "a158_063",
        "a360_256": "a158_058",
        "a360_257": "a158_058",
        "a360_258": "a158_058",
        "a360_259": "a158_058",
        "a360_260": "a158_058",
        "a360_261": "a158_058",
        "a360_262": "a158_058",
        "a360_263": "a158_058",
        "a360_264": "a158_058",
        "a360_265": "a158_058",
        "a360_266": "a158_058",
        "a360_267": "a158_058",
        "a360_268": "a158_058",
        "a360_269": "a158_058",
        "a360_270": "a158_063",
        "a360_271": "a158_058",
        "a360_272": "a158_058",
        "a360_273": "a360_220",
        "a360_274": "a360_220",
        "a360_275": "a360_220",
        "a360_276": "a360_220",
        "a360_277": "a360_220",
        "a360_278": "a360_220",
        "a360_279": "a360_220",
        "a360_280": "a360_220",
        "a360_281": "a360_220",
        "a360_282": "a360_220",
        "a360_283": "a360_220",
        "a360_284": "a360_220",
        "a360_285": "a158_056",
        "a360_286": "a158_056",
        "a360_287": "a158_056",
        "a360_288": "a158_056",
        "a360_289": "a158_056",
        "a360_290": "a360_232",
        "a360_291": "a360_232",
        "a360_292": "a360_232",
        "a360_293": "a360_235",
        "a360_294": "a360_235",
        "a360_295": "a360_235",
        "a360_296": "a360_235",
        "a360_297": "a360_238",
        "a360_298": "a360_238",
        "a360_299": "a360_239",
        "a360_300": "gtja_124",
        "a360_302": "a360_301",
        "a360_303": "a360_304",
        "a360_305": "a360_306",
        "a360_307": "a360_306",
        "a360_309": "a360_310",
        "a360_311": "a360_310",
        "a360_312": "a360_313",
        "a360_314": "a360_313",
        "a360_316": "a360_315",
        "a360_317": "a360_318",
        "a360_319": "a360_318",
        "a360_321": "a360_320",
        "a360_323": "a360_324",
        "a360_325": "a360_326",
        "a360_327": "a360_326",
        "a360_330": "a158_147",
        "a360_340": "a158_146",
        "a360_350": "a158_145",
        "a360_355": "a158_144",
        "gtja_004": "a101_021",
        "gtja_007": "a101_011",
        "gtja_010": "gtja_167",
        "gtja_014": "a360_235",
        "gtja_018": "a360_235",
        "gtja_019": "a360_235",
        "gtja_020": "a360_235",
        "gtja_021": "gtja_024",
        "gtja_027": "gtja_024",
        "gtja_031": "a360_235",
        "gtja_034": "a360_235",
        "gtja_037": "a101_008",
        "gtja_042": "a101_040",
        "gtja_046": "a360_235",
        "gtja_050": "gtja_049",
        "gtja_051": "gtja_049",
        "gtja_055": "gtja_059",
        "gtja_057": "gtja_028",
        "gtja_063": "a158_056",
        "gtja_065": "a360_238",
        "gtja_066": "a360_238",
        "gtja_067": "a158_063",
        "gtja_070": "gtja_150",
        "gtja_071": "a158_056",
        "gtja_078": "gtja_028",
        "gtja_079": "a158_056",
        "gtja_080": "a158_144",
        "gtja_086": "a101_046",
        "gtja_088": "a360_220",
        "gtja_095": "gtja_150",
        "gtja_096": "gtja_047",
        "gtja_106": "gtja_059",
        "gtja_110": "a360_220",
        "gtja_112": "a158_056",
        "gtja_114": "a101_083",
        "gtja_120": "a101_042",
        "gtja_132": "gtja_150",
        "gtja_158": "a158_002",
        "gtja_161": "gtja_167",
        "gtja_163": "a101_025",
        "gtja_164": "gtja_167",
        "gtja_168": "gtja_102",
        "gtja_174": "gtja_167",
        "gtja_175": "gtja_167",
        "gtja_177": "a158_076",
        "gtja_187": "gtja_167",
        "gtja_189": "gtja_167",
}

HQUALITY_SELECTION = FactorSelection(
    name="Hquality",
    source_factor_sets=("combined", "gtja191", "qlib_alpha360"),
    members=HQUALITY_MEMBERS,
    criteria={
        "min_year_days": 120,
        "min_positive_rank_ic_year_ratio": 0.7,
        "min_positive_long_short_year_ratio": 0.0,
        "min_abs_rank_ic": 0.01,
        "min_coverage": 0.8,
        "max_abs_rank_correlation": 0.8,
        "correlation_date_step": 5,
    },
    provenance={
        "horizon": "1d",
        "evaluation_start": "2010-01-04",
        "evaluation_end": "2026-10-19",
        "qlib_fingerprint": "4c48525d91abd4f4081a48179015fd9a9de899fed0158ef0b4db3a9880c009b8",
        "deduplicated": DEDUPLICATED,
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
