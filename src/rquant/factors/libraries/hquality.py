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
- Alpha360 is excluded (``excluded_factor_sets``): with its 42 surviving lag features the LGB backtest returned 16.2% a
  year with Sharpe 0.74, and without them 19.1% with Sharpe 0.83;
- coverage (published observations over the best-covered factor's, pooled over dates) is at least ``min_coverage``;
- duplicates are then pruned with ``prune_correlated``: the 268 factors passing the checks above are
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
    SelectedFactor("a158_052", 1),  # RankIC +0.0206, IC+ 14/17, cov 0.923
    SelectedFactor("a158_056", 1),  # RankIC +0.0227, IC+ 15/17, cov 0.948
    SelectedFactor("a158_058", 1),  # RankIC +0.0168, IC+ 14/17, cov 0.859
    SelectedFactor("a158_061", 1),  # RankIC +0.0219, IC+ 16/17, cov 0.948
    SelectedFactor("a158_063", 1),  # RankIC +0.0206, IC+ 16/17, cov 0.859
    SelectedFactor("a158_064", -1),  # RankIC -0.0201, IC+ 16/17, cov 0.989
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
    SelectedFactor("a158_114", -1),  # RankIC -0.0270, IC+ 15/17, cov 0.986
    SelectedFactor("a158_115", -1),  # RankIC -0.0161, IC+ 13/17, cov 0.972
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
    SelectedFactor("gtja_009", -1),  # RankIC -0.0262, IC+ 14/17, cov 0.997
    SelectedFactor("gtja_011", -1),  # RankIC -0.0218, IC+ 15/17, cov 0.984
    SelectedFactor("gtja_012", 1),  # RankIC +0.0219, IC+ 15/17, cov 0.974
    SelectedFactor("gtja_014", -1),  # RankIC -0.0327, IC+ 15/17, cov 0.994
    SelectedFactor("gtja_016", 1),  # RankIC +0.0135, IC+ 17/17, cov 0.977
    SelectedFactor("gtja_020", -1),  # RankIC -0.0292, IC+ 15/17, cov 0.994
    SelectedFactor("gtja_022", -1),  # RankIC -0.0223, IC+ 14/17, cov 0.977
    SelectedFactor("gtja_023", -1),  # RankIC -0.0143, IC+ 15/17, cov 0.948
    SelectedFactor("gtja_024", -1),  # RankIC -0.0212, IC+ 15/17, cov 0.994
    SelectedFactor("gtja_028", -1),  # RankIC -0.0204, IC+ 14/17, cov 0.977
    SelectedFactor("gtja_032", 1),  # RankIC +0.0114, IC+ 16/17, cov 0.989
    SelectedFactor("gtja_035", 1),  # RankIC +0.0138, IC+ 14/17, cov 0.940
    SelectedFactor("gtja_036", -1),  # RankIC -0.0126, IC+ 17/17, cov 0.983
    SelectedFactor("gtja_038", 1),  # RankIC +0.0217, IC+ 16/17, cov 0.948
    SelectedFactor("gtja_040", -1),  # RankIC -0.0178, IC+ 17/17, cov 0.930
    SelectedFactor("gtja_041", 1),  # RankIC +0.0254, IC+ 16/17, cov 0.980
    SelectedFactor("gtja_043", -1),  # RankIC -0.0238, IC+ 17/17, cov 0.983
    SelectedFactor("gtja_047", 1),  # RankIC +0.0115, IC+ 12/17, cov 0.986
    SelectedFactor("gtja_048", 1),  # RankIC +0.0225, IC+ 16/17, cov 0.948
    SelectedFactor("gtja_052", -1),  # RankIC -0.0149, IC+ 16/17, cov 0.930
    SelectedFactor("gtja_053", -1),  # RankIC -0.0150, IC+ 15/17, cov 0.966
    SelectedFactor("gtja_054", 1),  # RankIC +0.0162, IC+ 16/17, cov 0.974
    SelectedFactor("gtja_056", 1),  # RankIC +0.0111, IC+ 13/17, cov 0.841
    SelectedFactor("gtja_058", -1),  # RankIC -0.0126, IC+ 16/17, cov 0.945
    SelectedFactor("gtja_059", -1),  # RankIC -0.0167, IC+ 16/17, cov 0.945
    SelectedFactor("gtja_060", -1),  # RankIC -0.0117, IC+ 14/17, cov 0.942
    SelectedFactor("gtja_062", 1),  # RankIC +0.0115, IC+ 16/17, cov 0.989
    SelectedFactor("gtja_064", 1),  # RankIC +0.0114, IC+ 17/17, cov 0.813
    SelectedFactor("gtja_065", 1),  # RankIC +0.0286, IC+ 14/17, cov 0.986
    SelectedFactor("gtja_068", -1),  # RankIC -0.0201, IC+ 14/17, cov 0.997
    SelectedFactor("gtja_069", -1),  # RankIC -0.0130, IC+ 16/17, cov 0.945
    SelectedFactor("gtja_074", -1),  # RankIC -0.0121, IC+ 17/17, cov 0.850
    SelectedFactor("gtja_083", 1),  # RankIC +0.0120, IC+ 16/17, cov 0.989
    SelectedFactor("gtja_084", -1),  # RankIC -0.0166, IC+ 17/17, cov 0.945
    SelectedFactor("gtja_088", -1),  # RankIC -0.0176, IC+ 15/17, cov 0.990
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
    SelectedFactor("gtja_176", -1),  # RankIC -0.0149, IC+ 17/17, cov 0.955
    SelectedFactor("gtja_178", -1),  # RankIC -0.0227, IC+ 16/17, cov 0.997
    SelectedFactor("gtja_185", 1),  # RankIC +0.0114, IC+ 15/17, cov 1.000
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
        "a101_084": "gtja_014",
        "a101_088": "gtja_140",
        "a101_094": "gtja_121",
        "a101_101": "a158_001",
        "a158_003": "a158_001",
        "a158_008": "gtja_124",
        "a158_010": "a158_001",
        "a158_013": "gtja_124",
        "a158_014": "gtja_014",
        "a158_015": "a158_056",
        "a158_016": "gtja_088",
        "a158_017": "a158_063",
        "a158_019": "gtja_065",
        "a158_020": "gtja_020",
        "a158_021": "a158_056",
        "a158_022": "a158_061",
        "a158_023": "a158_063",
        "a158_027": "a158_026",
        "a158_029": "gtja_065",
        "a158_030": "gtja_024",
        "a158_031": "gtja_088",
        "a158_046": "a158_056",
        "a158_050": "a158_049",
        "a158_051": "a158_061",
        "a158_053": "a158_052",
        "a158_054": "gtja_065",
        "a158_055": "gtja_065",
        "a158_057": "a158_056",
        "a158_059": "gtja_065",
        "a158_060": "gtja_020",
        "a158_062": "a158_061",
        "a158_065": "gtja_065",
        "a158_066": "a158_061",
        "a158_067": "a158_072",
        "a158_068": "a158_063",
        "a158_069": "gtja_065",
        "a158_070": "gtja_028",
        "a158_071": "a158_061",
        "a158_073": "a158_063",
        "a158_074": "a158_084",
        "a158_099": "a158_109",
        "a158_100": "gtja_053",
        "a158_101": "gtja_058",
        "a158_104": "a158_109",
        "a158_110": "gtja_053",
        "a158_111": "gtja_058",
        "a158_116": "gtja_088",
        "a158_117": "a158_058",
        "a158_119": "a158_114",
        "a158_120": "a158_115",
        "a158_121": "gtja_088",
        "a158_122": "a158_058",
        "a158_124": "gtja_065",
        "a158_125": "a158_115",
        "a158_126": "gtja_088",
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
        "gtja_004": "a101_021",
        "gtja_007": "a101_011",
        "gtja_008": "gtja_014",
        "gtja_010": "gtja_167",
        "gtja_017": "gtja_014",
        "gtja_018": "gtja_014",
        "gtja_019": "gtja_014",
        "gtja_021": "gtja_024",
        "gtja_027": "a158_061",
        "gtja_029": "gtja_020",
        "gtja_031": "gtja_020",
        "gtja_034": "gtja_020",
        "gtja_037": "a101_008",
        "gtja_042": "a101_040",
        "gtja_046": "gtja_065",
        "gtja_049": "a158_115",
        "gtja_050": "a158_115",
        "gtja_051": "a158_115",
        "gtja_055": "gtja_088",
        "gtja_057": "gtja_028",
        "gtja_063": "a158_056",
        "gtja_066": "gtja_065",
        "gtja_067": "a158_063",
        "gtja_070": "gtja_150",
        "gtja_071": "a158_056",
        "gtja_078": "gtja_028",
        "gtja_079": "a158_061",
        "gtja_080": "a158_144",
        "gtja_086": "a101_046",
        "gtja_095": "gtja_150",
        "gtja_096": "gtja_047",
        "gtja_106": "gtja_088",
        "gtja_110": "gtja_088",
        "gtja_112": "a158_115",
        "gtja_114": "a101_083",
        "gtja_120": "a101_042",
        "gtja_132": "gtja_150",
        "gtja_158": "a158_002",
        "gtja_161": "gtja_167",
        "gtja_163": "a101_025",
        "gtja_164": "gtja_167",
        "gtja_168": "gtja_102",
        "gtja_170": "gtja_014",
        "gtja_174": "gtja_167",
        "gtja_175": "gtja_167",
        "gtja_177": "a158_076",
        "gtja_187": "gtja_167",
        "gtja_189": "gtja_167",
}

HQUALITY_SELECTION = FactorSelection(
    name="Hquality",
    source_factor_sets=("combined", "gtja191"),
    members=HQUALITY_MEMBERS,
    criteria={
        "min_year_days": 120,
        "min_positive_rank_ic_year_ratio": 0.7,
        "min_positive_long_short_year_ratio": 0.0,
        "min_abs_rank_ic": 0.01,
        "min_coverage": 0.8,
        "max_abs_rank_correlation": 0.8,
        "correlation_date_step": 5,
        "excluded_factor_sets": ("qlib_alpha360",),
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
