"""Locked membership of the recent_sh_main factor set.

recent_sh_main applies the recent rule to the Shanghai main-board universe (``sh_main``): it keeps factors whose
single-factor Rank IC on that universe is still significant and stable since 2020. It was selected with
``rquant.factors.selection`` from the 1d evaluations listed in ``provenance``
(``rquant --universe sh_main factors evaluate --horizon 1d --start 2020-01-01 --end 2026-10-21`` on each of the twelve
library factor sets):

- direction is the sign of the 2020+ mean daily Rank IC; every yearly statistic is oriented by it;
- years with at least ``min_year_days`` evaluated days are counted (2020-2026; 2026 is partial);
- the oriented annual Rank IC is positive in at least ``min_positive_rank_ic_year_ratio`` of counted years
  (5 of 7, one year looser than recent);
- the yearly long-short return is not required (``min_positive_long_short_year_ratio`` is 0);
- the long side must work, because the portfolio is long-only: the oriented long quantile (top 20% for direction +1,
  bottom 20% for -1) beats the equal-weight sh_main universe in at least ``min_positive_long_excess_year_ratio`` of
  counted years (5 of 7) and on average over them. On sh_main most of a factor's Rank IC comes from
  the short side (median yearly excess: long +1.6%, short +14.7% for the factors passing the Rank IC checks), which a
  long-only top-50 book cannot use;
- the absolute 2020+ Rank IC is at least ``min_abs_rank_ic`` and its t-statistic (daily Rank ICIR times the square root
  of the number of days) is at least ``min_abs_rank_ic_t_stat`` in absolute value;
- coverage (published observations over the best-covered factor's, pooled over dates and all twelve evaluations) is at
  least ``min_coverage``;
- duplicates are then pruned with ``prune_correlated``: the 491 factors passing the checks above are ordered
  by mean yearly long excess, and a factor is dropped when its absolute mean daily cross-sectional Spearman correlation
  (``mean_rank_correlation`` on every fifth trading date since 2020-01-02) with an already kept factor reaches
  ``max_abs_rank_correlation``. ``DEDUPLICATED`` maps each dropped factor to the kept one it duplicates.

Kept per source: qlib_alpha158 24, wq_alpha101 26, gtja191 27, qlib_alpha360 11, china_chars 7, size_value_china 2,
talib 9, jq240 13, oap 9, barra_cne6 2, tushare_tech 3, jkp153 7.

The selection uses 2020-2026 sh_main returns, so walk-forward test years from 2020 on are in-sample for membership, and
the membership is only meant for the ``sh_main`` universe.

Members keep their source canonical names and values. ``direction == -1`` marks a factor used inverted; the stored
column is not negated. Changing membership changes the catalog and engine fingerprints, so a stale store is refused
rather than reused.
"""

from __future__ import annotations

from rquant.factors.libraries.base import FactorSelection, SelectedFactor

RECENT_SH_MAIN_MEMBERS = (
    SelectedFactor("a158_001", -1),  # RankIC -0.0282, t -8.2, IC+ 7/7, cov 1.000
    SelectedFactor("a158_006", -1),  # RankIC -0.0247, t -9.4, IC+ 7/7, cov 1.000
    SelectedFactor("a158_012", 1),  # RankIC +0.0448, t +14.6, IC+ 7/7, cov 1.000
    SelectedFactor("a158_013", 1),  # RankIC +0.0272, t +8.2, IC+ 6/7, cov 1.000
    SelectedFactor("a158_024", -1),  # RankIC -0.0414, t -10.4, IC+ 7/7, cov 0.999
    SelectedFactor("a158_025", -1),  # RankIC -0.0398, t -9.3, IC+ 7/7, cov 0.998
    SelectedFactor("a158_028", -1),  # RankIC -0.0229, t -4.8, IC+ 7/7, cov 0.984
    SelectedFactor("a158_058", 1),  # RankIC +0.0296, t +6.6, IC+ 7/7, cov 0.984
    SelectedFactor("a158_079", 1),  # RankIC +0.0145, t +4.9, IC+ 6/7, cov 0.999
    SelectedFactor("a158_080", 1),  # RankIC +0.0127, t +4.2, IC+ 7/7, cov 0.998
    SelectedFactor("a158_084", -1),  # RankIC -0.0177, t -5.5, IC+ 7/7, cov 0.999
    SelectedFactor("a158_087", -1),  # RankIC -0.0243, t -7.1, IC+ 7/7, cov 0.992
    SelectedFactor("a158_088", -1),  # RankIC -0.0199, t -5.6, IC+ 7/7, cov 0.984
    SelectedFactor("a158_089", -1),  # RankIC -0.0259, t -12.1, IC+ 7/7, cov 1.000
    SelectedFactor("a158_090", -1),  # RankIC -0.0307, t -13.1, IC+ 7/7, cov 1.000
    SelectedFactor("a158_091", -1),  # RankIC -0.0272, t -10.6, IC+ 7/7, cov 1.000
    SelectedFactor("a158_098", -1),  # RankIC -0.0195, t -8.6, IC+ 7/7, cov 1.000
    SelectedFactor("a158_112", -1),  # RankIC -0.0124, t -4.0, IC+ 7/7, cov 1.000
    SelectedFactor("a158_134", -1),  # RankIC -0.0141, t -8.5, IC+ 7/7, cov 0.999
    SelectedFactor("a158_138", 1),  # RankIC +0.0240, t +8.4, IC+ 7/7, cov 0.984
    SelectedFactor("a158_140", -1),  # RankIC -0.0120, t -7.4, IC+ 7/7, cov 0.997
    SelectedFactor("a158_142", -1),  # RankIC -0.0109, t -6.2, IC+ 6/7, cov 0.992
    SelectedFactor("a158_146", -1),  # RankIC -0.0281, t -11.3, IC+ 7/7, cov 0.995
    SelectedFactor("a158_148", -1),  # RankIC -0.0333, t -12.5, IC+ 7/7, cov 0.984
    SelectedFactor("a101_002", 1),  # RankIC +0.0143, t +9.6, IC+ 7/7, cov 1.000
    SelectedFactor("a101_015", 1),  # RankIC +0.0219, t +15.2, IC+ 7/7, cov 1.000
    SelectedFactor("a101_018", 1),  # RankIC +0.0114, t +4.8, IC+ 7/7, cov 0.999
    SelectedFactor("a101_020", -1),  # RankIC -0.0134, t -5.2, IC+ 7/7, cov 1.000
    SelectedFactor("a101_026", 1),  # RankIC +0.0211, t +11.4, IC+ 7/7, cov 1.000
    SelectedFactor("a101_027", 1),  # RankIC +0.0198, t +12.9, IC+ 7/7, cov 1.000
    SelectedFactor("a101_034", 1),  # RankIC +0.0167, t +6.4, IC+ 6/7, cov 1.000
    SelectedFactor("a101_037", 1),  # RankIC +0.0235, t +9.2, IC+ 7/7, cov 0.933
    SelectedFactor("a101_038", 1),  # RankIC +0.0214, t +6.2, IC+ 7/7, cov 0.998
    SelectedFactor("a101_040", 1),  # RankIC +0.0418, t +14.7, IC+ 7/7, cov 0.998
    SelectedFactor("a101_044", 1),  # RankIC +0.0272, t +16.0, IC+ 7/7, cov 1.000
    SelectedFactor("a101_046", 1),  # RankIC +0.0169, t +6.4, IC+ 7/7, cov 1.000
    SelectedFactor("a101_056", 1),  # RankIC +0.0119, t +4.7, IC+ 6/7, cov 0.997
    SelectedFactor("a101_058", 1),  # RankIC +0.0102, t +8.6, IC+ 7/7, cov 0.992
    SelectedFactor("a101_059", 1),  # RankIC +0.0114, t +10.0, IC+ 7/7, cov 0.987
    SelectedFactor("a101_060", 1),  # RankIC +0.0123, t +4.3, IC+ 5/7, cov 0.998
    SelectedFactor("a101_061", 1),  # RankIC +0.0156, t +7.7, IC+ 7/7, cov 1.000
    SelectedFactor("a101_062", 1),  # RankIC +0.0101, t +6.1, IC+ 7/7, cov 1.000
    SelectedFactor("a101_065", 1),  # RankIC +0.0154, t +7.1, IC+ 7/7, cov 1.000
    SelectedFactor("a101_066", 1),  # RankIC +0.0167, t +6.5, IC+ 6/7, cov 0.997
    SelectedFactor("a101_069", 1),  # RankIC +0.0247, t +11.8, IC+ 7/7, cov 0.988
    SelectedFactor("a101_071", 1),  # RankIC +0.0155, t +7.5, IC+ 7/7, cov 0.995
    SelectedFactor("a101_074", 1),  # RankIC +0.0105, t +7.1, IC+ 7/7, cov 1.000
    SelectedFactor("a101_086", 1),  # RankIC +0.0121, t +6.1, IC+ 7/7, cov 1.000
    SelectedFactor("a101_088", 1),  # RankIC +0.0478, t +16.7, IC+ 7/7, cov 0.998
    SelectedFactor("a101_095", 1),  # RankIC +0.0111, t +6.9, IC+ 7/7, cov 1.000
    SelectedFactor("gtja_001", 1),  # RankIC +0.0188, t +10.9, IC+ 7/7, cov 0.998
    SelectedFactor("gtja_003", -1),  # RankIC -0.0265, t -8.1, IC+ 7/7, cov 0.998
    SelectedFactor("gtja_007", 1),  # RankIC +0.0130, t +6.7, IC+ 7/7, cov 0.999
    SelectedFactor("gtja_010", -1),  # RankIC -0.0216, t -6.5, IC+ 7/7, cov 0.994
    SelectedFactor("gtja_011", -1),  # RankIC -0.0156, t -5.3, IC+ 6/7, cov 0.992
    SelectedFactor("gtja_015", 1),  # RankIC +0.0126, t +5.8, IC+ 7/7, cov 1.000
    SelectedFactor("gtja_016", 1),  # RankIC +0.0235, t +13.9, IC+ 7/7, cov 0.998
    SelectedFactor("gtja_039", 1),  # RankIC +0.0122, t +4.4, IC+ 6/7, cov 0.919
    SelectedFactor("gtja_040", -1),  # RankIC -0.0255, t -8.3, IC+ 7/7, cov 0.993
    SelectedFactor("gtja_041", 1),  # RankIC +0.0382, t +11.8, IC+ 7/7, cov 0.998
    SelectedFactor("gtja_056", 1),  # RankIC +0.0149, t +7.1, IC+ 7/7, cov 0.981
    SelectedFactor("gtja_061", 1),  # RankIC +0.0268, t +10.1, IC+ 7/7, cov 0.968
    SelectedFactor("gtja_064", 1),  # RankIC +0.0206, t +14.5, IC+ 7/7, cov 0.974
    SelectedFactor("gtja_069", -1),  # RankIC -0.0291, t -8.7, IC+ 7/7, cov 0.995
    SelectedFactor("gtja_082", 1),  # RankIC +0.0139, t +3.6, IC+ 7/7, cov 0.999
    SelectedFactor("gtja_090", 1),  # RankIC +0.0236, t +14.6, IC+ 7/7, cov 0.999
    SelectedFactor("gtja_094", -1),  # RankIC -0.0249, t -8.3, IC+ 7/7, cov 0.992
    SelectedFactor("gtja_105", 1),  # RankIC +0.0170, t +10.5, IC+ 7/7, cov 0.998
    SelectedFactor("gtja_111", 1),  # RankIC +0.0185, t +6.5, IC+ 6/7, cov 0.998
    SelectedFactor("gtja_120", 1),  # RankIC +0.0357, t +11.5, IC+ 7/7, cov 1.000
    SelectedFactor("gtja_139", 1),  # RankIC +0.0185, t +8.5, IC+ 7/7, cov 0.998
    SelectedFactor("gtja_148", 1),  # RankIC +0.0165, t +7.6, IC+ 7/7, cov 0.980
    SelectedFactor("gtja_156", 1),  # RankIC +0.0313, t +11.2, IC+ 7/7, cov 0.998
    SelectedFactor("gtja_157", 1),  # RankIC +0.0235, t +7.9, IC+ 7/7, cov 0.997
    SelectedFactor("gtja_176", -1),  # RankIC -0.0214, t -14.8, IC+ 7/7, cov 0.996
    SelectedFactor("gtja_183", 1),  # RankIC +0.0105, t +4.7, IC+ 6/7, cov 0.981
    SelectedFactor("gtja_189", -1),  # RankIC -0.0370, t -9.6, IC+ 7/7, cov 0.997
    SelectedFactor("a360_071", 1),  # RankIC +0.0330, t +7.7, IC+ 7/7, cov 0.999
    SelectedFactor("a360_302", 1),  # RankIC +0.0350, t +13.2, IC+ 7/7, cov 0.999
    SelectedFactor("a360_306", 1),  # RankIC +0.0363, t +13.7, IC+ 7/7, cov 0.999
    SelectedFactor("a360_310", 1),  # RankIC +0.0369, t +13.7, IC+ 7/7, cov 0.999
    SelectedFactor("a360_313", 1),  # RankIC +0.0374, t +14.0, IC+ 7/7, cov 0.999
    SelectedFactor("a360_316", 1),  # RankIC +0.0355, t +13.2, IC+ 7/7, cov 0.999
    SelectedFactor("a360_319", 1),  # RankIC +0.0347, t +13.0, IC+ 7/7, cov 0.999
    SelectedFactor("a360_323", 1),  # RankIC +0.0321, t +12.3, IC+ 7/7, cov 0.999
    SelectedFactor("a360_326", 1),  # RankIC +0.0333, t +12.9, IC+ 7/7, cov 0.999
    SelectedFactor("a360_338", 1),  # RankIC +0.0314, t +12.7, IC+ 7/7, cov 0.999
    SelectedFactor("a360_343", 1),  # RankIC +0.0283, t +11.7, IC+ 7/7, cov 0.999
    SelectedFactor("cnc_006", -1),  # RankIC -0.0261, t -5.7, IC+ 7/7, cov 1.000
    SelectedFactor("cnc_008", -1),  # RankIC -0.0335, t -11.6, IC+ 7/7, cov 0.997
    SelectedFactor("cnc_019", -1),  # RankIC -0.0129, t -7.2, IC+ 7/7, cov 1.000
    SelectedFactor("cnc_038", -1),  # RankIC -0.0175, t -4.2, IC+ 6/7, cov 1.000
    SelectedFactor("cnc_068", 1),  # RankIC +0.0145, t +5.1, IC+ 7/7, cov 0.992
    SelectedFactor("cnc_078", -1),  # RankIC -0.0149, t -5.1, IC+ 7/7, cov 0.976
    SelectedFactor("cnc_106", 1),  # RankIC +0.0221, t +6.2, IC+ 7/7, cov 0.956
    SelectedFactor("svc_013", 1),  # RankIC +0.0301, t +11.9, IC+ 7/7, cov 1.000
    SelectedFactor("svc_021", 1),  # RankIC +0.0301, t +10.1, IC+ 7/7, cov 0.991
    SelectedFactor("ta_004", -1),  # RankIC -0.0234, t -7.9, IC+ 7/7, cov 1.000
    SelectedFactor("ta_008", -1),  # RankIC -0.0159, t -6.3, IC+ 7/7, cov 1.000
    SelectedFactor("ta_015", -1),  # RankIC -0.0411, t -11.8, IC+ 7/7, cov 1.000
    SelectedFactor("ta_058", -1),  # RankIC -0.0107, t -6.4, IC+ 6/7, cov 1.000
    SelectedFactor("ta_059", -1),  # RankIC -0.0253, t -9.6, IC+ 7/7, cov 1.000
    SelectedFactor("ta_063", -1),  # RankIC -0.0329, t -7.3, IC+ 7/7, cov 1.000
    SelectedFactor("ta_074", -1),  # RankIC -0.0104, t -5.5, IC+ 6/7, cov 1.000
    SelectedFactor("ta_076", -1),  # RankIC -0.0182, t -9.9, IC+ 7/7, cov 1.000
    SelectedFactor("ta_103", -1),  # RankIC -0.0130, t -6.2, IC+ 7/7, cov 1.000
    SelectedFactor("jq_002", 1),  # RankIC +0.0148, t +6.0, IC+ 7/7, cov 0.992
    SelectedFactor("jq_103", -1),  # RankIC -0.0346, t -10.4, IC+ 7/7, cov 1.000
    SelectedFactor("jq_104", -1),  # RankIC -0.0135, t -7.4, IC+ 7/7, cov 1.000
    SelectedFactor("jq_108", -1),  # RankIC -0.0360, t -12.2, IC+ 7/7, cov 0.993
    SelectedFactor("jq_111", -1),  # RankIC -0.0255, t -8.0, IC+ 7/7, cov 1.000
    SelectedFactor("jq_114", -1),  # RankIC -0.0286, t -14.3, IC+ 7/7, cov 1.000
    SelectedFactor("jq_116", -1),  # RankIC -0.0436, t -10.5, IC+ 7/7, cov 1.000
    SelectedFactor("jq_137", -1),  # RankIC -0.0346, t -11.6, IC+ 7/7, cov 1.000
    SelectedFactor("jq_154", -1),  # RankIC -0.0186, t -8.4, IC+ 7/7, cov 0.993
    SelectedFactor("jq_155", -1),  # RankIC -0.0198, t -10.3, IC+ 7/7, cov 1.000
    SelectedFactor("jq_159", -1),  # RankIC -0.0406, t -7.9, IC+ 7/7, cov 1.000
    SelectedFactor("jq_227", -1),  # RankIC -0.0422, t -12.9, IC+ 7/7, cov 1.000
    SelectedFactor("jq_230", -1),  # RankIC -0.0361, t -8.7, IC+ 7/7, cov 1.000
    SelectedFactor("oap_012", -1),  # RankIC -0.0105, t -3.9, IC+ 6/7, cov 0.901
    SelectedFactor("oap_018", -1),  # RankIC -0.0532, t -13.6, IC+ 7/7, cov 0.999
    SelectedFactor("oap_024", -1),  # RankIC -0.0135, t -4.1, IC+ 6/7, cov 0.989
    SelectedFactor("oap_043", -1),  # RankIC -0.0166, t -5.8, IC+ 7/7, cov 0.923
    SelectedFactor("oap_057", 1),  # RankIC +0.0222, t +7.2, IC+ 7/7, cov 1.000
    SelectedFactor("oap_060", -1),  # RankIC -0.0212, t -5.4, IC+ 7/7, cov 0.917
    SelectedFactor("oap_069", 1),  # RankIC +0.0240, t +8.8, IC+ 7/7, cov 0.986
    SelectedFactor("oap_123", -1),  # RankIC -0.0127, t -4.6, IC+ 7/7, cov 0.960
    SelectedFactor("oap_129", 1),  # RankIC +0.0102, t +5.1, IC+ 7/7, cov 0.817
    SelectedFactor("cne6_002", -1),  # RankIC -0.0139, t -3.5, IC+ 7/7, cov 1.000
    SelectedFactor("cne6_013", 1),  # RankIC +0.0328, t +10.5, IC+ 7/7, cov 0.996
    SelectedFactor("tst_002", -1),  # RankIC -0.0251, t -6.2, IC+ 7/7, cov 1.000
    SelectedFactor("tst_028", 1),  # RankIC +0.0321, t +7.4, IC+ 7/7, cov 0.977
    SelectedFactor("tst_066", -1),  # RankIC -0.0207, t -4.9, IC+ 7/7, cov 1.000
    SelectedFactor("jkp_016", 1),  # RankIC +0.0183, t +5.4, IC+ 7/7, cov 1.000
    SelectedFactor("jkp_017", 1),  # RankIC +0.0150, t +6.6, IC+ 7/7, cov 0.992
    SelectedFactor("jkp_026", -1),  # RankIC -0.0237, t -5.8, IC+ 7/7, cov 0.971
    SelectedFactor("jkp_111", -1),  # RankIC -0.0229, t -4.7, IC+ 7/7, cov 0.896
    SelectedFactor("jkp_124", -1),  # RankIC -0.0186, t -11.5, IC+ 7/7, cov 0.999
    SelectedFactor("jkp_133", 1),  # RankIC +0.0129, t +5.6, IC+ 7/7, cov 0.896
    SelectedFactor("jkp_135", -1),  # RankIC -0.0324, t -9.4, IC+ 7/7, cov 0.999
)

DEDUPLICATED = {
    "a158_003": "a158_001",
    "a158_008": "a158_013",
    "a158_010": "a158_001",
    "a158_018": "a360_071",
    "a158_023": "a158_058",
    "a158_026": "ta_063",
    "a158_027": "ta_063",
    "a158_033": "a360_071",
    "a158_063": "a360_071",
    "a158_092": "ta_059",
    "a158_117": "a158_058",
    "a158_118": "a360_071",
    "a158_122": "a158_058",
    "a158_123": "a360_071",
    "a158_127": "a158_058",
    "a158_128": "a360_071",
    "a158_133": "a158_138",
    "a158_151": "a158_146",
    "a158_153": "a158_148",
    "a158_156": "a158_146",
    "a158_158": "a158_148",
    "a101_003": "gtja_105",
    "a101_006": "gtja_139",
    "a101_011": "gtja_007",
    "a101_014": "gtja_139",
    "a101_033": "a158_001",
    "a101_042": "gtja_120",
    "a101_050": "gtja_016",
    "a101_055": "gtja_176",
    "a101_057": "a158_013",
    "a101_083": "a158_013",
    "a101_101": "a158_001",
    "gtja_005": "a101_026",
    "gtja_032": "a101_015",
    "gtja_036": "a101_027",
    "gtja_042": "a101_040",
    "gtja_054": "a101_018",
    "gtja_062": "a101_044",
    "gtja_067": "a158_058",
    "gtja_070": "jq_116",
    "gtja_081": "jq_137",
    "gtja_086": "a101_046",
    "gtja_093": "gtja_189",
    "gtja_095": "jq_116",
    "gtja_097": "jq_137",
    "gtja_100": "jq_137",
    "gtja_101": "a101_074",
    "gtja_107": "a101_020",
    "gtja_114": "a158_013",
    "gtja_118": "jq_103",
    "gtja_124": "a158_013",
    "gtja_126": "gtja_010",
    "gtja_129": "gtja_189",
    "gtja_132": "jq_116",
    "gtja_135": "a360_071",
    "gtja_136": "gtja_139",
    "gtja_144": "jq_116",
    "gtja_150": "jq_116",
    "gtja_151": "tst_066",
    "gtja_153": "gtja_010",
    "gtja_160": "gtja_189",
    "gtja_161": "gtja_189",
    "gtja_164": "gtja_189",
    "gtja_167": "gtja_189",
    "gtja_173": "gtja_010",
    "gtja_174": "gtja_189",
    "gtja_175": "gtja_189",
    "gtja_184": "a101_037",
    "gtja_187": "gtja_189",
    "a360_001": "a360_071",
    "a360_002": "a360_071",
    "a360_003": "a360_071",
    "a360_004": "a360_071",
    "a360_005": "a360_071",
    "a360_006": "a360_071",
    "a360_007": "a360_071",
    "a360_008": "a360_071",
    "a360_009": "a360_071",
    "a360_010": "a360_071",
    "a360_011": "a360_071",
    "a360_012": "a360_071",
    "a360_013": "a360_071",
    "a360_014": "a360_071",
    "a360_015": "a360_071",
    "a360_016": "a360_071",
    "a360_017": "a360_071",
    "a360_018": "a360_071",
    "a360_019": "a360_071",
    "a360_020": "a360_071",
    "a360_021": "a158_058",
    "a360_022": "a158_058",
    "a360_023": "a158_058",
    "a360_024": "a158_058",
    "a360_025": "a158_058",
    "a360_026": "tst_002",
    "a360_027": "tst_002",
    "a360_029": "tst_002",
    "a360_061": "a360_071",
    "a360_062": "a360_071",
    "a360_063": "a360_071",
    "a360_064": "a360_071",
    "a360_065": "a360_071",
    "a360_066": "a360_071",
    "a360_067": "a360_071",
    "a360_068": "a360_071",
    "a360_069": "a360_071",
    "a360_070": "a360_071",
    "a360_072": "a360_071",
    "a360_073": "a360_071",
    "a360_074": "a360_071",
    "a360_075": "a360_071",
    "a360_076": "a360_071",
    "a360_077": "a360_071",
    "a360_078": "a360_071",
    "a360_079": "a360_071",
    "a360_080": "a360_071",
    "a360_081": "a360_071",
    "a360_082": "a158_058",
    "a360_083": "a158_058",
    "a360_084": "a158_058",
    "a360_085": "a158_058",
    "a360_086": "tst_002",
    "a360_087": "tst_002",
    "a360_088": "tst_002",
    "a360_090": "tst_002",
    "a360_092": "tst_002",
    "a360_100": "svc_021",
    "a360_120": "a158_001",
    "a360_121": "a360_071",
    "a360_122": "a360_071",
    "a360_123": "a360_071",
    "a360_124": "a360_071",
    "a360_125": "a360_071",
    "a360_126": "a360_071",
    "a360_127": "a360_071",
    "a360_128": "a360_071",
    "a360_129": "a360_071",
    "a360_130": "a360_071",
    "a360_131": "a360_071",
    "a360_132": "a360_071",
    "a360_133": "a360_071",
    "a360_134": "a360_071",
    "a360_135": "a360_071",
    "a360_136": "a360_071",
    "a360_137": "a360_071",
    "a360_138": "a360_071",
    "a360_139": "a360_071",
    "a360_140": "a360_071",
    "a360_141": "a158_058",
    "a360_142": "a158_058",
    "a360_143": "a158_058",
    "a360_144": "a158_058",
    "a360_145": "a158_058",
    "a360_146": "tst_002",
    "a360_147": "tst_002",
    "a360_148": "tst_002",
    "a360_181": "a360_071",
    "a360_182": "a360_071",
    "a360_183": "a360_071",
    "a360_184": "a360_071",
    "a360_185": "a360_071",
    "a360_186": "a360_071",
    "a360_187": "a360_071",
    "a360_188": "a360_071",
    "a360_189": "a360_071",
    "a360_190": "a360_071",
    "a360_191": "a360_071",
    "a360_192": "a360_071",
    "a360_193": "a360_071",
    "a360_194": "a360_071",
    "a360_195": "a360_071",
    "a360_196": "a360_071",
    "a360_197": "a360_071",
    "a360_198": "a360_071",
    "a360_199": "a360_071",
    "a360_200": "a360_071",
    "a360_201": "a360_071",
    "a360_202": "a158_058",
    "a360_203": "a158_058",
    "a360_204": "a158_058",
    "a360_205": "a158_058",
    "a360_206": "tst_002",
    "a360_207": "tst_002",
    "a360_208": "tst_002",
    "a360_209": "tst_002",
    "a360_212": "tst_002",
    "a360_219": "jkp_135",
    "a360_240": "a158_012",
    "a360_241": "a360_071",
    "a360_242": "a360_071",
    "a360_243": "a360_071",
    "a360_244": "a360_071",
    "a360_245": "a360_071",
    "a360_246": "a360_071",
    "a360_247": "a360_071",
    "a360_248": "a360_071",
    "a360_249": "a360_071",
    "a360_250": "a360_071",
    "a360_251": "a360_071",
    "a360_252": "a360_071",
    "a360_253": "a360_071",
    "a360_254": "a360_071",
    "a360_255": "a360_071",
    "a360_256": "a360_071",
    "a360_257": "a360_071",
    "a360_258": "a360_071",
    "a360_259": "a360_071",
    "a360_260": "a360_071",
    "a360_261": "a360_071",
    "a360_262": "a158_058",
    "a360_263": "a158_058",
    "a360_264": "a158_058",
    "a360_265": "a158_058",
    "a360_266": "tst_002",
    "a360_267": "tst_002",
    "a360_268": "tst_002",
    "a360_269": "tst_002",
    "a360_300": "a158_013",
    "a360_301": "a360_302",
    "a360_303": "a360_302",
    "a360_304": "a360_306",
    "a360_305": "a360_306",
    "a360_307": "a360_306",
    "a360_308": "a360_306",
    "a360_309": "a360_310",
    "a360_311": "a360_310",
    "a360_312": "a360_313",
    "a360_314": "a360_313",
    "a360_315": "a360_313",
    "a360_317": "a360_316",
    "a360_318": "a360_316",
    "a360_320": "a360_319",
    "a360_321": "a360_319",
    "a360_325": "a360_326",
    "a360_327": "a360_326",
    "a360_339": "a360_338",
    "cnc_003": "jq_230",
    "cnc_004": "cnc_006",
    "cnc_009": "jq_116",
    "cnc_010": "jq_116",
    "cnc_011": "jq_116",
    "cnc_012": "jq_116",
    "cnc_014": "jq_116",
    "cnc_015": "jq_227",
    "cnc_016": "jq_227",
    "cnc_017": "jq_159",
    "cnc_020": "jq_154",
    "cnc_037": "oap_018",
    "cnc_061": "cnc_106",
    "cnc_063": "cnc_106",
    "cnc_065": "oap_060",
    "cnc_067": "oap_060",
    "cnc_069": "jq_002",
    "cnc_071": "cnc_106",
    "svc_003": "cnc_008",
    "svc_014": "svc_021",
    "ta_005": "ta_004",
    "ta_006": "a158_001",
    "ta_009": "a158_058",
    "ta_010": "tst_002",
    "ta_016": "oap_018",
    "ta_030": "ta_063",
    "ta_050": "a158_001",
    "ta_051": "a158_013",
    "jq_069": "jq_002",
    "jq_097": "cnc_068",
    "jq_109": "cnc_008",
    "jq_110": "jq_108",
    "jq_112": "jq_116",
    "jq_115": "jq_116",
    "jq_117": "jq_116",
    "jq_118": "jq_116",
    "jq_121": "jq_137",
    "jq_122": "jq_137",
    "jq_123": "jq_137",
    "jq_124": "jq_137",
    "jq_126": "jq_230",
    "jq_128": "jq_230",
    "jq_130": "jq_230",
    "jq_131": "cnc_006",
    "jq_136": "jq_137",
    "jq_138": "jq_111",
    "jq_153": "a360_071",
    "jq_157": "jq_159",
    "jq_158": "oap_018",
    "jq_179": "a360_071",
    "jq_183": "a360_071",
    "jq_186": "a158_058",
    "jq_189": "cnc_038",
    "jq_197": "a158_058",
    "jq_204": "tst_028",
    "jq_210": "tst_028",
    "jq_211": "a158_058",
    "jq_214": "tst_028",
    "jq_217": "a360_071",
    "jq_221": "a158_058",
    "jq_228": "cne6_002",
    "jq_229": "cnc_106",
    "jq_231": "oap_060",
    "oap_002": "cnc_038",
    "oap_015": "oap_018",
    "oap_016": "jq_155",
    "oap_019": "oap_018",
    "oap_022": "jkp_111",
    "oap_037": "jq_116",
    "oap_044": "jq_230",
    "oap_048": "cnc_106",
    "oap_054": "oap_060",
    "oap_055": "jq_002",
    "oap_056": "cnc_068",
    "oap_063": "jkp_017",
    "oap_064": "jkp_026",
    "cne6_004": "jq_159",
    "cne6_005": "jq_159",
    "cne6_007": "jq_230",
    "cne6_008": "cnc_006",
    "cne6_017": "cnc_106",
    "cne6_019": "jq_002",
    "cne6_020": "oap_060",
    "cne6_038": "jkp_016",
    "tst_001": "tst_002",
    "tst_010": "jq_103",
    "tst_014": "tst_002",
    "tst_015": "a360_071",
    "tst_026": "a158_058",
    "tst_027": "a360_071",
    "tst_032": "a158_058",
    "tst_043": "a158_058",
    "tst_044": "a360_071",
    "tst_045": "tst_028",
    "tst_046": "a158_058",
    "tst_047": "tst_002",
    "tst_061": "a158_058",
    "tst_065": "tst_002",
    "tst_067": "gtja_040",
    "jkp_010": "cnc_106",
    "jkp_012": "cnc_106",
    "jkp_021": "cnc_068",
    "jkp_022": "jq_002",
    "jkp_024": "cnc_106",
    "jkp_037": "cnc_078",
    "jkp_098": "cnc_106",
    "jkp_106": "oap_018",
    "jkp_108": "oap_018",
    "jkp_109": "oap_018",
    "jkp_115": "jq_116",
    "jkp_117": "cnc_038",
    "jkp_119": "jq_230",
    "jkp_122": "oap_018",
    "jkp_123": "jq_155",
    "jkp_134": "oap_018",
}

RECENT_SH_MAIN_SELECTION = FactorSelection(
    name="recent_sh_main",
    source_factor_sets=(
        "qlib_alpha158",
        "wq_alpha101",
        "gtja191",
        "qlib_alpha360",
        "china_chars",
        "size_value_china",
        "talib",
        "jq240",
        "oap",
        "barra_cne6",
        "tushare_tech",
        "jkp153",
    ),
    members=RECENT_SH_MAIN_MEMBERS,
    criteria={
        "universe": "sh_main",
        "min_year_days": 120,
        "min_positive_rank_ic_year_ratio": 0.7,
        "min_positive_long_short_year_ratio": 0.0,
        "min_abs_rank_ic": 0.01,
        "min_abs_rank_ic_t_stat": 3.0,
        "min_coverage": 0.8,
        "min_positive_long_excess_year_ratio": 0.7,
        "deduplication_order": "mean_long_excess",
        "max_abs_rank_correlation": 0.8,
        "correlation_date_step": 5,
        "excluded_factor_sets": (),
    },
    provenance={
        "universe": "sh_main",
        "horizon": "1d",
        "evaluation_start": "2020-01-02",
        "evaluation_end": "2026-10-21",
        "qlib_fingerprint": "5ed881c946a85998619e254b57b502487479cc62b01061a1d1cfba4af7e939ef",
        "universe_annual_returns": {
            2020: 0.167149,
            2021: 0.257479,
            2022: -0.049043,
            2023: 0.054111,
            2024: 0.052236,
            2025: 0.355265,
            2026: -0.056876,
        },
        "deduplicated": DEDUPLICATED,
        "evaluations": (
            {
                "factor_set": "qlib_alpha158",
                "run_id": "20261027T234941Z-6896e004",
                "factor_fingerprint": "45ed864f934a45c54198ce3675d4e96f0df912dd32a5bb39a67c429972d8e729",
            },
            {
                "factor_set": "wq_alpha101",
                "run_id": "20261027T235336Z-14a8ff72",
                "factor_fingerprint": "63161a6d392e074644074bbd0ed6e4783e1756779e492a54ebc0bae6787dc98b",
            },
            {
                "factor_set": "gtja191",
                "run_id": "20261027T235645Z-8eb43fcd",
                "factor_fingerprint": "29e0be1e6ecc1511b143e5cf7171729b6cd0ed84de0084072c6818fbbacb5630",
            },
            {
                "factor_set": "qlib_alpha360",
                "run_id": "20261028T001602Z-7ccb50e6",
                "factor_fingerprint": "7c92677621e8a8e80cba0a8500c32780604644faf945e4e93c7c9dc65deab3cd",
            },
            {
                "factor_set": "china_chars",
                "run_id": "20261028T000323Z-8c256bf0",
                "factor_fingerprint": "74c2452cd9ad8dd8acb1df04f88d9d17c983dd7b3d0982ec2c381a1d0f651d27",
            },
            {
                "factor_set": "size_value_china",
                "run_id": "20261028T000616Z-664854a9",
                "factor_fingerprint": "57774ab41c3fd474ff9f8cec485f85031bf14d4192ac19760e45c7acdee2a83d",
            },
            {
                "factor_set": "talib",
                "run_id": "20261028T000045Z-c05920bc",
                "factor_fingerprint": "8dd97162bcaa982017dd78d245651fcc21f3daf06d9ec8200878f647d5f1a6ad",
            },
            {
                "factor_set": "jq240",
                "run_id": "20261028T000743Z-b993ef2a",
                "factor_fingerprint": "9f9a8c1c93fe6538fa7bf7976191b6c6838d5bee20e414a9811de319208678f0",
            },
            {
                "factor_set": "oap",
                "run_id": "20261028T001237Z-25201200",
                "factor_fingerprint": "a4320d2dd1ba0e4a57c89204d44072355238f095b549a3ebb02d8ebefb6479b8",
            },
            {
                "factor_set": "barra_cne6",
                "run_id": "20261028T004124Z-200ca09f",
                "factor_fingerprint": "0cf3488358ba8c55ecd51feb6ddd3c8d6155cd88cced69d8fb94c1d5b73af4d0",
            },
            {
                "factor_set": "tushare_tech",
                "run_id": "20261028T003833Z-542331f0",
                "factor_fingerprint": "7bee112390ec566799884769c8a47ab493596d944b4ee219942ce4effd9978f0",
            },
            {
                "factor_set": "jkp153",
                "run_id": "20261028T005237Z-d16a0227",
                "factor_fingerprint": "120406e0898f977a8374f3d49055c7290746330f9c251a78e770dcb1fdc8d940",
            },
        ),
    },
)
