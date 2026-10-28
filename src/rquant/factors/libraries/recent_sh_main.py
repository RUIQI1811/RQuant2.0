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
- the absolute 2020+ Rank IC is at least ``min_abs_rank_ic`` and its t-statistic (daily Rank ICIR times the square root
  of the number of days) is at least ``min_abs_rank_ic_t_stat`` in absolute value;
- coverage (published observations over the best-covered factor's, pooled over dates and all twelve evaluations) is at
  least ``min_coverage``;
- duplicates are then pruned with ``prune_correlated``: the 1078 factors passing the checks above are ordered
  by absolute Rank IC, and a factor is dropped when its absolute mean daily cross-sectional Spearman correlation
  (``mean_rank_correlation`` on every fifth trading date since 2020-01-02) with an already kept factor reaches
  ``max_abs_rank_correlation``. ``DEDUPLICATED`` maps each dropped factor to the kept one it duplicates.

Kept per source: qlib_alpha158 46, wq_alpha101 38, gtja191 79, qlib_alpha360 30, china_chars 10, size_value_china 4,
talib 20, jq240 25, oap 18, barra_cne6 4, tushare_tech 9, jkp153 8.

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
    SelectedFactor("a158_004", -1),  # RankIC -0.0182, t -7.1, IC+ 7/7, cov 1.000
    SelectedFactor("a158_006", -1),  # RankIC -0.0247, t -9.4, IC+ 7/7, cov 1.000
    SelectedFactor("a158_009", -1),  # RankIC -0.0177, t -5.9, IC+ 6/7, cov 1.000
    SelectedFactor("a158_012", 1),  # RankIC +0.0448, t +14.6, IC+ 7/7, cov 1.000
    SelectedFactor("a158_024", -1),  # RankIC -0.0414, t -10.4, IC+ 7/7, cov 0.999
    SelectedFactor("a158_025", -1),  # RankIC -0.0398, t -9.3, IC+ 7/7, cov 0.998
    SelectedFactor("a158_028", -1),  # RankIC -0.0229, t -4.8, IC+ 7/7, cov 0.984
    SelectedFactor("a158_039", -1),  # RankIC -0.0175, t -5.3, IC+ 7/7, cov 1.000
    SelectedFactor("a158_040", -1),  # RankIC -0.0153, t -4.4, IC+ 6/7, cov 1.000
    SelectedFactor("a158_041", -1),  # RankIC -0.0158, t -4.4, IC+ 6/7, cov 1.000
    SelectedFactor("a158_042", -1),  # RankIC -0.0175, t -4.7, IC+ 7/7, cov 1.000
    SelectedFactor("a158_043", -1),  # RankIC -0.0257, t -6.7, IC+ 7/7, cov 1.000
    SelectedFactor("a158_049", 1),  # RankIC +0.0448, t +13.2, IC+ 7/7, cov 0.999
    SelectedFactor("a158_060", 1),  # RankIC +0.0367, t +9.8, IC+ 7/7, cov 0.998
    SelectedFactor("a158_064", -1),  # RankIC -0.0228, t -6.8, IC+ 7/7, cov 0.999
    SelectedFactor("a158_067", -1),  # RankIC -0.0300, t -7.7, IC+ 7/7, cov 0.992
    SelectedFactor("a158_075", -1),  # RankIC -0.0189, t -6.5, IC+ 7/7, cov 0.998
    SelectedFactor("a158_076", -1),  # RankIC -0.0242, t -8.2, IC+ 7/7, cov 0.995
    SelectedFactor("a158_077", -1),  # RankIC -0.0247, t -8.0, IC+ 7/7, cov 0.992
    SelectedFactor("a158_079", 1),  # RankIC +0.0145, t +4.9, IC+ 6/7, cov 0.999
    SelectedFactor("a158_080", 1),  # RankIC +0.0127, t +4.2, IC+ 7/7, cov 0.998
    SelectedFactor("a158_082", 1),  # RankIC +0.0164, t +5.1, IC+ 7/7, cov 0.992
    SelectedFactor("a158_083", 1),  # RankIC +0.0131, t +3.9, IC+ 7/7, cov 0.984
    SelectedFactor("a158_084", -1),  # RankIC -0.0177, t -5.5, IC+ 7/7, cov 0.999
    SelectedFactor("a158_088", -1),  # RankIC -0.0199, t -5.6, IC+ 7/7, cov 0.984
    SelectedFactor("a158_089", -1),  # RankIC -0.0259, t -12.1, IC+ 7/7, cov 1.000
    SelectedFactor("a158_090", -1),  # RankIC -0.0307, t -13.1, IC+ 7/7, cov 1.000
    SelectedFactor("a158_091", -1),  # RankIC -0.0272, t -10.6, IC+ 7/7, cov 1.000
    SelectedFactor("a158_093", -1),  # RankIC -0.0123, t -4.5, IC+ 7/7, cov 1.000
    SelectedFactor("a158_094", -1),  # RankIC -0.0230, t -10.1, IC+ 7/7, cov 1.000
    SelectedFactor("a158_095", -1),  # RankIC -0.0253, t -10.7, IC+ 7/7, cov 1.000
    SelectedFactor("a158_096", -1),  # RankIC -0.0266, t -11.1, IC+ 7/7, cov 1.000
    SelectedFactor("a158_098", -1),  # RankIC -0.0195, t -8.6, IC+ 7/7, cov 1.000
    SelectedFactor("a158_099", -1),  # RankIC -0.0183, t -6.3, IC+ 7/7, cov 1.000
    SelectedFactor("a158_103", -1),  # RankIC -0.0137, t -4.5, IC+ 7/7, cov 1.000
    SelectedFactor("a158_112", -1),  # RankIC -0.0124, t -4.0, IC+ 7/7, cov 1.000
    SelectedFactor("a158_129", 1),  # RankIC +0.0115, t +5.3, IC+ 7/7, cov 0.999
    SelectedFactor("a158_131", 1),  # RankIC +0.0250, t +9.7, IC+ 7/7, cov 0.995
    SelectedFactor("a158_133", 1),  # RankIC +0.0367, t +12.6, IC+ 7/7, cov 0.984
    SelectedFactor("a158_134", -1),  # RankIC -0.0141, t -8.5, IC+ 7/7, cov 0.999
    SelectedFactor("a158_137", 1),  # RankIC +0.0113, t +4.6, IC+ 7/7, cov 0.992
    SelectedFactor("a158_140", -1),  # RankIC -0.0120, t -7.4, IC+ 7/7, cov 0.997
    SelectedFactor("a158_141", -1),  # RankIC -0.0124, t -7.4, IC+ 7/7, cov 0.995
    SelectedFactor("a158_142", -1),  # RankIC -0.0109, t -6.2, IC+ 6/7, cov 0.992
    SelectedFactor("a158_148", -1),  # RankIC -0.0333, t -12.5, IC+ 7/7, cov 0.984
    SelectedFactor("a101_001", -1),  # RankIC -0.0168, t -6.5, IC+ 7/7, cov 0.994
    SelectedFactor("a101_002", 1),  # RankIC +0.0143, t +9.6, IC+ 7/7, cov 1.000
    SelectedFactor("a101_004", 1),  # RankIC +0.0241, t +6.9, IC+ 7/7, cov 0.998
    SelectedFactor("a101_005", 1),  # RankIC +0.0121, t +3.8, IC+ 5/7, cov 0.998
    SelectedFactor("a101_007", -1),  # RankIC -0.0141, t -7.4, IC+ 7/7, cov 0.990
    SelectedFactor("a101_008", 1),  # RankIC +0.0117, t +3.8, IC+ 5/7, cov 0.996
    SelectedFactor("a101_010", 1),  # RankIC +0.0119, t +4.4, IC+ 6/7, cov 1.000
    SelectedFactor("a101_011", 1),  # RankIC +0.0130, t +6.7, IC+ 7/7, cov 0.999
    SelectedFactor("a101_012", 1),  # RankIC +0.0214, t +12.7, IC+ 7/7, cov 1.000
    SelectedFactor("a101_016", 1),  # RankIC +0.0314, t +17.5, IC+ 7/7, cov 0.999
    SelectedFactor("a101_019", 1),  # RankIC +0.0189, t +6.4, IC+ 6/7, cov 0.916
    SelectedFactor("a101_034", 1),  # RankIC +0.0167, t +6.4, IC+ 6/7, cov 1.000
    SelectedFactor("a101_037", 1),  # RankIC +0.0235, t +9.2, IC+ 7/7, cov 0.933
    SelectedFactor("a101_038", 1),  # RankIC +0.0214, t +6.2, IC+ 7/7, cov 0.998
    SelectedFactor("a101_041", 1),  # RankIC +0.0106, t +3.8, IC+ 5/7, cov 1.000
    SelectedFactor("a101_049", 1),  # RankIC +0.0199, t +6.4, IC+ 6/7, cov 1.000
    SelectedFactor("a101_052", 1),  # RankIC +0.0145, t +4.7, IC+ 7/7, cov 0.919
    SelectedFactor("a101_053", 1),  # RankIC +0.0108, t +4.1, IC+ 5/7, cov 0.999
    SelectedFactor("a101_056", 1),  # RankIC +0.0119, t +4.7, IC+ 6/7, cov 0.997
    SelectedFactor("a101_057", 1),  # RankIC +0.0214, t +7.5, IC+ 6/7, cov 0.992
    SelectedFactor("a101_058", 1),  # RankIC +0.0102, t +8.6, IC+ 7/7, cov 0.992
    SelectedFactor("a101_059", 1),  # RankIC +0.0114, t +10.0, IC+ 7/7, cov 0.987
    SelectedFactor("a101_060", 1),  # RankIC +0.0123, t +4.3, IC+ 5/7, cov 0.998
    SelectedFactor("a101_061", 1),  # RankIC +0.0156, t +7.7, IC+ 7/7, cov 1.000
    SelectedFactor("a101_062", 1),  # RankIC +0.0101, t +6.1, IC+ 7/7, cov 1.000
    SelectedFactor("a101_065", 1),  # RankIC +0.0154, t +7.1, IC+ 7/7, cov 1.000
    SelectedFactor("a101_066", 1),  # RankIC +0.0167, t +6.5, IC+ 6/7, cov 0.997
    SelectedFactor("a101_069", 1),  # RankIC +0.0247, t +11.8, IC+ 7/7, cov 0.988
    SelectedFactor("a101_070", 1),  # RankIC +0.0119, t +5.0, IC+ 7/7, cov 0.969
    SelectedFactor("a101_071", 1),  # RankIC +0.0155, t +7.5, IC+ 7/7, cov 0.995
    SelectedFactor("a101_073", 1),  # RankIC +0.0181, t +6.4, IC+ 7/7, cov 0.995
    SelectedFactor("a101_076", 1),  # RankIC +0.0152, t +7.7, IC+ 7/7, cov 0.950
    SelectedFactor("a101_082", 1),  # RankIC +0.0114, t +5.0, IC+ 6/7, cov 0.987
    SelectedFactor("a101_083", 1),  # RankIC +0.0303, t +9.8, IC+ 7/7, cov 0.997
    SelectedFactor("a101_086", 1),  # RankIC +0.0121, t +6.1, IC+ 7/7, cov 1.000
    SelectedFactor("a101_087", 1),  # RankIC +0.0155, t +7.5, IC+ 7/7, cov 0.959
    SelectedFactor("a101_088", 1),  # RankIC +0.0478, t +16.7, IC+ 7/7, cov 0.998
    SelectedFactor("a101_095", 1),  # RankIC +0.0111, t +6.9, IC+ 7/7, cov 1.000
    SelectedFactor("gtja_001", 1),  # RankIC +0.0188, t +10.9, IC+ 7/7, cov 0.998
    SelectedFactor("gtja_002", 1),  # RankIC +0.0110, t +3.8, IC+ 5/7, cov 0.997
    SelectedFactor("gtja_003", -1),  # RankIC -0.0265, t -8.1, IC+ 7/7, cov 0.998
    SelectedFactor("gtja_005", 1),  # RankIC +0.0215, t +11.5, IC+ 7/7, cov 0.997
    SelectedFactor("gtja_006", 1),  # RankIC +0.0139, t +4.7, IC+ 7/7, cov 0.999
    SelectedFactor("gtja_010", -1),  # RankIC -0.0216, t -6.5, IC+ 7/7, cov 0.994
    SelectedFactor("gtja_011", -1),  # RankIC -0.0156, t -5.3, IC+ 6/7, cov 0.992
    SelectedFactor("gtja_012", 1),  # RankIC +0.0202, t +7.1, IC+ 7/7, cov 0.998
    SelectedFactor("gtja_014", -1),  # RankIC -0.0245, t -7.0, IC+ 7/7, cov 0.999
    SelectedFactor("gtja_015", 1),  # RankIC +0.0126, t +5.8, IC+ 7/7, cov 1.000
    SelectedFactor("gtja_016", 1),  # RankIC +0.0235, t +13.9, IC+ 7/7, cov 0.998
    SelectedFactor("gtja_024", -1),  # RankIC -0.0245, t -6.7, IC+ 7/7, cov 0.999
    SelectedFactor("gtja_025", 1),  # RankIC +0.0179, t +5.4, IC+ 7/7, cov 0.916
    SelectedFactor("gtja_032", 1),  # RankIC +0.0220, t +15.4, IC+ 7/7, cov 0.999
    SelectedFactor("gtja_035", 1),  # RankIC +0.0170, t +5.9, IC+ 7/7, cov 0.994
    SelectedFactor("gtja_036", -1),  # RankIC -0.0237, t -13.3, IC+ 7/7, cov 0.998
    SelectedFactor("gtja_038", 1),  # RankIC +0.0154, t +6.4, IC+ 7/7, cov 0.995
    SelectedFactor("gtja_039", 1),  # RankIC +0.0122, t +4.4, IC+ 6/7, cov 0.919
    SelectedFactor("gtja_040", -1),  # RankIC -0.0255, t -8.3, IC+ 7/7, cov 0.993
    SelectedFactor("gtja_041", 1),  # RankIC +0.0382, t +11.8, IC+ 7/7, cov 0.998
    SelectedFactor("gtja_042", 1),  # RankIC +0.0418, t +14.7, IC+ 7/7, cov 0.998
    SelectedFactor("gtja_043", -1),  # RankIC -0.0230, t -8.0, IC+ 7/7, cov 0.998
    SelectedFactor("gtja_045", -1),  # RankIC -0.0151, t -5.7, IC+ 6/7, cov 0.946
    SelectedFactor("gtja_047", 1),  # RankIC +0.0151, t +4.0, IC+ 7/7, cov 0.999
    SelectedFactor("gtja_048", 1),  # RankIC +0.0223, t +7.2, IC+ 7/7, cov 0.995
    SelectedFactor("gtja_053", -1),  # RankIC -0.0160, t -5.6, IC+ 7/7, cov 0.997
    SelectedFactor("gtja_054", 1),  # RankIC +0.0115, t +4.8, IC+ 7/7, cov 0.998
    SelectedFactor("gtja_056", 1),  # RankIC +0.0149, t +7.1, IC+ 7/7, cov 0.981
    SelectedFactor("gtja_058", -1),  # RankIC -0.0165, t -5.6, IC+ 7/7, cov 0.995
    SelectedFactor("gtja_059", -1),  # RankIC -0.0280, t -8.1, IC+ 7/7, cov 0.995
    SelectedFactor("gtja_060", -1),  # RankIC -0.0140, t -5.0, IC+ 7/7, cov 0.973
    SelectedFactor("gtja_061", 1),  # RankIC +0.0268, t +10.1, IC+ 7/7, cov 0.968
    SelectedFactor("gtja_062", 1),  # RankIC +0.0274, t +16.1, IC+ 7/7, cov 0.999
    SelectedFactor("gtja_064", 1),  # RankIC +0.0206, t +14.5, IC+ 7/7, cov 0.974
    SelectedFactor("gtja_068", -1),  # RankIC -0.0221, t -6.6, IC+ 7/7, cov 1.000
    SelectedFactor("gtja_069", -1),  # RankIC -0.0291, t -8.7, IC+ 7/7, cov 0.995
    SelectedFactor("gtja_074", -1),  # RankIC -0.0220, t -11.3, IC+ 7/7, cov 0.983
    SelectedFactor("gtja_075", -1),  # RankIC -0.0154, t -4.8, IC+ 7/7, cov 0.987
    SelectedFactor("gtja_077", 1),  # RankIC +0.0144, t +6.5, IC+ 7/7, cov 0.988
    SelectedFactor("gtja_080", -1),  # RankIC -0.0196, t -8.7, IC+ 7/7, cov 0.999
    SelectedFactor("gtja_084", -1),  # RankIC -0.0245, t -8.4, IC+ 7/7, cov 0.995
    SelectedFactor("gtja_086", 1),  # RankIC +0.0170, t +6.4, IC+ 7/7, cov 0.997
    SelectedFactor("gtja_090", 1),  # RankIC +0.0236, t +14.6, IC+ 7/7, cov 0.999
    SelectedFactor("gtja_092", 1),  # RankIC +0.0167, t +7.6, IC+ 7/7, cov 0.930
    SelectedFactor("gtja_094", -1),  # RankIC -0.0249, t -8.3, IC+ 7/7, cov 0.992
    SelectedFactor("gtja_098", 1),  # RankIC +0.0233, t +8.2, IC+ 7/7, cov 0.933
    SelectedFactor("gtja_101", 1),  # RankIC +0.0115, t +7.5, IC+ 7/7, cov 0.977
    SelectedFactor("gtja_103", 1),  # RankIC +0.0163, t +5.3, IC+ 7/7, cov 0.995
    SelectedFactor("gtja_105", 1),  # RankIC +0.0170, t +10.5, IC+ 7/7, cov 0.998
    SelectedFactor("gtja_107", -1),  # RankIC -0.0134, t -5.2, IC+ 7/7, cov 1.000
    SelectedFactor("gtja_108", 1),  # RankIC +0.0152, t +7.3, IC+ 7/7, cov 0.960
    SelectedFactor("gtja_109", -1),  # RankIC -0.0287, t -10.3, IC+ 7/7, cov 1.000
    SelectedFactor("gtja_111", 1),  # RankIC +0.0185, t +6.5, IC+ 6/7, cov 0.998
    SelectedFactor("gtja_116", -1),  # RankIC -0.0238, t -6.5, IC+ 7/7, cov 0.995
    SelectedFactor("gtja_117", 1),  # RankIC +0.0134, t +4.8, IC+ 5/7, cov 0.991
    SelectedFactor("gtja_120", 1),  # RankIC +0.0357, t +11.5, IC+ 7/7, cov 1.000
    SelectedFactor("gtja_121", 1),  # RankIC +0.0331, t +10.4, IC+ 7/7, cov 0.977
    SelectedFactor("gtja_124", -1),  # RankIC -0.0253, t -8.4, IC+ 6/7, cov 0.992
    SelectedFactor("gtja_129", -1),  # RankIC -0.0267, t -6.6, IC+ 7/7, cov 0.997
    SelectedFactor("gtja_131", -1),  # RankIC -0.0122, t -4.8, IC+ 7/7, cov 0.975
    SelectedFactor("gtja_137", -1),  # RankIC -0.0257, t -7.8, IC+ 7/7, cov 1.000
    SelectedFactor("gtja_138", 1),  # RankIC +0.0132, t +5.6, IC+ 7/7, cov 0.962
    SelectedFactor("gtja_139", 1),  # RankIC +0.0185, t +8.5, IC+ 7/7, cov 0.998
    SelectedFactor("gtja_142", 1),  # RankIC +0.0127, t +4.5, IC+ 5/7, cov 0.994
    SelectedFactor("gtja_144", 1),  # RankIC +0.0204, t +4.7, IC+ 7/7, cov 0.995
    SelectedFactor("gtja_145", -1),  # RankIC -0.0244, t -9.7, IC+ 7/7, cov 0.993
    SelectedFactor("gtja_148", 1),  # RankIC +0.0165, t +7.6, IC+ 7/7, cov 0.980
    SelectedFactor("gtja_152", -1),  # RankIC -0.0121, t -3.6, IC+ 7/7, cov 0.991
    SelectedFactor("gtja_156", 1),  # RankIC +0.0313, t +11.2, IC+ 7/7, cov 0.998
    SelectedFactor("gtja_157", 1),  # RankIC +0.0235, t +7.9, IC+ 7/7, cov 0.997
    SelectedFactor("gtja_162", -1),  # RankIC -0.0139, t -4.0, IC+ 6/7, cov 0.997
    SelectedFactor("gtja_166", 1),  # RankIC +0.0213, t +8.2, IC+ 7/7, cov 0.990
    SelectedFactor("gtja_167", -1),  # RankIC -0.0442, t -11.7, IC+ 7/7, cov 0.997
    SelectedFactor("gtja_169", -1),  # RankIC -0.0162, t -5.2, IC+ 7/7, cov 0.993
    SelectedFactor("gtja_176", -1),  # RankIC -0.0214, t -14.8, IC+ 7/7, cov 0.996
    SelectedFactor("gtja_179", -1),  # RankIC -0.0217, t -12.4, IC+ 7/7, cov 0.984
    SelectedFactor("gtja_183", 1),  # RankIC +0.0105, t +4.7, IC+ 6/7, cov 0.981
    SelectedFactor("gtja_185", 1),  # RankIC +0.0293, t +10.3, IC+ 7/7, cov 1.000
    SelectedFactor("gtja_188", -1),  # RankIC -0.0129, t -5.6, IC+ 7/7, cov 1.000
    SelectedFactor("a360_051", 1),  # RankIC +0.0269, t +6.9, IC+ 7/7, cov 0.999
    SelectedFactor("a360_090", 1),  # RankIC +0.0333, t +7.8, IC+ 7/7, cov 0.999
    SelectedFactor("a360_117", 1),  # RankIC +0.0231, t +6.0, IC+ 6/7, cov 0.999
    SelectedFactor("a360_219", 1),  # RankIC +0.0361, t +8.7, IC+ 7/7, cov 0.999
    SelectedFactor("a360_228", 1),  # RankIC +0.0341, t +8.8, IC+ 7/7, cov 0.999
    SelectedFactor("a360_238", 1),  # RankIC +0.0338, t +9.3, IC+ 7/7, cov 1.000
    SelectedFactor("a360_239", 1),  # RankIC +0.0361, t +10.5, IC+ 7/7, cov 1.000
    SelectedFactor("a360_302", 1),  # RankIC +0.0350, t +13.2, IC+ 7/7, cov 0.999
    SelectedFactor("a360_306", 1),  # RankIC +0.0363, t +13.7, IC+ 7/7, cov 0.999
    SelectedFactor("a360_310", 1),  # RankIC +0.0369, t +13.7, IC+ 7/7, cov 0.999
    SelectedFactor("a360_314", 1),  # RankIC +0.0378, t +14.1, IC+ 7/7, cov 0.999
    SelectedFactor("a360_318", 1),  # RankIC +0.0351, t +13.1, IC+ 7/7, cov 0.999
    SelectedFactor("a360_321", 1),  # RankIC +0.0339, t +12.8, IC+ 7/7, cov 0.999
    SelectedFactor("a360_324", 1),  # RankIC +0.0328, t +12.7, IC+ 7/7, cov 0.999
    SelectedFactor("a360_326", 1),  # RankIC +0.0333, t +12.9, IC+ 7/7, cov 0.999
    SelectedFactor("a360_329", 1),  # RankIC +0.0325, t +12.5, IC+ 7/7, cov 0.999
    SelectedFactor("a360_332", 1),  # RankIC +0.0302, t +11.7, IC+ 7/7, cov 0.999
    SelectedFactor("a360_334", 1),  # RankIC +0.0314, t +12.4, IC+ 7/7, cov 0.999
    SelectedFactor("a360_336", 1),  # RankIC +0.0309, t +12.3, IC+ 7/7, cov 0.999
    SelectedFactor("a360_338", 1),  # RankIC +0.0314, t +12.7, IC+ 7/7, cov 0.999
    SelectedFactor("a360_340", 1),  # RankIC +0.0293, t +11.8, IC+ 7/7, cov 0.999
    SelectedFactor("a360_342", 1),  # RankIC +0.0285, t +11.6, IC+ 7/7, cov 0.999
    SelectedFactor("a360_345", 1),  # RankIC +0.0283, t +11.8, IC+ 7/7, cov 0.999
    SelectedFactor("a360_347", 1),  # RankIC +0.0251, t +10.5, IC+ 7/7, cov 0.999
    SelectedFactor("a360_349", 1),  # RankIC +0.0237, t +10.1, IC+ 7/7, cov 0.999
    SelectedFactor("a360_350", 1),  # RankIC +0.0237, t +10.2, IC+ 7/7, cov 0.999
    SelectedFactor("a360_351", 1),  # RankIC +0.0220, t +9.6, IC+ 7/7, cov 0.999
    SelectedFactor("a360_352", 1),  # RankIC +0.0203, t +9.0, IC+ 7/7, cov 0.999
    SelectedFactor("a360_353", 1),  # RankIC +0.0212, t +9.2, IC+ 7/7, cov 0.999
    SelectedFactor("a360_356", 1),  # RankIC +0.0145, t +6.5, IC+ 7/7, cov 0.999
    SelectedFactor("cnc_006", -1),  # RankIC -0.0261, t -5.7, IC+ 7/7, cov 1.000
    SelectedFactor("cnc_012", -1),  # RankIC -0.0288, t -7.8, IC+ 7/7, cov 1.000
    SelectedFactor("cnc_013", -1),  # RankIC -0.0182, t -5.5, IC+ 7/7, cov 1.000
    SelectedFactor("cnc_019", -1),  # RankIC -0.0129, t -7.2, IC+ 7/7, cov 1.000
    SelectedFactor("cnc_038", -1),  # RankIC -0.0175, t -4.2, IC+ 6/7, cov 1.000
    SelectedFactor("cnc_061", 1),  # RankIC +0.0263, t +6.4, IC+ 7/7, cov 1.000
    SelectedFactor("cnc_062", 1),  # RankIC +0.0123, t +3.5, IC+ 6/7, cov 1.000
    SelectedFactor("cnc_068", 1),  # RankIC +0.0145, t +5.1, IC+ 7/7, cov 0.992
    SelectedFactor("cnc_069", 1),  # RankIC +0.0153, t +5.1, IC+ 7/7, cov 0.992
    SelectedFactor("cnc_078", -1),  # RankIC -0.0149, t -5.1, IC+ 7/7, cov 0.976
    SelectedFactor("svc_002", 1),  # RankIC +0.0188, t +4.6, IC+ 7/7, cov 0.838
    SelectedFactor("svc_003", -1),  # RankIC -0.0348, t -11.7, IC+ 7/7, cov 0.992
    SelectedFactor("svc_013", 1),  # RankIC +0.0301, t +11.9, IC+ 7/7, cov 1.000
    SelectedFactor("svc_014", -1),  # RankIC -0.0240, t -8.6, IC+ 7/7, cov 1.000
    SelectedFactor("ta_001", -1),  # RankIC -0.0105, t -4.1, IC+ 6/7, cov 1.000
    SelectedFactor("ta_003", 1),  # RankIC +0.0161, t +5.4, IC+ 7/7, cov 1.000
    SelectedFactor("ta_004", -1),  # RankIC -0.0234, t -7.9, IC+ 7/7, cov 1.000
    SelectedFactor("ta_008", -1),  # RankIC -0.0159, t -6.3, IC+ 7/7, cov 1.000
    SelectedFactor("ta_013", 1),  # RankIC +0.0287, t +7.4, IC+ 7/7, cov 1.000
    SelectedFactor("ta_015", -1),  # RankIC -0.0411, t -11.8, IC+ 7/7, cov 1.000
    SelectedFactor("ta_016", -1),  # RankIC -0.0543, t -12.2, IC+ 7/7, cov 1.000
    SelectedFactor("ta_021", -1),  # RankIC -0.0110, t -3.1, IC+ 6/7, cov 1.000
    SelectedFactor("ta_025", -1),  # RankIC -0.0109, t -3.2, IC+ 5/7, cov 1.000
    SelectedFactor("ta_026", -1),  # RankIC -0.0184, t -4.3, IC+ 7/7, cov 0.997
    SelectedFactor("ta_027", -1),  # RankIC -0.0197, t -5.8, IC+ 7/7, cov 1.000
    SelectedFactor("ta_030", -1),  # RankIC -0.0353, t -7.9, IC+ 7/7, cov 1.000
    SelectedFactor("ta_039", 1),  # RankIC +0.0190, t +5.3, IC+ 7/7, cov 1.000
    SelectedFactor("ta_049", -1),  # RankIC -0.0493, t -12.2, IC+ 7/7, cov 1.000
    SelectedFactor("ta_058", -1),  # RankIC -0.0107, t -6.4, IC+ 6/7, cov 1.000
    SelectedFactor("ta_059", -1),  # RankIC -0.0253, t -9.6, IC+ 7/7, cov 1.000
    SelectedFactor("ta_060", 1),  # RankIC +0.0154, t +4.4, IC+ 5/7, cov 1.000
    SelectedFactor("ta_074", -1),  # RankIC -0.0104, t -5.5, IC+ 6/7, cov 1.000
    SelectedFactor("ta_076", -1),  # RankIC -0.0182, t -9.9, IC+ 7/7, cov 1.000
    SelectedFactor("ta_103", -1),  # RankIC -0.0130, t -6.2, IC+ 7/7, cov 1.000
    SelectedFactor("jq_103", -1),  # RankIC -0.0346, t -10.4, IC+ 7/7, cov 1.000
    SelectedFactor("jq_104", -1),  # RankIC -0.0135, t -7.4, IC+ 7/7, cov 1.000
    SelectedFactor("jq_107", -1),  # RankIC -0.0305, t -8.6, IC+ 7/7, cov 1.000
    SelectedFactor("jq_110", -1),  # RankIC -0.0394, t -13.2, IC+ 7/7, cov 0.993
    SelectedFactor("jq_114", -1),  # RankIC -0.0286, t -14.3, IC+ 7/7, cov 1.000
    SelectedFactor("jq_118", -1),  # RankIC -0.0491, t -12.7, IC+ 7/7, cov 1.000
    SelectedFactor("jq_120", -1),  # RankIC -0.0283, t -10.0, IC+ 7/7, cov 1.000
    SelectedFactor("jq_125", -1),  # RankIC -0.0157, t -6.3, IC+ 7/7, cov 1.000
    SelectedFactor("jq_130", -1),  # RankIC -0.0449, t -10.1, IC+ 7/7, cov 1.000
    SelectedFactor("jq_135", -1),  # RankIC -0.0213, t -9.2, IC+ 7/7, cov 1.000
    SelectedFactor("jq_136", -1),  # RankIC -0.0384, t -12.7, IC+ 7/7, cov 1.000
    SelectedFactor("jq_138", -1),  # RankIC -0.0282, t -8.9, IC+ 7/7, cov 1.000
    SelectedFactor("jq_148", 1),  # RankIC +0.0110, t +3.6, IC+ 7/7, cov 0.993
    SelectedFactor("jq_151", -1),  # RankIC -0.0247, t -6.2, IC+ 7/7, cov 0.993
    SelectedFactor("jq_154", -1),  # RankIC -0.0186, t -8.4, IC+ 7/7, cov 0.993
    SelectedFactor("jq_156", -1),  # RankIC -0.0199, t -9.3, IC+ 7/7, cov 1.000
    SelectedFactor("jq_188", -1),  # RankIC -0.0236, t -7.3, IC+ 7/7, cov 1.000
    SelectedFactor("jq_190", 1),  # RankIC +0.0161, t +5.1, IC+ 7/7, cov 1.000
    SelectedFactor("jq_191", -1),  # RankIC -0.0242, t -7.9, IC+ 7/7, cov 1.000
    SelectedFactor("jq_204", -1),  # RankIC -0.0243, t -6.3, IC+ 7/7, cov 0.977
    SelectedFactor("jq_205", -1),  # RankIC -0.0259, t -8.3, IC+ 7/7, cov 1.000
    SelectedFactor("jq_218", -1),  # RankIC -0.0232, t -7.4, IC+ 7/7, cov 1.000
    SelectedFactor("jq_219", -1),  # RankIC -0.0331, t -10.3, IC+ 7/7, cov 1.000
    SelectedFactor("jq_220", -1),  # RankIC -0.0294, t -9.0, IC+ 7/7, cov 1.000
    SelectedFactor("jq_227", -1),  # RankIC -0.0422, t -12.9, IC+ 7/7, cov 1.000
    SelectedFactor("oap_012", -1),  # RankIC -0.0105, t -3.9, IC+ 6/7, cov 0.901
    SelectedFactor("oap_015", -1),  # RankIC -0.0466, t -12.2, IC+ 7/7, cov 0.999
    SelectedFactor("oap_016", -1),  # RankIC -0.0197, t -10.4, IC+ 7/7, cov 0.999
    SelectedFactor("oap_017", -1),  # RankIC -0.0131, t -10.9, IC+ 7/7, cov 0.999
    SelectedFactor("oap_021", -1),  # RankIC -0.0125, t -3.5, IC+ 7/7, cov 0.954
    SelectedFactor("oap_022", -1),  # RankIC -0.0229, t -4.7, IC+ 7/7, cov 0.895
    SelectedFactor("oap_024", -1),  # RankIC -0.0135, t -4.1, IC+ 6/7, cov 0.989
    SelectedFactor("oap_030", 1),  # RankIC +0.0142, t +4.1, IC+ 6/7, cov 0.880
    SelectedFactor("oap_040", -1),  # RankIC -0.0125, t -3.2, IC+ 7/7, cov 0.943
    SelectedFactor("oap_042", -1),  # RankIC -0.0139, t -3.1, IC+ 7/7, cov 0.993
    SelectedFactor("oap_043", -1),  # RankIC -0.0166, t -5.8, IC+ 7/7, cov 0.923
    SelectedFactor("oap_046", 1),  # RankIC +0.0151, t +3.9, IC+ 7/7, cov 1.000
    SelectedFactor("oap_047", -1),  # RankIC -0.0347, t -8.8, IC+ 7/7, cov 0.999
    SelectedFactor("oap_057", 1),  # RankIC +0.0222, t +7.2, IC+ 7/7, cov 1.000
    SelectedFactor("oap_060", -1),  # RankIC -0.0212, t -5.4, IC+ 7/7, cov 0.917
    SelectedFactor("oap_069", 1),  # RankIC +0.0240, t +8.8, IC+ 7/7, cov 0.986
    SelectedFactor("oap_123", -1),  # RankIC -0.0127, t -4.6, IC+ 7/7, cov 0.960
    SelectedFactor("oap_129", 1),  # RankIC +0.0102, t +5.1, IC+ 7/7, cov 0.817
    SelectedFactor("cne6_002", -1),  # RankIC -0.0139, t -3.5, IC+ 7/7, cov 1.000
    SelectedFactor("cne6_005", -1),  # RankIC -0.0413, t -7.7, IC+ 7/7, cov 1.000
    SelectedFactor("cne6_006", -1),  # RankIC -0.0146, t -3.7, IC+ 7/7, cov 0.981
    SelectedFactor("cne6_025", -1),  # RankIC -0.0101, t -4.4, IC+ 7/7, cov 0.998
    SelectedFactor("tst_015", -1),  # RankIC -0.0233, t -5.5, IC+ 7/7, cov 1.000
    SelectedFactor("tst_027", 1),  # RankIC +0.0374, t +8.5, IC+ 7/7, cov 0.997
    SelectedFactor("tst_030", -1),  # RankIC -0.0201, t -5.2, IC+ 7/7, cov 1.000
    SelectedFactor("tst_038", 1),  # RankIC +0.0524, t +13.3, IC+ 7/7, cov 1.000
    SelectedFactor("tst_069", 1),  # RankIC +0.0181, t +5.1, IC+ 6/7, cov 1.000
    SelectedFactor("tst_072", -1),  # RankIC -0.0230, t -8.4, IC+ 7/7, cov 1.000
    SelectedFactor("tst_073", 1),  # RankIC +0.0142, t +5.0, IC+ 5/7, cov 1.000
    SelectedFactor("tst_074", -1),  # RankIC -0.0185, t -6.5, IC+ 7/7, cov 1.000
    SelectedFactor("tst_075", 1),  # RankIC +0.0158, t +5.4, IC+ 7/7, cov 1.000
    SelectedFactor("jkp_016", 1),  # RankIC +0.0183, t +5.4, IC+ 7/7, cov 1.000
    SelectedFactor("jkp_017", 1),  # RankIC +0.0150, t +6.6, IC+ 7/7, cov 0.992
    SelectedFactor("jkp_020", 1),  # RankIC +0.0162, t +4.5, IC+ 7/7, cov 0.977
    SelectedFactor("jkp_098", 1),  # RankIC +0.0117, t +4.0, IC+ 6/7, cov 0.930
    SelectedFactor("jkp_100", -1),  # RankIC -0.0223, t -5.8, IC+ 7/7, cov 0.979
    SelectedFactor("jkp_124", -1),  # RankIC -0.0186, t -11.5, IC+ 7/7, cov 0.999
    SelectedFactor("jkp_133", 1),  # RankIC +0.0129, t +5.6, IC+ 7/7, cov 0.896
    SelectedFactor("jkp_144", 1),  # RankIC +0.0105, t +4.2, IC+ 7/7, cov 0.970
)

DEDUPLICATED = {
    "a158_002": "ta_049",
    "a158_003": "a158_001",
    "a158_008": "gtja_124",
    "a158_010": "a158_001",
    "a158_013": "a101_083",
    "a158_014": "a158_060",
    "a158_015": "a360_228",
    "a158_016": "a360_219",
    "a158_017": "a360_090",
    "a158_018": "tst_027",
    "a158_019": "a360_238",
    "a158_020": "a158_060",
    "a158_021": "a360_228",
    "a158_022": "a360_219",
    "a158_023": "tst_027",
    "a158_026": "ta_030",
    "a158_027": "ta_030",
    "a158_029": "a360_117",
    "a158_030": "a360_051",
    "a158_031": "a360_219",
    "a158_032": "a360_219",
    "a158_033": "tst_015",
    "a158_050": "a158_049",
    "a158_051": "tst_038",
    "a158_052": "tst_038",
    "a158_053": "tst_038",
    "a158_055": "a360_117",
    "a158_056": "a360_228",
    "a158_057": "a158_067",
    "a158_058": "tst_027",
    "a158_059": "a360_238",
    "a158_061": "tst_038",
    "a158_062": "tst_038",
    "a158_063": "tst_038",
    "a158_065": "a158_060",
    "a158_066": "a158_067",
    "a158_068": "tst_027",
    "a158_069": "tst_069",
    "a158_070": "a158_060",
    "a158_071": "a158_067",
    "a158_072": "a158_067",
    "a158_073": "tst_027",
    "a158_074": "a158_084",
    "a158_078": "a158_088",
    "a158_081": "gtja_103",
    "a158_085": "a158_075",
    "a158_086": "a158_076",
    "a158_087": "a158_077",
    "a158_092": "ta_059",
    "a158_097": "a158_096",
    "a158_100": "gtja_053",
    "a158_101": "gtja_058",
    "a158_102": "gtja_058",
    "a158_104": "a158_099",
    "a158_109": "a158_099",
    "a158_110": "gtja_053",
    "a158_111": "gtja_058",
    "a158_114": "gtja_014",
    "a158_115": "a360_051",
    "a158_116": "a360_219",
    "a158_117": "a360_090",
    "a158_118": "tst_027",
    "a158_119": "gtja_014",
    "a158_120": "a360_051",
    "a158_121": "a360_219",
    "a158_122": "a360_090",
    "a158_123": "tst_027",
    "a158_124": "gtja_014",
    "a158_125": "a360_051",
    "a158_126": "a360_219",
    "a158_127": "a360_090",
    "a158_128": "tst_027",
    "a158_130": "a158_131",
    "a158_132": "a158_133",
    "a158_138": "a158_133",
    "a158_144": "gtja_080",
    "a158_145": "a360_350",
    "a158_146": "a360_340",
    "a158_147": "a360_329",
    "a158_149": "gtja_080",
    "a158_150": "a360_350",
    "a158_151": "a360_340",
    "a158_152": "a360_329",
    "a158_153": "a158_148",
    "a158_154": "gtja_080",
    "a158_155": "a360_350",
    "a158_156": "a360_340",
    "a158_157": "a360_329",
    "a158_158": "a158_148",
    "a101_003": "gtja_105",
    "a101_006": "gtja_139",
    "a101_009": "a101_049",
    "a101_013": "a101_016",
    "a101_014": "gtja_139",
    "a101_015": "gtja_032",
    "a101_017": "gtja_142",
    "a101_018": "gtja_054",
    "a101_020": "gtja_107",
    "a101_023": "gtja_038",
    "a101_024": "gtja_098",
    "a101_025": "a101_049",
    "a101_026": "gtja_005",
    "a101_027": "gtja_036",
    "a101_029": "gtja_157",
    "a101_033": "a158_001",
    "a101_035": "gtja_117",
    "a101_039": "gtja_025",
    "a101_040": "gtja_042",
    "a101_042": "gtja_120",
    "a101_044": "gtja_062",
    "a101_045": "a101_012",
    "a101_046": "gtja_086",
    "a101_050": "gtja_016",
    "a101_051": "a101_049",
    "a101_055": "gtja_176",
    "a101_074": "gtja_101",
    "a101_077": "gtja_077",
    "a101_084": "gtja_014",
    "a101_094": "gtja_121",
    "a101_101": "a158_001",
    "gtja_007": "a101_011",
    "gtja_008": "gtja_014",
    "gtja_009": "gtja_068",
    "gtja_013": "a101_041",
    "gtja_017": "gtja_014",
    "gtja_018": "a158_060",
    "gtja_019": "a158_060",
    "gtja_020": "a158_060",
    "gtja_021": "gtja_024",
    "gtja_022": "a158_041",
    "gtja_023": "gtja_058",
    "gtja_027": "a360_228",
    "gtja_028": "tst_069",
    "gtja_029": "jq_220",
    "gtja_031": "a158_060",
    "gtja_033": "a101_052",
    "gtja_034": "a158_060",
    "gtja_037": "a101_008",
    "gtja_046": "a158_060",
    "gtja_049": "a360_228",
    "gtja_050": "a360_228",
    "gtja_051": "a360_228",
    "gtja_052": "a360_090",
    "gtja_055": "gtja_059",
    "gtja_057": "a360_051",
    "gtja_063": "a158_067",
    "gtja_065": "a360_238",
    "gtja_066": "a360_238",
    "gtja_067": "tst_027",
    "gtja_070": "jq_118",
    "gtja_071": "a360_219",
    "gtja_072": "gtja_047",
    "gtja_078": "a158_060",
    "gtja_079": "tst_038",
    "gtja_081": "jq_136",
    "gtja_082": "gtja_047",
    "gtja_083": "a101_016",
    "gtja_088": "a360_219",
    "gtja_089": "gtja_024",
    "gtja_093": "gtja_129",
    "gtja_095": "jq_118",
    "gtja_096": "gtja_047",
    "gtja_097": "jq_136",
    "gtja_099": "a101_016",
    "gtja_100": "jq_136",
    "gtja_102": "a158_131",
    "gtja_106": "a360_219",
    "gtja_110": "jq_107",
    "gtja_112": "a360_228",
    "gtja_113": "a101_012",
    "gtja_114": "a101_083",
    "gtja_118": "jq_103",
    "gtja_126": "gtja_010",
    "gtja_128": "jq_188",
    "gtja_132": "jq_118",
    "gtja_133": "a158_076",
    "gtja_134": "jq_219",
    "gtja_135": "tst_027",
    "gtja_136": "gtja_139",
    "gtja_140": "a101_088",
    "gtja_147": "gtja_116",
    "gtja_150": "jq_118",
    "gtja_151": "tst_015",
    "gtja_153": "gtja_010",
    "gtja_155": "jq_125",
    "gtja_158": "ta_049",
    "gtja_159": "ta_016",
    "gtja_160": "gtja_129",
    "gtja_161": "gtja_167",
    "gtja_163": "a101_049",
    "gtja_164": "gtja_167",
    "gtja_168": "a158_131",
    "gtja_173": "gtja_010",
    "gtja_174": "gtja_167",
    "gtja_175": "gtja_167",
    "gtja_177": "a158_076",
    "gtja_178": "jq_218",
    "gtja_184": "a101_037",
    "gtja_187": "gtja_167",
    "gtja_189": "gtja_167",
    "gtja_190": "gtja_166",
    "a360_001": "tst_027",
    "a360_002": "tst_027",
    "a360_003": "tst_027",
    "a360_004": "tst_027",
    "a360_005": "tst_027",
    "a360_006": "tst_027",
    "a360_007": "tst_027",
    "a360_008": "tst_027",
    "a360_009": "tst_027",
    "a360_010": "tst_027",
    "a360_011": "tst_027",
    "a360_012": "tst_027",
    "a360_013": "tst_027",
    "a360_014": "tst_027",
    "a360_015": "tst_027",
    "a360_016": "tst_027",
    "a360_017": "tst_027",
    "a360_018": "tst_027",
    "a360_019": "tst_027",
    "a360_020": "tst_027",
    "a360_021": "tst_027",
    "a360_022": "tst_027",
    "a360_023": "a360_090",
    "a360_024": "a360_090",
    "a360_025": "a360_090",
    "a360_026": "tst_027",
    "a360_027": "a360_090",
    "a360_028": "a360_090",
    "a360_029": "a360_090",
    "a360_030": "a360_090",
    "a360_031": "a360_090",
    "a360_032": "a360_219",
    "a360_033": "a360_219",
    "a360_034": "a360_219",
    "a360_035": "a360_219",
    "a360_036": "a360_219",
    "a360_037": "a360_219",
    "a360_038": "a360_219",
    "a360_039": "a360_219",
    "a360_040": "a360_219",
    "a360_041": "a360_219",
    "a360_042": "a360_219",
    "a360_043": "a360_219",
    "a360_044": "a360_228",
    "a360_045": "a360_228",
    "a360_046": "a360_228",
    "a360_047": "a360_228",
    "a360_048": "a360_228",
    "a360_049": "a360_228",
    "a360_050": "a360_228",
    "a360_052": "a158_060",
    "a360_053": "a158_060",
    "a360_054": "a158_060",
    "a360_055": "a158_060",
    "a360_056": "a360_117",
    "a360_057": "a360_238",
    "a360_058": "a360_238",
    "a360_059": "a158_001",
    "a360_061": "tst_027",
    "a360_062": "tst_027",
    "a360_063": "tst_027",
    "a360_064": "tst_027",
    "a360_065": "tst_027",
    "a360_066": "tst_027",
    "a360_067": "tst_027",
    "a360_068": "tst_027",
    "a360_069": "tst_027",
    "a360_070": "tst_027",
    "a360_071": "tst_027",
    "a360_072": "tst_027",
    "a360_073": "tst_027",
    "a360_074": "tst_027",
    "a360_075": "tst_027",
    "a360_076": "tst_027",
    "a360_077": "tst_027",
    "a360_078": "tst_027",
    "a360_079": "tst_027",
    "a360_080": "tst_027",
    "a360_081": "tst_027",
    "a360_082": "tst_027",
    "a360_083": "tst_027",
    "a360_084": "a360_090",
    "a360_085": "tst_027",
    "a360_086": "tst_027",
    "a360_087": "tst_027",
    "a360_088": "tst_027",
    "a360_089": "a360_090",
    "a360_091": "a360_090",
    "a360_092": "a360_090",
    "a360_093": "a360_219",
    "a360_094": "a360_219",
    "a360_095": "a360_219",
    "a360_096": "a360_219",
    "a360_097": "a360_219",
    "a360_098": "a360_219",
    "a360_099": "a360_219",
    "a360_100": "a360_219",
    "a360_101": "a360_219",
    "a360_102": "a360_219",
    "a360_103": "a360_219",
    "a360_104": "a360_219",
    "a360_105": "a360_228",
    "a360_106": "a360_228",
    "a360_107": "a360_228",
    "a360_108": "a360_228",
    "a360_109": "a360_228",
    "a360_110": "a360_228",
    "a360_111": "a360_228",
    "a360_112": "a158_060",
    "a360_113": "a158_060",
    "a360_114": "a158_060",
    "a360_115": "a158_060",
    "a360_116": "a158_060",
    "a360_118": "a360_238",
    "a360_119": "gtja_137",
    "a360_120": "a158_001",
    "a360_121": "tst_027",
    "a360_122": "tst_027",
    "a360_123": "tst_027",
    "a360_124": "tst_027",
    "a360_125": "tst_027",
    "a360_126": "tst_027",
    "a360_127": "tst_027",
    "a360_128": "tst_027",
    "a360_129": "tst_027",
    "a360_130": "tst_027",
    "a360_131": "tst_027",
    "a360_132": "tst_027",
    "a360_133": "tst_027",
    "a360_134": "tst_027",
    "a360_135": "tst_027",
    "a360_136": "tst_027",
    "a360_137": "tst_027",
    "a360_138": "tst_027",
    "a360_139": "tst_027",
    "a360_140": "tst_027",
    "a360_141": "a360_090",
    "a360_142": "a360_090",
    "a360_143": "a360_090",
    "a360_144": "a360_090",
    "a360_145": "a360_090",
    "a360_146": "a360_090",
    "a360_147": "a360_090",
    "a360_148": "a360_090",
    "a360_149": "a360_090",
    "a360_150": "a360_090",
    "a360_151": "a360_090",
    "a360_152": "a360_090",
    "a360_153": "a360_090",
    "a360_154": "a360_090",
    "a360_155": "a360_219",
    "a360_156": "a360_219",
    "a360_157": "a360_219",
    "a360_158": "a360_219",
    "a360_159": "a360_219",
    "a360_160": "a360_219",
    "a360_161": "a360_219",
    "a360_162": "a360_219",
    "a360_163": "a360_219",
    "a360_164": "a360_228",
    "a360_165": "a360_228",
    "a360_166": "a360_228",
    "a360_167": "a360_228",
    "a360_168": "a360_228",
    "a360_169": "a360_228",
    "a360_170": "a360_051",
    "a360_171": "a360_051",
    "a360_172": "a360_051",
    "a360_173": "a360_051",
    "a360_174": "a158_060",
    "a360_175": "gtja_014",
    "a360_176": "a360_117",
    "a360_181": "tst_027",
    "a360_182": "tst_027",
    "a360_183": "tst_027",
    "a360_184": "tst_027",
    "a360_185": "tst_027",
    "a360_186": "tst_027",
    "a360_187": "tst_027",
    "a360_188": "tst_027",
    "a360_189": "tst_027",
    "a360_190": "tst_027",
    "a360_191": "tst_027",
    "a360_192": "tst_027",
    "a360_193": "tst_027",
    "a360_194": "tst_027",
    "a360_195": "tst_027",
    "a360_196": "tst_027",
    "a360_197": "tst_027",
    "a360_198": "tst_027",
    "a360_199": "tst_027",
    "a360_200": "tst_027",
    "a360_201": "tst_027",
    "a360_202": "tst_027",
    "a360_203": "tst_027",
    "a360_204": "tst_027",
    "a360_205": "tst_027",
    "a360_206": "tst_027",
    "a360_207": "tst_027",
    "a360_208": "tst_027",
    "a360_209": "tst_027",
    "a360_210": "tst_027",
    "a360_211": "a360_219",
    "a360_212": "a360_219",
    "a360_213": "a360_219",
    "a360_214": "a360_219",
    "a360_215": "a360_219",
    "a360_216": "a360_219",
    "a360_217": "a360_219",
    "a360_218": "a360_219",
    "a360_220": "a360_219",
    "a360_221": "a360_219",
    "a360_222": "a360_219",
    "a360_223": "a360_219",
    "a360_224": "a360_219",
    "a360_225": "a360_219",
    "a360_226": "tst_038",
    "a360_227": "tst_038",
    "a360_229": "a360_228",
    "a360_230": "a360_228",
    "a360_231": "a360_228",
    "a360_232": "a158_060",
    "a360_233": "a158_060",
    "a360_234": "a158_060",
    "a360_235": "a158_060",
    "a360_236": "a158_060",
    "a360_237": "a360_238",
    "a360_240": "a158_012",
    "a360_241": "tst_027",
    "a360_242": "tst_027",
    "a360_243": "tst_027",
    "a360_244": "tst_027",
    "a360_245": "tst_027",
    "a360_246": "tst_027",
    "a360_247": "tst_027",
    "a360_248": "tst_027",
    "a360_249": "tst_027",
    "a360_250": "tst_027",
    "a360_251": "tst_027",
    "a360_252": "tst_027",
    "a360_253": "tst_027",
    "a360_254": "tst_027",
    "a360_255": "tst_027",
    "a360_256": "tst_027",
    "a360_257": "tst_027",
    "a360_258": "tst_027",
    "a360_259": "tst_027",
    "a360_260": "tst_027",
    "a360_261": "tst_027",
    "a360_262": "tst_027",
    "a360_263": "a360_090",
    "a360_264": "a360_090",
    "a360_265": "a360_090",
    "a360_266": "tst_027",
    "a360_267": "a360_090",
    "a360_268": "a360_090",
    "a360_269": "a360_090",
    "a360_270": "a360_090",
    "a360_271": "a360_090",
    "a360_272": "a360_219",
    "a360_273": "a360_219",
    "a360_274": "a360_219",
    "a360_275": "a360_219",
    "a360_276": "a360_219",
    "a360_277": "a360_219",
    "a360_278": "a360_219",
    "a360_279": "a360_219",
    "a360_280": "a360_219",
    "a360_281": "a360_219",
    "a360_282": "a360_219",
    "a360_283": "a360_219",
    "a360_284": "a360_228",
    "a360_285": "a360_228",
    "a360_286": "a360_228",
    "a360_287": "a360_228",
    "a360_288": "a360_228",
    "a360_289": "a360_228",
    "a360_290": "a360_228",
    "a360_291": "a360_228",
    "a360_292": "a158_060",
    "a360_293": "a158_060",
    "a360_294": "a158_060",
    "a360_295": "a158_060",
    "a360_296": "a360_117",
    "a360_297": "a360_117",
    "a360_298": "a360_238",
    "a360_299": "a360_239",
    "a360_300": "a101_083",
    "a360_301": "a360_302",
    "a360_303": "a360_302",
    "a360_304": "a360_306",
    "a360_305": "a360_306",
    "a360_307": "a360_306",
    "a360_308": "a360_306",
    "a360_309": "a360_310",
    "a360_311": "a360_310",
    "a360_312": "a360_310",
    "a360_313": "a360_314",
    "a360_315": "a360_314",
    "a360_316": "a360_314",
    "a360_317": "a360_318",
    "a360_319": "a360_318",
    "a360_320": "a360_318",
    "a360_322": "a360_321",
    "a360_323": "a360_324",
    "a360_325": "a360_326",
    "a360_327": "a360_326",
    "a360_328": "a360_329",
    "a360_330": "a360_329",
    "a360_331": "a360_332",
    "a360_333": "a360_334",
    "a360_335": "a360_334",
    "a360_337": "a360_338",
    "a360_339": "a360_338",
    "a360_341": "a360_340",
    "a360_343": "a360_342",
    "a360_344": "a360_345",
    "a360_346": "a360_345",
    "a360_348": "a360_347",
    "a360_354": "jq_135",
    "a360_355": "gtja_080",
    "a360_357": "a158_129",
    "a360_358": "a158_129",
    "cnc_003": "jq_130",
    "cnc_004": "jq_130",
    "cnc_005": "cnc_006",
    "cnc_007": "cnc_013",
    "cnc_008": "svc_003",
    "cnc_009": "jq_118",
    "cnc_010": "cnc_012",
    "cnc_011": "cnc_012",
    "cnc_014": "gtja_144",
    "cnc_015": "cne6_005",
    "cnc_016": "cne6_005",
    "cnc_017": "cne6_005",
    "cnc_020": "jq_154",
    "cnc_026": "a360_219",
    "cnc_037": "ta_016",
    "cnc_063": "cnc_061",
    "cnc_064": "svc_002",
    "cnc_065": "oap_060",
    "cnc_067": "oap_060",
    "cnc_071": "cnc_061",
    "cnc_106": "cnc_061",
    "svc_007": "cne6_005",
    "svc_012": "cne6_005",
    "svc_019": "cne6_005",
    "svc_020": "jkp_100",
    "svc_021": "a360_219",
    "ta_005": "ta_004",
    "ta_006": "a158_001",
    "ta_007": "a158_060",
    "ta_009": "a360_090",
    "ta_010": "a360_090",
    "ta_011": "a158_043",
    "ta_012": "jq_188",
    "ta_017": "a360_219",
    "ta_018": "a360_228",
    "ta_019": "tst_038",
    "ta_022": "tst_069",
    "ta_024": "a158_064",
    "ta_028": "a158_067",
    "ta_029": "a158_067",
    "ta_031": "a360_051",
    "ta_032": "tst_038",
    "ta_033": "a360_228",
    "ta_034": "a360_219",
    "ta_035": "a158_060",
    "ta_036": "tst_038",
    "ta_037": "a360_228",
    "ta_038": "a360_051",
    "ta_040": "a360_219",
    "ta_041": "a360_117",
    "ta_042": "a158_042",
    "ta_043": "a360_219",
    "ta_044": "tst_038",
    "ta_045": "gtja_060",
    "ta_046": "ta_027",
    "ta_047": "gtja_084",
    "ta_048": "ta_016",
    "ta_050": "a158_001",
    "ta_051": "gtja_124",
    "ta_061": "a360_228",
    "ta_062": "a360_228",
    "ta_063": "ta_030",
    "ta_064": "ta_060",
    "jq_002": "cnc_069",
    "jq_069": "cnc_069",
    "jq_097": "cnc_068",
    "jq_105": "ta_016",
    "jq_106": "ta_016",
    "jq_108": "jq_110",
    "jq_109": "svc_003",
    "jq_111": "jq_138",
    "jq_112": "jq_118",
    "jq_113": "gtja_053",
    "jq_115": "jq_118",
    "jq_116": "jq_118",
    "jq_117": "jq_118",
    "jq_119": "jq_120",
    "jq_121": "jq_136",
    "jq_122": "jq_136",
    "jq_123": "jq_136",
    "jq_124": "jq_136",
    "jq_126": "jq_130",
    "jq_127": "cnc_006",
    "jq_128": "jq_130",
    "jq_129": "cnc_006",
    "jq_131": "jq_130",
    "jq_132": "gtja_145",
    "jq_133": "gtja_040",
    "jq_134": "a360_347",
    "jq_137": "jq_136",
    "jq_152": "a360_219",
    "jq_153": "tst_027",
    "jq_155": "gtja_166",
    "jq_157": "cne6_005",
    "jq_158": "oap_015",
    "jq_159": "cne6_005",
    "jq_175": "tst_038",
    "jq_177": "a158_060",
    "jq_178": "a158_060",
    "jq_179": "tst_027",
    "jq_180": "tst_038",
    "jq_181": "tst_038",
    "jq_182": "a158_060",
    "jq_183": "tst_027",
    "jq_184": "a360_228",
    "jq_185": "a360_238",
    "jq_186": "tst_027",
    "jq_187": "a158_043",
    "jq_189": "cnc_038",
    "jq_192": "a158_060",
    "jq_193": "a360_051",
    "jq_194": "a158_060",
    "jq_195": "a360_228",
    "jq_196": "a360_238",
    "jq_197": "tst_027",
    "jq_198": "tst_038",
    "jq_199": "a158_060",
    "jq_200": "a158_067",
    "jq_201": "a158_067",
    "jq_202": "tst_027",
    "jq_203": "a360_219",
    "jq_206": "a360_051",
    "jq_207": "a360_219",
    "jq_208": "a360_117",
    "jq_209": "a360_228",
    "jq_210": "tst_027",
    "jq_211": "tst_027",
    "jq_212": "a360_219",
    "jq_213": "a360_228",
    "jq_214": "jq_151",
    "jq_215": "a360_219",
    "jq_216": "a158_060",
    "jq_217": "tst_027",
    "jq_221": "a360_219",
    "jq_222": "a360_228",
    "jq_223": "a360_219",
    "jq_228": "cne6_002",
    "jq_229": "cnc_061",
    "jq_230": "jq_130",
    "jq_231": "svc_002",
    "oap_002": "cnc_038",
    "oap_003": "a360_219",
    "oap_009": "jkp_100",
    "oap_018": "ta_016",
    "oap_019": "oap_015",
    "oap_020": "cne6_005",
    "oap_037": "cnc_012",
    "oap_044": "jq_130",
    "oap_045": "cnc_006",
    "oap_048": "cnc_061",
    "oap_049": "cnc_061",
    "oap_050": "cnc_061",
    "oap_052": "cnc_062",
    "oap_053": "svc_002",
    "oap_054": "oap_060",
    "oap_055": "cnc_069",
    "oap_056": "cnc_068",
    "oap_061": "cnc_061",
    "oap_063": "jkp_017",
    "oap_064": "cnc_061",
    "cne6_004": "cne6_005",
    "cne6_007": "jq_130",
    "cne6_008": "jq_130",
    "cne6_009": "cnc_006",
    "cne6_010": "cnc_006",
    "cne6_011": "a360_228",
    "cne6_013": "tst_027",
    "cne6_017": "cnc_061",
    "cne6_018": "svc_002",
    "cne6_019": "cnc_069",
    "cne6_020": "svc_002",
    "cne6_038": "jkp_016",
    "tst_001": "a360_090",
    "tst_002": "a360_090",
    "tst_003": "ta_016",
    "tst_004": "a158_060",
    "tst_005": "a360_238",
    "tst_006": "a158_060",
    "tst_007": "a360_219",
    "tst_009": "tst_038",
    "tst_010": "jq_103",
    "tst_011": "jq_107",
    "tst_012": "a158_060",
    "tst_013": "a360_219",
    "tst_014": "a360_090",
    "tst_016": "ta_015",
    "tst_017": "ta_013",
    "tst_020": "a360_219",
    "tst_021": "a360_219",
    "tst_022": "a360_238",
    "tst_023": "a158_060",
    "tst_024": "tst_038",
    "tst_025": "tst_038",
    "tst_026": "tst_038",
    "tst_028": "tst_027",
    "tst_029": "a360_228",
    "tst_031": "a158_060",
    "tst_032": "tst_038",
    "tst_033": "a360_051",
    "tst_034": "gtja_047",
    "tst_035": "tst_069",
    "tst_037": "tst_038",
    "tst_039": "a360_238",
    "tst_040": "a158_060",
    "tst_041": "a360_228",
    "tst_042": "a360_219",
    "tst_043": "tst_027",
    "tst_044": "tst_027",
    "tst_045": "tst_027",
    "tst_046": "a360_090",
    "tst_047": "a360_090",
    "tst_048": "a158_043",
    "tst_049": "jq_205",
    "tst_050": "jq_205",
    "tst_051": "jq_188",
    "tst_052": "a360_228",
    "tst_053": "a360_228",
    "tst_054": "gtja_084",
    "tst_055": "gtja_053",
    "tst_056": "gtja_053",
    "tst_057": "a360_228",
    "tst_058": "a360_228",
    "tst_059": "a158_067",
    "tst_060": "tst_038",
    "tst_061": "tst_027",
    "tst_063": "a360_228",
    "tst_064": "tst_038",
    "tst_065": "a360_090",
    "tst_066": "tst_015",
    "tst_067": "gtja_040",
    "tst_068": "a158_060",
    "tst_070": "a360_238",
    "tst_071": "a158_067",
    "jkp_010": "cnc_061",
    "jkp_012": "cnc_061",
    "jkp_014": "svc_002",
    "jkp_021": "cnc_068",
    "jkp_022": "cnc_069",
    "jkp_023": "svc_002",
    "jkp_024": "cnc_061",
    "jkp_026": "cnc_061",
    "jkp_037": "cnc_078",
    "jkp_106": "ta_016",
    "jkp_107": "cne6_005",
    "jkp_108": "ta_016",
    "jkp_109": "oap_015",
    "jkp_110": "oap_021",
    "jkp_111": "oap_022",
    "jkp_113": "cnc_006",
    "jkp_114": "cnc_013",
    "jkp_115": "cnc_012",
    "jkp_116": "cnc_013",
    "jkp_117": "cnc_038",
    "jkp_119": "jq_130",
    "jkp_120": "cnc_006",
    "jkp_121": "oap_046",
    "jkp_122": "oap_015",
    "jkp_123": "oap_016",
    "jkp_125": "oap_017",
    "jkp_127": "a360_219",
    "jkp_129": "oap_047",
    "jkp_134": "ta_016",
    "jkp_135": "a360_219",
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
