"""Locked membership of the remove_duplicates factor set.

remove_duplicates keeps every factor of the twelve formula libraries except duplicates: no Rank IC, t-statistic,
yearly or long-side threshold removes a factor. Only the choice of which member of a group of near duplicates survives
uses returns:
- the absolute mean daily cross-sectional Spearman correlation of every pair of the 1696 library factors was measured
  on the ``sh_main`` stores on every 5th trading date from 2008-05-12 to 2026-10-20 (897 dates;
  ``rquant.factors.selection.RankCorrelationAccumulator``, scripts in ``scripts/remove_duplicates``);
- factors are ordered by their mean yearly long excess in the sh_main 1d evaluation of 2020-01-02 to 2026-10-21 listed
  in ``provenance`` (the evaluation recent_sh_main was selected from; the long side is the top or bottom 20% by the
  sign of the 2020+ Rank IC, against the equal-weight sh_main universe), factors without a score last;
- ``prune_correlated`` then drops a factor when its correlation with an already kept factor reaches
  ``max_abs_rank_correlation`` (0.8). ``DEDUPLICATED`` maps each dropped factor to the kept one it duplicates.

Kept per source: qlib_alpha158 66, wq_alpha101 75, gtja191 94, qlib_alpha360 43, china_chars 57, size_value_china 10,
talib 85, jq240 95, oap 65, barra_cne6 11, tushare_tech 13, jkp153 58.

The ordering uses 2020-2026 sh_main returns, so for walk-forward test years from 2020 on the choice among duplicates
is in-sample; membership is only meant for the ``sh_main`` universe. Members keep their source canonical names and
values; ``direction == -1`` marks a factor whose 2020+ Rank IC is negative, the stored column is not negated.
"""

from __future__ import annotations

from rquant.factors.libraries.base import FactorSelection, SelectedFactor

REMOVE_DUPLICATES_MEMBERS = (
    SelectedFactor("a158_001", -1),  # long excess +8.8%, RankIC -0.0282, cov 1.000
    SelectedFactor("a158_005", 1),  # long excess +2.2%, RankIC +0.0010, cov 1.000
    SelectedFactor("a158_006", -1),  # long excess +0.5%, RankIC -0.0247, cov 1.000
    SelectedFactor("a158_011", -1),  # long excess -21.5%, RankIC -0.0075, cov 1.000
    SelectedFactor("a158_012", 1),  # long excess +2.1%, RankIC +0.0448, cov 1.000
    SelectedFactor("a158_013", 1),  # long excess +12.9%, RankIC +0.0272, cov 1.000
    SelectedFactor("a158_024", -1),  # long excess +2.1%, RankIC -0.0414, cov 0.999
    SelectedFactor("a158_025", -1),  # long excess +3.6%, RankIC -0.0398, cov 0.998
    SelectedFactor("a158_028", -1),  # long excess +2.1%, RankIC -0.0229, cov 0.984
    SelectedFactor("a158_034", 1),  # long excess -4.2%, RankIC +0.0015, cov 1.000
    SelectedFactor("a158_035", -1),  # long excess +5.8%, RankIC -0.0018, cov 1.000
    SelectedFactor("a158_036", 1),  # long excess -0.8%, RankIC +0.0000, cov 1.000
    SelectedFactor("a158_037", 1),  # long excess +2.8%, RankIC +0.0019, cov 1.000
    SelectedFactor("a158_038", 1),  # long excess +4.8%, RankIC +0.0020, cov 1.000
    SelectedFactor("a158_039", -1),  # long excess -0.2%, RankIC -0.0175, cov 1.000
    SelectedFactor("a158_040", -1),  # long excess -3.7%, RankIC -0.0153, cov 1.000
    SelectedFactor("a158_041", -1),  # long excess -3.6%, RankIC -0.0158, cov 1.000
    SelectedFactor("a158_042", -1),  # long excess -4.0%, RankIC -0.0175, cov 1.000
    SelectedFactor("a158_043", -1),  # long excess -3.5%, RankIC -0.0257, cov 1.000
    SelectedFactor("a158_044", -1),  # long excess -18.2%, RankIC -0.0099, cov 0.999
    SelectedFactor("a158_045", -1),  # long excess -16.1%, RankIC -0.0066, cov 0.998
    SelectedFactor("a158_049", 1),  # long excess +0.3%, RankIC +0.0448, cov 0.999
    SelectedFactor("a158_053", 1),  # long excess -7.3%, RankIC +0.0396, cov 0.984
    SelectedFactor("a158_058", 1),  # long excess +6.2%, RankIC +0.0296, cov 0.984
    SelectedFactor("a158_064", -1),  # long excess -2.5%, RankIC -0.0228, cov 0.999
    SelectedFactor("a158_072", -1),  # long excess -4.4%, RankIC -0.0240, cov 1.000
    SelectedFactor("a158_077", -1),  # long excess +3.6%, RankIC -0.0247, cov 0.992
    SelectedFactor("a158_079", 1),  # long excess +4.3%, RankIC +0.0145, cov 0.999
    SelectedFactor("a158_080", 1),  # long excess +2.8%, RankIC +0.0127, cov 0.998
    SelectedFactor("a158_081", 1),  # long excess +0.5%, RankIC +0.0153, cov 0.995
    SelectedFactor("a158_082", 1),  # long excess +1.2%, RankIC +0.0164, cov 0.992
    SelectedFactor("a158_083", 1),  # long excess +1.6%, RankIC +0.0131, cov 0.984
    SelectedFactor("a158_084", -1),  # long excess +1.2%, RankIC -0.0177, cov 0.999
    SelectedFactor("a158_085", -1),  # long excess +1.3%, RankIC -0.0183, cov 0.998
    SelectedFactor("a158_088", -1),  # long excess +3.8%, RankIC -0.0199, cov 0.984
    SelectedFactor("a158_089", -1),  # long excess +2.5%, RankIC -0.0259, cov 1.000
    SelectedFactor("a158_090", -1),  # long excess +5.9%, RankIC -0.0307, cov 1.000
    SelectedFactor("a158_091", -1),  # long excess +5.6%, RankIC -0.0272, cov 1.000
    SelectedFactor("a158_093", -1),  # long excess +0.2%, RankIC -0.0123, cov 1.000
    SelectedFactor("a158_094", -1),  # long excess -0.6%, RankIC -0.0230, cov 1.000
    SelectedFactor("a158_095", -1),  # long excess -0.4%, RankIC -0.0253, cov 1.000
    SelectedFactor("a158_096", -1),  # long excess +1.6%, RankIC -0.0266, cov 1.000
    SelectedFactor("a158_097", -1),  # long excess +2.2%, RankIC -0.0249, cov 1.000
    SelectedFactor("a158_098", -1),  # long excess +2.1%, RankIC -0.0195, cov 1.000
    SelectedFactor("a158_102", -1),  # long excess +2.2%, RankIC -0.0164, cov 1.000
    SelectedFactor("a158_103", -1),  # long excess +3.1%, RankIC -0.0137, cov 1.000
    SelectedFactor("a158_104", 1),  # long excess +1.8%, RankIC +0.0132, cov 1.000
    SelectedFactor("a158_105", 1),  # long excess +0.4%, RankIC +0.0095, cov 1.000
    SelectedFactor("a158_106", 1),  # long excess +1.2%, RankIC +0.0088, cov 1.000
    SelectedFactor("a158_107", 1),  # long excess +2.6%, RankIC +0.0079, cov 1.000
    SelectedFactor("a158_108", 1),  # long excess +2.0%, RankIC +0.0034, cov 1.000
    SelectedFactor("a158_117", -1),  # long excess +4.2%, RankIC -0.0303, cov 0.992
    SelectedFactor("a158_134", -1),  # long excess +2.0%, RankIC -0.0141, cov 0.999
    SelectedFactor("a158_135", -1),  # long excess +0.3%, RankIC -0.0066, cov 0.998
    SelectedFactor("a158_137", 1),  # long excess -1.8%, RankIC +0.0113, cov 0.992
    SelectedFactor("a158_138", 1),  # long excess +3.7%, RankIC +0.0240, cov 0.984
    SelectedFactor("a158_139", -1),  # long excess +1.7%, RankIC -0.0094, cov 0.999
    SelectedFactor("a158_140", -1),  # long excess +3.7%, RankIC -0.0120, cov 0.997
    SelectedFactor("a158_141", -1),  # long excess +4.4%, RankIC -0.0124, cov 0.995
    SelectedFactor("a158_142", -1),  # long excess +4.2%, RankIC -0.0109, cov 0.992
    SelectedFactor("a158_143", -1),  # long excess +2.8%, RankIC -0.0058, cov 0.984
    SelectedFactor("a158_144", -1),  # long excess +0.9%, RankIC -0.0192, cov 0.999
    SelectedFactor("a158_145", -1),  # long excess -0.4%, RankIC -0.0233, cov 0.997
    SelectedFactor("a158_146", -1),  # long excess +2.0%, RankIC -0.0281, cov 0.995
    SelectedFactor("a158_147", -1),  # long excess +0.9%, RankIC -0.0307, cov 0.992
    SelectedFactor("a158_148", -1),  # long excess +3.2%, RankIC -0.0333, cov 0.984
    SelectedFactor("a101_001", -1),  # long excess +0.2%, RankIC -0.0168, cov 0.994
    SelectedFactor("a101_002", 1),  # long excess +1.6%, RankIC +0.0143, cov 1.000
    SelectedFactor("a101_004", 1),  # long excess +3.9%, RankIC +0.0241, cov 0.998
    SelectedFactor("a101_005", 1),  # long excess -0.8%, RankIC +0.0121, cov 0.998
    SelectedFactor("a101_007", -1),  # long excess -4.4%, RankIC -0.0141, cov 0.990
    SelectedFactor("a101_008", 1),  # long excess -6.1%, RankIC +0.0117, cov 0.996
    SelectedFactor("a101_010", 1),  # long excess -7.9%, RankIC +0.0119, cov 1.000
    SelectedFactor("a101_015", 1),  # long excess +3.7%, RankIC +0.0219, cov 1.000
    SelectedFactor("a101_018", 1),  # long excess +3.9%, RankIC +0.0114, cov 0.999
    SelectedFactor("a101_019", 1),  # long excess -0.3%, RankIC +0.0189, cov 0.916
    SelectedFactor("a101_020", -1),  # long excess +1.1%, RankIC -0.0134, cov 1.000
    SelectedFactor("a101_022", 1),  # long excess -3.0%, RankIC +0.0051, cov 0.995
    SelectedFactor("a101_023", 1),  # long excess -0.1%, RankIC +0.0152, cov 1.000
    SelectedFactor("a101_024", 1),  # long excess +2.5%, RankIC +0.0215, cov 0.999
    SelectedFactor("a101_026", 1),  # long excess +1.4%, RankIC +0.0211, cov 1.000
    SelectedFactor("a101_027", 1),  # long excess +4.2%, RankIC +0.0198, cov 1.000
    SelectedFactor("a101_030", 1),  # long excess -0.3%, RankIC +0.0033, cov 0.999
    SelectedFactor("a101_031", 1),  # long excess +0.9%, RankIC +0.0073, cov 0.995
    SelectedFactor("a101_032", 1),  # long excess +3.6%, RankIC +0.0001, cov 0.921
    SelectedFactor("a101_034", 1),  # long excess +3.8%, RankIC +0.0167, cov 1.000
    SelectedFactor("a101_036", 1),  # long excess -1.2%, RankIC +0.0085, cov 0.933
    SelectedFactor("a101_037", 1),  # long excess +6.4%, RankIC +0.0235, cov 0.933
    SelectedFactor("a101_038", 1),  # long excess +3.3%, RankIC +0.0214, cov 0.998
    SelectedFactor("a101_039", 1),  # long excess -3.9%, RankIC +0.0179, cov 0.916
    SelectedFactor("a101_040", 1),  # long excess +9.7%, RankIC +0.0418, cov 0.998
    SelectedFactor("a101_041", 1),  # long excess -2.4%, RankIC +0.0106, cov 1.000
    SelectedFactor("a101_044", 1),  # long excess +4.9%, RankIC +0.0272, cov 1.000
    SelectedFactor("a101_046", 1),  # long excess +1.5%, RankIC +0.0169, cov 1.000
    SelectedFactor("a101_047", 1),  # long excess +10.5%, RankIC +0.0033, cov 1.000
    SelectedFactor("a101_048", -1),  # long excess +1.4%, RankIC -0.0036, cov 0.912
    SelectedFactor("a101_049", 1),  # long excess -4.4%, RankIC +0.0199, cov 1.000
    SelectedFactor("a101_052", 1),  # long excess -2.2%, RankIC +0.0145, cov 0.919
    SelectedFactor("a101_053", 1),  # long excess +2.8%, RankIC +0.0108, cov 0.999
    SelectedFactor("a101_056", 1),  # long excess +2.5%, RankIC +0.0119, cov 0.997
    SelectedFactor("a101_058", 1),  # long excess +1.4%, RankIC +0.0102, cov 0.992
    SelectedFactor("a101_059", 1),  # long excess +2.4%, RankIC +0.0114, cov 0.987
    SelectedFactor("a101_060", 1),  # long excess +8.2%, RankIC +0.0123, cov 0.998
    SelectedFactor("a101_061", 1),  # long excess +4.5%, RankIC +0.0156, cov 1.000
    SelectedFactor("a101_062", 1),  # long excess +3.8%, RankIC +0.0101, cov 1.000
    SelectedFactor("a101_063", 1),  # long excess +5.5%, RankIC +0.0088, cov 0.916
    SelectedFactor("a101_064", 1),  # long excess +1.5%, RankIC +0.0022, cov 1.000
    SelectedFactor("a101_065", 1),  # long excess +5.1%, RankIC +0.0154, cov 1.000
    SelectedFactor("a101_066", 1),  # long excess +0.6%, RankIC +0.0167, cov 0.997
    SelectedFactor("a101_067", -1),  # long excess -3.1%, RankIC -0.0011, cov 0.958
    SelectedFactor("a101_068", 1),  # long excess +2.3%, RankIC +0.0080, cov 1.000
    SelectedFactor("a101_069", 1),  # long excess +6.4%, RankIC +0.0247, cov 0.988
    SelectedFactor("a101_070", 1),  # long excess -2.2%, RankIC +0.0119, cov 0.969
    SelectedFactor("a101_071", 1),  # long excess +4.7%, RankIC +0.0155, cov 0.995
    SelectedFactor("a101_072", 1),  # long excess +2.1%, RankIC +0.0071, cov 0.955
    SelectedFactor("a101_073", 1),  # long excess +1.4%, RankIC +0.0181, cov 0.995
    SelectedFactor("a101_074", 1),  # long excess +3.8%, RankIC +0.0105, cov 1.000
    SelectedFactor("a101_075", 1),  # long excess +3.0%, RankIC +0.0092, cov 1.000
    SelectedFactor("a101_076", 1),  # long excess +0.6%, RankIC +0.0152, cov 0.950
    SelectedFactor("a101_077", 1),  # long excess +0.8%, RankIC +0.0132, cov 1.000
    SelectedFactor("a101_078", 1),  # long excess -1.1%, RankIC +0.0050, cov 0.983
    SelectedFactor("a101_079", 1),  # long excess +6.7%, RankIC +0.0097, cov 1.000
    SelectedFactor("a101_080", 1),  # long excess +4.6%, RankIC +0.0083, cov 0.990
    SelectedFactor("a101_081", 1),  # long excess +2.7%, RankIC +0.0092, cov 1.000
    SelectedFactor("a101_082", 1),  # long excess +0.2%, RankIC +0.0114, cov 0.987
    SelectedFactor("a101_083", 1),  # long excess +11.1%, RankIC +0.0303, cov 0.997
    SelectedFactor("a101_085", 1),  # long excess -3.1%, RankIC +0.0032, cov 1.000
    SelectedFactor("a101_086", 1),  # long excess +5.4%, RankIC +0.0121, cov 1.000
    SelectedFactor("a101_087", 1),  # long excess +2.9%, RankIC +0.0155, cov 0.959
    SelectedFactor("a101_088", 1),  # long excess +2.8%, RankIC +0.0478, cov 0.998
    SelectedFactor("a101_089", 1),  # long excess +2.4%, RankIC +0.0057, cov 0.989
    SelectedFactor("a101_090", 1),  # long excess -2.6%, RankIC +0.0022, cov 0.951
    SelectedFactor("a101_091", 1),  # long excess +0.3%, RankIC +0.0018, cov 0.984
    SelectedFactor("a101_092", 1),  # long excess +2.5%, RankIC +0.0043, cov 1.000
    SelectedFactor("a101_093", 1),  # long excess -0.7%, RankIC +0.0077, cov 0.955
    SelectedFactor("a101_095", 1),  # long excess +4.0%, RankIC +0.0111, cov 1.000
    SelectedFactor("a101_096", 1),  # long excess -0.8%, RankIC +0.0041, cov 1.000
    SelectedFactor("a101_097", -1),  # long excess +3.9%, RankIC -0.0055, cov 0.249
    SelectedFactor("a101_098", -1),  # long excess +0.8%, RankIC -0.0050, cov 0.990
    SelectedFactor("a101_099", 1),  # long excess +2.2%, RankIC +0.0045, cov 1.000
    SelectedFactor("a101_100", 1),  # long excess +7.6%, RankIC +0.0098, cov 0.986
    SelectedFactor("gtja_001", 1),  # long excess +3.8%, RankIC +0.0188, cov 0.998
    SelectedFactor("gtja_002", 1),  # long excess +2.4%, RankIC +0.0110, cov 0.997
    SelectedFactor("gtja_003", -1),  # long excess +1.0%, RankIC -0.0265, cov 0.998
    SelectedFactor("gtja_004", 1),  # long excess +2.9%, RankIC +0.0013, cov 0.996
    SelectedFactor("gtja_006", 1),  # long excess +1.6%, RankIC +0.0139, cov 0.999
    SelectedFactor("gtja_007", 1),  # long excess +9.5%, RankIC +0.0130, cov 0.999
    SelectedFactor("gtja_009", -1),  # long excess -0.4%, RankIC -0.0195, cov 1.000
    SelectedFactor("gtja_010", -1),  # long excess +9.8%, RankIC -0.0216, cov 0.994
    SelectedFactor("gtja_011", -1),  # long excess +2.0%, RankIC -0.0156, cov 0.992
    SelectedFactor("gtja_012", 1),  # long excess -5.2%, RankIC +0.0202, cov 0.998
    SelectedFactor("gtja_015", 1),  # long excess +3.5%, RankIC +0.0126, cov 1.000
    SelectedFactor("gtja_016", 1),  # long excess +2.8%, RankIC +0.0235, cov 0.998
    SelectedFactor("gtja_017", 1),  # long excess -3.0%, RankIC +0.0201, cov 0.996
    SelectedFactor("gtja_022", -1),  # long excess -4.3%, RankIC -0.0141, cov 0.998
    SelectedFactor("gtja_023", -1),  # long excess +0.8%, RankIC -0.0158, cov 0.995
    SelectedFactor("gtja_024", -1),  # long excess -2.5%, RankIC -0.0245, cov 0.999
    SelectedFactor("gtja_026", 1),  # long excess +0.3%, RankIC +0.0088, cov 0.921
    SelectedFactor("gtja_029", -1),  # long excess -2.8%, RankIC -0.0264, cov 0.999
    SelectedFactor("gtja_035", 1),  # long excess -1.0%, RankIC +0.0170, cov 0.994
    SelectedFactor("gtja_039", 1),  # long excess +0.7%, RankIC +0.0122, cov 0.919
    SelectedFactor("gtja_040", -1),  # long excess +2.9%, RankIC -0.0255, cov 0.993
    SelectedFactor("gtja_041", 1),  # long excess +3.7%, RankIC +0.0382, cov 0.998
    SelectedFactor("gtja_043", -1),  # long excess -1.4%, RankIC -0.0230, cov 0.998
    SelectedFactor("gtja_044", -1),  # long excess +0.8%, RankIC -0.0090, cov 0.993
    SelectedFactor("gtja_045", -1),  # long excess -0.5%, RankIC -0.0151, cov 0.946
    SelectedFactor("gtja_048", 1),  # long excess +1.1%, RankIC +0.0223, cov 0.995
    SelectedFactor("gtja_056", 1),  # long excess +3.3%, RankIC +0.0149, cov 0.981
    SelectedFactor("gtja_059", -1),  # long excess +0.4%, RankIC -0.0280, cov 0.995
    SelectedFactor("gtja_061", 1),  # long excess +2.2%, RankIC +0.0268, cov 0.968
    SelectedFactor("gtja_064", 1),  # long excess +4.3%, RankIC +0.0206, cov 0.974
    SelectedFactor("gtja_068", -1),  # long excess +1.4%, RankIC -0.0221, cov 1.000
    SelectedFactor("gtja_069", -1),  # long excess +3.3%, RankIC -0.0291, cov 0.995
    SelectedFactor("gtja_073", 1),  # long excess -0.7%, RankIC +0.0020, cov 0.991
    SelectedFactor("gtja_074", -1),  # long excess +3.0%, RankIC -0.0220, cov 0.983
    SelectedFactor("gtja_075", -1),  # long excess +1.2%, RankIC -0.0154, cov 0.987
    SelectedFactor("gtja_076", -1),  # long excess +1.0%, RankIC -0.0058, cov 0.995
    SelectedFactor("gtja_079", -1),  # long excess +0.0%, RankIC -0.0334, cov 1.000
    SelectedFactor("gtja_082", 1),  # long excess +0.8%, RankIC +0.0139, cov 0.999
    SelectedFactor("gtja_084", -1),  # long excess -2.0%, RankIC -0.0245, cov 0.995
    SelectedFactor("gtja_085", 1),  # long excess +4.6%, RankIC +0.0057, cov 0.990
    SelectedFactor("gtja_090", 1),  # long excess +3.1%, RankIC +0.0236, cov 0.999
    SelectedFactor("gtja_091", 1),  # long excess -4.0%, RankIC +0.0025, cov 0.989
    SelectedFactor("gtja_092", 1),  # long excess +3.5%, RankIC +0.0167, cov 0.930
    SelectedFactor("gtja_094", -1),  # long excess +1.2%, RankIC -0.0249, cov 0.992
    SelectedFactor("gtja_099", 1),  # long excess +2.0%, RankIC +0.0309, cov 0.999
    SelectedFactor("gtja_102", -1),  # long excess -2.5%, RankIC -0.0216, cov 1.000
    SelectedFactor("gtja_105", 1),  # long excess +5.6%, RankIC +0.0170, cov 0.998
    SelectedFactor("gtja_108", 1),  # long excess +0.5%, RankIC +0.0152, cov 0.960
    SelectedFactor("gtja_109", -1),  # long excess -3.0%, RankIC -0.0287, cov 1.000
    SelectedFactor("gtja_111", 1),  # long excess +2.9%, RankIC +0.0185, cov 0.998
    SelectedFactor("gtja_113", 1),  # long excess -2.7%, RankIC +0.0141, cov 0.994
    SelectedFactor("gtja_117", 1),  # long excess -0.0%, RankIC +0.0134, cov 0.991
    SelectedFactor("gtja_119", -1),  # long excess +2.0%, RankIC -0.0027, cov 0.985
    SelectedFactor("gtja_120", 1),  # long excess +14.1%, RankIC +0.0357, cov 1.000
    SelectedFactor("gtja_121", 1),  # long excess -1.1%, RankIC +0.0331, cov 0.977
    SelectedFactor("gtja_122", -1),  # long excess +1.5%, RankIC -0.0043, cov 1.000
    SelectedFactor("gtja_125", 1),  # long excess -2.0%, RankIC +0.0044, cov 0.964
    SelectedFactor("gtja_127", -1),  # long excess -1.7%, RankIC -0.0105, cov 0.994
    SelectedFactor("gtja_130", 1),  # long excess +2.3%, RankIC +0.0080, cov 0.985
    SelectedFactor("gtja_131", -1),  # long excess -1.6%, RankIC -0.0122, cov 0.975
    SelectedFactor("gtja_135", -1),  # long excess +5.7%, RankIC -0.0273, cov 0.998
    SelectedFactor("gtja_137", -1),  # long excess -0.1%, RankIC -0.0257, cov 1.000
    SelectedFactor("gtja_138", 1),  # long excess +1.3%, RankIC +0.0132, cov 0.962
    SelectedFactor("gtja_139", 1),  # long excess +6.9%, RankIC +0.0185, cov 0.998
    SelectedFactor("gtja_141", 1),  # long excess +3.9%, RankIC +0.0097, cov 0.994
    SelectedFactor("gtja_142", 1),  # long excess -2.9%, RankIC +0.0127, cov 0.994
    SelectedFactor("gtja_143", -1),  # long excess +0.3%, RankIC -0.0057, cov 1.000
    SelectedFactor("gtja_146", -1),  # long excess +4.4%, RankIC -0.0093, cov 0.995
    SelectedFactor("gtja_147", -1),  # long excess -2.4%, RankIC -0.0210, cov 0.994
    SelectedFactor("gtja_148", 1),  # long excess +3.8%, RankIC +0.0165, cov 0.980
    SelectedFactor("gtja_149", -1),  # long excess -3.3%, RankIC -0.0064, cov 0.942
    SelectedFactor("gtja_152", -1),  # long excess -0.4%, RankIC -0.0121, cov 0.991
    SelectedFactor("gtja_154", -1),  # long excess -0.5%, RankIC -0.0057, cov 0.934
    SelectedFactor("gtja_156", 1),  # long excess +6.6%, RankIC +0.0313, cov 0.998
    SelectedFactor("gtja_157", 1),  # long excess +3.7%, RankIC +0.0235, cov 0.997
    SelectedFactor("gtja_160", -1),  # long excess +5.1%, RankIC -0.0251, cov 0.995
    SelectedFactor("gtja_162", -1),  # long excess -4.0%, RankIC -0.0139, cov 0.997
    SelectedFactor("gtja_164", 1),  # long excess +5.8%, RankIC +0.0312, cov 0.995
    SelectedFactor("gtja_165", 1),  # long excess +1.5%, RankIC +0.0041, cov 0.954
    SelectedFactor("gtja_169", -1),  # long excess -2.6%, RankIC -0.0162, cov 0.993
    SelectedFactor("gtja_170", 1),  # long excess +3.2%, RankIC +0.0080, cov 0.995
    SelectedFactor("gtja_171", 1),  # long excess +1.1%, RankIC +0.0098, cov 0.954
    SelectedFactor("gtja_172", -1),  # long excess +1.6%, RankIC -0.0050, cov 0.995
    SelectedFactor("gtja_174", -1),  # long excess +5.8%, RankIC -0.0304, cov 0.995
    SelectedFactor("gtja_176", -1),  # long excess +2.3%, RankIC -0.0214, cov 0.996
    SelectedFactor("gtja_177", -1),  # long excess +0.0%, RankIC -0.0231, cov 0.995
    SelectedFactor("gtja_179", -1),  # long excess +2.2%, RankIC -0.0217, cov 0.984
    SelectedFactor("gtja_181", 1),  # long excess -6.8%, RankIC +0.0006, cov 0.990
    SelectedFactor("gtja_182", 1),  # long excess +2.4%, RankIC +0.0065, cov 0.995
    SelectedFactor("gtja_183", 1),  # long excess +2.1%, RankIC +0.0105, cov 0.981
    SelectedFactor("gtja_185", 1),  # long excess -1.5%, RankIC +0.0293, cov 1.000
    SelectedFactor("gtja_188", -1),  # long excess -8.3%, RankIC -0.0129, cov 1.000
    SelectedFactor("gtja_189", -1),  # long excess +8.1%, RankIC -0.0370, cov 0.997
    SelectedFactor("gtja_191", -1),  # long excess +1.9%, RankIC -0.0076, cov 0.994
    SelectedFactor("a360_060", 1),  # long excess n/a, RankIC n/a, cov 1.000
    SelectedFactor("a360_071", 1),  # long excess +9.6%, RankIC +0.0330, cov 0.999
    SelectedFactor("a360_098", 1),  # long excess +2.0%, RankIC +0.0341, cov 0.999
    SelectedFactor("a360_117", 1),  # long excess -3.9%, RankIC +0.0231, cov 0.999
    SelectedFactor("a360_173", 1),  # long excess +0.1%, RankIC +0.0233, cov 0.999
    SelectedFactor("a360_175", 1),  # long excess -1.6%, RankIC +0.0208, cov 0.999
    SelectedFactor("a360_178", 1),  # long excess -2.8%, RankIC +0.0118, cov 1.000
    SelectedFactor("a360_179", 1),  # long excess -2.7%, RankIC +0.0061, cov 1.000
    SelectedFactor("a360_229", 1),  # long excess -1.1%, RankIC +0.0337, cov 0.999
    SelectedFactor("a360_238", 1),  # long excess -0.1%, RankIC +0.0338, cov 1.000
    SelectedFactor("a360_239", 1),  # long excess +1.8%, RankIC +0.0361, cov 1.000
    SelectedFactor("a360_284", 1),  # long excess +0.7%, RankIC +0.0329, cov 0.999
    SelectedFactor("a360_302", 1),  # long excess +3.5%, RankIC +0.0350, cov 0.999
    SelectedFactor("a360_304", 1),  # long excess +3.8%, RankIC +0.0353, cov 0.999
    SelectedFactor("a360_306", 1),  # long excess +4.2%, RankIC +0.0363, cov 0.999
    SelectedFactor("a360_308", 1),  # long excess +2.9%, RankIC +0.0349, cov 0.999
    SelectedFactor("a360_310", 1),  # long excess +5.2%, RankIC +0.0369, cov 0.999
    SelectedFactor("a360_313", 1),  # long excess +6.8%, RankIC +0.0374, cov 0.999
    SelectedFactor("a360_315", 1),  # long excess +4.6%, RankIC +0.0355, cov 0.999
    SelectedFactor("a360_318", 1),  # long excess +3.6%, RankIC +0.0351, cov 0.999
    SelectedFactor("a360_320", 1),  # long excess +2.4%, RankIC +0.0340, cov 0.999
    SelectedFactor("a360_322", 1),  # long excess +1.1%, RankIC +0.0331, cov 0.999
    SelectedFactor("a360_324", 1),  # long excess +1.5%, RankIC +0.0328, cov 0.999
    SelectedFactor("a360_326", 1),  # long excess +1.4%, RankIC +0.0333, cov 0.999
    SelectedFactor("a360_329", 1),  # long excess +0.6%, RankIC +0.0325, cov 0.999
    SelectedFactor("a360_332", 1),  # long excess -0.4%, RankIC +0.0302, cov 0.999
    SelectedFactor("a360_334", 1),  # long excess +0.7%, RankIC +0.0314, cov 0.999
    SelectedFactor("a360_336", 1),  # long excess -0.0%, RankIC +0.0309, cov 0.999
    SelectedFactor("a360_338", 1),  # long excess +1.6%, RankIC +0.0314, cov 0.999
    SelectedFactor("a360_341", 1),  # long excess -0.9%, RankIC +0.0281, cov 0.999
    SelectedFactor("a360_343", 1),  # long excess +0.8%, RankIC +0.0283, cov 0.999
    SelectedFactor("a360_345", 1),  # long excess -0.2%, RankIC +0.0283, cov 0.999
    SelectedFactor("a360_346", 1),  # long excess -1.6%, RankIC +0.0260, cov 0.999
    SelectedFactor("a360_347", 1),  # long excess -2.8%, RankIC +0.0251, cov 0.999
    SelectedFactor("a360_349", 1),  # long excess -3.2%, RankIC +0.0237, cov 0.999
    SelectedFactor("a360_351", 1),  # long excess -3.4%, RankIC +0.0220, cov 0.999
    SelectedFactor("a360_352", 1),  # long excess -2.7%, RankIC +0.0203, cov 0.999
    SelectedFactor("a360_353", 1),  # long excess -1.3%, RankIC +0.0212, cov 0.999
    SelectedFactor("a360_356", 1),  # long excess -3.2%, RankIC +0.0145, cov 0.999
    SelectedFactor("a360_357", 1),  # long excess -4.3%, RankIC +0.0112, cov 0.999
    SelectedFactor("a360_358", 1),  # long excess -1.6%, RankIC +0.0115, cov 1.000
    SelectedFactor("a360_359", 1),  # long excess -1.9%, RankIC +0.0081, cov 1.000
    SelectedFactor("a360_360", 1),  # long excess n/a, RankIC n/a, cov 1.000
    SelectedFactor("cnc_006", -1),  # long excess +2.6%, RankIC -0.0261, cov 1.000
    SelectedFactor("cnc_008", -1),  # long excess +8.4%, RankIC -0.0335, cov 0.997
    SelectedFactor("cnc_013", -1),  # long excess +3.7%, RankIC -0.0182, cov 1.000
    SelectedFactor("cnc_019", -1),  # long excess +4.2%, RankIC -0.0129, cov 1.000
    SelectedFactor("cnc_021", -1),  # long excess +2.9%, RankIC -0.0090, cov 1.000
    SelectedFactor("cnc_024", -1),  # long excess -6.3%, RankIC -0.0109, cov 1.000
    SelectedFactor("cnc_025", -1),  # long excess -7.2%, RankIC -0.0117, cov 1.000
    SelectedFactor("cnc_028", -1),  # long excess +5.4%, RankIC -0.0055, cov 0.773
    SelectedFactor("cnc_029", -1),  # long excess -2.6%, RankIC -0.0049, cov 0.992
    SelectedFactor("cnc_030", -1),  # long excess -0.6%, RankIC -0.0059, cov 1.000
    SelectedFactor("cnc_031", -1),  # long excess -0.6%, RankIC -0.0076, cov 0.996
    SelectedFactor("cnc_032", -1),  # long excess +3.5%, RankIC -0.0052, cov 0.896
    SelectedFactor("cnc_033", -1),  # long excess +1.4%, RankIC -0.0069, cov 0.976
    SelectedFactor("cnc_036", -1),  # long excess +7.6%, RankIC -0.0070, cov 0.993
    SelectedFactor("cnc_038", -1),  # long excess +5.4%, RankIC -0.0175, cov 1.000
    SelectedFactor("cnc_039", -1),  # long excess -3.1%, RankIC -0.0009, cov 0.999
    SelectedFactor("cnc_043", 1),  # long excess -0.0%, RankIC +0.0046, cov 0.996
    SelectedFactor("cnc_046", 1),  # long excess -2.3%, RankIC +0.0073, cov 0.999
    SelectedFactor("cnc_050", 1),  # long excess -1.0%, RankIC +0.0014, cov 0.953
    SelectedFactor("cnc_052", 1),  # long excess +1.1%, RankIC +0.0036, cov 0.953
    SelectedFactor("cnc_055", 1),  # long excess -3.6%, RankIC +0.0024, cov 1.000
    SelectedFactor("cnc_057", -1),  # long excess -0.3%, RankIC -0.0014, cov 1.000
    SelectedFactor("cnc_059", 1),  # long excess +4.8%, RankIC +0.0076, cov 0.936
    SelectedFactor("cnc_066", -1),  # long excess -0.3%, RankIC -0.0040, cov 0.971
    SelectedFactor("cnc_068", 1),  # long excess +3.4%, RankIC +0.0145, cov 0.992
    SelectedFactor("cnc_070", -1),  # long excess +0.1%, RankIC -0.0017, cov 1.000
    SelectedFactor("cnc_071", 1),  # long excess +3.5%, RankIC +0.0208, cov 0.978
    SelectedFactor("cnc_072", -1),  # long excess +3.0%, RankIC -0.0055, cov 0.978
    SelectedFactor("cnc_073", -1),  # long excess +0.2%, RankIC -0.0012, cov 0.987
    SelectedFactor("cnc_077", -1),  # long excess -0.3%, RankIC -0.0015, cov 0.999
    SelectedFactor("cnc_078", -1),  # long excess +3.9%, RankIC -0.0149, cov 0.976
    SelectedFactor("cnc_079", -1),  # long excess +0.8%, RankIC -0.0044, cov 0.972
    SelectedFactor("cnc_081", -1),  # long excess +0.2%, RankIC -0.0016, cov 0.947
    SelectedFactor("cnc_082", -1),  # long excess +2.1%, RankIC -0.0028, cov 1.000
    SelectedFactor("cnc_083", -1),  # long excess +2.7%, RankIC -0.0027, cov 0.956
    SelectedFactor("cnc_084", -1),  # long excess +1.1%, RankIC -0.0026, cov 0.956
    SelectedFactor("cnc_085", -1),  # long excess +0.6%, RankIC -0.0023, cov 0.956
    SelectedFactor("cnc_086", -1),  # long excess -1.7%, RankIC -0.0000, cov 0.956
    SelectedFactor("cnc_087", -1),  # long excess +1.4%, RankIC -0.0031, cov 0.956
    SelectedFactor("cnc_089", -1),  # long excess -0.9%, RankIC -0.0001, cov 0.956
    SelectedFactor("cnc_090", 1),  # long excess -1.2%, RankIC +0.0011, cov 1.000
    SelectedFactor("cnc_091", -1),  # long excess +1.9%, RankIC -0.0017, cov 1.000
    SelectedFactor("cnc_092", 1),  # long excess +10.0%, RankIC +0.0097, cov 0.844
    SelectedFactor("cnc_093", -1),  # long excess -2.7%, RankIC -0.0038, cov 0.844
    SelectedFactor("cnc_095", 1),  # long excess -2.8%, RankIC +0.0002, cov 1.000
    SelectedFactor("cnc_096", 1),  # long excess -0.4%, RankIC +0.0006, cov 0.947
    SelectedFactor("cnc_097", 1),  # long excess -0.8%, RankIC +0.0006, cov 0.950
    SelectedFactor("cnc_098", 1),  # long excess +1.0%, RankIC +0.0010, cov 0.943
    SelectedFactor("cnc_099", -1),  # long excess -0.0%, RankIC -0.0008, cov 1.000
    SelectedFactor("cnc_100", -1),  # long excess +1.5%, RankIC -0.0007, cov 0.994
    SelectedFactor("cnc_101", 1),  # long excess +1.4%, RankIC +0.0015, cov 0.703
    SelectedFactor("cnc_103", 1),  # long excess -1.6%, RankIC +0.0016, cov 0.978
    SelectedFactor("cnc_104", -1),  # long excess -1.3%, RankIC -0.0067, cov 0.966
    SelectedFactor("cnc_105", -1),  # long excess -0.9%, RankIC -0.0012, cov 0.953
    SelectedFactor("cnc_106", 1),  # long excess +4.3%, RankIC +0.0221, cov 0.956
    SelectedFactor("cnc_108", 1),  # long excess +3.8%, RankIC +0.0029, cov 0.963
    SelectedFactor("cnc_109", 1),  # long excess +1.6%, RankIC +0.0024, cov 0.964
    SelectedFactor("svc_001", -1),  # long excess +15.2%, RankIC -0.0106, cov 1.000
    SelectedFactor("svc_008", -1),  # long excess -5.0%, RankIC -0.0057, cov 1.000
    SelectedFactor("svc_009", 1),  # long excess +5.9%, RankIC +0.0027, cov 1.000
    SelectedFactor("svc_011", -1),  # long excess +0.0%, RankIC -0.0024, cov 1.000
    SelectedFactor("svc_013", 1),  # long excess +8.6%, RankIC +0.0301, cov 1.000
    SelectedFactor("svc_015", -1),  # long excess -0.6%, RankIC -0.0075, cov 0.993
    SelectedFactor("svc_017", 1),  # long excess -2.8%, RankIC +0.0151, cov 0.993
    SelectedFactor("svc_018", -1),  # long excess +3.2%, RankIC -0.0036, cov 0.993
    SelectedFactor("svc_019", -1),  # long excess +2.7%, RankIC -0.0343, cov 0.993
    SelectedFactor("svc_021", 1),  # long excess +5.2%, RankIC +0.0301, cov 0.991
    SelectedFactor("ta_002", -1),  # long excess -0.6%, RankIC -0.0092, cov 1.000
    SelectedFactor("ta_003", 1),  # long excess +0.8%, RankIC +0.0161, cov 1.000
    SelectedFactor("ta_004", -1),  # long excess +3.6%, RankIC -0.0234, cov 1.000
    SelectedFactor("ta_008", -1),  # long excess +2.0%, RankIC -0.0159, cov 1.000
    SelectedFactor("ta_013", 1),  # long excess +1.8%, RankIC +0.0287, cov 1.000
    SelectedFactor("ta_015", -1),  # long excess +6.0%, RankIC -0.0411, cov 1.000
    SelectedFactor("ta_016", -1),  # long excess +3.1%, RankIC -0.0543, cov 1.000
    SelectedFactor("ta_021", -1),  # long excess -2.1%, RankIC -0.0110, cov 1.000
    SelectedFactor("ta_025", -1),  # long excess -1.1%, RankIC -0.0109, cov 1.000
    SelectedFactor("ta_026", -1),  # long excess +3.4%, RankIC -0.0184, cov 0.997
    SelectedFactor("ta_027", -1),  # long excess +0.8%, RankIC -0.0197, cov 1.000
    SelectedFactor("ta_028", -1),  # long excess -5.2%, RankIC -0.0210, cov 1.000
    SelectedFactor("ta_034", 1),  # long excess +0.7%, RankIC +0.0350, cov 1.000
    SelectedFactor("ta_035", 1),  # long excess -2.2%, RankIC +0.0269, cov 1.000
    SelectedFactor("ta_039", 1),  # long excess -5.1%, RankIC +0.0190, cov 1.000
    SelectedFactor("ta_045", -1),  # long excess +1.5%, RankIC -0.0133, cov 1.000
    SelectedFactor("ta_049", -1),  # long excess -0.5%, RankIC -0.0493, cov 1.000
    SelectedFactor("ta_052", -1),  # long excess -1.5%, RankIC -0.0009, cov 1.000
    SelectedFactor("ta_053", -1),  # long excess -1.5%, RankIC -0.0083, cov 1.000
    SelectedFactor("ta_054", -1),  # long excess -2.2%, RankIC -0.0085, cov 1.000
    SelectedFactor("ta_055", -1),  # long excess +0.6%, RankIC -0.0096, cov 1.000
    SelectedFactor("ta_056", -1),  # long excess -2.5%, RankIC -0.0048, cov 1.000
    SelectedFactor("ta_057", 1),  # long excess -2.3%, RankIC +0.0062, cov 1.000
    SelectedFactor("ta_058", -1),  # long excess +2.9%, RankIC -0.0107, cov 1.000
    SelectedFactor("ta_059", -1),  # long excess +5.8%, RankIC -0.0253, cov 1.000
    SelectedFactor("ta_060", 1),  # long excess -3.7%, RankIC +0.0154, cov 1.000
    SelectedFactor("ta_063", -1),  # long excess +3.4%, RankIC -0.0329, cov 1.000
    SelectedFactor("ta_065", 1),  # long excess n/a, RankIC +0.0003, cov 1.000
    SelectedFactor("ta_066", -1),  # long excess n/a, RankIC -0.0010, cov 1.000
    SelectedFactor("ta_067", -1),  # long excess -1.2%, RankIC -0.0009, cov 1.000
    SelectedFactor("ta_068", 1),  # long excess -2.0%, RankIC +0.0026, cov 1.000
    SelectedFactor("ta_069", -1),  # long excess -0.5%, RankIC -0.0064, cov 1.000
    SelectedFactor("ta_070", 1),  # long excess n/a, RankIC n/a, cov 1.000
    SelectedFactor("ta_071", -1),  # long excess n/a, RankIC -0.0032, cov 1.000
    SelectedFactor("ta_072", 1),  # long excess n/a, RankIC +0.0052, cov 1.000
    SelectedFactor("ta_073", -1),  # long excess -0.8%, RankIC -0.0003, cov 1.000
    SelectedFactor("ta_074", -1),  # long excess +2.2%, RankIC -0.0104, cov 1.000
    SelectedFactor("ta_075", -1),  # long excess n/a, RankIC -0.0036, cov 1.000
    SelectedFactor("ta_076", -1),  # long excess +4.7%, RankIC -0.0182, cov 1.000
    SelectedFactor("ta_077", -1),  # long excess n/a, RankIC -0.0064, cov 1.000
    SelectedFactor("ta_078", -1),  # long excess -1.5%, RankIC -0.0001, cov 1.000
    SelectedFactor("ta_079", 1),  # long excess -3.8%, RankIC +0.0014, cov 1.000
    SelectedFactor("ta_081", -1),  # long excess -0.6%, RankIC -0.0035, cov 1.000
    SelectedFactor("ta_082", -1),  # long excess -2.1%, RankIC -0.0011, cov 1.000
    SelectedFactor("ta_083", -1),  # long excess +0.5%, RankIC -0.0087, cov 1.000
    SelectedFactor("ta_084", 1),  # long excess -9.7%, RankIC +0.0007, cov 1.000
    SelectedFactor("ta_085", 1),  # long excess -5.2%, RankIC +0.0004, cov 1.000
    SelectedFactor("ta_086", 1),  # long excess +1.3%, RankIC +0.0005, cov 1.000
    SelectedFactor("ta_087", -1),  # long excess +1.7%, RankIC -0.0026, cov 1.000
    SelectedFactor("ta_088", -1),  # long excess -7.5%, RankIC -0.0016, cov 1.000
    SelectedFactor("ta_089", 1),  # long excess +2.8%, RankIC +0.0010, cov 1.000
    SelectedFactor("ta_090", 1),  # long excess +5.6%, RankIC +0.0048, cov 1.000
    SelectedFactor("ta_091", 1),  # long excess +4.0%, RankIC +0.0004, cov 1.000
    SelectedFactor("ta_092", 1),  # long excess +3.2%, RankIC +0.0017, cov 1.000
    SelectedFactor("ta_093", 1),  # long excess +3.8%, RankIC +0.0005, cov 1.000
    SelectedFactor("ta_094", 1),  # long excess +1.9%, RankIC +0.0011, cov 1.000
    SelectedFactor("ta_095", 1),  # long excess -3.9%, RankIC +0.0018, cov 1.000
    SelectedFactor("ta_096", -1),  # long excess n/a, RankIC -0.0003, cov 1.000
    SelectedFactor("ta_097", 1),  # long excess -15.8%, RankIC +0.0019, cov 1.000
    SelectedFactor("ta_098", -1),  # long excess -0.5%, RankIC -0.0029, cov 1.000
    SelectedFactor("ta_099", 1),  # long excess n/a, RankIC +0.0026, cov 1.000
    SelectedFactor("ta_100", 1),  # long excess n/a, RankIC +0.0049, cov 1.000
    SelectedFactor("ta_101", -1),  # long excess n/a, RankIC -0.0024, cov 1.000
    SelectedFactor("ta_103", -1),  # long excess +5.6%, RankIC -0.0130, cov 1.000
    SelectedFactor("ta_104", -1),  # long excess +1.0%, RankIC -0.0096, cov 1.000
    SelectedFactor("ta_105", 1),  # long excess +0.5%, RankIC +0.0003, cov 1.000
    SelectedFactor("ta_106", -1),  # long excess n/a, RankIC -0.0032, cov 1.000
    SelectedFactor("ta_107", -1),  # long excess -14.3%, RankIC -0.0006, cov 1.000
    SelectedFactor("ta_108", -1),  # long excess -8.4%, RankIC -0.0026, cov 1.000
    SelectedFactor("ta_109", 1),  # long excess -8.4%, RankIC +0.0020, cov 1.000
    SelectedFactor("ta_110", -1),  # long excess -11.4%, RankIC -0.0024, cov 1.000
    SelectedFactor("ta_111", 1),  # long excess +1.1%, RankIC +0.0000, cov 1.000
    SelectedFactor("ta_112", -1),  # long excess n/a, RankIC -0.0018, cov 1.000
    SelectedFactor("ta_113", -1),  # long excess -3.8%, RankIC -0.0007, cov 1.000
    SelectedFactor("ta_114", 1),  # long excess +10.8%, RankIC +0.0006, cov 1.000
    SelectedFactor("ta_115", -1),  # long excess -1.1%, RankIC -0.0026, cov 1.000
    SelectedFactor("ta_116", -1),  # long excess +1.3%, RankIC -0.0006, cov 1.000
    SelectedFactor("ta_117", 1),  # long excess -1.7%, RankIC +0.0004, cov 1.000
    SelectedFactor("ta_118", -1),  # long excess -10.3%, RankIC -0.0015, cov 1.000
    SelectedFactor("ta_120", 1),  # long excess -5.6%, RankIC +0.0017, cov 1.000
    SelectedFactor("ta_121", 1),  # long excess -11.8%, RankIC +0.0031, cov 1.000
    SelectedFactor("ta_122", 1),  # long excess -13.4%, RankIC +0.0007, cov 1.000
    SelectedFactor("ta_123", -1),  # long excess +9.4%, RankIC -0.0016, cov 1.000
    SelectedFactor("ta_124", -1),  # long excess n/a, RankIC -0.0012, cov 1.000
    SelectedFactor("ta_125", 1),  # long excess -6.2%, RankIC +0.0018, cov 1.000
    SelectedFactor("jq_001", 1),  # long excess -3.4%, RankIC +0.0060, cov 0.992
    SelectedFactor("jq_002", 1),  # long excess +5.1%, RankIC +0.0148, cov 0.992
    SelectedFactor("jq_003", -1),  # long excess +0.6%, RankIC -0.0038, cov 0.947
    SelectedFactor("jq_005", -1),  # long excess +2.3%, RankIC -0.0017, cov 0.992
    SelectedFactor("jq_006", -1),  # long excess -2.0%, RankIC -0.0010, cov 1.000
    SelectedFactor("jq_008", 1),  # long excess +2.2%, RankIC +0.0024, cov 0.932
    SelectedFactor("jq_009", -1),  # long excess -2.3%, RankIC -0.0060, cov 0.992
    SelectedFactor("jq_010", 1),  # long excess -1.2%, RankIC +0.0016, cov 0.992
    SelectedFactor("jq_013", 1),  # long excess +1.1%, RankIC +0.0063, cov 0.992
    SelectedFactor("jq_014", 1),  # long excess +1.4%, RankIC +0.0068, cov 0.992
    SelectedFactor("jq_016", -1),  # long excess -3.9%, RankIC -0.0017, cov 1.000
    SelectedFactor("jq_017", 1),  # long excess +0.9%, RankIC +0.0069, cov 0.951
    SelectedFactor("jq_019", 1),  # long excess -1.1%, RankIC +0.0018, cov 0.956
    SelectedFactor("jq_020", 1),  # long excess -1.4%, RankIC +0.0037, cov 0.943
    SelectedFactor("jq_021", -1),  # long excess +0.3%, RankIC -0.0002, cov 0.951
    SelectedFactor("jq_022", 1),  # long excess -1.0%, RankIC +0.0009, cov 0.956
    SelectedFactor("jq_023", -1),  # long excess -1.8%, RankIC -0.0021, cov 1.000
    SelectedFactor("jq_028", -1),  # long excess -0.8%, RankIC -0.0002, cov 0.992
    SelectedFactor("jq_029", -1),  # long excess +1.3%, RankIC -0.0012, cov 0.877
    SelectedFactor("jq_033", 1),  # long excess -1.2%, RankIC +0.0039, cov 0.956
    SelectedFactor("jq_034", -1),  # long excess +1.0%, RankIC -0.0037, cov 0.992
    SelectedFactor("jq_036", 1),  # long excess -0.4%, RankIC +0.0000, cov 0.818
    SelectedFactor("jq_037", -1),  # long excess +0.4%, RankIC -0.0034, cov 0.868
    SelectedFactor("jq_038", 1),  # long excess -0.0%, RankIC +0.0044, cov 0.868
    SelectedFactor("jq_039", 1),  # long excess -1.8%, RankIC +0.0026, cov 0.992
    SelectedFactor("jq_040", 1),  # long excess -2.1%, RankIC +0.0024, cov 0.908
    SelectedFactor("jq_042", 1),  # long excess -3.7%, RankIC +0.0032, cov 0.908
    SelectedFactor("jq_045", 1),  # long excess +3.0%, RankIC +0.0009, cov 0.987
    SelectedFactor("jq_046", -1),  # long excess +0.6%, RankIC -0.0001, cov 0.975
    SelectedFactor("jq_048", -1),  # long excess -0.6%, RankIC -0.0008, cov 0.926
    SelectedFactor("jq_049", -1),  # long excess +1.7%, RankIC -0.0018, cov 0.943
    SelectedFactor("jq_050", 1),  # long excess -0.7%, RankIC +0.0033, cov 0.992
    SelectedFactor("jq_053", 1),  # long excess -3.4%, RankIC +0.0028, cov 0.947
    SelectedFactor("jq_054", 1),  # long excess -0.4%, RankIC +0.0035, cov 0.949
    SelectedFactor("jq_057", -1),  # long excess +0.5%, RankIC -0.0005, cov 0.992
    SelectedFactor("jq_060", -1),  # long excess -2.8%, RankIC -0.0023, cov 0.992
    SelectedFactor("jq_061", 1),  # long excess -0.2%, RankIC +0.0011, cov 0.948
    SelectedFactor("jq_063", 1),  # long excess -0.4%, RankIC +0.0069, cov 0.992
    SelectedFactor("jq_067", 1),  # long excess -3.2%, RankIC +0.0027, cov 0.992
    SelectedFactor("jq_068", 1),  # long excess -3.0%, RankIC +0.0021, cov 0.992
    SelectedFactor("jq_073", 1),  # long excess -4.0%, RankIC +0.0045, cov 1.000
    SelectedFactor("jq_074", -1),  # long excess -0.9%, RankIC -0.0018, cov 0.992
    SelectedFactor("jq_078", -1),  # long excess +1.0%, RankIC -0.0001, cov 1.000
    SelectedFactor("jq_082", -1),  # long excess -2.1%, RankIC -0.0053, cov 0.928
    SelectedFactor("jq_084", -1),  # long excess -2.4%, RankIC -0.0017, cov 0.988
    SelectedFactor("jq_087", 1),  # long excess -4.0%, RankIC +0.0006, cov 0.956
    SelectedFactor("jq_088", -1),  # long excess -0.7%, RankIC -0.0004, cov 0.992
    SelectedFactor("jq_090", 1),  # long excess -3.4%, RankIC +0.0074, cov 0.992
    SelectedFactor("jq_092", 1),  # long excess -1.6%, RankIC +0.0021, cov 0.948
    SelectedFactor("jq_096", 1),  # long excess -3.3%, RankIC +0.0083, cov 1.000
    SelectedFactor("jq_098", -1),  # long excess -1.4%, RankIC -0.0041, cov 0.992
    SelectedFactor("jq_102", 1),  # long excess -4.0%, RankIC +0.0046, cov 0.992
    SelectedFactor("jq_103", -1),  # long excess +5.1%, RankIC -0.0346, cov 1.000
    SelectedFactor("jq_104", -1),  # long excess +1.3%, RankIC -0.0135, cov 1.000
    SelectedFactor("jq_107", -1),  # long excess +3.5%, RankIC -0.0305, cov 1.000
    SelectedFactor("jq_108", -1),  # long excess +6.8%, RankIC -0.0360, cov 0.993
    SelectedFactor("jq_111", -1),  # long excess +1.6%, RankIC -0.0255, cov 1.000
    SelectedFactor("jq_114", -1),  # long excess +6.1%, RankIC -0.0286, cov 1.000
    SelectedFactor("jq_116", -1),  # long excess +18.0%, RankIC -0.0436, cov 1.000
    SelectedFactor("jq_119", -1),  # long excess -1.7%, RankIC -0.0258, cov 1.000
    SelectedFactor("jq_125", -1),  # long excess -5.6%, RankIC -0.0157, cov 1.000
    SelectedFactor("jq_132", -1),  # long excess +1.2%, RankIC -0.0242, cov 1.000
    SelectedFactor("jq_134", -1),  # long excess -2.0%, RankIC -0.0250, cov 1.000
    SelectedFactor("jq_135", -1),  # long excess -1.0%, RankIC -0.0213, cov 1.000
    SelectedFactor("jq_137", -1),  # long excess +12.4%, RankIC -0.0346, cov 1.000
    SelectedFactor("jq_139", -1),  # long excess +2.5%, RankIC -0.0034, cov 0.884
    SelectedFactor("jq_142", 1),  # long excess +1.2%, RankIC +0.0012, cov 0.976
    SelectedFactor("jq_144", -1),  # long excess -0.2%, RankIC -0.0011, cov 0.975
    SelectedFactor("jq_145", 1),  # long excess -2.8%, RankIC +0.0002, cov 0.814
    SelectedFactor("jq_148", 1),  # long excess -3.6%, RankIC +0.0110, cov 0.993
    SelectedFactor("jq_149", -1),  # long excess -1.9%, RankIC -0.0014, cov 1.000
    SelectedFactor("jq_150", 1),  # long excess -2.7%, RankIC +0.0053, cov 1.000
    SelectedFactor("jq_154", -1),  # long excess +4.8%, RankIC -0.0186, cov 0.993
    SelectedFactor("jq_155", -1),  # long excess +2.2%, RankIC -0.0198, cov 1.000
    SelectedFactor("jq_156", -1),  # long excess +2.6%, RankIC -0.0199, cov 1.000
    SelectedFactor("jq_159", -1),  # long excess +2.5%, RankIC -0.0406, cov 1.000
    SelectedFactor("jq_160", -1),  # long excess -0.4%, RankIC -0.0026, cov 1.000
    SelectedFactor("jq_161", 1),  # long excess -2.6%, RankIC +0.0036, cov 1.000
    SelectedFactor("jq_162", 1),  # long excess -0.6%, RankIC +0.0006, cov 0.991
    SelectedFactor("jq_164", 1),  # long excess -1.8%, RankIC +0.0046, cov 1.000
    SelectedFactor("jq_167", 1),  # long excess -2.6%, RankIC +0.0051, cov 0.992
    SelectedFactor("jq_168", 1),  # long excess +0.1%, RankIC +0.0009, cov 0.999
    SelectedFactor("jq_172", 1),  # long excess -1.5%, RankIC +0.0090, cov 1.000
    SelectedFactor("jq_175", 1),  # long excess -3.9%, RankIC +0.0421, cov 1.000
    SelectedFactor("jq_188", -1),  # long excess -0.1%, RankIC -0.0236, cov 1.000
    SelectedFactor("jq_190", 1),  # long excess +0.4%, RankIC +0.0161, cov 1.000
    SelectedFactor("jq_191", -1),  # long excess +3.2%, RankIC -0.0242, cov 1.000
    SelectedFactor("jq_199", -1),  # long excess -0.8%, RankIC -0.0240, cov 1.000
    SelectedFactor("jq_218", -1),  # long excess -0.6%, RankIC -0.0232, cov 1.000
    SelectedFactor("jq_219", -1),  # long excess -2.7%, RankIC -0.0331, cov 1.000
    SelectedFactor("jq_226", -1),  # long excess -1.6%, RankIC -0.0098, cov 0.974
    SelectedFactor("jq_227", -1),  # long excess +5.6%, RankIC -0.0422, cov 1.000
    SelectedFactor("jq_228", 1),  # long excess +7.4%, RankIC +0.0109, cov 1.000
    SelectedFactor("jq_230", -1),  # long excess +6.6%, RankIC -0.0361, cov 1.000
    SelectedFactor("jq_232", 1),  # long excess -0.6%, RankIC +0.0014, cov 0.985
    SelectedFactor("oap_006", 1),  # long excess +0.8%, RankIC +0.0020, cov 0.976
    SelectedFactor("oap_007", -1),  # long excess +1.8%, RankIC -0.0042, cov 0.960
    SelectedFactor("oap_010", 1),  # long excess +2.3%, RankIC +0.0035, cov 0.976
    SelectedFactor("oap_013", -1),  # long excess +3.3%, RankIC -0.0034, cov 0.658
    SelectedFactor("oap_014", -1),  # long excess +2.5%, RankIC -0.0038, cov 0.637
    SelectedFactor("oap_015", -1),  # long excess +3.9%, RankIC -0.0466, cov 0.999
    SelectedFactor("oap_017", -1),  # long excess +0.9%, RankIC -0.0131, cov 0.999
    SelectedFactor("oap_018", -1),  # long excess +6.9%, RankIC -0.0532, cov 0.999
    SelectedFactor("oap_021", -1),  # long excess -0.6%, RankIC -0.0125, cov 0.954
    SelectedFactor("oap_023", -1),  # long excess +0.5%, RankIC -0.0034, cov 0.896
    SelectedFactor("oap_024", -1),  # long excess +2.3%, RankIC -0.0135, cov 0.989
    SelectedFactor("oap_025", 1),  # long excess +0.9%, RankIC +0.0033, cov 0.942
    SelectedFactor("oap_026", -1),  # long excess +1.0%, RankIC -0.0086, cov 0.989
    SelectedFactor("oap_027", -1),  # long excess -2.7%, RankIC -0.0011, cov 0.989
    SelectedFactor("oap_029", 1),  # long excess +2.9%, RankIC +0.0027, cov 0.942
    SelectedFactor("oap_030", 1),  # long excess -1.8%, RankIC +0.0142, cov 0.880
    SelectedFactor("oap_033", -1),  # long excess +1.0%, RankIC -0.0114, cov 0.688
    SelectedFactor("oap_040", -1),  # long excess -1.0%, RankIC -0.0125, cov 0.943
    SelectedFactor("oap_041", -1),  # long excess +5.2%, RankIC -0.0067, cov 0.943
    SelectedFactor("oap_042", -1),  # long excess -1.8%, RankIC -0.0139, cov 0.993
    SelectedFactor("oap_043", -1),  # long excess +2.6%, RankIC -0.0166, cov 0.923
    SelectedFactor("oap_044", 1),  # long excess +1.6%, RankIC +0.0362, cov 1.000
    SelectedFactor("oap_045", 1),  # long excess +0.9%, RankIC +0.0208, cov 1.000
    SelectedFactor("oap_047", -1),  # long excess -0.3%, RankIC -0.0347, cov 0.999
    SelectedFactor("oap_057", 1),  # long excess +3.8%, RankIC +0.0222, cov 1.000
    SelectedFactor("oap_058", -1),  # long excess -2.0%, RankIC -0.0003, cov 1.000
    SelectedFactor("oap_060", -1),  # long excess +2.8%, RankIC -0.0212, cov 0.917
    SelectedFactor("oap_068", -1),  # long excess +9.8%, RankIC -0.0207, cov 0.719
    SelectedFactor("oap_069", 1),  # long excess +11.1%, RankIC +0.0240, cov 0.986
    SelectedFactor("oap_070", -1),  # long excess +0.1%, RankIC -0.0006, cov 0.927
    SelectedFactor("oap_075", 1),  # long excess -2.1%, RankIC +0.0014, cov 0.940
    SelectedFactor("oap_078", 1),  # long excess +0.2%, RankIC +0.0007, cov 0.948
    SelectedFactor("oap_080", -1),  # long excess +2.0%, RankIC -0.0000, cov 0.968
    SelectedFactor("oap_081", 1),  # long excess +1.8%, RankIC +0.0054, cov 0.897
    SelectedFactor("oap_082", 1),  # long excess +3.6%, RankIC +0.0068, cov 0.992
    SelectedFactor("oap_083", -1),  # long excess -2.0%, RankIC -0.0061, cov 0.933
    SelectedFactor("oap_084", -1),  # long excess -7.7%, RankIC -0.0045, cov 0.051
    SelectedFactor("oap_085", 1),  # long excess +5.1%, RankIC +0.0032, cov 0.959
    SelectedFactor("oap_086", 1),  # long excess +6.4%, RankIC +0.0028, cov 0.959
    SelectedFactor("oap_087", 1),  # long excess -0.5%, RankIC +0.0024, cov 0.991
    SelectedFactor("oap_088", 1),  # long excess +2.7%, RankIC +0.0005, cov 0.688
    SelectedFactor("oap_090", 1),  # long excess -1.0%, RankIC +0.0007, cov 0.928
    SelectedFactor("oap_092", 1),  # long excess +2.0%, RankIC +0.0026, cov 0.930
    SelectedFactor("oap_093", 1),  # long excess +1.4%, RankIC +0.0001, cov 0.996
    SelectedFactor("oap_097", -1),  # long excess +0.2%, RankIC -0.0012, cov 0.936
    SelectedFactor("oap_101", -1),  # long excess +2.4%, RankIC -0.0019, cov 0.943
    SelectedFactor("oap_109", -1),  # long excess +1.3%, RankIC -0.0006, cov 0.931
    SelectedFactor("oap_111", -1),  # long excess +4.0%, RankIC -0.0040, cov 0.975
    SelectedFactor("oap_115", -1),  # long excess +2.9%, RankIC -0.0026, cov 0.918
    SelectedFactor("oap_117", -1),  # long excess +0.6%, RankIC -0.0009, cov 0.905
    SelectedFactor("oap_118", -1),  # long excess +0.0%, RankIC -0.0013, cov 0.969
    SelectedFactor("oap_119", -1),  # long excess +1.1%, RankIC -0.0022, cov 0.940
    SelectedFactor("oap_121", -1),  # long excess -0.1%, RankIC -0.0027, cov 0.688
    SelectedFactor("oap_122", -1),  # long excess +1.1%, RankIC -0.0120, cov 0.773
    SelectedFactor("oap_123", -1),  # long excess +1.2%, RankIC -0.0127, cov 0.960
    SelectedFactor("oap_128", -1),  # long excess -0.3%, RankIC -0.0007, cov 0.992
    SelectedFactor("oap_130", -1),  # long excess -3.4%, RankIC -0.0013, cov 0.817
    SelectedFactor("oap_131", 1),  # long excess +6.0%, RankIC +0.0030, cov 0.817
    SelectedFactor("oap_132", -1),  # long excess +0.7%, RankIC -0.0001, cov 0.997
    SelectedFactor("oap_136", -1),  # long excess -1.9%, RankIC -0.0028, cov 0.999
    SelectedFactor("oap_137", -1),  # long excess -1.5%, RankIC -0.0006, cov 1.000
    SelectedFactor("oap_138", -1),  # long excess -1.4%, RankIC -0.0010, cov 0.990
    SelectedFactor("oap_139", -1),  # long excess -2.1%, RankIC -0.0002, cov 0.976
    SelectedFactor("oap_140", 1),  # long excess +4.6%, RankIC +0.0012, cov 0.942
    SelectedFactor("oap_142", 1),  # long excess -1.1%, RankIC +0.0007, cov 0.732
    SelectedFactor("cne6_006", -1),  # long excess -1.7%, RankIC -0.0146, cov 0.981
    SelectedFactor("cne6_013", 1),  # long excess +9.1%, RankIC +0.0328, cov 0.996
    SelectedFactor("cne6_016", 1),  # long excess +2.7%, RankIC +0.0065, cov 0.889
    SelectedFactor("cne6_024", -1),  # long excess +1.1%, RankIC -0.0080, cov 0.998
    SelectedFactor("cne6_025", -1),  # long excess -1.2%, RankIC -0.0101, cov 0.998
    SelectedFactor("cne6_026", -1),  # long excess +0.9%, RankIC -0.0020, cov 0.998
    SelectedFactor("cne6_028", 1),  # long excess +1.5%, RankIC +0.0041, cov 0.988
    SelectedFactor("cne6_033", 1),  # long excess +0.5%, RankIC +0.0034, cov 0.998
    SelectedFactor("cne6_034", 1),  # long excess +1.5%, RankIC +0.0048, cov 0.998
    SelectedFactor("cne6_035", 1),  # long excess -0.3%, RankIC +0.0026, cov 0.998
    SelectedFactor("cne6_036", 1),  # long excess -1.2%, RankIC +0.0014, cov 0.998
    SelectedFactor("tst_002", -1),  # long excess +5.8%, RankIC -0.0251, cov 1.000
    SelectedFactor("tst_028", 1),  # long excess +6.0%, RankIC +0.0321, cov 0.977
    SelectedFactor("tst_030", -1),  # long excess -0.7%, RankIC -0.0201, cov 1.000
    SelectedFactor("tst_034", -1),  # long excess -3.7%, RankIC -0.0144, cov 1.000
    SelectedFactor("tst_036", 1),  # long excess -1.1%, RankIC +0.0033, cov 1.000
    SelectedFactor("tst_038", 1),  # long excess +0.2%, RankIC +0.0524, cov 1.000
    SelectedFactor("tst_050", -1),  # long excess +1.0%, RankIC -0.0202, cov 1.000
    SelectedFactor("tst_056", -1),  # long excess -0.2%, RankIC -0.0123, cov 1.000
    SelectedFactor("tst_069", 1),  # long excess -4.6%, RankIC +0.0181, cov 1.000
    SelectedFactor("tst_072", -1),  # long excess +2.0%, RankIC -0.0230, cov 1.000
    SelectedFactor("tst_073", 1),  # long excess +1.9%, RankIC +0.0142, cov 1.000
    SelectedFactor("tst_074", -1),  # long excess +0.8%, RankIC -0.0185, cov 1.000
    SelectedFactor("tst_075", 1),  # long excess +2.3%, RankIC +0.0158, cov 1.000
    SelectedFactor("jkp_005", 1),  # long excess -0.0%, RankIC +0.0008, cov 0.944
    SelectedFactor("jkp_015", 1),  # long excess +3.3%, RankIC +0.0094, cov 0.992
    SelectedFactor("jkp_016", 1),  # long excess +3.3%, RankIC +0.0183, cov 1.000
    SelectedFactor("jkp_017", 1),  # long excess +3.3%, RankIC +0.0150, cov 0.992
    SelectedFactor("jkp_018", -1),  # long excess -0.1%, RankIC -0.0026, cov 0.906
    SelectedFactor("jkp_020", 1),  # long excess +1.0%, RankIC +0.0162, cov 0.977
    SelectedFactor("jkp_023", 1),  # long excess -0.1%, RankIC +0.0171, cov 0.826
    SelectedFactor("jkp_026", -1),  # long excess +1.5%, RankIC -0.0237, cov 0.971
    SelectedFactor("jkp_029", -1),  # long excess +0.6%, RankIC -0.0022, cov 0.987
    SelectedFactor("jkp_030", -1),  # long excess -1.3%, RankIC -0.0001, cov 0.944
    SelectedFactor("jkp_031", -1),  # long excess +4.6%, RankIC -0.0046, cov 0.943
    SelectedFactor("jkp_033", -1),  # long excess +0.6%, RankIC -0.0013, cov 0.968
    SelectedFactor("jkp_035", -1),  # long excess +1.0%, RankIC -0.0029, cov 0.899
    SelectedFactor("jkp_036", -1),  # long excess +1.2%, RankIC -0.0042, cov 0.976
    SelectedFactor("jkp_038", -1),  # long excess -0.3%, RankIC -0.0019, cov 0.814
    SelectedFactor("jkp_042", -1),  # long excess +2.1%, RankIC -0.0008, cov 0.984
    SelectedFactor("jkp_044", -1),  # long excess +0.2%, RankIC -0.0012, cov 0.943
    SelectedFactor("jkp_045", 1),  # long excess +3.4%, RankIC +0.0016, cov 0.943
    SelectedFactor("jkp_047", -1),  # long excess +2.8%, RankIC -0.0032, cov 0.943
    SelectedFactor("jkp_048", -1),  # long excess +1.2%, RankIC -0.0012, cov 0.943
    SelectedFactor("jkp_049", 1),  # long excess +0.4%, RankIC +0.0015, cov 0.987
    SelectedFactor("jkp_050", 1),  # long excess +0.3%, RankIC +0.0005, cov 0.995
    SelectedFactor("jkp_051", 1),  # long excess -1.6%, RankIC +0.0020, cov 0.987
    SelectedFactor("jkp_053", -1),  # long excess +2.1%, RankIC -0.0021, cov 0.984
    SelectedFactor("jkp_056", -1),  # long excess +1.1%, RankIC -0.0052, cov 0.992
    SelectedFactor("jkp_057", -1),  # long excess -0.6%, RankIC -0.0046, cov 0.992
    SelectedFactor("jkp_060", 1),  # long excess +5.7%, RankIC +0.0055, cov 0.985
    SelectedFactor("jkp_064", 1),  # long excess -4.3%, RankIC +0.0041, cov 0.991
    SelectedFactor("jkp_069", 1),  # long excess -3.3%, RankIC +0.0040, cov 0.991
    SelectedFactor("jkp_072", 1),  # long excess -0.4%, RankIC +0.0007, cov 0.984
    SelectedFactor("jkp_073", 1),  # long excess +1.5%, RankIC +0.0027, cov 0.992
    SelectedFactor("jkp_075", 1),  # long excess +3.0%, RankIC +0.0053, cov 0.914
    SelectedFactor("jkp_077", -1),  # long excess +1.7%, RankIC -0.0072, cov 0.949
    SelectedFactor("jkp_087", 1),  # long excess +1.6%, RankIC +0.0017, cov 0.903
    SelectedFactor("jkp_089", -1),  # long excess -1.2%, RankIC -0.0005, cov 0.946
    SelectedFactor("jkp_091", -1),  # long excess -4.7%, RankIC -0.0049, cov 0.761
    SelectedFactor("jkp_092", -1),  # long excess +1.5%, RankIC -0.0036, cov 0.976
    SelectedFactor("jkp_094", 1),  # long excess -1.7%, RankIC +0.0009, cov 0.794
    SelectedFactor("jkp_095", -1),  # long excess -1.7%, RankIC -0.0113, cov 0.794
    SelectedFactor("jkp_096", -1),  # long excess -1.6%, RankIC -0.0072, cov 0.852
    SelectedFactor("jkp_097", -1),  # long excess +0.4%, RankIC -0.0025, cov 0.944
    SelectedFactor("jkp_100", -1),  # long excess +2.1%, RankIC -0.0223, cov 0.979
    SelectedFactor("jkp_101", -1),  # long excess +2.5%, RankIC -0.0021, cov 0.773
    SelectedFactor("jkp_111", -1),  # long excess +0.9%, RankIC -0.0229, cov 0.896
    SelectedFactor("jkp_112", -1),  # long excess -8.9%, RankIC -0.0027, cov 0.999
    SelectedFactor("jkp_118", 1),  # long excess +10.9%, RankIC +0.0085, cov 1.000
    SelectedFactor("jkp_121", 1),  # long excess -0.6%, RankIC +0.0140, cov 0.976
    SelectedFactor("jkp_124", -1),  # long excess +4.7%, RankIC -0.0186, cov 0.999
    SelectedFactor("jkp_126", -1),  # long excess -0.1%, RankIC -0.0019, cov 0.999
    SelectedFactor("jkp_128", -1),  # long excess -5.8%, RankIC -0.0051, cov 0.992
    SelectedFactor("jkp_133", 1),  # long excess +4.3%, RankIC +0.0129, cov 0.896
    SelectedFactor("jkp_135", -1),  # long excess +2.6%, RankIC -0.0324, cov 0.999
    SelectedFactor("jkp_138", 1),  # long excess +2.2%, RankIC +0.0023, cov 0.975
    SelectedFactor("jkp_139", -1),  # long excess -0.5%, RankIC -0.0001, cov 0.981
    SelectedFactor("jkp_140", 1),  # long excess +1.5%, RankIC +0.0073, cov 0.987
    SelectedFactor("jkp_141", 1),  # long excess +0.2%, RankIC +0.0093, cov 0.794
    SelectedFactor("jkp_143", 1),  # long excess +0.6%, RankIC +0.0047, cov 0.795
    SelectedFactor("jkp_144", 1),  # long excess -0.2%, RankIC +0.0105, cov 0.970
)

DEDUPLICATED = {
    "a101_003": "gtja_105",
    "a101_006": "gtja_139",
    "a101_009": "a101_049",
    "a101_011": "gtja_007",
    "a101_012": "gtja_113",
    "a101_013": "gtja_099",
    "a101_014": "gtja_139",
    "a101_016": "gtja_099",
    "a101_017": "gtja_142",
    "a101_021": "gtja_004",
    "a101_025": "jq_218",
    "a101_028": "gtja_191",
    "a101_029": "gtja_157",
    "a101_033": "a158_001",
    "a101_035": "gtja_117",
    "a101_042": "gtja_120",
    "a101_043": "gtja_085",
    "a101_045": "gtja_113",
    "a101_050": "gtja_016",
    "a101_051": "a101_049",
    "a101_054": "gtja_171",
    "a101_055": "gtja_176",
    "a101_057": "a158_013",
    "a101_084": "gtja_017",
    "a101_094": "gtja_121",
    "a101_101": "a158_001",
    "a158_002": "ta_049",
    "a158_003": "a158_001",
    "a158_004": "a158_005",
    "a158_007": "a158_006",
    "a158_008": "a158_013",
    "a158_009": "a158_013",
    "a158_010": "a158_001",
    "a158_014": "a360_175",
    "a158_015": "a360_229",
    "a158_016": "a360_098",
    "a158_017": "a158_117",
    "a158_018": "a360_071",
    "a158_019": "a360_238",
    "a158_020": "a360_173",
    "a158_021": "a360_284",
    "a158_022": "a360_098",
    "a158_023": "a158_058",
    "a158_026": "ta_063",
    "a158_027": "ta_063",
    "a158_029": "a360_117",
    "a158_030": "a360_173",
    "a158_031": "a360_284",
    "a158_032": "tst_002",
    "a158_033": "a360_071",
    "a158_046": "tst_036",
    "a158_047": "tst_036",
    "a158_048": "a158_058",
    "a158_050": "a158_049",
    "a158_051": "jq_175",
    "a158_052": "tst_038",
    "a158_054": "a360_178",
    "a158_055": "a360_175",
    "a158_056": "tst_036",
    "a158_057": "ta_034",
    "a158_059": "a360_238",
    "a158_060": "jq_199",
    "a158_061": "tst_038",
    "a158_062": "tst_038",
    "a158_063": "a360_071",
    "a158_065": "jq_199",
    "a158_066": "gtja_079",
    "a158_067": "gtja_079",
    "a158_068": "gtja_079",
    "a158_069": "tst_069",
    "a158_070": "ta_028",
    "a158_071": "a158_072",
    "a158_073": "a158_058",
    "a158_074": "a158_084",
    "a158_075": "a158_085",
    "a158_076": "gtja_177",
    "a158_078": "a158_088",
    "a158_086": "gtja_177",
    "a158_087": "a158_077",
    "a158_092": "ta_059",
    "a158_099": "a158_104",
    "a158_100": "a158_105",
    "a158_101": "a158_102",
    "a158_109": "a158_104",
    "a158_110": "a158_105",
    "a158_111": "a158_106",
    "a158_112": "a158_102",
    "a158_113": "a158_103",
    "a158_114": "ta_025",
    "a158_115": "a360_229",
    "a158_116": "a360_098",
    "a158_118": "a360_071",
    "a158_119": "ta_025",
    "a158_120": "a360_229",
    "a158_121": "a360_098",
    "a158_122": "a158_117",
    "a158_123": "a360_071",
    "a158_124": "ta_025",
    "a158_125": "a360_229",
    "a158_126": "a360_098",
    "a158_127": "a158_117",
    "a158_128": "a360_071",
    "a158_129": "a360_357",
    "a158_130": "gtja_102",
    "a158_131": "gtja_102",
    "a158_132": "gtja_102",
    "a158_133": "a158_138",
    "a158_136": "a158_137",
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
    "a360_025": "a158_117",
    "a360_026": "tst_002",
    "a360_027": "a158_117",
    "a360_028": "a158_117",
    "a360_029": "tst_002",
    "a360_030": "a158_117",
    "a360_031": "a158_117",
    "a360_032": "a158_117",
    "a360_033": "a158_117",
    "a360_034": "a158_117",
    "a360_035": "a360_098",
    "a360_036": "a360_098",
    "a360_037": "a360_098",
    "a360_038": "a360_098",
    "a360_039": "a360_098",
    "a360_040": "a360_098",
    "a360_041": "a360_284",
    "a360_042": "a360_284",
    "a360_043": "a360_284",
    "a360_044": "a360_284",
    "a360_045": "a360_284",
    "a360_046": "a360_284",
    "a360_047": "a360_284",
    "a360_048": "a360_229",
    "a360_049": "a360_229",
    "a360_050": "a360_229",
    "a360_051": "a360_173",
    "a360_052": "a360_173",
    "a360_053": "a360_173",
    "a360_054": "a360_173",
    "a360_055": "a360_175",
    "a360_056": "a360_117",
    "a360_057": "a360_238",
    "a360_058": "a360_238",
    "a360_059": "a158_001",
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
    "a360_088": "a158_117",
    "a360_089": "a158_117",
    "a360_090": "tst_002",
    "a360_091": "a158_117",
    "a360_092": "a158_117",
    "a360_093": "a158_117",
    "a360_094": "a158_117",
    "a360_095": "a158_117",
    "a360_096": "a158_117",
    "a360_097": "a360_098",
    "a360_099": "a360_098",
    "a360_100": "a360_098",
    "a360_101": "a360_098",
    "a360_102": "a360_284",
    "a360_103": "a360_284",
    "a360_104": "a360_284",
    "a360_105": "a360_284",
    "a360_106": "a360_284",
    "a360_107": "a360_284",
    "a360_108": "a360_284",
    "a360_109": "a360_229",
    "a360_110": "a360_229",
    "a360_111": "a360_229",
    "a360_112": "a360_173",
    "a360_113": "a360_173",
    "a360_114": "a360_173",
    "a360_115": "a360_173",
    "a360_116": "a360_175",
    "a360_118": "a360_238",
    "a360_119": "gtja_137",
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
    "a360_145": "a158_117",
    "a360_146": "a158_117",
    "a360_147": "a158_117",
    "a360_148": "a158_117",
    "a360_149": "a158_117",
    "a360_150": "a158_117",
    "a360_151": "a158_117",
    "a360_152": "a158_117",
    "a360_153": "a158_117",
    "a360_154": "a158_117",
    "a360_155": "a158_117",
    "a360_156": "a360_098",
    "a360_157": "a360_098",
    "a360_158": "a360_098",
    "a360_159": "a360_098",
    "a360_160": "a360_098",
    "a360_161": "a360_284",
    "a360_162": "a360_284",
    "a360_163": "a360_284",
    "a360_164": "a360_284",
    "a360_165": "a360_284",
    "a360_166": "a360_284",
    "a360_167": "a360_284",
    "a360_168": "a360_284",
    "a360_169": "a360_229",
    "a360_170": "a360_229",
    "a360_171": "a360_229",
    "a360_172": "a360_173",
    "a360_174": "a360_173",
    "a360_176": "a360_117",
    "a360_177": "a360_117",
    "a360_180": "a158_011",
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
    "a360_202": "a360_071",
    "a360_203": "a158_058",
    "a360_204": "a158_058",
    "a360_205": "tst_002",
    "a360_206": "tst_002",
    "a360_207": "tst_002",
    "a360_208": "tst_002",
    "a360_209": "tst_002",
    "a360_210": "a158_117",
    "a360_211": "a158_117",
    "a360_212": "a158_117",
    "a360_213": "a158_117",
    "a360_214": "a158_117",
    "a360_215": "a158_117",
    "a360_216": "a360_098",
    "a360_217": "a360_098",
    "a360_218": "a360_098",
    "a360_219": "a360_098",
    "a360_220": "a360_098",
    "a360_221": "a360_098",
    "a360_222": "a360_284",
    "a360_223": "a360_284",
    "a360_224": "a360_284",
    "a360_225": "a360_284",
    "a360_226": "a360_284",
    "a360_227": "a360_284",
    "a360_228": "a360_229",
    "a360_230": "a360_229",
    "a360_231": "a360_229",
    "a360_232": "a360_173",
    "a360_233": "a360_173",
    "a360_234": "a360_173",
    "a360_235": "a360_175",
    "a360_236": "a360_175",
    "a360_237": "a360_238",
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
    "a360_265": "a158_117",
    "a360_266": "tst_002",
    "a360_267": "a158_117",
    "a360_268": "a158_117",
    "a360_269": "a158_117",
    "a360_270": "a158_117",
    "a360_271": "a158_117",
    "a360_272": "a158_117",
    "a360_273": "a158_117",
    "a360_274": "a158_117",
    "a360_275": "a158_117",
    "a360_276": "a360_098",
    "a360_277": "a360_098",
    "a360_278": "a360_098",
    "a360_279": "a360_098",
    "a360_280": "a360_098",
    "a360_281": "a360_284",
    "a360_282": "a360_284",
    "a360_283": "a360_284",
    "a360_285": "a360_284",
    "a360_286": "a360_284",
    "a360_287": "a360_284",
    "a360_288": "a360_229",
    "a360_289": "a360_229",
    "a360_290": "a360_229",
    "a360_291": "a360_229",
    "a360_292": "a360_173",
    "a360_293": "a360_173",
    "a360_294": "a360_173",
    "a360_295": "a360_175",
    "a360_296": "a360_117",
    "a360_297": "a360_238",
    "a360_298": "a360_238",
    "a360_299": "a360_239",
    "a360_300": "a158_013",
    "a360_301": "a360_302",
    "a360_303": "a360_302",
    "a360_305": "a360_304",
    "a360_307": "a360_306",
    "a360_309": "a360_310",
    "a360_311": "a360_310",
    "a360_312": "a360_313",
    "a360_314": "a360_313",
    "a360_316": "a360_315",
    "a360_317": "a360_318",
    "a360_319": "a360_318",
    "a360_321": "a360_320",
    "a360_323": "a360_322",
    "a360_325": "a360_324",
    "a360_327": "a360_326",
    "a360_328": "a360_329",
    "a360_330": "a158_147",
    "a360_331": "a158_147",
    "a360_333": "a360_334",
    "a360_335": "a360_334",
    "a360_337": "a360_338",
    "a360_339": "a360_338",
    "a360_340": "a158_146",
    "a360_342": "a360_343",
    "a360_344": "a360_343",
    "a360_348": "jq_134",
    "a360_350": "a158_145",
    "a360_354": "jq_135",
    "a360_355": "a158_144",
    "cnc_001": "svc_001",
    "cnc_002": "svc_001",
    "cnc_003": "jq_230",
    "cnc_004": "cnc_006",
    "cnc_005": "cnc_006",
    "cnc_007": "cnc_013",
    "cnc_009": "jq_116",
    "cnc_010": "jq_116",
    "cnc_011": "jq_116",
    "cnc_012": "jq_116",
    "cnc_014": "jq_116",
    "cnc_015": "jq_227",
    "cnc_016": "jq_227",
    "cnc_017": "jq_159",
    "cnc_018": "cnc_019",
    "cnc_020": "jq_154",
    "cnc_022": "jkp_128",
    "cnc_023": "cnc_024",
    "cnc_026": "a360_098",
    "cnc_027": "jkp_100",
    "cnc_034": "oap_029",
    "cnc_035": "cnc_029",
    "cnc_037": "oap_015",
    "cnc_040": "jkp_101",
    "cnc_041": "cnc_043",
    "cnc_042": "jkp_060",
    "cnc_044": "jkp_060",
    "cnc_045": "cnc_043",
    "cnc_047": "oap_078",
    "cnc_048": "oap_078",
    "cnc_049": "cnc_050",
    "cnc_051": "jkp_069",
    "cnc_053": "jkp_072",
    "cnc_054": "cnc_052",
    "cnc_056": "cnc_070",
    "cnc_058": "cnc_108",
    "cnc_060": "oap_083",
    "cnc_061": "cnc_071",
    "cnc_062": "jkp_077",
    "cnc_063": "cnc_106",
    "cnc_064": "oap_060",
    "cnc_065": "oap_060",
    "cnc_067": "oap_060",
    "cnc_069": "jq_002",
    "cnc_074": "jkp_029",
    "cnc_075": "jkp_030",
    "cnc_076": "jkp_031",
    "cnc_080": "cnc_081",
    "cnc_088": "cnc_087",
    "cnc_094": "oap_078",
    "cnc_102": "jkp_092",
    "cnc_107": "jkp_060",
    "cne6_001": "svc_001",
    "cne6_002": "svc_001",
    "cne6_003": "cnc_025",
    "cne6_004": "svc_019",
    "cne6_005": "jq_159",
    "cne6_007": "oap_044",
    "cne6_008": "cnc_006",
    "cne6_009": "cnc_006",
    "cne6_010": "cnc_006",
    "cne6_011": "a360_229",
    "cne6_012": "jkp_101",
    "cne6_014": "jq_226",
    "cne6_015": "cne6_016",
    "cne6_017": "cnc_071",
    "cne6_018": "oap_060",
    "cne6_019": "jq_002",
    "cne6_020": "oap_060",
    "cne6_021": "jq_017",
    "cne6_022": "jq_017",
    "cne6_023": "cnc_070",
    "cne6_027": "jkp_031",
    "cne6_029": "oap_078",
    "cne6_030": "cnc_050",
    "cne6_031": "jq_037",
    "cne6_032": "jq_039",
    "cne6_037": "jq_232",
    "cne6_038": "jkp_016",
    "gtja_005": "a101_026",
    "gtja_008": "a360_117",
    "gtja_013": "a101_041",
    "gtja_014": "a360_175",
    "gtja_018": "a360_175",
    "gtja_019": "a360_175",
    "gtja_020": "a360_173",
    "gtja_021": "gtja_024",
    "gtja_025": "a101_039",
    "gtja_027": "a360_229",
    "gtja_028": "ta_021",
    "gtja_031": "a360_173",
    "gtja_032": "a101_015",
    "gtja_033": "a101_052",
    "gtja_034": "a360_173",
    "gtja_036": "a101_027",
    "gtja_037": "a101_008",
    "gtja_038": "a101_023",
    "gtja_042": "a101_040",
    "gtja_046": "gtja_079",
    "gtja_047": "gtja_082",
    "gtja_049": "a360_229",
    "gtja_050": "a360_229",
    "gtja_051": "a360_229",
    "gtja_052": "a158_117",
    "gtja_053": "tst_056",
    "gtja_054": "a101_018",
    "gtja_055": "gtja_059",
    "gtja_057": "ta_021",
    "gtja_058": "a158_106",
    "gtja_060": "ta_045",
    "gtja_062": "a101_044",
    "gtja_063": "gtja_079",
    "gtja_065": "a360_238",
    "gtja_066": "a360_238",
    "gtja_067": "a158_058",
    "gtja_070": "jq_116",
    "gtja_071": "a360_284",
    "gtja_072": "gtja_082",
    "gtja_077": "a101_077",
    "gtja_078": "jq_199",
    "gtja_080": "a158_144",
    "gtja_081": "jq_137",
    "gtja_083": "gtja_099",
    "gtja_086": "a101_046",
    "gtja_087": "a101_066",
    "gtja_088": "a360_098",
    "gtja_089": "gtja_024",
    "gtja_093": "gtja_160",
    "gtja_095": "jq_116",
    "gtja_096": "tst_034",
    "gtja_097": "jq_137",
    "gtja_098": "a101_024",
    "gtja_100": "jq_137",
    "gtja_101": "a101_074",
    "gtja_103": "a158_081",
    "gtja_104": "a101_022",
    "gtja_106": "gtja_059",
    "gtja_107": "a101_020",
    "gtja_110": "jq_107",
    "gtja_112": "a360_229",
    "gtja_114": "a101_083",
    "gtja_115": "a101_085",
    "gtja_116": "gtja_147",
    "gtja_118": "jq_103",
    "gtja_123": "a101_099",
    "gtja_124": "a158_013",
    "gtja_126": "gtja_010",
    "gtja_128": "jq_188",
    "gtja_129": "gtja_160",
    "gtja_132": "jq_116",
    "gtja_133": "gtja_177",
    "gtja_134": "jq_219",
    "gtja_136": "gtja_139",
    "gtja_140": "a101_088",
    "gtja_144": "jq_116",
    "gtja_145": "jq_132",
    "gtja_150": "jq_116",
    "gtja_151": "gtja_135",
    "gtja_153": "gtja_010",
    "gtja_155": "jq_125",
    "gtja_158": "ta_049",
    "gtja_159": "ta_016",
    "gtja_161": "gtja_189",
    "gtja_163": "jq_218",
    "gtja_166": "jq_155",
    "gtja_167": "gtja_189",
    "gtja_168": "gtja_102",
    "gtja_173": "gtja_010",
    "gtja_175": "gtja_189",
    "gtja_178": "jq_218",
    "gtja_180": "a101_007",
    "gtja_184": "a101_037",
    "gtja_186": "gtja_172",
    "gtja_187": "gtja_189",
    "gtja_190": "jq_155",
    "jkp_001": "oap_085",
    "jkp_002": "cnc_029",
    "jkp_003": "jkp_100",
    "jkp_004": "oap_086",
    "jkp_006": "oap_087",
    "jkp_007": "cnc_036",
    "jkp_008": "cnc_029",
    "jkp_009": "oap_029",
    "jkp_010": "cnc_071",
    "jkp_011": "jkp_077",
    "jkp_012": "cnc_106",
    "jkp_013": "cnc_028",
    "jkp_014": "oap_060",
    "jkp_019": "jq_144",
    "jkp_021": "cnc_068",
    "jkp_022": "jq_002",
    "jkp_024": "cnc_071",
    "jkp_025": "cnc_072",
    "jkp_027": "oap_117",
    "jkp_028": "cnc_073",
    "jkp_032": "jkp_047",
    "jkp_034": "oap_119",
    "jkp_037": "cnc_078",
    "jkp_039": "oap_097",
    "jkp_040": "oap_097",
    "jkp_041": "oap_111",
    "jkp_043": "oap_101",
    "jkp_046": "jkp_047",
    "jkp_052": "jkp_049",
    "jkp_054": "oap_111",
    "jkp_055": "jkp_042",
    "jkp_058": "jkp_056",
    "jkp_059": "cnc_043",
    "jkp_061": "cnc_043",
    "jkp_062": "jkp_060",
    "jkp_063": "jkp_072",
    "jkp_065": "oap_078",
    "jkp_066": "oap_078",
    "jkp_067": "cnc_050",
    "jkp_068": "cnc_050",
    "jkp_070": "jkp_069",
    "jkp_071": "jkp_072",
    "jkp_074": "jkp_073",
    "jkp_076": "jkp_144",
    "jkp_078": "oap_080",
    "jkp_079": "cnc_070",
    "jkp_080": "cnc_108",
    "jkp_081": "cnc_092",
    "jkp_082": "cnc_093",
    "jkp_083": "oap_078",
    "jkp_084": "oap_130",
    "jkp_085": "cnc_095",
    "jkp_086": "oap_090",
    "jkp_088": "jq_008",
    "jkp_090": "cnc_101",
    "jkp_093": "cnc_103",
    "jkp_098": "cnc_106",
    "jkp_099": "oap_010",
    "jkp_102": "cnc_028",
    "jkp_103": "oap_013",
    "jkp_104": "oap_014",
    "jkp_105": "svc_001",
    "jkp_106": "oap_018",
    "jkp_107": "svc_019",
    "jkp_108": "oap_018",
    "jkp_109": "oap_018",
    "jkp_110": "oap_021",
    "jkp_113": "cnc_006",
    "jkp_114": "cnc_013",
    "jkp_115": "jq_116",
    "jkp_116": "cnc_013",
    "jkp_117": "cnc_038",
    "jkp_119": "oap_044",
    "jkp_120": "oap_045",
    "jkp_122": "oap_015",
    "jkp_123": "jq_155",
    "jkp_125": "oap_017",
    "jkp_127": "a360_098",
    "jkp_129": "oap_047",
    "jkp_130": "cnc_030",
    "jkp_131": "jq_226",
    "jkp_132": "oap_006",
    "jkp_134": "oap_015",
    "jkp_136": "jq_039",
    "jkp_137": "jkp_073",
    "jkp_142": "jkp_141",
    "jq_004": "jq_001",
    "jq_007": "jq_053",
    "jq_011": "jq_020",
    "jq_012": "jq_020",
    "jq_015": "jq_022",
    "jq_018": "cnc_070",
    "jq_024": "jq_037",
    "jq_025": "cnc_105",
    "jq_026": "jq_039",
    "jq_027": "jq_039",
    "jq_030": "jq_001",
    "jq_031": "jq_037",
    "jq_032": "cnc_070",
    "jq_035": "cnc_070",
    "jq_041": "jq_039",
    "jq_043": "oap_111",
    "jq_044": "jq_008",
    "jq_047": "jq_144",
    "jq_051": "jq_006",
    "jq_052": "jq_003",
    "jq_055": "oap_078",
    "jq_056": "jkp_073",
    "jq_058": "jkp_073",
    "jq_059": "jkp_073",
    "jq_062": "jq_039",
    "jq_064": "jq_017",
    "jq_065": "cnc_070",
    "jq_066": "jq_039",
    "jq_069": "jq_002",
    "jq_070": "svc_001",
    "jq_071": "jq_090",
    "jq_072": "jq_096",
    "jq_075": "jq_078",
    "jq_076": "jq_092",
    "jq_077": "jq_067",
    "jq_079": "jq_092",
    "jq_080": "svc_001",
    "jq_081": "cnc_072",
    "jq_083": "jq_074",
    "jq_085": "jq_002",
    "jq_086": "jq_090",
    "jq_089": "jq_090",
    "jq_091": "jq_067",
    "jq_093": "jq_067",
    "jq_094": "jq_090",
    "jq_095": "jq_092",
    "jq_097": "cnc_068",
    "jq_099": "jq_092",
    "jq_100": "jq_092",
    "jq_101": "jq_090",
    "jq_105": "oap_018",
    "jq_106": "ta_016",
    "jq_109": "cnc_008",
    "jq_110": "jq_108",
    "jq_112": "jq_116",
    "jq_113": "tst_056",
    "jq_115": "jq_116",
    "jq_117": "jq_116",
    "jq_118": "jq_116",
    "jq_120": "jq_119",
    "jq_121": "jq_137",
    "jq_122": "jq_137",
    "jq_123": "jq_137",
    "jq_124": "jq_137",
    "jq_126": "jq_230",
    "jq_127": "cnc_006",
    "jq_128": "jq_230",
    "jq_129": "cnc_006",
    "jq_130": "jq_230",
    "jq_131": "cnc_006",
    "jq_133": "gtja_040",
    "jq_136": "jq_137",
    "jq_138": "jq_111",
    "jq_140": "jkp_053",
    "jq_141": "jkp_138",
    "jq_143": "jq_142",
    "jq_146": "cnc_073",
    "jq_147": "jq_142",
    "jq_151": "tst_028",
    "jq_152": "a360_098",
    "jq_153": "a360_071",
    "jq_157": "jq_159",
    "jq_158": "oap_018",
    "jq_163": "jq_167",
    "jq_165": "jq_002",
    "jq_166": "cnc_043",
    "jq_169": "jq_168",
    "jq_170": "jq_164",
    "jq_171": "jq_164",
    "jq_173": "jq_168",
    "jq_174": "jq_168",
    "jq_176": "tst_036",
    "jq_177": "a360_173",
    "jq_178": "a360_173",
    "jq_179": "a360_071",
    "jq_180": "ta_034",
    "jq_181": "a158_058",
    "jq_182": "a360_173",
    "jq_183": "a360_071",
    "jq_184": "a360_284",
    "jq_185": "a360_238",
    "jq_186": "a158_058",
    "jq_187": "a158_043",
    "jq_189": "cnc_038",
    "jq_192": "a360_173",
    "jq_193": "tst_036",
    "jq_194": "a360_173",
    "jq_195": "a360_284",
    "jq_196": "a360_238",
    "jq_197": "a158_058",
    "jq_198": "tst_038",
    "jq_200": "jq_199",
    "jq_201": "jq_199",
    "jq_202": "a158_058",
    "jq_203": "a360_098",
    "jq_204": "tst_028",
    "jq_205": "tst_050",
    "jq_206": "a360_229",
    "jq_207": "a360_098",
    "jq_208": "a360_175",
    "jq_209": "a360_284",
    "jq_210": "tst_028",
    "jq_211": "a158_058",
    "jq_212": "a360_098",
    "jq_213": "a360_229",
    "jq_214": "tst_028",
    "jq_215": "a360_098",
    "jq_216": "a360_173",
    "jq_217": "a360_071",
    "jq_220": "gtja_029",
    "jq_221": "a360_098",
    "jq_222": "a360_229",
    "jq_223": "a360_098",
    "jq_224": "svc_001",
    "jq_225": "cnc_025",
    "jq_229": "cnc_071",
    "jq_231": "oap_060",
    "jq_233": "cnc_070",
    "oap_001": "svc_001",
    "oap_002": "cnc_038",
    "oap_003": "a360_098",
    "oap_004": "cnc_029",
    "oap_005": "jkp_100",
    "oap_008": "cnc_032",
    "oap_009": "jkp_100",
    "oap_011": "jkp_101",
    "oap_012": "cnc_028",
    "oap_016": "jq_155",
    "oap_019": "oap_018",
    "oap_020": "svc_019",
    "oap_022": "jkp_111",
    "oap_028": "oap_027",
    "oap_031": "cnc_036",
    "oap_032": "cnc_031",
    "oap_034": "cnc_029",
    "oap_035": "cnc_032",
    "oap_036": "cnc_029",
    "oap_037": "jkp_118",
    "oap_038": "jkp_118",
    "oap_039": "cnc_006",
    "oap_046": "jkp_121",
    "oap_048": "cnc_106",
    "oap_049": "cnc_106",
    "oap_050": "cnc_071",
    "oap_051": "cnc_070",
    "oap_052": "jkp_077",
    "oap_053": "oap_060",
    "oap_054": "oap_060",
    "oap_055": "jq_002",
    "oap_056": "cnc_068",
    "oap_059": "cnc_072",
    "oap_061": "cnc_106",
    "oap_062": "jkp_016",
    "oap_063": "jkp_017",
    "oap_064": "jkp_026",
    "oap_065": "oap_068",
    "oap_066": "oap_068",
    "oap_067": "oap_068",
    "oap_071": "cnc_103",
    "oap_072": "cnc_050",
    "oap_073": "jkp_072",
    "oap_074": "jkp_072",
    "oap_076": "jq_039",
    "oap_077": "cnc_043",
    "oap_079": "cnc_101",
    "oap_089": "jkp_018",
    "oap_091": "jq_046",
    "oap_094": "cnc_073",
    "oap_095": "jkp_053",
    "oap_096": "oap_097",
    "oap_098": "jkp_029",
    "oap_099": "jkp_031",
    "oap_100": "jkp_030",
    "oap_102": "oap_109",
    "oap_103": "jkp_044",
    "oap_104": "jkp_045",
    "oap_105": "jkp_053",
    "oap_106": "jkp_049",
    "oap_107": "jkp_051",
    "oap_108": "jkp_049",
    "oap_110": "oap_101",
    "oap_112": "oap_069",
    "oap_113": "oap_111",
    "oap_114": "oap_115",
    "oap_116": "cnc_109",
    "oap_120": "oap_119",
    "oap_124": "oap_122",
    "oap_125": "jkp_057",
    "oap_126": "jkp_056",
    "oap_127": "jkp_056",
    "oap_129": "cnc_092",
    "oap_133": "oap_132",
    "oap_134": "oap_132",
    "oap_135": "cnc_095",
    "oap_141": "oap_142",
    "svc_002": "oap_060",
    "svc_003": "cnc_008",
    "svc_004": "svc_008",
    "svc_005": "svc_009",
    "svc_006": "svc_017",
    "svc_007": "jq_227",
    "svc_010": "svc_017",
    "svc_012": "jq_227",
    "svc_014": "svc_021",
    "svc_016": "svc_009",
    "svc_020": "jkp_100",
    "ta_001": "ta_002",
    "ta_005": "ta_004",
    "ta_006": "a158_001",
    "ta_007": "jq_199",
    "ta_009": "a158_058",
    "ta_010": "tst_002",
    "ta_011": "a158_043",
    "ta_012": "jq_188",
    "ta_014": "tst_036",
    "ta_017": "a360_098",
    "ta_018": "a360_229",
    "ta_019": "a158_058",
    "ta_020": "ta_021",
    "ta_022": "tst_069",
    "ta_023": "ta_021",
    "ta_024": "a158_064",
    "ta_029": "jq_199",
    "ta_030": "ta_063",
    "ta_031": "ta_035",
    "ta_032": "a158_058",
    "ta_033": "a360_284",
    "ta_036": "ta_034",
    "ta_037": "a360_229",
    "ta_038": "a360_229",
    "ta_040": "a360_098",
    "ta_041": "a158_041",
    "ta_042": "a158_042",
    "ta_043": "a360_284",
    "ta_044": "a360_284",
    "ta_046": "ta_027",
    "ta_047": "gtja_040",
    "ta_048": "jq_159",
    "ta_050": "a158_001",
    "ta_051": "a158_013",
    "ta_061": "a360_229",
    "ta_062": "a360_229",
    "ta_064": "ta_060",
    "ta_080": "ta_111",
    "ta_102": "ta_111",
    "ta_119": "ta_082",
    "tst_001": "tst_002",
    "tst_003": "oap_018",
    "tst_004": "a360_173",
    "tst_005": "a360_238",
    "tst_006": "a360_173",
    "tst_007": "a360_284",
    "tst_008": "tst_036",
    "tst_009": "jq_175",
    "tst_010": "jq_103",
    "tst_011": "jq_107",
    "tst_012": "jq_199",
    "tst_013": "a360_098",
    "tst_014": "tst_002",
    "tst_015": "a360_071",
    "tst_016": "ta_015",
    "tst_017": "ta_013",
    "tst_018": "gtja_172",
    "tst_019": "gtja_172",
    "tst_020": "a360_098",
    "tst_021": "a360_098",
    "tst_022": "a360_238",
    "tst_023": "a360_173",
    "tst_024": "ta_034",
    "tst_025": "a158_058",
    "tst_026": "a158_058",
    "tst_027": "a360_071",
    "tst_029": "a360_284",
    "tst_031": "a360_173",
    "tst_032": "a158_058",
    "tst_033": "ta_021",
    "tst_035": "ta_021",
    "tst_037": "ta_034",
    "tst_039": "a360_238",
    "tst_040": "a360_173",
    "tst_041": "a360_284",
    "tst_042": "a360_098",
    "tst_043": "a158_058",
    "tst_044": "a360_071",
    "tst_045": "tst_028",
    "tst_046": "a158_058",
    "tst_047": "tst_002",
    "tst_048": "a158_043",
    "tst_049": "tst_050",
    "tst_051": "jq_188",
    "tst_052": "a360_229",
    "tst_053": "a360_284",
    "tst_054": "gtja_040",
    "tst_055": "tst_056",
    "tst_057": "a360_229",
    "tst_058": "a360_284",
    "tst_059": "jq_199",
    "tst_060": "gtja_079",
    "tst_061": "a158_058",
    "tst_062": "tst_036",
    "tst_063": "a360_284",
    "tst_064": "jq_175",
    "tst_065": "tst_002",
    "tst_066": "gtja_135",
    "tst_067": "gtja_040",
    "tst_068": "ta_028",
    "tst_070": "a360_238",
    "tst_071": "ta_034",
}

REMOVE_DUPLICATES_SELECTION = FactorSelection(
    name="remove_duplicates",
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
    members=REMOVE_DUPLICATES_MEMBERS,
    criteria={
        "universe": "sh_main",
        "thresholds": None,
        "deduplication_order": "mean_long_excess",
        "max_abs_rank_correlation": 0.8,
        "correlation_date_step": 5,
        "correlation_start": "2008-05-12",
        "correlation_end": "2026-10-20",
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
