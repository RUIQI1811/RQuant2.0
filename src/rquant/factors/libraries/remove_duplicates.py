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
  ``max_abs_rank_correlation`` (0.5). ``DEDUPLICATED`` maps each dropped factor to the kept one it duplicates.

Kept per source: qlib_alpha158 28, wq_alpha101 57, gtja191 41, qlib_alpha360 5, china_chars 25, size_value_china 5,
talib 63, jq240 39, oap 44, barra_cne6 7, tushare_tech 5, jkp153 26.

The ordering uses 2020-2026 sh_main returns, so for walk-forward test years from 2020 on the choice among duplicates
is in-sample; membership is only meant for the ``sh_main`` universe. Members keep their source canonical names and
values; ``direction == -1`` marks a factor whose 2020+ Rank IC is negative, the stored column is not negated.
"""

from __future__ import annotations

from rquant.factors.libraries.base import FactorSelection, SelectedFactor

REMOVE_DUPLICATES_MEMBERS = (
    SelectedFactor("a158_005", 1),  # long excess +2.2%, RankIC +0.0010, cov 1.000
    SelectedFactor("a158_006", -1),  # long excess +0.5%, RankIC -0.0247, cov 1.000
    SelectedFactor("a158_024", -1),  # long excess +2.1%, RankIC -0.0414, cov 0.999
    SelectedFactor("a158_027", -1),  # long excess +2.7%, RankIC -0.0287, cov 0.992
    SelectedFactor("a158_034", 1),  # long excess -4.2%, RankIC +0.0015, cov 1.000
    SelectedFactor("a158_035", -1),  # long excess +5.8%, RankIC -0.0018, cov 1.000
    SelectedFactor("a158_036", 1),  # long excess -0.8%, RankIC +0.0000, cov 1.000
    SelectedFactor("a158_037", 1),  # long excess +2.8%, RankIC +0.0019, cov 1.000
    SelectedFactor("a158_038", 1),  # long excess +4.8%, RankIC +0.0020, cov 1.000
    SelectedFactor("a158_039", -1),  # long excess -0.2%, RankIC -0.0175, cov 1.000
    SelectedFactor("a158_074", -1),  # long excess -1.6%, RankIC -0.0166, cov 0.999
    SelectedFactor("a158_077", -1),  # long excess +3.6%, RankIC -0.0247, cov 0.992
    SelectedFactor("a158_079", 1),  # long excess +4.3%, RankIC +0.0145, cov 0.999
    SelectedFactor("a158_080", 1),  # long excess +2.8%, RankIC +0.0127, cov 0.998
    SelectedFactor("a158_082", 1),  # long excess +1.2%, RankIC +0.0164, cov 0.992
    SelectedFactor("a158_083", 1),  # long excess +1.6%, RankIC +0.0131, cov 0.984
    SelectedFactor("a158_093", -1),  # long excess +0.2%, RankIC -0.0123, cov 1.000
    SelectedFactor("a158_095", -1),  # long excess -0.4%, RankIC -0.0253, cov 1.000
    SelectedFactor("a158_097", -1),  # long excess +2.2%, RankIC -0.0249, cov 1.000
    SelectedFactor("a158_103", -1),  # long excess +3.1%, RankIC -0.0137, cov 1.000
    SelectedFactor("a158_104", 1),  # long excess +1.8%, RankIC +0.0132, cov 1.000
    SelectedFactor("a158_107", 1),  # long excess +2.6%, RankIC +0.0079, cov 1.000
    SelectedFactor("a158_134", -1),  # long excess +2.0%, RankIC -0.0141, cov 0.999
    SelectedFactor("a158_140", -1),  # long excess +3.7%, RankIC -0.0120, cov 0.997
    SelectedFactor("a158_143", -1),  # long excess +2.8%, RankIC -0.0058, cov 0.984
    SelectedFactor("a158_144", -1),  # long excess +0.9%, RankIC -0.0192, cov 0.999
    SelectedFactor("a158_145", -1),  # long excess -0.4%, RankIC -0.0233, cov 0.997
    SelectedFactor("a158_146", -1),  # long excess +2.0%, RankIC -0.0281, cov 0.995
    SelectedFactor("a101_001", -1),  # long excess +0.2%, RankIC -0.0168, cov 0.994
    SelectedFactor("a101_004", 1),  # long excess +3.9%, RankIC +0.0241, cov 0.998
    SelectedFactor("a101_008", 1),  # long excess -6.1%, RankIC +0.0117, cov 0.996
    SelectedFactor("a101_015", 1),  # long excess +3.7%, RankIC +0.0219, cov 1.000
    SelectedFactor("a101_019", 1),  # long excess -0.3%, RankIC +0.0189, cov 0.916
    SelectedFactor("a101_022", 1),  # long excess -3.0%, RankIC +0.0051, cov 0.995
    SelectedFactor("a101_023", 1),  # long excess -0.1%, RankIC +0.0152, cov 1.000
    SelectedFactor("a101_024", 1),  # long excess +2.5%, RankIC +0.0215, cov 0.999
    SelectedFactor("a101_026", 1),  # long excess +1.4%, RankIC +0.0211, cov 1.000
    SelectedFactor("a101_027", 1),  # long excess +4.2%, RankIC +0.0198, cov 1.000
    SelectedFactor("a101_031", 1),  # long excess +0.9%, RankIC +0.0073, cov 0.995
    SelectedFactor("a101_032", 1),  # long excess +3.6%, RankIC +0.0001, cov 0.921
    SelectedFactor("a101_034", 1),  # long excess +3.8%, RankIC +0.0167, cov 1.000
    SelectedFactor("a101_036", 1),  # long excess -1.2%, RankIC +0.0085, cov 0.933
    SelectedFactor("a101_037", 1),  # long excess +6.4%, RankIC +0.0235, cov 0.933
    SelectedFactor("a101_040", 1),  # long excess +9.7%, RankIC +0.0418, cov 0.998
    SelectedFactor("a101_041", 1),  # long excess -2.4%, RankIC +0.0106, cov 1.000
    SelectedFactor("a101_044", 1),  # long excess +4.9%, RankIC +0.0272, cov 1.000
    SelectedFactor("a101_046", 1),  # long excess +1.5%, RankIC +0.0169, cov 1.000
    SelectedFactor("a101_047", 1),  # long excess +10.5%, RankIC +0.0033, cov 1.000
    SelectedFactor("a101_048", -1),  # long excess +1.4%, RankIC -0.0036, cov 0.912
    SelectedFactor("a101_053", 1),  # long excess +2.8%, RankIC +0.0108, cov 0.999
    SelectedFactor("a101_056", 1),  # long excess +2.5%, RankIC +0.0119, cov 0.997
    SelectedFactor("a101_059", 1),  # long excess +2.4%, RankIC +0.0114, cov 0.987
    SelectedFactor("a101_061", 1),  # long excess +4.5%, RankIC +0.0156, cov 1.000
    SelectedFactor("a101_062", 1),  # long excess +3.8%, RankIC +0.0101, cov 1.000
    SelectedFactor("a101_063", 1),  # long excess +5.5%, RankIC +0.0088, cov 0.916
    SelectedFactor("a101_064", 1),  # long excess +1.5%, RankIC +0.0022, cov 1.000
    SelectedFactor("a101_065", 1),  # long excess +5.1%, RankIC +0.0154, cov 1.000
    SelectedFactor("a101_066", 1),  # long excess +0.6%, RankIC +0.0167, cov 0.997
    SelectedFactor("a101_068", 1),  # long excess +2.3%, RankIC +0.0080, cov 1.000
    SelectedFactor("a101_069", 1),  # long excess +6.4%, RankIC +0.0247, cov 0.988
    SelectedFactor("a101_071", 1),  # long excess +4.7%, RankIC +0.0155, cov 0.995
    SelectedFactor("a101_072", 1),  # long excess +2.1%, RankIC +0.0071, cov 0.955
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
    SelectedFactor("a101_086", 1),  # long excess +5.4%, RankIC +0.0121, cov 1.000
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
    SelectedFactor("gtja_004", 1),  # long excess +2.9%, RankIC +0.0013, cov 0.996
    SelectedFactor("gtja_007", 1),  # long excess +9.5%, RankIC +0.0130, cov 0.999
    SelectedFactor("gtja_011", -1),  # long excess +2.0%, RankIC -0.0156, cov 0.992
    SelectedFactor("gtja_012", 1),  # long excess -5.2%, RankIC +0.0202, cov 0.998
    SelectedFactor("gtja_015", 1),  # long excess +3.5%, RankIC +0.0126, cov 1.000
    SelectedFactor("gtja_044", -1),  # long excess +0.8%, RankIC -0.0090, cov 0.993
    SelectedFactor("gtja_045", -1),  # long excess -0.5%, RankIC -0.0151, cov 0.946
    SelectedFactor("gtja_056", 1),  # long excess +3.3%, RankIC +0.0149, cov 0.981
    SelectedFactor("gtja_061", 1),  # long excess +2.2%, RankIC +0.0268, cov 0.968
    SelectedFactor("gtja_064", 1),  # long excess +4.3%, RankIC +0.0206, cov 0.974
    SelectedFactor("gtja_068", -1),  # long excess +1.4%, RankIC -0.0221, cov 1.000
    SelectedFactor("gtja_075", -1),  # long excess +1.2%, RankIC -0.0154, cov 0.987
    SelectedFactor("gtja_076", -1),  # long excess +1.0%, RankIC -0.0058, cov 0.995
    SelectedFactor("gtja_085", 1),  # long excess +4.6%, RankIC +0.0057, cov 0.990
    SelectedFactor("gtja_092", 1),  # long excess +3.5%, RankIC +0.0167, cov 0.930
    SelectedFactor("gtja_105", 1),  # long excess +5.6%, RankIC +0.0170, cov 0.998
    SelectedFactor("gtja_113", 1),  # long excess -2.7%, RankIC +0.0141, cov 0.994
    SelectedFactor("gtja_117", 1),  # long excess -0.0%, RankIC +0.0134, cov 0.991
    SelectedFactor("gtja_119", -1),  # long excess +2.0%, RankIC -0.0027, cov 0.985
    SelectedFactor("gtja_120", 1),  # long excess +14.1%, RankIC +0.0357, cov 1.000
    SelectedFactor("gtja_122", -1),  # long excess +1.5%, RankIC -0.0043, cov 1.000
    SelectedFactor("gtja_125", 1),  # long excess -2.0%, RankIC +0.0044, cov 0.964
    SelectedFactor("gtja_131", -1),  # long excess -1.6%, RankIC -0.0122, cov 0.975
    SelectedFactor("gtja_141", 1),  # long excess +3.9%, RankIC +0.0097, cov 0.994
    SelectedFactor("gtja_143", -1),  # long excess +0.3%, RankIC -0.0057, cov 1.000
    SelectedFactor("gtja_146", -1),  # long excess +4.4%, RankIC -0.0093, cov 0.995
    SelectedFactor("gtja_149", -1),  # long excess -3.3%, RankIC -0.0064, cov 0.942
    SelectedFactor("gtja_156", 1),  # long excess +6.6%, RankIC +0.0313, cov 0.998
    SelectedFactor("gtja_157", 1),  # long excess +3.7%, RankIC +0.0235, cov 0.997
    SelectedFactor("gtja_165", 1),  # long excess +1.5%, RankIC +0.0041, cov 0.954
    SelectedFactor("gtja_176", -1),  # long excess +2.3%, RankIC -0.0214, cov 0.996
    SelectedFactor("gtja_179", -1),  # long excess +2.2%, RankIC -0.0217, cov 0.984
    SelectedFactor("gtja_180", -1),  # long excess -8.5%, RankIC -0.0032, cov 0.982
    SelectedFactor("gtja_181", 1),  # long excess -6.8%, RankIC +0.0006, cov 0.990
    SelectedFactor("gtja_182", 1),  # long excess +2.4%, RankIC +0.0065, cov 0.995
    SelectedFactor("gtja_183", 1),  # long excess +2.1%, RankIC +0.0105, cov 0.981
    SelectedFactor("gtja_185", 1),  # long excess -1.5%, RankIC +0.0293, cov 1.000
    SelectedFactor("gtja_190", 1),  # long excess +1.1%, RankIC +0.0200, cov 0.990
    SelectedFactor("gtja_191", -1),  # long excess +1.9%, RankIC -0.0076, cov 0.994
    SelectedFactor("a360_060", 1),  # long excess n/a, RankIC n/a, cov 1.000
    SelectedFactor("a360_071", 1),  # long excess +9.6%, RankIC +0.0330, cov 0.999
    SelectedFactor("a360_313", 1),  # long excess +6.8%, RankIC +0.0374, cov 0.999
    SelectedFactor("a360_358", 1),  # long excess -1.6%, RankIC +0.0115, cov 1.000
    SelectedFactor("a360_360", 1),  # long excess n/a, RankIC n/a, cov 1.000
    SelectedFactor("cnc_008", -1),  # long excess +8.4%, RankIC -0.0335, cov 0.997
    SelectedFactor("cnc_013", -1),  # long excess +3.7%, RankIC -0.0182, cov 1.000
    SelectedFactor("cnc_021", -1),  # long excess +2.9%, RankIC -0.0090, cov 1.000
    SelectedFactor("cnc_024", -1),  # long excess -6.3%, RankIC -0.0109, cov 1.000
    SelectedFactor("cnc_031", -1),  # long excess -0.6%, RankIC -0.0076, cov 0.996
    SelectedFactor("cnc_032", -1),  # long excess +3.5%, RankIC -0.0052, cov 0.896
    SelectedFactor("cnc_033", -1),  # long excess +1.4%, RankIC -0.0069, cov 0.976
    SelectedFactor("cnc_038", -1),  # long excess +5.4%, RankIC -0.0175, cov 1.000
    SelectedFactor("cnc_047", 1),  # long excess -0.4%, RankIC +0.0004, cov 0.974
    SelectedFactor("cnc_052", 1),  # long excess +1.1%, RankIC +0.0036, cov 0.953
    SelectedFactor("cnc_055", 1),  # long excess -3.6%, RankIC +0.0024, cov 1.000
    SelectedFactor("cnc_068", 1),  # long excess +3.4%, RankIC +0.0145, cov 0.992
    SelectedFactor("cnc_072", -1),  # long excess +3.0%, RankIC -0.0055, cov 0.978
    SelectedFactor("cnc_078", -1),  # long excess +3.9%, RankIC -0.0149, cov 0.976
    SelectedFactor("cnc_079", -1),  # long excess +0.8%, RankIC -0.0044, cov 0.972
    SelectedFactor("cnc_081", -1),  # long excess +0.2%, RankIC -0.0016, cov 0.947
    SelectedFactor("cnc_083", -1),  # long excess +2.7%, RankIC -0.0027, cov 0.956
    SelectedFactor("cnc_084", -1),  # long excess +1.1%, RankIC -0.0026, cov 0.956
    SelectedFactor("cnc_087", -1),  # long excess +1.4%, RankIC -0.0031, cov 0.956
    SelectedFactor("cnc_090", 1),  # long excess -1.2%, RankIC +0.0011, cov 1.000
    SelectedFactor("cnc_092", 1),  # long excess +10.0%, RankIC +0.0097, cov 0.844
    SelectedFactor("cnc_097", 1),  # long excess -0.8%, RankIC +0.0006, cov 0.950
    SelectedFactor("cnc_100", -1),  # long excess +1.5%, RankIC -0.0007, cov 0.994
    SelectedFactor("cnc_101", 1),  # long excess +1.4%, RankIC +0.0015, cov 0.703
    SelectedFactor("cnc_104", -1),  # long excess -1.3%, RankIC -0.0067, cov 0.966
    SelectedFactor("svc_009", 1),  # long excess +5.9%, RankIC +0.0027, cov 1.000
    SelectedFactor("svc_013", 1),  # long excess +8.6%, RankIC +0.0301, cov 1.000
    SelectedFactor("svc_015", -1),  # long excess -0.6%, RankIC -0.0075, cov 0.993
    SelectedFactor("svc_017", 1),  # long excess -2.8%, RankIC +0.0151, cov 0.993
    SelectedFactor("svc_018", -1),  # long excess +3.2%, RankIC -0.0036, cov 0.993
    SelectedFactor("ta_002", -1),  # long excess -0.6%, RankIC -0.0092, cov 1.000
    SelectedFactor("ta_004", -1),  # long excess +3.6%, RankIC -0.0234, cov 1.000
    SelectedFactor("ta_008", -1),  # long excess +2.0%, RankIC -0.0159, cov 1.000
    SelectedFactor("ta_015", -1),  # long excess +6.0%, RankIC -0.0411, cov 1.000
    SelectedFactor("ta_045", -1),  # long excess +1.5%, RankIC -0.0133, cov 1.000
    SelectedFactor("ta_052", -1),  # long excess -1.5%, RankIC -0.0009, cov 1.000
    SelectedFactor("ta_053", -1),  # long excess -1.5%, RankIC -0.0083, cov 1.000
    SelectedFactor("ta_055", -1),  # long excess +0.6%, RankIC -0.0096, cov 1.000
    SelectedFactor("ta_056", -1),  # long excess -2.5%, RankIC -0.0048, cov 1.000
    SelectedFactor("ta_058", -1),  # long excess +2.9%, RankIC -0.0107, cov 1.000
    SelectedFactor("ta_059", -1),  # long excess +5.8%, RankIC -0.0253, cov 1.000
    SelectedFactor("ta_064", 1),  # long excess -4.8%, RankIC +0.0108, cov 1.000
    SelectedFactor("ta_065", 1),  # long excess n/a, RankIC +0.0003, cov 1.000
    SelectedFactor("ta_066", -1),  # long excess n/a, RankIC -0.0010, cov 1.000
    SelectedFactor("ta_067", -1),  # long excess -1.2%, RankIC -0.0009, cov 1.000
    SelectedFactor("ta_068", 1),  # long excess -2.0%, RankIC +0.0026, cov 1.000
    SelectedFactor("ta_069", -1),  # long excess -0.5%, RankIC -0.0064, cov 1.000
    SelectedFactor("ta_070", 1),  # long excess n/a, RankIC n/a, cov 1.000
    SelectedFactor("ta_071", -1),  # long excess n/a, RankIC -0.0032, cov 1.000
    SelectedFactor("ta_072", 1),  # long excess n/a, RankIC +0.0052, cov 1.000
    SelectedFactor("ta_073", -1),  # long excess -0.8%, RankIC -0.0003, cov 1.000
    SelectedFactor("ta_075", -1),  # long excess n/a, RankIC -0.0036, cov 1.000
    SelectedFactor("ta_077", -1),  # long excess n/a, RankIC -0.0064, cov 1.000
    SelectedFactor("ta_078", -1),  # long excess -1.5%, RankIC -0.0001, cov 1.000
    SelectedFactor("ta_079", 1),  # long excess -3.8%, RankIC +0.0014, cov 1.000
    SelectedFactor("ta_081", -1),  # long excess -0.6%, RankIC -0.0035, cov 1.000
    SelectedFactor("ta_082", -1),  # long excess -2.1%, RankIC -0.0011, cov 1.000
    SelectedFactor("ta_083", -1),  # long excess +0.5%, RankIC -0.0087, cov 1.000
    SelectedFactor("ta_085", 1),  # long excess -5.2%, RankIC +0.0004, cov 1.000
    SelectedFactor("ta_086", 1),  # long excess +1.3%, RankIC +0.0005, cov 1.000
    SelectedFactor("ta_087", -1),  # long excess +1.7%, RankIC -0.0026, cov 1.000
    SelectedFactor("ta_088", -1),  # long excess -7.5%, RankIC -0.0016, cov 1.000
    SelectedFactor("ta_089", 1),  # long excess +2.8%, RankIC +0.0010, cov 1.000
    SelectedFactor("ta_090", 1),  # long excess +5.6%, RankIC +0.0048, cov 1.000
    SelectedFactor("ta_092", 1),  # long excess +3.2%, RankIC +0.0017, cov 1.000
    SelectedFactor("ta_093", 1),  # long excess +3.8%, RankIC +0.0005, cov 1.000
    SelectedFactor("ta_094", 1),  # long excess +1.9%, RankIC +0.0011, cov 1.000
    SelectedFactor("ta_095", 1),  # long excess -3.9%, RankIC +0.0018, cov 1.000
    SelectedFactor("ta_096", -1),  # long excess n/a, RankIC -0.0003, cov 1.000
    SelectedFactor("ta_097", 1),  # long excess -15.8%, RankIC +0.0019, cov 1.000
    SelectedFactor("ta_098", -1),  # long excess -0.5%, RankIC -0.0029, cov 1.000
    SelectedFactor("ta_099", 1),  # long excess n/a, RankIC +0.0026, cov 1.000
    SelectedFactor("ta_101", -1),  # long excess n/a, RankIC -0.0024, cov 1.000
    SelectedFactor("ta_103", -1),  # long excess +5.6%, RankIC -0.0130, cov 1.000
    SelectedFactor("ta_104", -1),  # long excess +1.0%, RankIC -0.0096, cov 1.000
    SelectedFactor("ta_105", 1),  # long excess +0.5%, RankIC +0.0003, cov 1.000
    SelectedFactor("ta_106", -1),  # long excess n/a, RankIC -0.0032, cov 1.000
    SelectedFactor("ta_108", -1),  # long excess -8.4%, RankIC -0.0026, cov 1.000
    SelectedFactor("ta_109", 1),  # long excess -8.4%, RankIC +0.0020, cov 1.000
    SelectedFactor("ta_110", -1),  # long excess -11.4%, RankIC -0.0024, cov 1.000
    SelectedFactor("ta_111", 1),  # long excess +1.1%, RankIC +0.0000, cov 1.000
    SelectedFactor("ta_112", -1),  # long excess n/a, RankIC -0.0018, cov 1.000
    SelectedFactor("ta_113", -1),  # long excess -3.8%, RankIC -0.0007, cov 1.000
    SelectedFactor("ta_114", 1),  # long excess +10.8%, RankIC +0.0006, cov 1.000
    SelectedFactor("ta_115", -1),  # long excess -1.1%, RankIC -0.0026, cov 1.000
    SelectedFactor("ta_117", 1),  # long excess -1.7%, RankIC +0.0004, cov 1.000
    SelectedFactor("ta_118", -1),  # long excess -10.3%, RankIC -0.0015, cov 1.000
    SelectedFactor("ta_120", 1),  # long excess -5.6%, RankIC +0.0017, cov 1.000
    SelectedFactor("ta_121", 1),  # long excess -11.8%, RankIC +0.0031, cov 1.000
    SelectedFactor("ta_122", 1),  # long excess -13.4%, RankIC +0.0007, cov 1.000
    SelectedFactor("ta_123", -1),  # long excess +9.4%, RankIC -0.0016, cov 1.000
    SelectedFactor("ta_124", -1),  # long excess n/a, RankIC -0.0012, cov 1.000
    SelectedFactor("ta_125", 1),  # long excess -6.2%, RankIC +0.0018, cov 1.000
    SelectedFactor("jq_002", 1),  # long excess +5.1%, RankIC +0.0148, cov 0.992
    SelectedFactor("jq_003", -1),  # long excess +0.6%, RankIC -0.0038, cov 0.947
    SelectedFactor("jq_005", -1),  # long excess +2.3%, RankIC -0.0017, cov 0.992
    SelectedFactor("jq_008", 1),  # long excess +2.2%, RankIC +0.0024, cov 0.932
    SelectedFactor("jq_010", 1),  # long excess -1.2%, RankIC +0.0016, cov 0.992
    SelectedFactor("jq_013", 1),  # long excess +1.1%, RankIC +0.0063, cov 0.992
    SelectedFactor("jq_016", -1),  # long excess -3.9%, RankIC -0.0017, cov 1.000
    SelectedFactor("jq_020", 1),  # long excess -1.4%, RankIC +0.0037, cov 0.943
    SelectedFactor("jq_021", -1),  # long excess +0.3%, RankIC -0.0002, cov 0.951
    SelectedFactor("jq_029", -1),  # long excess +1.3%, RankIC -0.0012, cov 0.877
    SelectedFactor("jq_032", -1),  # long excess -0.5%, RankIC -0.0011, cov 1.000
    SelectedFactor("jq_036", 1),  # long excess -0.4%, RankIC +0.0000, cov 0.818
    SelectedFactor("jq_038", 1),  # long excess -0.0%, RankIC +0.0044, cov 0.868
    SelectedFactor("jq_045", 1),  # long excess +3.0%, RankIC +0.0009, cov 0.987
    SelectedFactor("jq_046", -1),  # long excess +0.6%, RankIC -0.0001, cov 0.975
    SelectedFactor("jq_049", -1),  # long excess +1.7%, RankIC -0.0018, cov 0.943
    SelectedFactor("jq_050", 1),  # long excess -0.7%, RankIC +0.0033, cov 0.992
    SelectedFactor("jq_053", 1),  # long excess -3.4%, RankIC +0.0028, cov 0.947
    SelectedFactor("jq_057", -1),  # long excess +0.5%, RankIC -0.0005, cov 0.992
    SelectedFactor("jq_060", -1),  # long excess -2.8%, RankIC -0.0023, cov 0.992
    SelectedFactor("jq_061", 1),  # long excess -0.2%, RankIC +0.0011, cov 0.948
    SelectedFactor("jq_068", 1),  # long excess -3.0%, RankIC +0.0021, cov 0.992
    SelectedFactor("jq_084", -1),  # long excess -2.4%, RankIC -0.0017, cov 0.988
    SelectedFactor("jq_098", -1),  # long excess -1.4%, RankIC -0.0041, cov 0.992
    SelectedFactor("jq_102", 1),  # long excess -4.0%, RankIC +0.0046, cov 0.992
    SelectedFactor("jq_104", -1),  # long excess +1.3%, RankIC -0.0135, cov 1.000
    SelectedFactor("jq_111", -1),  # long excess +1.6%, RankIC -0.0255, cov 1.000
    SelectedFactor("jq_114", -1),  # long excess +6.1%, RankIC -0.0286, cov 1.000
    SelectedFactor("jq_116", -1),  # long excess +18.0%, RankIC -0.0436, cov 1.000
    SelectedFactor("jq_139", -1),  # long excess +2.5%, RankIC -0.0034, cov 0.884
    SelectedFactor("jq_145", 1),  # long excess -2.8%, RankIC +0.0002, cov 0.814
    SelectedFactor("jq_148", 1),  # long excess -3.6%, RankIC +0.0110, cov 0.993
    SelectedFactor("jq_149", -1),  # long excess -1.9%, RankIC -0.0014, cov 1.000
    SelectedFactor("jq_154", -1),  # long excess +4.8%, RankIC -0.0186, cov 0.993
    SelectedFactor("jq_160", -1),  # long excess -0.4%, RankIC -0.0026, cov 1.000
    SelectedFactor("jq_161", 1),  # long excess -2.6%, RankIC +0.0036, cov 1.000
    SelectedFactor("jq_162", 1),  # long excess -0.6%, RankIC +0.0006, cov 0.991
    SelectedFactor("jq_172", 1),  # long excess -1.5%, RankIC +0.0090, cov 1.000
    SelectedFactor("jq_230", -1),  # long excess +6.6%, RankIC -0.0361, cov 1.000
    SelectedFactor("oap_007", -1),  # long excess +1.8%, RankIC -0.0042, cov 0.960
    SelectedFactor("oap_010", 1),  # long excess +2.3%, RankIC +0.0035, cov 0.976
    SelectedFactor("oap_013", -1),  # long excess +3.3%, RankIC -0.0034, cov 0.658
    SelectedFactor("oap_014", -1),  # long excess +2.5%, RankIC -0.0038, cov 0.637
    SelectedFactor("oap_015", -1),  # long excess +3.9%, RankIC -0.0466, cov 0.999
    SelectedFactor("oap_021", -1),  # long excess -0.6%, RankIC -0.0125, cov 0.954
    SelectedFactor("oap_023", -1),  # long excess +0.5%, RankIC -0.0034, cov 0.896
    SelectedFactor("oap_025", 1),  # long excess +0.9%, RankIC +0.0033, cov 0.942
    SelectedFactor("oap_026", -1),  # long excess +1.0%, RankIC -0.0086, cov 0.989
    SelectedFactor("oap_027", -1),  # long excess -2.7%, RankIC -0.0011, cov 0.989
    SelectedFactor("oap_029", 1),  # long excess +2.9%, RankIC +0.0027, cov 0.942
    SelectedFactor("oap_030", 1),  # long excess -1.8%, RankIC +0.0142, cov 0.880
    SelectedFactor("oap_033", -1),  # long excess +1.0%, RankIC -0.0114, cov 0.688
    SelectedFactor("oap_040", -1),  # long excess -1.0%, RankIC -0.0125, cov 0.943
    SelectedFactor("oap_041", -1),  # long excess +5.2%, RankIC -0.0067, cov 0.943
    SelectedFactor("oap_043", -1),  # long excess +2.6%, RankIC -0.0166, cov 0.923
    SelectedFactor("oap_047", -1),  # long excess -0.3%, RankIC -0.0347, cov 0.999
    SelectedFactor("oap_057", 1),  # long excess +3.8%, RankIC +0.0222, cov 1.000
    SelectedFactor("oap_060", -1),  # long excess +2.8%, RankIC -0.0212, cov 0.917
    SelectedFactor("oap_068", -1),  # long excess +9.8%, RankIC -0.0207, cov 0.719
    SelectedFactor("oap_069", 1),  # long excess +11.1%, RankIC +0.0240, cov 0.986
    SelectedFactor("oap_070", -1),  # long excess +0.1%, RankIC -0.0006, cov 0.927
    SelectedFactor("oap_075", 1),  # long excess -2.1%, RankIC +0.0014, cov 0.940
    SelectedFactor("oap_080", -1),  # long excess +2.0%, RankIC -0.0000, cov 0.968
    SelectedFactor("oap_086", 1),  # long excess +6.4%, RankIC +0.0028, cov 0.959
    SelectedFactor("oap_088", 1),  # long excess +2.7%, RankIC +0.0005, cov 0.688
    SelectedFactor("oap_090", 1),  # long excess -1.0%, RankIC +0.0007, cov 0.928
    SelectedFactor("oap_092", 1),  # long excess +2.0%, RankIC +0.0026, cov 0.930
    SelectedFactor("oap_093", 1),  # long excess +1.4%, RankIC +0.0001, cov 0.996
    SelectedFactor("oap_098", -1),  # long excess +0.1%, RankIC -0.0021, cov 0.987
    SelectedFactor("oap_101", -1),  # long excess +2.4%, RankIC -0.0019, cov 0.943
    SelectedFactor("oap_109", -1),  # long excess +1.3%, RankIC -0.0006, cov 0.931
    SelectedFactor("oap_115", -1),  # long excess +2.9%, RankIC -0.0026, cov 0.918
    SelectedFactor("oap_119", -1),  # long excess +1.1%, RankIC -0.0022, cov 0.940
    SelectedFactor("oap_126", -1),  # long excess +0.1%, RankIC -0.0021, cov 0.984
    SelectedFactor("oap_128", -1),  # long excess -0.3%, RankIC -0.0007, cov 0.992
    SelectedFactor("oap_131", 1),  # long excess +6.0%, RankIC +0.0030, cov 0.817
    SelectedFactor("oap_132", -1),  # long excess +0.7%, RankIC -0.0001, cov 0.997
    SelectedFactor("oap_136", -1),  # long excess -1.9%, RankIC -0.0028, cov 0.999
    SelectedFactor("oap_137", -1),  # long excess -1.5%, RankIC -0.0006, cov 1.000
    SelectedFactor("oap_138", -1),  # long excess -1.4%, RankIC -0.0010, cov 0.990
    SelectedFactor("oap_139", -1),  # long excess -2.1%, RankIC -0.0002, cov 0.976
    SelectedFactor("oap_140", 1),  # long excess +4.6%, RankIC +0.0012, cov 0.942
    SelectedFactor("oap_142", 1),  # long excess -1.1%, RankIC +0.0007, cov 0.732
    SelectedFactor("cne6_002", -1),  # long excess +15.0%, RankIC -0.0139, cov 1.000
    SelectedFactor("cne6_006", -1),  # long excess -1.7%, RankIC -0.0146, cov 0.981
    SelectedFactor("cne6_024", -1),  # long excess +1.1%, RankIC -0.0080, cov 0.998
    SelectedFactor("cne6_025", -1),  # long excess -1.2%, RankIC -0.0101, cov 0.998
    SelectedFactor("cne6_026", -1),  # long excess +0.9%, RankIC -0.0020, cov 0.998
    SelectedFactor("cne6_034", 1),  # long excess +1.5%, RankIC +0.0048, cov 0.998
    SelectedFactor("cne6_035", 1),  # long excess -0.3%, RankIC +0.0026, cov 0.998
    SelectedFactor("tst_030", -1),  # long excess -0.7%, RankIC -0.0201, cov 1.000
    SelectedFactor("tst_050", -1),  # long excess +1.0%, RankIC -0.0202, cov 1.000
    SelectedFactor("tst_072", -1),  # long excess +2.0%, RankIC -0.0230, cov 1.000
    SelectedFactor("tst_074", -1),  # long excess +0.8%, RankIC -0.0185, cov 1.000
    SelectedFactor("tst_075", 1),  # long excess +2.3%, RankIC +0.0158, cov 1.000
    SelectedFactor("jkp_005", 1),  # long excess -0.0%, RankIC +0.0008, cov 0.944
    SelectedFactor("jkp_018", -1),  # long excess -0.1%, RankIC -0.0026, cov 0.906
    SelectedFactor("jkp_030", -1),  # long excess -1.3%, RankIC -0.0001, cov 0.944
    SelectedFactor("jkp_031", -1),  # long excess +4.6%, RankIC -0.0046, cov 0.943
    SelectedFactor("jkp_036", -1),  # long excess +1.2%, RankIC -0.0042, cov 0.976
    SelectedFactor("jkp_042", -1),  # long excess +2.1%, RankIC -0.0008, cov 0.984
    SelectedFactor("jkp_045", 1),  # long excess +3.4%, RankIC +0.0016, cov 0.943
    SelectedFactor("jkp_048", -1),  # long excess +1.2%, RankIC -0.0012, cov 0.943
    SelectedFactor("jkp_050", 1),  # long excess +0.3%, RankIC +0.0005, cov 0.995
    SelectedFactor("jkp_051", 1),  # long excess -1.6%, RankIC +0.0020, cov 0.987
    SelectedFactor("jkp_053", -1),  # long excess +2.1%, RankIC -0.0021, cov 0.984
    SelectedFactor("jkp_057", -1),  # long excess -0.6%, RankIC -0.0046, cov 0.992
    SelectedFactor("jkp_060", 1),  # long excess +5.7%, RankIC +0.0055, cov 0.985
    SelectedFactor("jkp_075", 1),  # long excess +3.0%, RankIC +0.0053, cov 0.914
    SelectedFactor("jkp_087", 1),  # long excess +1.6%, RankIC +0.0017, cov 0.903
    SelectedFactor("jkp_092", -1),  # long excess +1.5%, RankIC -0.0036, cov 0.976
    SelectedFactor("jkp_094", 1),  # long excess -1.7%, RankIC +0.0009, cov 0.794
    SelectedFactor("jkp_097", -1),  # long excess +0.4%, RankIC -0.0025, cov 0.944
    SelectedFactor("jkp_101", -1),  # long excess +2.5%, RankIC -0.0021, cov 0.773
    SelectedFactor("jkp_112", -1),  # long excess -8.9%, RankIC -0.0027, cov 0.999
    SelectedFactor("jkp_121", 1),  # long excess -0.6%, RankIC +0.0140, cov 0.976
    SelectedFactor("jkp_124", -1),  # long excess +4.7%, RankIC -0.0186, cov 0.999
    SelectedFactor("jkp_126", -1),  # long excess -0.1%, RankIC -0.0019, cov 0.999
    SelectedFactor("jkp_133", 1),  # long excess +4.3%, RankIC +0.0129, cov 0.896
    SelectedFactor("jkp_138", 1),  # long excess +2.2%, RankIC +0.0023, cov 0.975
    SelectedFactor("jkp_143", 1),  # long excess +0.6%, RankIC +0.0047, cov 0.795
)

DEDUPLICATED = {
    "a101_002": "gtja_001",
    "a101_003": "gtja_105",
    "a101_005": "a101_083",
    "a101_006": "a101_040",
    "a101_007": "a158_145",
    "a101_009": "a158_039",
    "a101_010": "a158_039",
    "a101_011": "gtja_007",
    "a101_012": "gtja_113",
    "a101_013": "a101_044",
    "a101_014": "a101_040",
    "a101_016": "a101_015",
    "a101_017": "tst_072",
    "a101_018": "a158_035",
    "a101_020": "gtja_015",
    "a101_021": "gtja_004",
    "a101_025": "tst_072",
    "a101_028": "gtja_191",
    "a101_029": "gtja_157",
    "a101_030": "tst_072",
    "a101_033": "a101_083",
    "a101_035": "gtja_117",
    "a101_038": "a101_004",
    "a101_039": "a101_019",
    "a101_042": "gtja_120",
    "a101_043": "gtja_085",
    "a101_045": "gtja_113",
    "a101_049": "a158_039",
    "a101_050": "a101_027",
    "a101_051": "a158_039",
    "a101_052": "a158_080",
    "a101_054": "gtja_002",
    "a101_055": "gtja_176",
    "a101_057": "a101_083",
    "a101_058": "a101_059",
    "a101_060": "a101_083",
    "a101_067": "tst_074",
    "a101_070": "gtja_131",
    "a101_073": "gtja_156",
    "a101_084": "a101_004",
    "a101_085": "a101_072",
    "a101_087": "gtja_092",
    "a101_094": "a101_004",
    "a101_101": "ta_103",
    "a158_001": "a101_083",
    "a158_002": "gtja_185",
    "a158_003": "ta_103",
    "a158_004": "a158_005",
    "a158_007": "a158_006",
    "a158_008": "a101_083",
    "a158_009": "gtja_002",
    "a158_010": "a101_083",
    "a158_011": "a101_047",
    "a158_012": "a101_083",
    "a158_013": "gtja_120",
    "a158_014": "a101_004",
    "a158_015": "a101_004",
    "a158_016": "ta_015",
    "a158_017": "a360_071",
    "a158_018": "a360_071",
    "a158_019": "a101_090",
    "a158_020": "a101_004",
    "a158_021": "ta_015",
    "a158_022": "ta_015",
    "a158_023": "a360_071",
    "a158_025": "oap_015",
    "a158_026": "oap_015",
    "a158_028": "a158_027",
    "a158_029": "a101_004",
    "a158_030": "a101_004",
    "a158_031": "tst_030",
    "a158_032": "a360_071",
    "a158_033": "a360_071",
    "a158_040": "a101_090",
    "a158_041": "a101_004",
    "a158_042": "a101_004",
    "a158_043": "a101_004",
    "a158_044": "a101_090",
    "a158_045": "a101_090",
    "a158_046": "a158_027",
    "a158_047": "a158_027",
    "a158_048": "a360_071",
    "a158_049": "a101_088",
    "a158_050": "a101_088",
    "a158_051": "ta_015",
    "a158_052": "ta_015",
    "a158_053": "a360_071",
    "a158_054": "a101_090",
    "a158_055": "a101_004",
    "a158_056": "a101_004",
    "a158_057": "ta_015",
    "a158_058": "a360_071",
    "a158_059": "a101_004",
    "a158_060": "a101_004",
    "a158_061": "ta_015",
    "a158_062": "ta_015",
    "a158_063": "a360_071",
    "a158_064": "tst_072",
    "a158_065": "a101_004",
    "a158_066": "ta_015",
    "a158_067": "ta_015",
    "a158_068": "a360_071",
    "a158_069": "a101_090",
    "a158_070": "a101_004",
    "a158_071": "a101_004",
    "a158_072": "a158_077",
    "a158_073": "a360_071",
    "a158_075": "ta_004",
    "a158_076": "ta_004",
    "a158_078": "a360_071",
    "a158_081": "a158_082",
    "a158_084": "a158_079",
    "a158_085": "a158_080",
    "a158_086": "ta_004",
    "a158_087": "a158_077",
    "a158_088": "a360_071",
    "a158_089": "a101_044",
    "a158_090": "a101_040",
    "a158_091": "ta_059",
    "a158_092": "ta_059",
    "a158_094": "a158_095",
    "a158_096": "a158_097",
    "a158_098": "a158_097",
    "a158_099": "a158_104",
    "a158_100": "a158_104",
    "a158_101": "a158_107",
    "a158_102": "a158_107",
    "a158_105": "a158_104",
    "a158_106": "a158_107",
    "a158_108": "a158_107",
    "a158_109": "a158_104",
    "a158_110": "a158_104",
    "a158_111": "a158_107",
    "a158_112": "a158_107",
    "a158_113": "a158_103",
    "a158_114": "a158_104",
    "a158_115": "ta_015",
    "a158_116": "ta_015",
    "a158_117": "a360_071",
    "a158_118": "a360_071",
    "a158_119": "a158_104",
    "a158_120": "ta_015",
    "a158_121": "ta_015",
    "a158_122": "a360_071",
    "a158_123": "a360_071",
    "a158_124": "a158_104",
    "a158_125": "ta_015",
    "a158_126": "ta_015",
    "a158_127": "a360_071",
    "a158_128": "a360_071",
    "a158_129": "a360_358",
    "a158_130": "a158_144",
    "a158_131": "a158_145",
    "a158_132": "a158_146",
    "a158_133": "a360_313",
    "a158_135": "a158_134",
    "a158_136": "jq_114",
    "a158_137": "a158_146",
    "a158_138": "a360_313",
    "a158_139": "a158_140",
    "a158_141": "jq_114",
    "a158_142": "jq_114",
    "a158_147": "a158_146",
    "a158_148": "a360_313",
    "a158_149": "a158_144",
    "a158_150": "a158_145",
    "a158_151": "a158_146",
    "a158_152": "a158_146",
    "a158_153": "a360_313",
    "a158_154": "a158_144",
    "a158_155": "a158_145",
    "a158_156": "a158_146",
    "a158_157": "a158_146",
    "a158_158": "a360_313",
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
    "a360_021": "a360_071",
    "a360_022": "a360_071",
    "a360_023": "a360_071",
    "a360_024": "a360_071",
    "a360_025": "a360_071",
    "a360_026": "a360_071",
    "a360_027": "a360_071",
    "a360_028": "a360_071",
    "a360_029": "a360_071",
    "a360_030": "a360_071",
    "a360_031": "a360_071",
    "a360_032": "a360_071",
    "a360_033": "a360_071",
    "a360_034": "a360_071",
    "a360_035": "ta_015",
    "a360_036": "ta_015",
    "a360_037": "ta_015",
    "a360_038": "ta_015",
    "a360_039": "ta_015",
    "a360_040": "ta_015",
    "a360_041": "tst_030",
    "a360_042": "tst_030",
    "a360_043": "ta_015",
    "a360_044": "ta_015",
    "a360_045": "ta_015",
    "a360_046": "ta_015",
    "a360_047": "ta_015",
    "a360_048": "ta_015",
    "a360_049": "ta_015",
    "a360_050": "a101_004",
    "a360_051": "a101_004",
    "a360_052": "a101_004",
    "a360_053": "a101_004",
    "a360_054": "a101_004",
    "a360_055": "a101_004",
    "a360_056": "a101_004",
    "a360_057": "a101_004",
    "a360_058": "tst_074",
    "a360_059": "a158_039",
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
    "a360_082": "a360_071",
    "a360_083": "a360_071",
    "a360_084": "a360_071",
    "a360_085": "a360_071",
    "a360_086": "a360_071",
    "a360_087": "a360_071",
    "a360_088": "a360_071",
    "a360_089": "a360_071",
    "a360_090": "a360_071",
    "a360_091": "a360_071",
    "a360_092": "a360_071",
    "a360_093": "a360_071",
    "a360_094": "a360_071",
    "a360_095": "a360_071",
    "a360_096": "ta_015",
    "a360_097": "ta_015",
    "a360_098": "ta_015",
    "a360_099": "ta_015",
    "a360_100": "ta_015",
    "a360_101": "ta_015",
    "a360_102": "tst_030",
    "a360_103": "tst_030",
    "a360_104": "ta_015",
    "a360_105": "ta_015",
    "a360_106": "ta_015",
    "a360_107": "ta_015",
    "a360_108": "ta_015",
    "a360_109": "ta_015",
    "a360_110": "ta_015",
    "a360_111": "a101_004",
    "a360_112": "a101_004",
    "a360_113": "a101_004",
    "a360_114": "a101_004",
    "a360_115": "a101_004",
    "a360_116": "a101_004",
    "a360_117": "a101_004",
    "a360_118": "a101_004",
    "a360_119": "tst_074",
    "a360_120": "a101_083",
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
    "a360_141": "a360_071",
    "a360_142": "a360_071",
    "a360_143": "a360_071",
    "a360_144": "a360_071",
    "a360_145": "a360_071",
    "a360_146": "a360_071",
    "a360_147": "a360_071",
    "a360_148": "a360_071",
    "a360_149": "a360_071",
    "a360_150": "a360_071",
    "a360_151": "a360_071",
    "a360_152": "a360_071",
    "a360_153": "a360_071",
    "a360_154": "a360_071",
    "a360_155": "ta_015",
    "a360_156": "ta_015",
    "a360_157": "ta_015",
    "a360_158": "ta_015",
    "a360_159": "ta_015",
    "a360_160": "ta_015",
    "a360_161": "ta_015",
    "a360_162": "tst_030",
    "a360_163": "ta_015",
    "a360_164": "ta_015",
    "a360_165": "ta_015",
    "a360_166": "ta_015",
    "a360_167": "ta_015",
    "a360_168": "ta_015",
    "a360_169": "ta_015",
    "a360_170": "a101_004",
    "a360_171": "a101_004",
    "a360_172": "a101_004",
    "a360_173": "a101_004",
    "a360_174": "a101_004",
    "a360_175": "a101_004",
    "a360_176": "a101_004",
    "a360_177": "a101_090",
    "a360_178": "a101_090",
    "a360_179": "a158_039",
    "a360_180": "a101_047",
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
    "a360_203": "a360_071",
    "a360_204": "a360_071",
    "a360_205": "a360_071",
    "a360_206": "a360_071",
    "a360_207": "a360_071",
    "a360_208": "a360_071",
    "a360_209": "a360_071",
    "a360_210": "a360_071",
    "a360_211": "a360_071",
    "a360_212": "a360_071",
    "a360_213": "a360_071",
    "a360_214": "a360_071",
    "a360_215": "a360_071",
    "a360_216": "ta_015",
    "a360_217": "ta_015",
    "a360_218": "ta_015",
    "a360_219": "ta_015",
    "a360_220": "ta_015",
    "a360_221": "tst_030",
    "a360_222": "tst_030",
    "a360_223": "tst_030",
    "a360_224": "ta_015",
    "a360_225": "ta_015",
    "a360_226": "ta_015",
    "a360_227": "ta_015",
    "a360_228": "ta_015",
    "a360_229": "ta_015",
    "a360_230": "ta_015",
    "a360_231": "a101_004",
    "a360_232": "a101_004",
    "a360_233": "a101_004",
    "a360_234": "a101_004",
    "a360_235": "a101_004",
    "a360_236": "a101_004",
    "a360_237": "a101_004",
    "a360_238": "a101_004",
    "a360_239": "tst_075",
    "a360_240": "a101_083",
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
    "a360_262": "a360_071",
    "a360_263": "a360_071",
    "a360_264": "a360_071",
    "a360_265": "a360_071",
    "a360_266": "a360_071",
    "a360_267": "a360_071",
    "a360_268": "a360_071",
    "a360_269": "a360_071",
    "a360_270": "a360_071",
    "a360_271": "a360_071",
    "a360_272": "a360_071",
    "a360_273": "a360_071",
    "a360_274": "a360_071",
    "a360_275": "ta_015",
    "a360_276": "ta_015",
    "a360_277": "ta_015",
    "a360_278": "ta_015",
    "a360_279": "ta_015",
    "a360_280": "ta_015",
    "a360_281": "tst_030",
    "a360_282": "tst_030",
    "a360_283": "ta_015",
    "a360_284": "ta_015",
    "a360_285": "ta_015",
    "a360_286": "ta_015",
    "a360_287": "ta_015",
    "a360_288": "ta_015",
    "a360_289": "ta_015",
    "a360_290": "a101_004",
    "a360_291": "a101_004",
    "a360_292": "a101_004",
    "a360_293": "a101_004",
    "a360_294": "a101_004",
    "a360_295": "a101_004",
    "a360_296": "a101_004",
    "a360_297": "a101_004",
    "a360_298": "tst_074",
    "a360_299": "a158_039",
    "a360_300": "gtja_120",
    "a360_301": "a360_313",
    "a360_302": "a360_313",
    "a360_303": "a360_313",
    "a360_304": "a360_313",
    "a360_305": "a360_313",
    "a360_306": "a360_313",
    "a360_307": "a360_313",
    "a360_308": "a360_313",
    "a360_309": "a360_313",
    "a360_310": "a360_313",
    "a360_311": "a360_313",
    "a360_312": "a360_313",
    "a360_314": "a360_313",
    "a360_315": "a360_313",
    "a360_316": "a360_313",
    "a360_317": "a360_313",
    "a360_318": "a360_313",
    "a360_319": "a360_313",
    "a360_320": "a360_313",
    "a360_321": "a360_313",
    "a360_322": "a360_313",
    "a360_323": "a360_313",
    "a360_324": "a360_313",
    "a360_325": "a360_313",
    "a360_326": "a360_313",
    "a360_327": "a360_313",
    "a360_328": "a360_313",
    "a360_329": "a360_313",
    "a360_330": "a360_313",
    "a360_331": "a158_146",
    "a360_332": "a158_146",
    "a360_333": "a158_146",
    "a360_334": "a158_146",
    "a360_335": "a158_146",
    "a360_336": "a158_146",
    "a360_337": "a158_146",
    "a360_338": "a158_146",
    "a360_339": "a158_146",
    "a360_340": "a158_146",
    "a360_341": "a158_146",
    "a360_342": "a158_146",
    "a360_343": "a158_146",
    "a360_344": "a158_146",
    "a360_345": "a158_146",
    "a360_346": "a158_146",
    "a360_347": "a158_145",
    "a360_348": "a158_145",
    "a360_349": "a158_145",
    "a360_350": "a158_145",
    "a360_351": "a158_145",
    "a360_352": "a158_145",
    "a360_353": "a158_145",
    "a360_354": "a158_144",
    "a360_355": "a158_144",
    "a360_356": "a158_144",
    "a360_357": "gtja_007",
    "a360_359": "a360_358",
    "cnc_001": "cne6_002",
    "cnc_002": "cne6_002",
    "cnc_003": "jq_230",
    "cnc_004": "jq_230",
    "cnc_005": "jq_230",
    "cnc_006": "jq_230",
    "cnc_007": "cnc_013",
    "cnc_009": "jq_116",
    "cnc_010": "jq_116",
    "cnc_011": "jq_116",
    "cnc_012": "jq_116",
    "cnc_014": "jq_116",
    "cnc_015": "svc_013",
    "cnc_016": "svc_013",
    "cnc_017": "jq_230",
    "cnc_018": "jq_154",
    "cnc_019": "jq_154",
    "cnc_020": "jq_154",
    "cnc_022": "gtja_149",
    "cnc_023": "cnc_024",
    "cnc_025": "cnc_024",
    "cnc_026": "ta_015",
    "cnc_027": "oap_029",
    "cnc_028": "oap_068",
    "cnc_029": "cnc_033",
    "cnc_030": "a360_071",
    "cnc_034": "oap_029",
    "cnc_035": "cnc_033",
    "cnc_036": "a360_071",
    "cnc_037": "oap_015",
    "cnc_039": "oap_136",
    "cnc_040": "jkp_101",
    "cnc_041": "cnc_052",
    "cnc_042": "jkp_060",
    "cnc_043": "cnc_052",
    "cnc_044": "jkp_060",
    "cnc_045": "cnc_052",
    "cnc_046": "cnc_052",
    "cnc_048": "cnc_047",
    "cnc_049": "cnc_052",
    "cnc_050": "cnc_052",
    "cnc_051": "cnc_052",
    "cnc_053": "cnc_052",
    "cnc_054": "cnc_052",
    "cnc_056": "jq_032",
    "cnc_057": "jkp_018",
    "cnc_058": "oap_086",
    "cnc_059": "jkp_060",
    "cnc_060": "jq_032",
    "cnc_061": "oap_057",
    "cnc_062": "cnc_068",
    "cnc_063": "cnc_068",
    "cnc_064": "oap_060",
    "cnc_065": "oap_060",
    "cnc_066": "jkp_018",
    "cnc_067": "oap_060",
    "cnc_069": "jq_002",
    "cnc_070": "cnc_072",
    "cnc_071": "cnc_038",
    "cnc_073": "jkp_097",
    "cnc_074": "oap_098",
    "cnc_075": "jkp_030",
    "cnc_076": "jkp_031",
    "cnc_077": "oap_119",
    "cnc_080": "cnc_081",
    "cnc_082": "jq_002",
    "cnc_085": "cnc_084",
    "cnc_086": "jkp_045",
    "cnc_088": "cnc_087",
    "cnc_089": "jkp_048",
    "cnc_091": "jkp_053",
    "cnc_093": "cnc_092",
    "cnc_094": "cnc_068",
    "cnc_095": "gtja_143",
    "cnc_096": "cnc_081",
    "cnc_098": "jq_008",
    "cnc_099": "jq_046",
    "cnc_102": "jkp_092",
    "cnc_103": "cnc_072",
    "cnc_105": "jkp_097",
    "cnc_106": "cnc_038",
    "cnc_107": "jkp_060",
    "cnc_108": "oap_086",
    "cnc_109": "jkp_060",
    "cne6_001": "jq_116",
    "cne6_003": "cnc_024",
    "cne6_004": "jq_230",
    "cne6_005": "a158_027",
    "cne6_007": "jq_230",
    "cne6_008": "jq_230",
    "cne6_009": "jq_230",
    "cne6_010": "jq_230",
    "cne6_011": "tst_030",
    "cne6_012": "jkp_101",
    "cne6_013": "a360_071",
    "cne6_014": "oap_029",
    "cne6_015": "cnc_032",
    "cne6_016": "cnc_032",
    "cne6_017": "oap_057",
    "cne6_018": "oap_060",
    "cne6_019": "jq_002",
    "cne6_020": "oap_060",
    "cne6_021": "cnc_072",
    "cne6_022": "cnc_072",
    "cne6_023": "jq_032",
    "cne6_027": "jkp_031",
    "cne6_028": "jkp_031",
    "cne6_029": "cnc_047",
    "cne6_030": "cnc_052",
    "cne6_031": "cnc_068",
    "cne6_032": "jkp_053",
    "cne6_033": "cnc_079",
    "cne6_036": "jkp_143",
    "cne6_037": "jkp_018",
    "cne6_038": "cnc_078",
    "gtja_003": "a158_104",
    "gtja_005": "a101_026",
    "gtja_006": "a158_079",
    "gtja_008": "a101_004",
    "gtja_009": "gtja_068",
    "gtja_010": "gtja_120",
    "gtja_013": "a101_041",
    "gtja_014": "a101_004",
    "gtja_016": "a101_027",
    "gtja_017": "a101_004",
    "gtja_018": "a101_004",
    "gtja_019": "a101_004",
    "gtja_020": "a101_004",
    "gtja_021": "gtja_157",
    "gtja_022": "a101_004",
    "gtja_023": "a158_103",
    "gtja_024": "gtja_068",
    "gtja_025": "a101_019",
    "gtja_026": "a101_032",
    "gtja_027": "ta_015",
    "gtja_028": "a101_004",
    "gtja_029": "gtja_011",
    "gtja_031": "a101_004",
    "gtja_032": "a101_015",
    "gtja_033": "a158_080",
    "gtja_034": "a101_004",
    "gtja_035": "a101_082",
    "gtja_036": "a101_027",
    "gtja_037": "a101_008",
    "gtja_038": "a101_023",
    "gtja_039": "a101_063",
    "gtja_040": "a158_103",
    "gtja_041": "gtja_157",
    "gtja_042": "a101_040",
    "gtja_043": "a158_104",
    "gtja_046": "a101_004",
    "gtja_047": "ta_045",
    "gtja_048": "a158_104",
    "gtja_049": "ta_004",
    "gtja_050": "ta_004",
    "gtja_051": "ta_004",
    "gtja_052": "ta_015",
    "gtja_053": "a158_104",
    "gtja_054": "a158_035",
    "gtja_055": "tst_030",
    "gtja_057": "a101_004",
    "gtja_058": "a158_107",
    "gtja_059": "ta_045",
    "gtja_060": "ta_045",
    "gtja_062": "a101_044",
    "gtja_063": "a101_004",
    "gtja_065": "a101_004",
    "gtja_066": "a101_004",
    "gtja_067": "a360_071",
    "gtja_069": "ta_015",
    "gtja_070": "jq_116",
    "gtja_071": "ta_015",
    "gtja_072": "ta_045",
    "gtja_073": "a101_091",
    "gtja_074": "a101_027",
    "gtja_077": "a101_077",
    "gtja_078": "a101_004",
    "gtja_079": "ta_015",
    "gtja_080": "a158_144",
    "gtja_081": "jq_116",
    "gtja_082": "ta_045",
    "gtja_083": "a101_015",
    "gtja_084": "jq_111",
    "gtja_086": "a101_046",
    "gtja_087": "a101_066",
    "gtja_088": "ta_015",
    "gtja_089": "gtja_068",
    "gtja_090": "a101_015",
    "gtja_091": "a101_090",
    "gtja_093": "gtja_120",
    "gtja_094": "jq_111",
    "gtja_095": "jq_116",
    "gtja_096": "ta_004",
    "gtja_097": "jq_116",
    "gtja_098": "a101_024",
    "gtja_099": "a101_044",
    "gtja_100": "jq_116",
    "gtja_101": "a101_074",
    "gtja_102": "a158_144",
    "gtja_103": "a158_082",
    "gtja_104": "a101_022",
    "gtja_106": "tst_030",
    "gtja_107": "gtja_015",
    "gtja_108": "tst_074",
    "gtja_109": "a158_145",
    "gtja_110": "ta_015",
    "gtja_111": "a101_083",
    "gtja_112": "ta_015",
    "gtja_114": "a101_083",
    "gtja_115": "a101_072",
    "gtja_116": "tst_030",
    "gtja_118": "ta_015",
    "gtja_121": "a101_004",
    "gtja_123": "a101_099",
    "gtja_124": "a101_083",
    "gtja_126": "gtja_120",
    "gtja_127": "tst_030",
    "gtja_128": "ta_004",
    "gtja_129": "gtja_120",
    "gtja_130": "a101_027",
    "gtja_132": "jq_116",
    "gtja_133": "ta_004",
    "gtja_134": "ta_015",
    "gtja_135": "a360_071",
    "gtja_136": "a101_040",
    "gtja_137": "a101_034",
    "gtja_138": "a101_097",
    "gtja_139": "a101_040",
    "gtja_140": "a101_088",
    "gtja_142": "tst_072",
    "gtja_144": "jq_116",
    "gtja_145": "a158_146",
    "gtja_147": "tst_030",
    "gtja_148": "a101_065",
    "gtja_150": "jq_116",
    "gtja_151": "a360_071",
    "gtja_152": "tst_050",
    "gtja_153": "gtja_120",
    "gtja_154": "a101_061",
    "gtja_155": "a158_145",
    "gtja_158": "gtja_185",
    "gtja_159": "oap_015",
    "gtja_160": "gtja_120",
    "gtja_161": "gtja_120",
    "gtja_162": "a101_090",
    "gtja_163": "tst_072",
    "gtja_164": "gtja_120",
    "gtja_166": "gtja_190",
    "gtja_167": "gtja_120",
    "gtja_168": "a158_145",
    "gtja_169": "tst_030",
    "gtja_170": "a101_047",
    "gtja_171": "gtja_002",
    "gtja_172": "ta_008",
    "gtja_173": "gtja_120",
    "gtja_174": "gtja_120",
    "gtja_175": "gtja_120",
    "gtja_177": "ta_004",
    "gtja_178": "tst_072",
    "gtja_184": "a101_037",
    "gtja_186": "a158_037",
    "gtja_187": "gtja_120",
    "gtja_188": "gtja_185",
    "gtja_189": "gtja_120",
    "jkp_001": "jkp_060",
    "jkp_002": "cnc_033",
    "jkp_003": "oap_029",
    "jkp_004": "oap_086",
    "jkp_006": "jkp_060",
    "jkp_007": "a360_071",
    "jkp_008": "cnc_033",
    "jkp_009": "oap_029",
    "jkp_010": "oap_057",
    "jkp_011": "cnc_072",
    "jkp_012": "cnc_068",
    "jkp_013": "oap_068",
    "jkp_014": "oap_060",
    "jkp_015": "jq_002",
    "jkp_016": "cnc_078",
    "jkp_017": "cnc_078",
    "jkp_019": "oap_092",
    "jkp_020": "oap_060",
    "jkp_021": "cnc_068",
    "jkp_022": "jq_002",
    "jkp_023": "oap_060",
    "jkp_024": "cnc_038",
    "jkp_025": "cnc_072",
    "jkp_026": "oap_060",
    "jkp_027": "oap_119",
    "jkp_028": "jkp_097",
    "jkp_029": "jkp_031",
    "jkp_032": "jkp_031",
    "jkp_033": "oap_119",
    "jkp_034": "oap_119",
    "jkp_035": "oap_119",
    "jkp_037": "cnc_078",
    "jkp_038": "cnc_079",
    "jkp_039": "cnc_081",
    "jkp_040": "oap_098",
    "jkp_041": "jq_002",
    "jkp_043": "oap_101",
    "jkp_044": "jkp_031",
    "jkp_046": "jkp_031",
    "jkp_047": "jkp_031",
    "jkp_049": "jkp_042",
    "jkp_052": "jkp_031",
    "jkp_054": "jq_002",
    "jkp_055": "jkp_042",
    "jkp_056": "jq_139",
    "jkp_058": "oap_126",
    "jkp_059": "cnc_052",
    "jkp_061": "cnc_052",
    "jkp_062": "jkp_060",
    "jkp_063": "cnc_052",
    "jkp_064": "cnc_052",
    "jkp_065": "cnc_047",
    "jkp_066": "cnc_047",
    "jkp_067": "cnc_052",
    "jkp_068": "cnc_052",
    "jkp_069": "cnc_052",
    "jkp_070": "cnc_052",
    "jkp_071": "cnc_052",
    "jkp_072": "cnc_052",
    "jkp_073": "jq_002",
    "jkp_074": "jq_002",
    "jkp_076": "cnc_072",
    "jkp_077": "cnc_072",
    "jkp_078": "oap_080",
    "jkp_079": "jq_032",
    "jkp_080": "oap_086",
    "jkp_081": "cnc_092",
    "jkp_082": "cnc_092",
    "jkp_083": "cnc_047",
    "jkp_084": "cnc_092",
    "jkp_085": "gtja_143",
    "jkp_086": "oap_090",
    "jkp_088": "jq_008",
    "jkp_089": "jq_046",
    "jkp_090": "cnc_101",
    "jkp_091": "jq_029",
    "jkp_093": "cnc_072",
    "jkp_095": "cnc_104",
    "jkp_096": "cnc_104",
    "jkp_098": "cnc_068",
    "jkp_099": "oap_010",
    "jkp_100": "oap_029",
    "jkp_102": "oap_068",
    "jkp_103": "oap_013",
    "jkp_104": "oap_014",
    "jkp_105": "jq_116",
    "jkp_106": "cnc_008",
    "jkp_107": "jq_230",
    "jkp_108": "cnc_008",
    "jkp_109": "oap_015",
    "jkp_110": "oap_021",
    "jkp_111": "jq_230",
    "jkp_113": "jq_230",
    "jkp_114": "cnc_013",
    "jkp_115": "jq_116",
    "jkp_116": "cnc_013",
    "jkp_117": "cnc_038",
    "jkp_118": "jq_116",
    "jkp_119": "jq_230",
    "jkp_120": "jq_230",
    "jkp_122": "oap_015",
    "jkp_123": "jkp_124",
    "jkp_125": "jkp_124",
    "jkp_127": "ta_015",
    "jkp_128": "gtja_149",
    "jkp_129": "oap_047",
    "jkp_130": "a360_071",
    "jkp_131": "oap_029",
    "jkp_132": "cnc_033",
    "jkp_134": "oap_015",
    "jkp_135": "ta_015",
    "jkp_136": "cnc_052",
    "jkp_137": "jq_002",
    "jkp_139": "cnc_052",
    "jkp_140": "jkp_031",
    "jkp_141": "jkp_143",
    "jkp_142": "jq_002",
    "jkp_144": "cnc_072",
    "jq_001": "jkp_053",
    "jq_004": "jkp_053",
    "jq_006": "jq_029",
    "jq_007": "jq_053",
    "jq_009": "jq_098",
    "jq_011": "jq_020",
    "jq_012": "jq_020",
    "jq_014": "jq_002",
    "jq_015": "jq_032",
    "jq_017": "cnc_072",
    "jq_018": "jq_032",
    "jq_019": "jq_032",
    "jq_022": "jq_032",
    "jq_023": "cnc_072",
    "jq_024": "cnc_068",
    "jq_025": "jq_029",
    "jq_026": "jkp_053",
    "jq_027": "jkp_053",
    "jq_028": "cnc_068",
    "jq_030": "jkp_053",
    "jq_031": "cnc_068",
    "jq_033": "cnc_072",
    "jq_034": "cnc_072",
    "jq_035": "jq_032",
    "jq_037": "cnc_068",
    "jq_039": "jkp_053",
    "jq_040": "cne6_025",
    "jq_041": "jkp_053",
    "jq_042": "cne6_025",
    "jq_043": "jq_002",
    "jq_044": "jq_008",
    "jq_047": "oap_092",
    "jq_048": "jkp_087",
    "jq_051": "jq_029",
    "jq_052": "jq_003",
    "jq_054": "jkp_092",
    "jq_055": "cnc_047",
    "jq_056": "jq_002",
    "jq_058": "jq_002",
    "jq_059": "jq_002",
    "jq_062": "jkp_053",
    "jq_063": "jq_002",
    "jq_064": "cnc_072",
    "jq_065": "jq_032",
    "jq_066": "jkp_053",
    "jq_067": "cne6_002",
    "jq_069": "jq_002",
    "jq_070": "cne6_002",
    "jq_071": "oap_060",
    "jq_072": "oap_060",
    "jq_073": "jq_161",
    "jq_074": "cnc_072",
    "jq_075": "jq_032",
    "jq_076": "cnc_068",
    "jq_077": "jq_098",
    "jq_078": "cnc_072",
    "jq_079": "cnc_068",
    "jq_080": "jq_116",
    "jq_081": "cnc_072",
    "jq_082": "oap_126",
    "jq_083": "cnc_072",
    "jq_085": "jq_002",
    "jq_086": "oap_060",
    "jq_087": "jq_021",
    "jq_088": "jq_005",
    "jq_089": "oap_060",
    "jq_090": "cnc_052",
    "jq_091": "cne6_002",
    "jq_092": "cnc_068",
    "jq_093": "jq_032",
    "jq_094": "oap_060",
    "jq_095": "cnc_068",
    "jq_096": "oap_060",
    "jq_097": "cnc_068",
    "jq_099": "cnc_068",
    "jq_100": "cnc_068",
    "jq_101": "oap_060",
    "jq_103": "ta_015",
    "jq_105": "oap_015",
    "jq_106": "a158_024",
    "jq_107": "ta_015",
    "jq_108": "cnc_008",
    "jq_109": "cnc_008",
    "jq_110": "cnc_008",
    "jq_112": "jq_116",
    "jq_113": "a158_104",
    "jq_115": "jq_116",
    "jq_117": "jq_116",
    "jq_118": "jq_116",
    "jq_119": "tst_050",
    "jq_120": "a158_146",
    "jq_121": "jq_116",
    "jq_122": "jq_116",
    "jq_123": "jq_116",
    "jq_124": "jq_116",
    "jq_125": "a158_145",
    "jq_126": "jq_230",
    "jq_127": "jq_230",
    "jq_128": "jq_230",
    "jq_129": "jq_230",
    "jq_130": "jq_230",
    "jq_131": "jq_230",
    "jq_132": "a158_146",
    "jq_133": "a158_107",
    "jq_134": "a158_145",
    "jq_135": "a158_144",
    "jq_136": "jq_116",
    "jq_137": "jq_116",
    "jq_138": "jq_111",
    "jq_140": "jkp_053",
    "jq_141": "jkp_138",
    "jq_142": "jkp_075",
    "jq_143": "jkp_075",
    "jq_144": "oap_092",
    "jq_146": "jkp_097",
    "jq_147": "jkp_075",
    "jq_150": "a158_143",
    "jq_151": "a360_071",
    "jq_152": "ta_015",
    "jq_153": "a360_071",
    "jq_155": "jkp_124",
    "jq_156": "jq_154",
    "jq_157": "jq_230",
    "jq_158": "oap_015",
    "jq_159": "a158_027",
    "jq_163": "cnc_052",
    "jq_164": "jq_160",
    "jq_165": "jq_002",
    "jq_166": "cnc_052",
    "jq_167": "cnc_052",
    "jq_168": "cnc_068",
    "jq_169": "cnc_068",
    "jq_170": "jq_172",
    "jq_171": "jq_172",
    "jq_173": "cnc_068",
    "jq_174": "cnc_068",
    "jq_175": "ta_015",
    "jq_176": "ta_004",
    "jq_177": "a101_004",
    "jq_178": "a101_004",
    "jq_179": "a360_071",
    "jq_180": "ta_015",
    "jq_181": "ta_015",
    "jq_182": "a101_004",
    "jq_183": "a360_071",
    "jq_184": "ta_015",
    "jq_185": "a101_090",
    "jq_186": "a360_071",
    "jq_187": "a101_004",
    "jq_188": "ta_004",
    "jq_189": "cnc_038",
    "jq_190": "a158_082",
    "jq_191": "a158_077",
    "jq_192": "a101_004",
    "jq_193": "a101_004",
    "jq_194": "a101_004",
    "jq_195": "ta_015",
    "jq_196": "a101_090",
    "jq_197": "a360_071",
    "jq_198": "ta_015",
    "jq_199": "a101_004",
    "jq_200": "a101_004",
    "jq_201": "a101_004",
    "jq_202": "a360_071",
    "jq_203": "ta_015",
    "jq_204": "a360_071",
    "jq_205": "tst_050",
    "jq_206": "ta_015",
    "jq_207": "ta_015",
    "jq_208": "a101_004",
    "jq_209": "ta_015",
    "jq_210": "a360_071",
    "jq_211": "a360_071",
    "jq_212": "ta_015",
    "jq_213": "ta_015",
    "jq_214": "a360_071",
    "jq_215": "ta_015",
    "jq_216": "a101_004",
    "jq_217": "a360_071",
    "jq_218": "tst_072",
    "jq_219": "ta_015",
    "jq_220": "gtja_011",
    "jq_221": "a360_071",
    "jq_222": "ta_015",
    "jq_223": "ta_015",
    "jq_224": "jq_116",
    "jq_225": "cnc_024",
    "jq_226": "oap_029",
    "jq_227": "svc_013",
    "jq_228": "cne6_002",
    "jq_229": "oap_057",
    "jq_231": "oap_060",
    "jq_232": "jkp_018",
    "jq_233": "jq_032",
    "oap_001": "jq_116",
    "oap_002": "cnc_038",
    "oap_003": "ta_015",
    "oap_004": "cnc_033",
    "oap_005": "oap_029",
    "oap_006": "cnc_033",
    "oap_008": "cnc_032",
    "oap_009": "oap_029",
    "oap_011": "jkp_101",
    "oap_012": "oap_068",
    "oap_016": "jkp_124",
    "oap_017": "jkp_124",
    "oap_018": "cnc_008",
    "oap_019": "oap_015",
    "oap_020": "jq_230",
    "oap_022": "jq_230",
    "oap_024": "cnc_021",
    "oap_028": "oap_027",
    "oap_031": "a360_071",
    "oap_032": "cnc_031",
    "oap_034": "cnc_033",
    "oap_035": "cnc_032",
    "oap_036": "cnc_033",
    "oap_037": "jq_116",
    "oap_038": "jq_116",
    "oap_039": "jq_230",
    "oap_042": "oap_040",
    "oap_044": "jq_230",
    "oap_045": "jq_230",
    "oap_046": "jkp_121",
    "oap_048": "cnc_068",
    "oap_049": "oap_057",
    "oap_050": "oap_057",
    "oap_051": "jq_032",
    "oap_052": "cnc_068",
    "oap_053": "oap_060",
    "oap_054": "oap_060",
    "oap_055": "jq_002",
    "oap_056": "cnc_068",
    "oap_058": "cnc_072",
    "oap_059": "cnc_072",
    "oap_061": "cnc_068",
    "oap_062": "cnc_078",
    "oap_063": "cnc_078",
    "oap_064": "oap_060",
    "oap_065": "oap_068",
    "oap_066": "oap_068",
    "oap_067": "oap_068",
    "oap_071": "cnc_072",
    "oap_072": "cnc_052",
    "oap_073": "cnc_052",
    "oap_074": "cnc_052",
    "oap_076": "cnc_052",
    "oap_077": "cnc_052",
    "oap_078": "cnc_068",
    "oap_079": "cnc_101",
    "oap_081": "jkp_075",
    "oap_082": "jq_002",
    "oap_083": "jq_032",
    "oap_084": "cne6_025",
    "oap_085": "jkp_060",
    "oap_087": "jkp_060",
    "oap_089": "jkp_018",
    "oap_091": "jq_046",
    "oap_094": "jkp_097",
    "oap_095": "jkp_053",
    "oap_096": "oap_098",
    "oap_097": "cnc_081",
    "oap_099": "jkp_031",
    "oap_100": "jkp_030",
    "oap_102": "oap_109",
    "oap_103": "jkp_031",
    "oap_104": "jkp_045",
    "oap_105": "jkp_053",
    "oap_106": "oap_126",
    "oap_107": "jkp_051",
    "oap_108": "jkp_042",
    "oap_110": "oap_101",
    "oap_111": "jq_002",
    "oap_112": "oap_069",
    "oap_113": "jq_002",
    "oap_114": "oap_115",
    "oap_116": "jkp_060",
    "oap_117": "oap_119",
    "oap_118": "oap_119",
    "oap_120": "oap_119",
    "oap_121": "cnc_079",
    "oap_122": "cne6_034",
    "oap_123": "cnc_078",
    "oap_124": "cne6_034",
    "oap_125": "jkp_057",
    "oap_127": "jq_139",
    "oap_129": "cnc_092",
    "oap_130": "cnc_092",
    "oap_133": "oap_132",
    "oap_134": "oap_132",
    "oap_135": "gtja_143",
    "oap_141": "oap_142",
    "svc_001": "jq_116",
    "svc_002": "oap_060",
    "svc_003": "cnc_008",
    "svc_004": "svc_015",
    "svc_005": "svc_009",
    "svc_006": "svc_017",
    "svc_007": "svc_013",
    "svc_008": "svc_015",
    "svc_010": "svc_017",
    "svc_011": "svc_018",
    "svc_012": "svc_013",
    "svc_014": "ta_015",
    "svc_016": "svc_009",
    "svc_019": "jq_230",
    "svc_020": "oap_029",
    "svc_021": "ta_015",
    "ta_001": "ta_002",
    "ta_003": "a158_080",
    "ta_005": "ta_004",
    "ta_006": "ta_103",
    "ta_007": "a101_004",
    "ta_009": "a360_071",
    "ta_010": "a360_071",
    "ta_011": "a101_004",
    "ta_012": "ta_004",
    "ta_013": "ta_015",
    "ta_014": "a158_027",
    "ta_016": "ta_015",
    "ta_017": "ta_015",
    "ta_018": "a101_004",
    "ta_019": "ta_015",
    "ta_020": "a101_004",
    "ta_021": "a101_004",
    "ta_022": "a101_090",
    "ta_023": "a101_004",
    "ta_024": "a101_090",
    "ta_025": "a158_104",
    "ta_026": "a360_071",
    "ta_027": "gtja_011",
    "ta_028": "a101_004",
    "ta_029": "ta_015",
    "ta_030": "oap_015",
    "ta_031": "a101_004",
    "ta_032": "ta_015",
    "ta_033": "ta_015",
    "ta_034": "ta_015",
    "ta_035": "a101_004",
    "ta_036": "ta_015",
    "ta_037": "a101_004",
    "ta_038": "a101_004",
    "ta_039": "a101_004",
    "ta_040": "ta_015",
    "ta_041": "a101_004",
    "ta_042": "a101_004",
    "ta_043": "tst_030",
    "ta_044": "ta_015",
    "ta_046": "ta_045",
    "ta_047": "a158_107",
    "ta_048": "a158_027",
    "ta_049": "a158_024",
    "ta_050": "a101_083",
    "ta_051": "a101_083",
    "ta_054": "gtja_157",
    "ta_057": "ta_053",
    "ta_060": "a101_090",
    "ta_061": "ta_015",
    "ta_062": "ta_015",
    "ta_063": "oap_015",
    "ta_074": "ta_103",
    "ta_076": "ta_103",
    "ta_080": "ta_111",
    "ta_084": "ta_085",
    "ta_091": "ta_090",
    "ta_100": "ta_099",
    "ta_102": "ta_111",
    "ta_107": "ta_108",
    "ta_116": "ta_092",
    "ta_119": "ta_082",
    "tst_001": "a360_071",
    "tst_002": "a360_071",
    "tst_003": "a158_027",
    "tst_004": "a101_004",
    "tst_005": "a101_004",
    "tst_006": "a101_004",
    "tst_007": "ta_015",
    "tst_008": "ta_004",
    "tst_009": "ta_015",
    "tst_010": "ta_015",
    "tst_011": "ta_015",
    "tst_012": "a101_004",
    "tst_013": "ta_015",
    "tst_014": "a360_071",
    "tst_015": "a360_071",
    "tst_016": "ta_015",
    "tst_017": "ta_004",
    "tst_018": "ta_008",
    "tst_019": "a158_037",
    "tst_020": "ta_015",
    "tst_021": "a360_071",
    "tst_022": "a101_004",
    "tst_023": "a101_004",
    "tst_024": "ta_015",
    "tst_025": "ta_015",
    "tst_026": "a360_071",
    "tst_027": "a360_071",
    "tst_028": "a360_071",
    "tst_029": "tst_030",
    "tst_031": "a101_004",
    "tst_032": "a360_071",
    "tst_033": "a101_004",
    "tst_034": "ta_004",
    "tst_035": "a101_004",
    "tst_036": "a101_004",
    "tst_037": "ta_015",
    "tst_038": "ta_015",
    "tst_039": "a101_090",
    "tst_040": "a101_004",
    "tst_041": "ta_015",
    "tst_042": "ta_015",
    "tst_043": "a360_071",
    "tst_044": "a360_071",
    "tst_045": "a360_071",
    "tst_046": "a360_071",
    "tst_047": "a360_071",
    "tst_048": "a101_004",
    "tst_049": "tst_050",
    "tst_051": "ta_004",
    "tst_052": "ta_015",
    "tst_053": "tst_030",
    "tst_054": "a158_107",
    "tst_055": "a158_104",
    "tst_056": "a158_107",
    "tst_057": "ta_015",
    "tst_058": "tst_030",
    "tst_059": "a101_004",
    "tst_060": "ta_015",
    "tst_061": "a360_071",
    "tst_062": "a158_027",
    "tst_063": "gtja_068",
    "tst_064": "ta_015",
    "tst_065": "a360_071",
    "tst_066": "a360_071",
    "tst_067": "a158_107",
    "tst_068": "a101_004",
    "tst_069": "a101_090",
    "tst_070": "a101_004",
    "tst_071": "ta_015",
    "tst_073": "tst_072",
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
        "max_abs_rank_correlation": 0.5,
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
