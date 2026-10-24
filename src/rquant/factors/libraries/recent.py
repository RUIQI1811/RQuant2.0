"""Locked membership of the recent factor set.

recent keeps factors whose single-factor Rank IC is still significant and stable since 2022, the period in which the
LGB predictions of every other factor set decayed to a Rank IC of 0.01-0.02. It was selected with
``rquant.factors.selection`` from the 1d evaluations listed in ``provenance``
(``rquant factors evaluate --horizon 1d --start 2022-01-01 --end 2026-10-21`` on each of the seven library factor sets):

- direction is the sign of the 2022+ mean daily Rank IC; every yearly statistic is oriented by it;
- years with at least ``min_year_days`` evaluated days are counted (2022-2026; 2026 is partial with 189 days);
- the oriented annual Rank IC is positive in at least ``min_positive_rank_ic_year_ratio`` of counted years (4 of 5);
- the yearly long-short return is not required (``min_positive_long_short_year_ratio`` is 0), as for Hquality;
- the absolute 2022+ Rank IC is at least ``min_abs_rank_ic`` and its t-statistic (daily Rank ICIR times the square root
  of the number of days) is at least ``min_abs_rank_ic_t_stat`` in absolute value;
- coverage (published observations over the best-covered factor's, pooled over dates and all seven evaluations) is at
  least ``min_coverage``;
- duplicates are then pruned with ``prune_correlated``: the 183 factors passing the checks above are ordered
  by absolute Rank IC, and a factor is dropped when its absolute mean daily cross-sectional Spearman correlation
  (``mean_rank_correlation`` on every fifth trading date since 2022-01-04) with an already kept factor reaches
  ``max_abs_rank_correlation``. ``DEDUPLICATED`` maps each dropped factor to the kept one it duplicates.

The selection uses 2022-2026 returns, so walk-forward test years from 2022 on are in-sample for membership.

Members keep their source canonical names and values. ``direction == -1`` marks a factor used inverted; the stored
column is not negated. Changing membership changes the recent catalog and engine fingerprints, so a stale
``data/factors/recent`` store is refused rather than reused.
"""

from __future__ import annotations

from rquant.factors.libraries.base import FactorSelection, SelectedFactor

RECENT_MEMBERS = (
    SelectedFactor("a158_002", -1),  # RankIC -0.0336, t -5.3, IC+ 5/5, cov 1.000
    SelectedFactor("a158_004", -1),  # RankIC -0.0164, t -3.4, IC+ 5/5, cov 1.000
    SelectedFactor("a158_006", -1),  # RankIC -0.0241, t -5.1, IC+ 5/5, cov 1.000
    SelectedFactor("a158_012", 1),  # RankIC +0.0380, t +7.1, IC+ 5/5, cov 1.000
    SelectedFactor("a158_024", -1),  # RankIC -0.0304, t -4.9, IC+ 5/5, cov 1.000
    SelectedFactor("a158_025", -1),  # RankIC -0.0372, t -5.6, IC+ 5/5, cov 0.999
    SelectedFactor("a158_028", -1),  # RankIC -0.0308, t -4.0, IC+ 5/5, cov 0.993
    SelectedFactor("a158_039", -1),  # RankIC -0.0186, t -3.3, IC+ 5/5, cov 1.000
    SelectedFactor("a158_049", 1),  # RankIC +0.0314, t +5.6, IC+ 5/5, cov 1.000
    SelectedFactor("a158_052", 1),  # RankIC +0.0221, t +3.5, IC+ 5/5, cov 0.997
    SelectedFactor("a158_090", -1),  # RankIC -0.0177, t -4.0, IC+ 5/5, cov 1.000
    SelectedFactor("a158_094", -1),  # RankIC -0.0133, t -3.1, IC+ 5/5, cov 1.000
    SelectedFactor("a158_095", -1),  # RankIC -0.0139, t -3.0, IC+ 5/5, cov 1.000
    SelectedFactor("a158_097", -1),  # RankIC -0.0187, t -4.0, IC+ 5/5, cov 1.000
    SelectedFactor("a158_098", -1),  # RankIC -0.0183, t -3.8, IC+ 5/5, cov 1.000
    SelectedFactor("a101_010", 1),  # RankIC +0.0147, t +3.1, IC+ 5/5, cov 1.000
    SelectedFactor("a101_020", -1),  # RankIC -0.0194, t -3.7, IC+ 5/5, cov 1.000
    SelectedFactor("a101_024", 1),  # RankIC +0.0261, t +5.3, IC+ 5/5, cov 1.000
    SelectedFactor("a101_025", 1),  # RankIC +0.0230, t +4.3, IC+ 5/5, cov 0.998
    SelectedFactor("a101_026", 1),  # RankIC +0.0109, t +3.1, IC+ 5/5, cov 1.000
    SelectedFactor("a101_036", 1),  # RankIC +0.0115, t +3.1, IC+ 4/5, cov 0.975
    SelectedFactor("a101_040", 1),  # RankIC +0.0292, t +5.9, IC+ 5/5, cov 0.999
    SelectedFactor("a101_042", 1),  # RankIC +0.0353, t +7.0, IC+ 5/5, cov 1.000
    SelectedFactor("a101_047", 1),  # RankIC +0.0128, t +3.2, IC+ 5/5, cov 1.000
    SelectedFactor("a101_056", 1),  # RankIC +0.0197, t +4.4, IC+ 5/5, cov 0.999
    SelectedFactor("a101_060", 1),  # RankIC +0.0169, t +3.5, IC+ 5/5, cov 0.999
    SelectedFactor("a101_069", 1),  # RankIC +0.0243, t +5.9, IC+ 5/5, cov 0.996
    SelectedFactor("a101_071", 1),  # RankIC +0.0119, t +3.3, IC+ 5/5, cov 0.998
    SelectedFactor("a101_083", 1),  # RankIC +0.0347, t +6.1, IC+ 5/5, cov 0.999
    SelectedFactor("a101_086", 1),  # RankIC +0.0123, t +3.1, IC+ 5/5, cov 1.000
    SelectedFactor("a101_088", 1),  # RankIC +0.0273, t +6.6, IC+ 5/5, cov 0.999
    SelectedFactor("gtja_001", 1),  # RankIC +0.0122, t +3.4, IC+ 5/5, cov 0.999
    SelectedFactor("gtja_002", 1),  # RankIC +0.0159, t +3.0, IC+ 4/5, cov 1.000
    SelectedFactor("gtja_041", 1),  # RankIC +0.0261, t +4.9, IC+ 5/5, cov 0.999
    SelectedFactor("gtja_054", 1),  # RankIC +0.0226, t +5.2, IC+ 5/5, cov 0.999
    SelectedFactor("gtja_070", -1),  # RankIC -0.0200, t -4.0, IC+ 5/5, cov 0.999
    SelectedFactor("gtja_083", 1),  # RankIC +0.0112, t +3.9, IC+ 5/5, cov 1.000
    SelectedFactor("gtja_092", 1),  # RankIC +0.0126, t +3.4, IC+ 4/5, cov 0.974
    SelectedFactor("gtja_099", 1),  # RankIC +0.0101, t +3.3, IC+ 5/5, cov 1.000
    SelectedFactor("gtja_111", 1),  # RankIC +0.0175, t +3.6, IC+ 5/5, cov 1.000
    SelectedFactor("gtja_121", 1),  # RankIC +0.0219, t +4.2, IC+ 5/5, cov 0.991
    SelectedFactor("gtja_143", -1),  # RankIC -0.0118, t -3.2, IC+ 4/5, cov 1.000
    SelectedFactor("gtja_149", -1),  # RankIC -0.0218, t -3.1, IC+ 4/5, cov 0.979
    SelectedFactor("gtja_156", 1),  # RankIC +0.0226, t +5.0, IC+ 5/5, cov 0.999
    SelectedFactor("gtja_159", -1),  # RankIC -0.0484, t -6.2, IC+ 5/5, cov 0.997
    SelectedFactor("gtja_178", -1),  # RankIC -0.0240, t -4.6, IC+ 5/5, cov 1.000
    SelectedFactor("gtja_185", 1),  # RankIC +0.0153, t +3.3, IC+ 5/5, cov 1.000
    SelectedFactor("gtja_187", -1),  # RankIC -0.0358, t -5.7, IC+ 5/5, cov 0.998
    SelectedFactor("a360_238", 1),  # RankIC +0.0216, t +3.7, IC+ 5/5, cov 1.000
    SelectedFactor("a360_239", 1),  # RankIC +0.0262, t +4.6, IC+ 5/5, cov 1.000
    SelectedFactor("a360_306", 1),  # RankIC +0.0143, t +3.3, IC+ 5/5, cov 0.999
    SelectedFactor("a360_310", 1),  # RankIC +0.0144, t +3.4, IC+ 5/5, cov 0.999
    SelectedFactor("a360_312", 1),  # RankIC +0.0143, t +3.4, IC+ 5/5, cov 0.999
    SelectedFactor("a360_314", 1),  # RankIC +0.0151, t +3.5, IC+ 5/5, cov 0.999
    SelectedFactor("cnc_003", -1),  # RankIC -0.0400, t -5.1, IC+ 5/5, cov 0.999
    SelectedFactor("cnc_019", -1),  # RankIC -0.0207, t -5.0, IC+ 5/5, cov 1.000
    SelectedFactor("cnc_020", -1),  # RankIC -0.0228, t -5.5, IC+ 5/5, cov 1.000
    SelectedFactor("cnc_025", -1),  # RankIC -0.0300, t -3.1, IC+ 4/5, cov 1.000
    SelectedFactor("cnc_027", 1),  # RankIC +0.0248, t +3.5, IC+ 4/5, cov 0.996
    SelectedFactor("cnc_034", 1),  # RankIC +0.0188, t +3.8, IC+ 4/5, cov 0.991
    SelectedFactor("cnc_035", 1),  # RankIC +0.0152, t +3.0, IC+ 4/5, cov 0.991
    SelectedFactor("cnc_036", 1),  # RankIC +0.0263, t +3.4, IC+ 4/5, cov 1.000
    SelectedFactor("cnc_038", -1),  # RankIC -0.0342, t -5.1, IC+ 5/5, cov 1.000
    SelectedFactor("cnc_044", 1),  # RankIC +0.0128, t +3.4, IC+ 4/5, cov 0.999
    SelectedFactor("cnc_046", 1),  # RankIC +0.0163, t +4.0, IC+ 5/5, cov 1.000
    SelectedFactor("cnc_048", -1),  # RankIC -0.0185, t -4.0, IC+ 4/5, cov 1.000
    SelectedFactor("cnc_056", 1),  # RankIC +0.0216, t +3.9, IC+ 5/5, cov 1.000
    SelectedFactor("cnc_057", -1),  # RankIC -0.0177, t -3.7, IC+ 4/5, cov 1.000
    SelectedFactor("cnc_059", 1),  # RankIC +0.0214, t +5.9, IC+ 5/5, cov 0.842
    SelectedFactor("cnc_061", 1),  # RankIC +0.0349, t +4.7, IC+ 5/5, cov 1.000
    SelectedFactor("cnc_065", 1),  # RankIC +0.0334, t +4.4, IC+ 5/5, cov 0.999
    SelectedFactor("cnc_066", -1),  # RankIC -0.0294, t -5.2, IC+ 4/5, cov 0.991
    SelectedFactor("cnc_067", -1),  # RankIC -0.0290, t -4.6, IC+ 5/5, cov 0.894
    SelectedFactor("cnc_069", 1),  # RankIC +0.0267, t +4.7, IC+ 5/5, cov 1.000
    SelectedFactor("cnc_076", -1),  # RankIC -0.0180, t -4.3, IC+ 4/5, cov 0.924
    SelectedFactor("cnc_078", -1),  # RankIC -0.0311, t -5.0, IC+ 5/5, cov 0.996
    SelectedFactor("cnc_079", -1),  # RankIC -0.0300, t -5.9, IC+ 5/5, cov 0.991
    SelectedFactor("cnc_081", -1),  # RankIC -0.0153, t -3.6, IC+ 4/5, cov 0.827
    SelectedFactor("cnc_082", -1),  # RankIC -0.0106, t -3.6, IC+ 4/5, cov 0.999
    SelectedFactor("cnc_085", -1),  # RankIC -0.0174, t -4.0, IC+ 4/5, cov 0.844
    SelectedFactor("cnc_088", -1),  # RankIC -0.0166, t -4.4, IC+ 4/5, cov 0.844
    SelectedFactor("cnc_091", -1),  # RankIC -0.0248, t -4.5, IC+ 4/5, cov 1.000
    SelectedFactor("cnc_095", 1),  # RankIC +0.0108, t +3.3, IC+ 4/5, cov 1.000
    SelectedFactor("cnc_104", -1),  # RankIC -0.0169, t -4.3, IC+ 5/5, cov 0.995
    SelectedFactor("cnc_105", -1),  # RankIC -0.0138, t -3.2, IC+ 4/5, cov 0.844
    SelectedFactor("svc_001", 1),  # RankIC +0.0213, t +3.8, IC+ 5/5, cov 1.000
    SelectedFactor("svc_003", -1),  # RankIC -0.0158, t -3.5, IC+ 5/5, cov 0.998
    SelectedFactor("svc_013", 1),  # RankIC +0.0169, t +3.6, IC+ 4/5, cov 1.000
    SelectedFactor("svc_015", -1),  # RankIC -0.0227, t -3.8, IC+ 4/5, cov 1.000
    SelectedFactor("svc_017", 1),  # RankIC +0.0367, t +3.6, IC+ 5/5, cov 1.000
    SelectedFactor("svc_018", -1),  # RankIC -0.0233, t -3.5, IC+ 5/5, cov 1.000
    SelectedFactor("svc_019", -1),  # RankIC -0.0464, t -5.7, IC+ 5/5, cov 1.000
    SelectedFactor("ta_014", -1),  # RankIC -0.0293, t -3.6, IC+ 5/5, cov 1.000
    SelectedFactor("ta_030", -1),  # RankIC -0.0324, t -4.6, IC+ 5/5, cov 1.000
    SelectedFactor("ta_050", 1),  # RankIC +0.0261, t +4.4, IC+ 5/5, cov 1.000
    SelectedFactor("ta_076", -1),  # RankIC -0.0107, t -3.2, IC+ 5/5, cov 1.000
)

DEDUPLICATED = {
    "a158_001": "ta_050",
    "a158_003": "ta_050",
    "a158_008": "ta_050",
    "a158_009": "ta_050",
    "a158_010": "ta_050",
    "a158_013": "a101_083",
    "a158_026": "ta_030",
    "a158_027": "ta_030",
    "a158_050": "a158_049",
    "a158_051": "a158_052",
    "a158_053": "a158_052",
    "a158_059": "a360_238",
    "a158_096": "a158_097",
    "a101_009": "a101_025",
    "a101_013": "gtja_099",
    "a101_016": "gtja_083",
    "a101_018": "gtja_054",
    "a101_033": "ta_050",
    "a101_057": "a101_083",
    "a101_094": "gtja_121",
    "a101_101": "ta_050",
    "gtja_005": "a101_026",
    "gtja_010": "gtja_187",
    "gtja_042": "a101_040",
    "gtja_093": "gtja_187",
    "gtja_095": "gtja_070",
    "gtja_098": "a101_024",
    "gtja_107": "a101_020",
    "gtja_114": "a101_083",
    "gtja_120": "a101_042",
    "gtja_124": "a101_083",
    "gtja_126": "gtja_187",
    "gtja_129": "gtja_187",
    "gtja_140": "a101_088",
    "gtja_150": "gtja_070",
    "gtja_153": "gtja_187",
    "gtja_158": "a158_002",
    "gtja_160": "gtja_187",
    "gtja_161": "gtja_187",
    "gtja_163": "a101_025",
    "gtja_164": "gtja_187",
    "gtja_167": "gtja_187",
    "gtja_171": "ta_050",
    "gtja_173": "gtja_187",
    "gtja_174": "gtja_187",
    "gtja_175": "gtja_187",
    "gtja_189": "gtja_187",
    "a360_059": "ta_050",
    "a360_120": "ta_050",
    "a360_240": "a158_012",
    "a360_300": "a101_083",
    "a360_305": "a360_306",
    "a360_313": "a360_312",
    "cnc_001": "svc_001",
    "cnc_002": "svc_001",
    "cnc_004": "cnc_003",
    "cnc_005": "cnc_003",
    "cnc_006": "cnc_003",
    "cnc_008": "svc_003",
    "cnc_015": "svc_019",
    "cnc_016": "gtja_159",
    "cnc_017": "svc_019",
    "cnc_018": "cnc_019",
    "cnc_023": "cnc_025",
    "cnc_024": "cnc_025",
    "cnc_037": "gtja_159",
    "cnc_042": "cnc_044",
    "cnc_062": "cnc_061",
    "cnc_063": "cnc_061",
    "cnc_064": "cnc_065",
    "cnc_070": "cnc_056",
    "cnc_071": "cnc_061",
    "cnc_080": "cnc_081",
    "cnc_087": "cnc_088",
    "cnc_106": "cnc_061",
    "svc_002": "cnc_065",
    "svc_006": "svc_017",
    "svc_007": "gtja_159",
    "svc_008": "svc_015",
    "svc_010": "svc_017",
    "svc_012": "gtja_159",
    "ta_006": "ta_050",
    "ta_016": "gtja_159",
    "ta_048": "gtja_159",
    "ta_049": "a158_002",
    "ta_051": "ta_050",
    "ta_063": "ta_030",
}

RECENT_SELECTION = FactorSelection(
    name="recent",
    source_factor_sets=(
        "qlib_alpha158",
        "wq_alpha101",
        "gtja191",
        "qlib_alpha360",
        "china_chars",
        "size_value_china",
        "talib",
    ),
    members=RECENT_MEMBERS,
    criteria={
        "min_year_days": 120,
        "min_positive_rank_ic_year_ratio": 0.8,
        "min_positive_long_short_year_ratio": 0.0,
        "min_abs_rank_ic": 0.01,
        "min_abs_rank_ic_t_stat": 3.0,
        "min_coverage": 0.8,
        "max_abs_rank_correlation": 0.8,
        "correlation_date_step": 5,
        "excluded_factor_sets": (),
    },
    provenance={
        "horizon": "1d",
        "evaluation_start": "2022-01-04",
        "evaluation_end": "2026-10-21",
        "qlib_fingerprint": "4c48525d91abd4f4081a48179015fd9a9de899fed0158ef0b4db3a9880c009b8",
        "deduplicated": DEDUPLICATED,
        "evaluations": (
            {
                "factor_set": "qlib_alpha158",
                "run_id": "20261024T091047Z-6c2d69c9",
                "factor_fingerprint": "71aafb5d55a082d3eab4ca7bdc2c90ec0d9d24c08216a25468bb2285ac7e2a6e",
            },
            {
                "factor_set": "wq_alpha101",
                "run_id": "20261024T091148Z-0cb4d976",
                "factor_fingerprint": "21a0f59aaa2616eb5b494657e83582dc155f8fa9cf46f686efbdde0b0c435029",
            },
            {
                "factor_set": "gtja191",
                "run_id": "20261024T091244Z-fc7dfb1f",
                "factor_fingerprint": "bc9d664f354f5cf78265f528eca208e5b85e69963cdaec2e2989df19a6a459d5",
            },
            {
                "factor_set": "qlib_alpha360",
                "run_id": "20261024T091348Z-e459b8ea",
                "factor_fingerprint": "d6cff8fc9b2f68f6cf605fd45a4a6ec3fd05859e81e310939b44a7e162200511",
            },
            {
                "factor_set": "china_chars",
                "run_id": "20261024T091502Z-4861e6d8",
                "factor_fingerprint": "7c3ba1d9cb85222fd295cd1222166987b88184e579198add02595dcf0f054a60",
            },
            {
                "factor_set": "size_value_china",
                "run_id": "20261024T091550Z-c6536f91",
                "factor_fingerprint": "967bf775908bcb969b03d11fd9df9e5e3fc3362aab7765800465ab3ecba54773",
            },
            {
                "factor_set": "talib",
                "run_id": "20261024T091634Z-df715b28",
                "factor_fingerprint": "d6801ecdeb333fb69b0e1d3b89ca1d1f86d1950fedfc1a93c05754af4bed2ebf",
            },
        ),
    },
)
