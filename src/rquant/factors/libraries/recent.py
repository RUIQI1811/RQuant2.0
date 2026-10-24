"""Locked membership of the recent factor set.

recent keeps factors whose single-factor Rank IC is still significant and stable since 2020; from 2022 on the LGB
predictions of every other factor set decayed to a Rank IC of 0.01-0.02. It was selected with
``rquant.factors.selection`` from the 1d evaluations listed in ``provenance``
(``rquant factors evaluate --horizon 1d --start 2020-01-01 --end 2026-10-21`` on each of the seven library factor sets):

- direction is the sign of the 2020+ mean daily Rank IC; every yearly statistic is oriented by it;
- years with at least ``min_year_days`` evaluated days are counted (2020-2026; 2026 is partial with 189 days);
- the oriented annual Rank IC is positive in at least ``min_positive_rank_ic_year_ratio`` of counted years (6 of 7);
- the yearly long-short return is not required (``min_positive_long_short_year_ratio`` is 0), as for Hquality;
- the absolute 2020+ Rank IC is at least ``min_abs_rank_ic`` and its t-statistic (daily Rank ICIR times the square root
  of the number of days) is at least ``min_abs_rank_ic_t_stat`` in absolute value;
- coverage (published observations over the best-covered factor's, pooled over dates and all seven evaluations) is at
  least ``min_coverage``;
- duplicates are then pruned with ``prune_correlated``: the 213 factors passing the checks above are ordered
  by absolute Rank IC, and a factor is dropped when its absolute mean daily cross-sectional Spearman correlation
  (``mean_rank_correlation`` on every fifth trading date since 2020-01-02) with an already kept factor reaches
  ``max_abs_rank_correlation``. ``DEDUPLICATED`` maps each dropped factor to the kept one it duplicates.

The selection uses 2020-2026 returns, so walk-forward test years from 2020 on are in-sample for membership.

Members keep their source canonical names and values. ``direction == -1`` marks a factor used inverted; the stored
column is not negated. Changing membership changes the recent catalog and engine fingerprints, so a stale
``data/factors/recent`` store is refused rather than reused.
"""

from __future__ import annotations

from rquant.factors.libraries.base import FactorSelection, SelectedFactor

RECENT_MEMBERS = (
    SelectedFactor("a158_002", -1),  # RankIC -0.0270, t -5.2, IC+ 7/7, cov 1.000
    SelectedFactor("a158_004", -1),  # RankIC -0.0138, t -3.6, IC+ 7/7, cov 1.000
    SelectedFactor("a158_006", -1),  # RankIC -0.0213, t -5.5, IC+ 7/7, cov 1.000
    SelectedFactor("a158_012", 1),  # RankIC +0.0363, t +8.3, IC+ 7/7, cov 1.000
    SelectedFactor("a158_024", -1),  # RankIC -0.0241, t -4.7, IC+ 6/7, cov 0.999
    SelectedFactor("a158_025", -1),  # RankIC -0.0297, t -5.4, IC+ 7/7, cov 0.999
    SelectedFactor("a158_028", -1),  # RankIC -0.0218, t -3.6, IC+ 6/7, cov 0.991
    SelectedFactor("a158_039", -1),  # RankIC -0.0200, t -4.4, IC+ 7/7, cov 1.000
    SelectedFactor("a158_040", -1),  # RankIC -0.0180, t -3.8, IC+ 7/7, cov 1.000
    SelectedFactor("a158_049", 1),  # RankIC +0.0307, t +6.6, IC+ 7/7, cov 0.999
    SelectedFactor("a158_051", 1),  # RankIC +0.0218, t +4.3, IC+ 7/7, cov 0.997
    SelectedFactor("a158_053", 1),  # RankIC +0.0199, t +3.7, IC+ 6/7, cov 0.991
    SelectedFactor("a158_060", 1),  # RankIC +0.0210, t +4.3, IC+ 7/7, cov 0.999
    SelectedFactor("a158_064", -1),  # RankIC -0.0176, t -4.0, IC+ 7/7, cov 0.999
    SelectedFactor("a158_090", -1),  # RankIC -0.0160, t -4.3, IC+ 7/7, cov 1.000
    SelectedFactor("a158_094", -1),  # RankIC -0.0129, t -3.6, IC+ 7/7, cov 1.000
    SelectedFactor("a158_095", -1),  # RankIC -0.0126, t -3.3, IC+ 7/7, cov 1.000
    SelectedFactor("a158_097", -1),  # RankIC -0.0164, t -4.1, IC+ 7/7, cov 1.000
    SelectedFactor("a158_098", -1),  # RankIC -0.0160, t -3.9, IC+ 7/7, cov 1.000
    SelectedFactor("a158_133", 1),  # RankIC +0.0120, t +3.2, IC+ 7/7, cov 0.991
    SelectedFactor("a158_148", -1),  # RankIC -0.0138, t -3.9, IC+ 7/7, cov 0.991
    SelectedFactor("a101_001", -1),  # RankIC -0.0125, t -3.4, IC+ 6/7, cov 0.997
    SelectedFactor("a101_010", 1),  # RankIC +0.0160, t +4.1, IC+ 7/7, cov 1.000
    SelectedFactor("a101_012", 1),  # RankIC +0.0112, t +4.2, IC+ 7/7, cov 1.000
    SelectedFactor("a101_025", 1),  # RankIC +0.0257, t +5.9, IC+ 7/7, cov 0.997
    SelectedFactor("a101_034", 1),  # RankIC +0.0147, t +3.9, IC+ 7/7, cov 1.000
    SelectedFactor("a101_036", 1),  # RankIC +0.0111, t +3.5, IC+ 6/7, cov 0.964
    SelectedFactor("a101_038", 1),  # RankIC +0.0199, t +4.4, IC+ 7/7, cov 0.999
    SelectedFactor("a101_040", 1),  # RankIC +0.0254, t +6.5, IC+ 7/7, cov 0.999
    SelectedFactor("a101_042", 1),  # RankIC +0.0348, t +8.2, IC+ 7/7, cov 1.000
    SelectedFactor("a101_047", 1),  # RankIC +0.0115, t +3.4, IC+ 6/7, cov 1.000
    SelectedFactor("a101_053", 1),  # RankIC +0.0135, t +3.5, IC+ 7/7, cov 0.999
    SelectedFactor("a101_054", 1),  # RankIC +0.0146, t +3.6, IC+ 6/7, cov 1.000
    SelectedFactor("a101_056", 1),  # RankIC +0.0218, t +5.9, IC+ 7/7, cov 0.999
    SelectedFactor("a101_060", 1),  # RankIC +0.0180, t +4.5, IC+ 7/7, cov 0.999
    SelectedFactor("a101_068", 1),  # RankIC +0.0111, t +3.9, IC+ 7/7, cov 1.000
    SelectedFactor("a101_069", 1),  # RankIC +0.0184, t +5.3, IC+ 6/7, cov 0.992
    SelectedFactor("a101_071", 1),  # RankIC +0.0149, t +5.2, IC+ 7/7, cov 0.997
    SelectedFactor("a101_083", 1),  # RankIC +0.0390, t +8.5, IC+ 7/7, cov 0.999
    SelectedFactor("a101_086", 1),  # RankIC +0.0150, t +4.7, IC+ 7/7, cov 1.000
    SelectedFactor("a101_087", 1),  # RankIC +0.0115, t +3.8, IC+ 6/7, cov 0.965
    SelectedFactor("a101_088", 1),  # RankIC +0.0247, t +7.2, IC+ 7/7, cov 0.999
    SelectedFactor("gtja_001", 1),  # RankIC +0.0133, t +4.4, IC+ 7/7, cov 0.999
    SelectedFactor("gtja_002", 1),  # RankIC +0.0145, t +3.5, IC+ 6/7, cov 0.999
    SelectedFactor("gtja_005", 1),  # RankIC +0.0120, t +4.0, IC+ 7/7, cov 0.999
    SelectedFactor("gtja_029", -1),  # RankIC -0.0151, t -3.2, IC+ 6/7, cov 0.999
    SelectedFactor("gtja_036", -1),  # RankIC -0.0106, t -4.1, IC+ 7/7, cov 0.999
    SelectedFactor("gtja_041", 1),  # RankIC +0.0220, t +4.9, IC+ 6/7, cov 0.999
    SelectedFactor("gtja_043", -1),  # RankIC -0.0145, t -3.7, IC+ 7/7, cov 0.999
    SelectedFactor("gtja_045", -1),  # RankIC -0.0141, t -3.5, IC+ 7/7, cov 0.972
    SelectedFactor("gtja_048", 1),  # RankIC +0.0142, t +3.6, IC+ 6/7, cov 0.997
    SelectedFactor("gtja_054", 1),  # RankIC +0.0207, t +5.7, IC+ 7/7, cov 0.999
    SelectedFactor("gtja_061", 1),  # RankIC +0.0134, t +3.4, IC+ 6/7, cov 0.984
    SelectedFactor("gtja_064", 1),  # RankIC +0.0100, t +4.6, IC+ 7/7, cov 0.986
    SelectedFactor("gtja_070", -1),  # RankIC -0.0164, t -3.8, IC+ 6/7, cov 0.999
    SelectedFactor("gtja_083", 1),  # RankIC +0.0137, t +5.7, IC+ 7/7, cov 0.999
    SelectedFactor("gtja_092", 1),  # RankIC +0.0141, t +4.6, IC+ 6/7, cov 0.962
    SelectedFactor("gtja_098", 1),  # RankIC +0.0275, t +6.8, IC+ 7/7, cov 0.964
    SelectedFactor("gtja_099", 1),  # RankIC +0.0128, t +5.0, IC+ 7/7, cov 0.999
    SelectedFactor("gtja_111", 1),  # RankIC +0.0203, t +5.1, IC+ 7/7, cov 1.000
    SelectedFactor("gtja_121", 1),  # RankIC +0.0193, t +4.4, IC+ 6/7, cov 0.987
    SelectedFactor("gtja_136", 1),  # RankIC +0.0118, t +3.8, IC+ 6/7, cov 0.999
    SelectedFactor("gtja_137", -1),  # RankIC -0.0224, t -4.9, IC+ 7/7, cov 1.000
    SelectedFactor("gtja_156", 1),  # RankIC +0.0173, t +4.5, IC+ 6/7, cov 0.999
    SelectedFactor("gtja_159", -1),  # RankIC -0.0379, t -5.9, IC+ 6/7, cov 0.996
    SelectedFactor("gtja_167", -1),  # RankIC -0.0264, t -4.9, IC+ 6/7, cov 0.998
    SelectedFactor("gtja_176", -1),  # RankIC -0.0110, t -4.4, IC+ 7/7, cov 0.998
    SelectedFactor("gtja_178", -1),  # RankIC -0.0257, t -5.9, IC+ 7/7, cov 1.000
    SelectedFactor("gtja_185", 1),  # RankIC +0.0124, t +3.3, IC+ 7/7, cov 1.000
    SelectedFactor("a360_238", 1),  # RankIC +0.0256, t +5.4, IC+ 7/7, cov 1.000
    SelectedFactor("a360_239", 1),  # RankIC +0.0294, t +6.4, IC+ 7/7, cov 1.000
    SelectedFactor("a360_301", 1),  # RankIC +0.0125, t +3.5, IC+ 7/7, cov 0.999
    SelectedFactor("a360_304", 1),  # RankIC +0.0121, t +3.4, IC+ 7/7, cov 0.999
    SelectedFactor("a360_306", 1),  # RankIC +0.0151, t +4.3, IC+ 7/7, cov 0.999
    SelectedFactor("a360_308", 1),  # RankIC +0.0116, t +3.3, IC+ 7/7, cov 0.999
    SelectedFactor("a360_310", 1),  # RankIC +0.0156, t +4.4, IC+ 7/7, cov 0.999
    SelectedFactor("a360_312", 1),  # RankIC +0.0151, t +4.3, IC+ 7/7, cov 0.999
    SelectedFactor("a360_314", 1),  # RankIC +0.0162, t +4.5, IC+ 7/7, cov 0.999
    SelectedFactor("a360_316", 1),  # RankIC +0.0125, t +3.5, IC+ 7/7, cov 0.999
    SelectedFactor("a360_319", 1),  # RankIC +0.0128, t +3.5, IC+ 7/7, cov 0.999
    SelectedFactor("a360_321", 1),  # RankIC +0.0111, t +3.1, IC+ 7/7, cov 0.999
    SelectedFactor("a360_324", 1),  # RankIC +0.0116, t +3.3, IC+ 7/7, cov 0.999
    SelectedFactor("a360_326", 1),  # RankIC +0.0124, t +3.5, IC+ 7/7, cov 0.999
    SelectedFactor("a360_327", 1),  # RankIC +0.0127, t +3.5, IC+ 7/7, cov 0.999
    SelectedFactor("a360_329", 1),  # RankIC +0.0117, t +3.2, IC+ 7/7, cov 0.999
    SelectedFactor("a360_334", 1),  # RankIC +0.0114, t +3.2, IC+ 7/7, cov 0.999
    SelectedFactor("a360_335", 1),  # RankIC +0.0106, t +3.0, IC+ 6/7, cov 0.999
    SelectedFactor("a360_339", 1),  # RankIC +0.0104, t +3.0, IC+ 7/7, cov 0.999
    SelectedFactor("a360_345", 1),  # RankIC +0.0110, t +3.3, IC+ 6/7, cov 0.999
    SelectedFactor("cnc_001", 1),  # RankIC +0.0179, t +4.4, IC+ 7/7, cov 1.000
    SelectedFactor("cnc_003", -1),  # RankIC -0.0323, t -5.2, IC+ 7/7, cov 0.999
    SelectedFactor("cnc_016", -1),  # RankIC -0.0361, t -5.7, IC+ 6/7, cov 1.000
    SelectedFactor("cnc_019", -1),  # RankIC -0.0172, t -5.1, IC+ 7/7, cov 1.000
    SelectedFactor("cnc_020", -1),  # RankIC -0.0177, t -4.8, IC+ 7/7, cov 1.000
    SelectedFactor("cnc_024", -1),  # RankIC -0.0212, t -3.1, IC+ 6/7, cov 1.000
    SelectedFactor("cnc_027", 1),  # RankIC +0.0231, t +4.0, IC+ 6/7, cov 0.994
    SelectedFactor("cnc_034", 1),  # RankIC +0.0154, t +3.9, IC+ 6/7, cov 0.985
    SelectedFactor("cnc_035", 1),  # RankIC +0.0135, t +3.3, IC+ 6/7, cov 0.985
    SelectedFactor("cnc_038", -1),  # RankIC -0.0214, t -3.7, IC+ 6/7, cov 1.000
    SelectedFactor("cnc_044", 1),  # RankIC +0.0119, t +3.9, IC+ 6/7, cov 0.998
    SelectedFactor("cnc_046", 1),  # RankIC +0.0152, t +4.6, IC+ 7/7, cov 1.000
    SelectedFactor("cnc_059", 1),  # RankIC +0.0192, t +6.8, IC+ 7/7, cov 0.830
    SelectedFactor("cnc_061", 1),  # RankIC +0.0225, t +3.5, IC+ 6/7, cov 1.000
    SelectedFactor("cnc_064", 1),  # RankIC +0.0219, t +3.7, IC+ 6/7, cov 0.999
    SelectedFactor("cnc_069", 1),  # RankIC +0.0175, t +3.9, IC+ 6/7, cov 0.999
    SelectedFactor("cnc_078", -1),  # RankIC -0.0222, t -4.5, IC+ 6/7, cov 0.994
    SelectedFactor("cnc_079", -1),  # RankIC -0.0220, t -5.4, IC+ 6/7, cov 0.989
    SelectedFactor("cnc_085", -1),  # RankIC -0.0131, t -3.9, IC+ 6/7, cov 0.833
    SelectedFactor("cnc_104", -1),  # RankIC -0.0124, t -3.6, IC+ 6/7, cov 0.993
    SelectedFactor("svc_003", -1),  # RankIC -0.0158, t -4.3, IC+ 7/7, cov 0.998
    SelectedFactor("svc_015", -1),  # RankIC -0.0171, t -3.5, IC+ 6/7, cov 1.000
    SelectedFactor("svc_018", -1),  # RankIC -0.0180, t -3.5, IC+ 6/7, cov 1.000
    SelectedFactor("svc_021", 1),  # RankIC +0.0144, t +3.5, IC+ 6/7, cov 0.999
    SelectedFactor("ta_014", -1),  # RankIC -0.0200, t -3.1, IC+ 6/7, cov 1.000
    SelectedFactor("ta_030", -1),  # RankIC -0.0248, t -4.4, IC+ 7/7, cov 1.000
    SelectedFactor("ta_050", 1),  # RankIC +0.0307, t +6.5, IC+ 7/7, cov 1.000
    SelectedFactor("ta_076", -1),  # RankIC -0.0114, t -4.2, IC+ 7/7, cov 1.000
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
    "a158_052": "a158_051",
    "a158_059": "a360_238",
    "a158_061": "a158_051",
    "a158_096": "a158_097",
    "a158_153": "a158_148",
    "a158_158": "a158_148",
    "a101_006": "gtja_136",
    "a101_009": "a101_025",
    "a101_013": "gtja_099",
    "a101_014": "gtja_136",
    "a101_016": "gtja_083",
    "a101_018": "gtja_054",
    "a101_024": "gtja_098",
    "a101_026": "gtja_005",
    "a101_033": "ta_050",
    "a101_049": "a101_025",
    "a101_055": "gtja_176",
    "a101_057": "a101_083",
    "a101_094": "gtja_121",
    "a101_101": "ta_050",
    "gtja_010": "gtja_167",
    "gtja_042": "a101_040",
    "gtja_093": "gtja_167",
    "gtja_095": "gtja_070",
    "gtja_114": "a101_083",
    "gtja_120": "a101_042",
    "gtja_124": "a101_083",
    "gtja_129": "gtja_167",
    "gtja_139": "gtja_136",
    "gtja_140": "a101_088",
    "gtja_150": "gtja_070",
    "gtja_158": "a158_002",
    "gtja_161": "gtja_167",
    "gtja_163": "a101_025",
    "gtja_164": "gtja_167",
    "gtja_171": "ta_050",
    "gtja_174": "gtja_167",
    "gtja_175": "gtja_167",
    "gtja_187": "gtja_167",
    "gtja_189": "gtja_167",
    "a360_058": "a360_238",
    "a360_059": "gtja_178",
    "a360_119": "gtja_137",
    "a360_120": "ta_050",
    "a360_233": "a158_060",
    "a360_234": "a158_060",
    "a360_240": "a158_012",
    "a360_298": "a360_238",
    "a360_299": "a360_239",
    "a360_300": "a101_083",
    "a360_305": "a360_306",
    "a360_307": "a360_306",
    "a360_309": "a360_310",
    "a360_311": "a360_310",
    "a360_313": "a360_314",
    "a360_315": "a360_314",
    "a360_317": "a360_316",
    "a360_318": "a360_319",
    "a360_320": "a360_319",
    "cnc_002": "cnc_001",
    "cnc_004": "cnc_003",
    "cnc_005": "cnc_003",
    "cnc_006": "cnc_003",
    "cnc_008": "svc_003",
    "cnc_015": "cnc_016",
    "cnc_017": "cnc_016",
    "cnc_018": "cnc_019",
    "cnc_037": "gtja_159",
    "cnc_042": "cnc_044",
    "cnc_062": "cnc_061",
    "cnc_063": "cnc_061",
    "cnc_065": "cnc_064",
    "cnc_067": "cnc_064",
    "svc_001": "cnc_001",
    "svc_002": "cnc_064",
    "svc_007": "gtja_159",
    "svc_008": "svc_015",
    "svc_012": "cnc_016",
    "svc_019": "cnc_016",
    "ta_006": "ta_050",
    "ta_016": "gtja_159",
    "ta_024": "a158_064",
    "ta_031": "a158_060",
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
        "evaluation_start": "2020-01-02",
        "evaluation_end": "2026-10-21",
        "qlib_fingerprint": "4c48525d91abd4f4081a48179015fd9a9de899fed0158ef0b4db3a9880c009b8",
        "deduplicated": DEDUPLICATED,
        "evaluations": (
            {
                "factor_set": "qlib_alpha158",
                "run_id": "20261024T095112Z-33c4d56a",
                "factor_fingerprint": "71aafb5d55a082d3eab4ca7bdc2c90ec0d9d24c08216a25468bb2285ac7e2a6e",
            },
            {
                "factor_set": "wq_alpha101",
                "run_id": "20261024T095254Z-260dc27f",
                "factor_fingerprint": "21a0f59aaa2616eb5b494657e83582dc155f8fa9cf46f686efbdde0b0c435029",
            },
            {
                "factor_set": "gtja191",
                "run_id": "20261024T095428Z-48fef8b5",
                "factor_fingerprint": "bc9d664f354f5cf78265f528eca208e5b85e69963cdaec2e2989df19a6a459d5",
            },
            {
                "factor_set": "qlib_alpha360",
                "run_id": "20261024T095610Z-9d26f205",
                "factor_fingerprint": "d6cff8fc9b2f68f6cf605fd45a4a6ec3fd05859e81e310939b44a7e162200511",
            },
            {
                "factor_set": "china_chars",
                "run_id": "20261024T095827Z-764fcd8d",
                "factor_fingerprint": "7c3ba1d9cb85222fd295cd1222166987b88184e579198add02595dcf0f054a60",
            },
            {
                "factor_set": "size_value_china",
                "run_id": "20261024T095956Z-18081ff5",
                "factor_fingerprint": "967bf775908bcb969b03d11fd9df9e5e3fc3362aab7765800465ab3ecba54773",
            },
            {
                "factor_set": "talib",
                "run_id": "20261024T100110Z-5f721730",
                "factor_fingerprint": "d6801ecdeb333fb69b0e1d3b89ca1d1f86d1950fedfc1a93c05754af4bed2ebf",
            },
        ),
    },
)
