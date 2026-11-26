# Main-board blend live strategy: the jq240 live strategy above (ptrade_live.py, built with LIVE_UNIVERSE = "main",
# LIVE_TOPK = 50, LIVE_N_DROP = 5 and no rank exit, as backtested) trading the blend of two LightGBM windows on
# RQuant's ``main`` universe (Shanghai 60xxxx and Shenzhen 00xxxx): jq240 (MODEL) and size_value_china (SVC_MODEL),
# each z-scored across the day's stocks and averaged (backtest main-blend-jq240-size_value_china, 2026-11-19: 200k
# top50 drop5 from 2020, 28.9% a year, Sharpe 1.01). Only the scoring differs from the jq240 strategy.


def plan_scores(inputs, last, clock):
    """The blend's scores for ``last``; the two models' top picks are logged too."""
    blend, jq, svc = main_blend_scores(inputs, last, clock[:8], rq_log, _pt["cache"])
    rq_log("live.top10_jq240", ",".join(to_rquant(code) for code in ranked(jq)[:10]))
    rq_log("live.top10_svc", ",".join(to_rquant(code) for code in ranked(svc)[:10]))
    rq_log("live.blend_stocks", len(blend))
    return blend
