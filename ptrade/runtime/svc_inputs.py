# size_value_china and the main-board blend for PTrade (Python 3.11 / numpy 1.23 / pandas 1.5).
# Runs after the jq240 core, the size_value_china reference (SizeValueChina), CH4FactorBuilder, ptrade_inputs.py and
# lgb_numpy.py in the bundled file (ptrade/build_main_blend.py). PTrade research rejects the system modules, so none
# are used.
#
# RQuant conventions reproduced here:
# * the CH-3/CH-4 factor returns are rquant.data.ch4_factors' CH4FactorBuilder fed day by day, as
#   rquant.data.reference feeds it from Tushare: each day's close / preclose - 1 of every A share with a traded bar,
#   the day's total market value (raw close x total shares; only ratios matter) and pe_ttm, and turnover as traded
#   shares over total shares. Listing dates are PTrade's; stocks delisted before the run are absent (as in the jq240
#   inputs' market return);
# * size_value_china reads Tushare's daily_basic units: total_mv in 10,000 yuan (cap / 1e4), pe_ttm and
#   turnover_total as a fraction; its panel only needs the last SVC_DAYS rows (250-day fits and turnover);
# * the blend is scripts/analysis/blend_runs.py's: each model's scores z-scored across the day's stocks (standard
#   deviation with ddof 1; a day whose scores are all equal counts 0), then averaged over the stocks both score.
SVC_DAYS = 260
CH4_STREAM_DAYS = 2 * CH4_YEAR_DAYS + 30  # 250 days of factor returns, each month formed on a year of history


def all_a_daily(codes, days, end, calendar, cache=None):
    """{"close", "ret", "volume"}: ``days`` x codes panels of every A share's traded daily bars (raw close, close over
    preclose - 1, shares). With ``cache`` only the days after the cached ones are read again."""
    key = ("all_a_daily",)
    kept = cache.get(key) if cache is not None else None
    old = None
    start = days[0]
    if kept is not None and len(kept["close"]) and kept["close"].index[0] <= days[0]:
        old = dict((name, panel[panel.index >= days[0]]) for name, panel in kept.items())
        later = days[days > old["close"].index[-1]]
        start = later[0] if len(later) else None
    fresh = {"close": {}, "ret": {}, "volume": {}}
    if start is not None:
        for code, index, bars in _bars(codes, ["close", "preclose", "volume"], start.strftime("%Y%m%d"), end,
                                       calendar):
            traded = _traded(bars)
            at = index[traded]
            close = np.asarray(bars["close"], dtype=float)[traded]
            pre_close = np.asarray(bars["preclose"], dtype=float)[traded]
            fresh["close"][code] = pd.Series(close, index=at)
            fresh["ret"][code] = pd.Series(close / np.where(pre_close > 0, pre_close, np.nan) - 1.0, index=at)
            fresh["volume"][code] = pd.Series(np.asarray(bars["volume"], dtype=float)[traded], index=at)
    panels = {}
    for name in fresh:
        panel = pd.DataFrame(fresh[name])
        if old is not None:
            panel = pd.concat([old[name], panel[panel.index > old[name].index[-1]] if len(panel) else panel])
        panels[name] = panel.reindex(index=days, columns=sorted(codes))
    if cache is not None:
        keep = _cacheable(days)
        cache[key] = dict((name, panel[panel.index.isin(keep)]) for name, panel in panels.items())
    return panels


def ch4_factor_returns(inputs, end, log=print, cache=None):
    """Daily CH-3/CH-4 factor returns (columns CH4_FACTORS and ch4_stocks) over the last CH4_STREAM_DAYS sessions up
    to the inputs' last day, from PTrade's all-A bars and valuation table."""
    began = time.time()
    calendar, every, listed = inputs["calendar"], inputs["all_a"], inputs["listed"]
    last = calendar.searchsorted(inputs["close"].index[-1])
    days = calendar[max(0, last - CH4_STREAM_DAYS + 1):last + 1]
    daily = all_a_daily(every, days, end, calendar, cache)
    codes = daily["close"].columns
    shares = valuation_panels(every, days, ["total_shares", "a_floats"], log, cache)["total_shares"]
    shares = shares.reindex(index=days, columns=codes)
    pe = valuation_panels(every, days, ["pe_ttm"], log, cache, positive=False)["pe_ttm"].reindex(
        index=days, columns=codes)
    log("svc.ch4_inputs_seconds", round(time.time() - began, 2))
    cap = daily["close"].ffill() * shares
    turnover = daily["volume"] / shares.where(shares > 0)
    builder = CH4FactorBuilder(dict((code, listed.get(code, pd.NaT)) for code in codes))
    rows = []
    for day in days:
        traded = daily["close"].loc[day].dropna().index
        state = pd.DataFrame({"total_mv": cap.loc[day], "pe_ttm": pe.loc[day]}).dropna(subset=["total_mv"])
        rows.append(builder.update(day, daily["ret"].loc[day].reindex(traded), state,
                                   turnover.loc[day].reindex(traded)))
    factors = pd.DataFrame(rows).set_index("datetime")
    log("svc.ch4_last", "%s %s" % (days[-1].strftime("%Y%m%d"), " ".join(
        "%s=%.5f" % (name, factors[name].iloc[-1]) for name in CH4_FACTORS)))
    log("svc.ch4_stocks_last", int(factors["ch4_stocks"].iloc[-1]))
    log("svc.ch4_seconds", round(time.time() - began, 2))
    return factors


def svc_inputs(inputs, factors):
    """size_value_china's input dict over the last SVC_DAYS rows of the jq240 inputs (the same stocks)."""
    tail = inputs["close"].index[-SVC_DAYS:]
    result = {
        "close": inputs["close"].reindex(tail),
        "basic_total_mv": inputs["cap"].reindex(tail) / 1e4,
        "basic_pe_ttm": inputs["basic_pe_ttm"].reindex(tail),
        "basic_turnover_total": inputs["basic_turnover_total"].reindex(tail),
    }
    for name in CH4_FACTORS:
        result["ch4_" + name] = factors[name].reindex(tail)
    return result


def svc_features(inputs, date, names, codes):
    """Rows x ``names`` (svc_NNN, the factor store's columns) for ``codes`` on ``date``."""
    calculator = SizeValueChina(inputs)
    columns = []
    for name in names:
        source = SIZE_VALUE_CHINA_NAMES[int(name.split("_")[1]) - 1]
        columns.append(calculator.calculate(source).loc[date].reindex(codes).values.astype(float))
    return np.column_stack(columns)


def model_scores(model, matrix, codes):
    prepared = prepare_model(model)
    return pd.Series(predict(prepared, normalise(matrix, prepared)), index=codes)


def blend_scores(members):
    """The mean daily z-score of the Series ``members`` over the codes all of them score."""
    joined = pd.concat([member.dropna() for member in members], axis=1, join="inner").sort_index()
    spread = joined.std(ddof=1).replace(0.0, np.nan)
    return ((joined - joined.mean()) / spread).fillna(0.0).mean(axis=1)


def main_blend_scores(inputs, date, end, log=print, cache=None):
    """(blend, jq240 scores, size_value_china scores) of the day's universe on ``date`` (the inputs' last day)."""
    began = time.time()
    member = inputs["universe"].loc[date]
    codes = sorted(code for code, inside in zip(member.index, member.values) if inside)
    jq = daily_scores(inputs, pd.DatetimeIndex([date]))[date]
    log("svc.jq240_seconds", round(time.time() - began, 2))
    factors = ch4_factor_returns(inputs, end, log, cache)
    svc = model_scores(SVC_MODEL, svc_features(svc_inputs(inputs, factors), date, SVC_MODEL["features"], codes), codes)
    log("svc.models", "%s year=%s trees=%d + %s year=%s trees=%d" % (
        MODEL["run"], MODEL["prediction_year"], len(MODEL["trees"]), SVC_MODEL["run"], SVC_MODEL["prediction_year"],
        len(SVC_MODEL["trees"])))
    log("svc.score_seconds", round(time.time() - began, 2))
    return blend_scores([jq, svc]), jq, svc
