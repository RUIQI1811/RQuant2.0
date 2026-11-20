# Main-board blend parity run for PTrade's research notebook: builds the jq240 inputs for RQuant's ``main`` universe
# (Shanghai 60xxxx and Shenzhen 00xxxx) as of AS_OF from PTrade data, scores the day with the jq240 and the
# size_value_china LightGBM windows, blends them as the live strategy does, and compares all three with RQuant's own
# backtest scores for that day (EXPECTED, from the walk-forward runs). READ-ONLY: no orders. Results stay in RESULT.
AS_OF = "20261021"
DAYS = 1100
TOP_N = 50
TAG = "[RQ]"
OUT = []
RESULT = {}


def log(key, value=""):
    line = "%s %s=%s" % (TAG, key, value)
    OUT.append(line)
    print(line)


def compare(name, mine, expected):
    """Rank correlation, top-N overlap and the largest gap of ``mine`` against RQuant's scores of the same day."""
    mine = pd.Series(mine.values, index=[to_rquant(code) for code in mine.index])
    expected = pd.Series(expected, dtype=float)
    expected.index = [code[2:] + (".SH" if code.startswith("SH") else ".SZ") for code in expected.index]
    common = mine.index.intersection(expected.index)
    log("compare.%s.stocks" % name, "%d ours, %d RQuant, %d both" % (len(mine), len(expected), len(common)))
    if not len(common):
        return
    ours, theirs = mine[common], expected[common]
    log("compare.%s.rank_corr" % name, "%.4f" % ours.rank().corr(theirs.rank()))
    log("compare.%s.max_abs_gap" % name, "%.6g" % float((ours - theirs).abs().max()))
    top_ours, top_theirs = set(ranked(mine)[:TOP_N]), set(ranked(expected)[:TOP_N])
    log("compare.%s.top%d_overlap" % (name, TOP_N), len(top_ours & top_theirs))
    log("compare.%s.top%d_only_ours" % (name, TOP_N), ",".join(sorted(top_ours - top_theirs)))
    log("compare.%s.top%d_only_rquant" % (name, TOP_N), ",".join(sorted(top_theirs - top_ours)))


def run_parity():
    began = time.time()
    cache = {}
    inputs = build_inputs(AS_OF, DAYS, log, codes=main_codes(AS_OF), cache=cache)
    RESULT["inputs"] = inputs
    last = inputs["close"].index[-1]
    log("parity.date", last.strftime("%Y%m%d"))
    log("parity.universe", int(inputs["universe"].loc[last].sum()))
    blend, jq, svc = main_blend_scores(inputs, last, AS_OF, log, cache)
    RESULT.update(blend=blend, jq240=jq, svc=svc, cache=cache)
    log("score.top%d" % TOP_N, ",".join(to_rquant(code) for code in ranked(blend)[:TOP_N]))
    for name, mine in (("jq240", jq), ("svc", svc), ("blend", blend)):
        compare(name, mine, EXPECTED[name])
    log("parity.seconds", round(time.time() - began, 2))


log("parity", "start " + time.strftime("%Y-%m-%d %H:%M:%S"))
try:
    run_parity()
except Exception as exc:
    log("parity.error", "%s: %s" % (type(exc).__name__, exc))
    raise
log("parity", "end " + time.strftime("%Y-%m-%d %H:%M:%S"))
