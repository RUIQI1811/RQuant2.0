# jq240 parity run for PTrade's research notebook: builds the jq240 inputs for sh_main as of AS_OF from PTrade data,
# computes every factor for the last day, scores the universe with the exported LightGBM window and prints the top
# picks, to compare with the QMT live strategy's live.top30 and RQuant's own scores for the same day.
# READ-ONLY: no orders. Results stay in the notebook variable RESULT for follow-up cells.
AS_OF = "20261110"
DAYS = 1100
TOP_N = 30
TAG = "[RQ]"
OUT = []
RESULT = {}


def log(key, value=""):
    line = "%s %s=%s" % (TAG, key, value)
    OUT.append(line)
    print(line)


def run_parity():
    began = time.time()
    inputs = build_inputs(AS_OF, DAYS, log)
    RESULT["inputs"] = inputs
    calculator = JQFactors(inputs)
    member = inputs["universe"].iloc[-1]
    codes = [code for code, inside in zip(member.index, member.values) if inside]
    log("parity.date", inputs["close"].index[-1].strftime("%Y%m%d"))
    log("parity.universe", len(codes))
    last, failed, sparse = {}, [], []
    for ordinal, name in enumerate(JQ240_NAMES, 1):
        key = "jq_%03d" % ordinal
        try:
            values = calculator.calculate(name).iloc[-1].reindex(codes).values.astype(float)
        except Exception as exc:
            failed.append("%s(%s): %s" % (key, name, exc))
            continue
        last[key] = values
        if np.isfinite(values).mean() < 0.5:
            sparse.append("%s(%s) %.0f%%" % (key, name, 100 * np.isfinite(values).mean()))
    log("parity.factor_errors", len(failed))
    for line in failed[:20]:
        log("parity.factor_error", line)
    log("parity.sparse_factors", "; ".join(sparse) if sparse else "none")
    log("parity.factor_seconds", round(time.time() - began, 2))
    RESULT["codes"] = codes
    RESULT["factors"] = pd.DataFrame(last, index=codes)
    prepared = prepare_model(MODEL)
    missing = [name for name in prepared["features"] if name not in last]
    if missing:
        log("score.missing_factors", ",".join(missing))
    nan = np.full(len(codes), np.nan)
    matrix = np.column_stack([last.get(name, nan) for name in prepared["features"]])
    scores = predict(prepared, normalise(matrix, prepared))
    RESULT["scores"] = pd.Series(scores, index=[to_rquant(code) for code in codes])
    log("score.model", "%s year=%s trees=%d" % (MODEL["run"], MODEL["prediction_year"], len(prepared["trees"])))
    order = np.argsort(-scores, kind="mergesort")
    log("score.top%d" % TOP_N, ",".join(to_rquant(codes[i]) for i in order[:TOP_N]))
    log("score.top50_values", ",".join("%s:%.6g" % (to_rquant(codes[i]), scores[i]) for i in order[:50]))
    log("score.stats", "mean %.6g std %.6g min %.6g max %.6g" % (scores.mean(), scores.std(), scores.min(),
                                                                  scores.max()))
    log("parity.seconds", round(time.time() - began, 2))


log("parity", "start " + time.strftime("%Y-%m-%d %H:%M:%S"))
try:
    run_parity()
except Exception as exc:
    log("parity.error", "%s: %s" % (type(exc).__name__, exc))
    raise
log("parity", "end " + time.strftime("%Y-%m-%d %H:%M:%S"))
