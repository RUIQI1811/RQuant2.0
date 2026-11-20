# 日内择时下单回测（qlib NestedExecutor + Tushare 1 分钟线）

目的：日频选股不变（walk-forward 的预测 + TopkDropout），只改「当天什么时候、按什么价格成交」，
看开盘后一段时间 TWAP/VWAP 买卖是否好过开盘价成交。2022 年以后的超额主要丢在买入当天开盘到收盘这一段
（`docs/crowding-diagnostics.md`，PR #32），改成收盘买已经测过没有提升，所以预期不要太高。

## 做法

- 外层：qlib `NestedExecutor`，每天一步，`TopkDropoutStrategy`（和 `rquant report` 相同的成本、100 股、最后一天清仓）。
  T+1 不用另外补：外层每天只决策一次、`hold_thresh=1`，内层只执行当天的单子，不会当天买当天卖。
- 内层：每 1 分钟一步，`WindowTWAPStrategy` 把卖单平均分到卖出时间窗、买单平均分到买入时间窗，按每根 bar 的 VWAP 成交。
  这根 bar 收盘价在涨停价就买不进、在跌停价就卖不出，这一份顺延到窗口里后面的 bar，窗口结束还没成交的当天作废。
  下单股数按当天第一根 bar 的价格算（日频回测按开盘价算）。
- 时间窗写 bar 的开始时间，两端都含：`09:30-09:59` 是开盘后半小时；`09:30-09:30` 只有第一根，
  里面是开盘集合竞价加 09:30–09:31 的连续竞价；`14:59-14:59` 是收盘集合竞价（14:57–15:00）。
  买入窗口比卖出窗口早时，买单只能用手里的现金（卖出的钱还没回来），会买不满。
- 从 `--start` 那天起步时，默认接着日频回测前一天收盘的持仓和现金（run 的 `positions.parquet`、`portfolio.parquet`），
  两本账持有同样的股票，按日频账下的分钟线才能覆盖日内的账。`--from-empty` 改成从空仓起步。
  2026-11-20 第一次基准回测是从空仓起步的：持仓很快和日频不同，43% 的持仓日没有分钟线、卖不掉，结果作废。
- 某只股票某天还是没有分钟线时，这一天按日线处理：整笔在 09:30 按日线开盘价成交（开盘涨跌停照日频规则不能买卖），
  按日线收盘价估值。`intraday.json` 里 `uncovered_holding_days` / `uncovered_share` 是这样的持仓日数和占比，
  `fallback_trades` / `fallback_trade_value_share` 是这样成交的笔数和金额占比；`execution_vs_open_bp` 只算真正用分钟线成交的单子。
  占比高时，用 `plan --book <intraday 输出目录>` / `sync --book ...` 把日内账自己的持仓和交易也补下分钟线，再重跑。
- qlib 嵌套执行器只在内层账户记成本和换手，外层日报的 `cost`、`turnover` 是 0（收益已经扣了成本）。
  这里按成交记录补回 `cost`、`turnover`，`return` 改回未扣成本的收益，`return - cost` 和 `rquant report` 一样是净收益。
- 分钟行情不写成 qlib 的分钟 bin（全市场要几十 GB），而是每月一个 parquet（`data/minutes/1min/month=YYYYMM/bars.parquet`），
  回测一次跑一个月，月末把现金、持仓、最新价、持有天数带到下个月，每次只载入这个月有分钟线的股票（几百只，内存几百 MB）。
- 价格用日线同一个 `factor` 复权；成交量单位是股、成交额单位是元（2026-11-19 在 Windows 上对过 600000.SH，分钟加总和日线完全一致）。
  Tushare 的 bar 用结束时间标记：09:30 那根是开盘集合竞价，合并进 qlib 的 09:30 bar；15:00 那根是收盘集合竞价，对应 qlib 的 14:59 bar。

## 只下需要的分钟线

`plan` 按 run 的预测和日频回测的持仓算出需要的股票日：每天前一天预测排名前 `--top`（默认 200）的股票，
加上日频回测当天和前一天持有、当天交易的股票。每只股票每个月请求一次（约 5,500 行，Tushare 一页 8,000 行）。
嵌套回测里的持仓和日频回测会有少量差别；如果某天持仓没有分钟线，`intraday.json` 的 `uncovered_holding_days` 会列出来。

量级（沪深主板 main，2020 年以后）：估计每月几百只股票，总共几万次请求；按默认每分钟 190 次约几个小时，
parquet 几 GB。准确数字以 `plan` 的输出为准。

## Windows 上的命令

在项目根目录、rquant 环境里，用 PR 分支的代码（worktree + `PYTHONPATH=<worktree>\src`，`--root D:\自己瞎搞\RQuant2.0`）。
`<run>` 是 walk-forward run 目录，比如 main jq240 的 `runs\walk-forward\20261119T182443Z-30e72d4e`；先用同样的 top-k 跑过 `rquant report`。

```powershell
# 1. 看要下多少（不调接口）
python scripts\analysis\intraday_execution.py --root D:\自己瞎搞\RQuant2.0 plan --run <run> --start 2020-01-01
# 2. 先试 50 次请求，再去掉 --limit 下全部（有缓存，中断后重跑会接着下）
python scripts\analysis\intraday_execution.py --root D:\自己瞎搞\RQuant2.0 sync --run <run> --start 2020-01-01 --limit 50
# 3. 整理成每月文件
python scripts\analysis\intraday_execution.py --root D:\自己瞎搞\RQuant2.0 build
# 4. 回测一个方案（每个方案写到 <run>\intraday\<标签>\）
python scripts\analysis\intraday_execution.py --root D:\自己瞎搞\RQuant2.0 backtest --run <run> --start 2020-01-02 `
    --sell-window 09:30-09:59 --buy-window 09:30-09:59
```

建议的对照组：

| 方案 | 卖出窗口 | 买入窗口 |
| --- | --- | --- |
| 接近开盘价（基准） | 09:30-09:30 | 09:30-09:30 |
| 开盘后半小时 | 09:30-09:59 | 09:30-09:59 |
| 上午 | 09:30-11:29 | 09:30-11:29 |
| 全天 | 09:30-14:59 | 09:30-14:59 |
| 早卖晚买 | 09:30-09:59 | 10:00-11:29 |
| 收盘集合竞价 | 14:59-14:59 | 14:59-14:59 |

基准方案应该和 `rquant report` 的日频结果接近（价格是第一根 bar 的 VWAP 而不是开盘价）；差得多说明数据或撮合有问题，要先查。
`intraday.json` 里 `execution_vs_open_bp` 是每笔成交价相对当天开盘价的偏离（基点，正数表示更差）。
`portfolio.parquet` 和 `rquant report` 的格式一样，可以用 `scripts/analysis/window_metrics.py` 算 2022 年以后的指标。

内存和速度（2026-11-20 Windows 实测，main jq240 top50 换 5，2020-01 至 2026-10）：峰值约 4.4 GB，每月约 20 秒，一个方案约 25 分钟。
跑之前看空闲内存。
