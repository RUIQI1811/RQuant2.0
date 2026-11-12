# PTrade 实盘（国金证券，jq240 + LightGBM）

2026-11-11 起实盘从 QMT 迁到国金 PTrade。PTrade 的策略跑在券商服务器上，电脑关机也照常运行。
服务器上是 Python 3.11、numpy 1.23、pandas 1.5，没有 lightgbm，所以和 QMT 一样用 `qmt/runtime/lgb_numpy.py` 打分。
本目录的文件都由 `python qmt/build_jq240.py` 生成或打包，不属于 `rquant` 包。

## 规则

- 账号、密码只在 PTrade 客户端里登录，不写进本仓库的任何文件。日志里账号只显示后 4 位。
- `LIVE_DRY_RUN = True`（默认）只打印不下单。只有用户自己看过模拟输出后，才在 PTrade 里改成 `False`。
- PTrade 研究模块禁止 `os`、`sys` 等系统模块；生成的文件不含这些词，构建脚本会检查。

## 文件

| 文件 | 用途 |
| --- | --- |
| `probe_research.py` | 研究模块里跑的只读探测（库版本、取数接口和耗时） |
| `runtime/ptrade_inputs.py` | 用 PTrade 自带数据拼出 jq240 的输入（与 QMT 适配器口径一致，测试对齐） |
| `runtime/research_parity.py` → `generated/jq240_research_parity.py` | 研究模块里的对账：给定日期的 top30 |
| `runtime/ptrade_live.py` → `generated/jq240_ptrade_live.py` | 实盘策略（交易模块） |

实盘策略的下单规模和 QMT 完全是同一份代码：构建时从 `qmt/runtime/live_shell.py` 取出 `live_sell`、`live_buy`，
只把账户、行情快照、下单这几个接口换成 PTrade 的（`context.portfolio`、`get_snapshot`、`order`）。

## 对账结果（2026-11-12）

研究模块里用 2026-11-10 的数据打分，top30 与 QMT 11-11 实盘日志的 `live.top30` 是同一组股票（只有同分股票的先后不同）。
冷启动 1100 个交易日的输入要约 19 分钟，主要花在逐日的估值和 ST 状态查询上。
`jq_029`、`jq_051`（固定资产）曾经全空：PTrade 三季报的 `fixed_assets` 是空的，现在空时改用 `total_fixed_asset`（2026-11-12 修复）。

## 实盘策略的时间表（北京时间）

- 启动时、每天 08:00（`LIVE_PLAN_AT`）：用到上一交易日为止的数据打分。第一次约 20 分钟；之后逐日查询结果留在内存里，
  只补新的几天。09:10 的 `before_trading_start` 会在 08:00 没成功时再试一次。
- 09:21（`LIVE_AUCTION_AT`）：集合竞价下单。卖单挂跌停价，买单最高挂昨收 +3%，开盘价成交。
  买单只用竞价前的现金，并留出约 `LIVE_N_DROP` 份现金给第二天的竞价买入。
- 09:31 / 09:32：只有竞价时计划还没好才用，开盘后先卖后买，挂最新价 ±1%。
- 每天最多交易一次；账户当天已经有沪市主板委托（比如服务器重启后），当天就不再下单。
- 参数与 QMT 相同：top30、每天最多换 3 只、`LIVE_TOPUP = False`（闲钱只给当天新买的股票，每只最多 1.3 份）。

## 使用步骤

1. 停掉 QMT 里的实盘策略，避免两边同时下单。
2. PTrade → 量化 → 新建策略（股票、日线），把 `generated/jq240_ptrade_live.py` 全部粘贴进去，保存。
3. 新建交易，选这个策略启动。日志里会马上出现 `[RQ] live.top30=...` 和一组 `live.dry_sell` / `live.dry_buy`。
4. 第二天 09:21 再看一次模拟下单。确认没问题后，把 `LIVE_DRY_RUN = True` 改成 `False`，重新启动交易。
