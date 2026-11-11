# RQuant2.0 项目进度（共享记忆）

最后更新：2026-11-11（美西时间）。

这份文件是 Claude 和 GPT/Codex 等 agent 共同读取的项目进度。规则见文末「怎么维护」。
研究口径、目录职责和安全边界以 `agent.md` 为准；本文件只记录「现在做到哪了」。

## 1. 项目和机器

- 基于 qlib 的 A 股日频量化系统。流程：`doctor` → `data sync`（Tushare）→ `data build-qlib` →
  `factors build/validate` → `walk-forward` → `report`。
- Windows 笔记本（`D:\自己瞎搞\RQuant2.0`）常开在家，是唯一的计算机器：ROG Zephyrus G16，24 线程，16 GB 内存（焊死，不能加）。
  内存是瓶颈，用户开游戏或桌面应用时空闲常不足 1 GB。**任何运行前先看空闲内存**；神经网络、walk-forward、
  甚至 NN 的 pytest 都要至少 6 GB 空闲，否则会卡死。
- Windows 主 checkout（`D:\自己瞎搞\RQuant2.0`）从 2026-11-08 起在 `main` 上。原来 `codex/rquant-subagent-rules` 上的未提交改动
  已存成那个分支上的本地提交 `f0891dd`（没有推送）。其他旧 worktree（`.claude\worktrees`、`RQuant2.0-oap` 等）可能还有未提交改动，不要删。
  跑某个未合并 PR 的代码用 worktree + `PYTHONPATH=<worktree>\src` + `--root D:\自己瞎搞\RQuant2.0`。
- Mac 用来讨论和审阅，不跑重任务。
- 用户在美西时间（UTC-8）。A 股时间是北京时间，两者相差 16 小时；给用户写时间时注明时区。
- 用户用中文交流。

## 2. 默认设置（实际在用的，和 `config/default.yaml` 不完全一样）

| 项 | 实际在用 | `config/default.yaml` 当前值 |
| --- | --- | --- |
| 因子集 | `jq240`（233 列） | `jq240` |
| 股票池 | `sh_main`（沪市主板 60 开头，逐日剔除 ST 和上市不满 60 个交易日） | `csi300`，运行时加 `--universe sh_main` |
| 模型 | LightGBM，1 日标签（下一开盘到再下一开盘） | 同 |
| 组合 | 每天调仓 TopkDropout：研究用 top50 每天最多换 5 只；实盘 top30 每天最多换 3 只 | top50、每 5 天全部重置（`n_drop 50`、`rebalance_days 5`） |

非 csi300 股票池的因子库在 `data/factors-<universe>/`（如 `data/factors-sh_main/jq240`）。

## 3. 因子库

- 全部 12 个公式库都用 KunQuant 实现（PR #17/#18/#19 已合并）：`qlib_alpha158`、`qlib_alpha360`、`wq_alpha101`、`gtja191`、
  `talib`、`china_chars`、`size_value_china`、`jq240`、`oap`、`barra_cne6`、`tushare_tech`、`jkp153`；另有选择型 `oap_jq_cnc`（254 列）。
- 按收益挑选的因子集（Hquality、recent、recent_sh_main、remove_duplicates）已按用户要求删除，以后也不要再按收益挑因子。
- 接入新因子集的步骤见 `docs/adding-factor-sets.md`。
- GPT/Codex 做的、还没合并的：`jkp406`（PR #25，364 列；362 列版本在 sh_main 上 2022 年以后 Sharpe 0.93，364 列版本因内存不足没建完），
  `china469_pit`（PR #27，18 列，单独回测很弱：全期 7.0%/0.39）。

## 4. 回测结论（sh_main，jq240，LGB 1d，2013-01 至 2026-10-21）

基准 run：`runs/walk-forward/20261027T202921Z-41fb3dde`（top50 每日换 5 的报告在 `...-top50-drop5`）。
qlib 原生化后的同一结果：`20261101T210410Z-aada1bdd`（预测逐位相同，带 feature importance）。

- top50 每天最多换 5 只：年化 39.0%，Sharpe 1.26，最大回撤 -46%；2022 年以后 28.8%/1.03/-35.5%。
- 每天调仓在 13 个因子集上都好于每 5 天全部重置；用 5 日模型配每 5 天调仓是备选（2022 年以后 0.87 对 1.03）。
- top30 到 top50 差不多，top10 明显更差。
- 2022 年以后 Sharpe：jq240 1.03 最好，其次 size_value_china 0.91、oap 0.90、gtja191 0.89、oap_jq_cnc 0.89。
- DoubleEnsemble 40.9%/1.30，和 LGB 的差别在噪声内，而且慢 8 倍；GRU 29.7%/1.07、ALSTM 28.4%/1.01 都不如 LGB。保留 LGB。
- 结构化风险模型（申万一级行业 + CNE6 风格）的指数增强只有 11.6%/0.66，不用。
- **实盘口径**（20 万、top30、每天最多换 3 只、最低佣金 5 元）：2022 年以后 32.7%/1.12/-35.1%；最低佣金每年拖累约 1%。
- 风险提示：2022 年以后的收益主要来自 2025 年；Sharpe 的标准误约 ±0.45；没有计冲击成本。
- 内置 LGB 参数（Qlib Alpha158 调的）在 jq240 + sh_main 上按 MSE 早停选出的树数很不稳定：2013 至 2021 年 144 到 741 棵，
  2022 年 22、2025 年 26、2026 年只有 8 棵（2026 年平均每天 31% 的股票同分）。调参方案（IC 早停、排名标签、多种子，
  只用 2013 至 2021 年挑参数）见 `docs/lgb-tuning.md`（PR #32）。第 1 步（2026-11-08，6 个配置，2013 至 2021 年，top50 每天换 5）
  没有配置比基线 Sharpe 1.38 高 0.1 以上：最好的是原参数加 IC 早停 1.48（`20261108T092004Z-ac9bfa3f`）；排名标签把 Rank IC
  从约 0.09 提到约 0.10，但 Sharpe 没有同样提升。按规则保留原参数。第 2 步（2026-11-09）`base_ic` 5 个种子 Sharpe
  1.42 至 1.50（均值 1.458，全部高于基线 1.384），5 个平均 1.458 不比单个好，最终用 1 个种子。第 3 步（全期，`20261109T055815Z-2a3c074e`）2022 年以后**没有提升**：研究口径
  Sharpe 0.96 对基线 1.03，实盘口径（20 万 top30 换 3，2022 年起步）0.92 对 1.12，差距在噪声内。2026 年模型从 8 棵树变成 90 棵、
  不再同分，但 2026 年收益两者差不多。实盘模型和内置 `lgb` 都不换；2022 年以后的数据已用过一次，不要再拿来调参。
- 拥挤诊断（2026-11-10，`docs/crowding-diagnostics.md`，run `aada1bdd`，只读）：模型 Rank IC 从 2013 年 0.124 逐年降到 2022 年以后 0.042，大中小盘同时减半；top50 的损失几乎全在买入当天开盘到收盘（24 bp/天降到 0.6），持有隔夜那段没变，买入前的隔夜跳空还在且分数对它的 IC 变大；jq240 单因子平均 IC 只掉 12%（量价类掉约四分之一）。和拥挤（别人更早按同样信号交易）一致，不像小盘风格崩盘。用现有预测回测了两个办法（2026-11-10，研究口径 top50 换 5）：开盘跳空超过 2% 或 3% 不买，2013 至 2021 年 Sharpe 1.42/1.43 对 1.38，没过 +0.1 门槛，2022 年以后反而 0.82/0.88 对 1.03（实盘竞价买入限价前收盘 +3% 可能有同样的拖累）；收盘价成交（上限）2013 至 2021 年 1.64，2022 年以后 1.02 对 1.03，没有提升。两个都不用。
- LGB jq240 sh_main walk-forward 的实测内存峰值：工作集约 3.9 GB、提交内存约 5.9 GB，空闲 5 GB 就能跑（用户 2026-11-08 选了 5 GB 门槛）。

## 5. 实盘（国金证券 QMT，PR #28 草稿）

- miniQMT/xtquant 外接在关停，大 QMT 内置 Python 从 2026-10-23 起禁止读写文件和联网，所以整个策略（因子计算 + numpy 版 LightGBM 打分）
  打包成一个文件粘贴进 QMT 内置 Python 3.6 运行。源码在 `qmt/`，说明在 `qmt/README.md`（都在 PR #28 分支上）。
- 模型：2026 年窗口（训练 2023-01-03 至 2025-12-30）导出到 `qmt/models/lgb_jq240_sh_main.json`，只有 8 棵树，很多股票同分，同分按代码排序。
- 一致性核对已通过（2026-11-05）：2026-10-21 的 top30/top50 和回测完全相同，排序相关系数 0.997。
  QMT 没有折旧摊销数据（EBITDA 实际是 EBIT），这两个因子重要性约 0.1%，可以接受。
- 实盘脚本：Windows 上 `D:\自己瞎搞\RQuant2.0\qmt_jq240_live.py`（由 `qmt/generated/jq240_live.py` 复制）。
  `LIVE_DRY_RUN = True` 只打印，`False` 真实下单。账号只在 QMT 编辑器里填，不进仓库。
- **2026-11-06 第一次实盘**：北京时间 13:00:27 下了 28 笔买单，全部成交，成本 113,388 元，剩约 3.6 万现金；
  603124、603986 一手超过单只额度，没买。当天是临时把下单时间改成 13:00 跑的。
- **接下来**：从 2026-11-09（周一）起，每个交易日北京时间 09:00 前（美西前一天 17:00 前）以「实盘交易」模式启动，
  09:30:05 卖、09:30:25 买，10:00 以后不再下单，每天只下一次。启动前要确认下单时间已改回 `093005` / `093025` / `100000`。
  计划先跑两周，再和回测对比。
- 每天第一次运行会先补最近 10 天日线；最新一天不到约 90% 的股票有日线就报 `live.error`、不下单。
- QMT 日志：`D:\国金证券QMT交易端\userdata\log\XtClient_FormulaOutput_YYYYMMDD.log`（GBK 编码）。
- 硬规则：没有用户明确确认，任何 agent 都不能下真实订单或改 `LIVE_DRY_RUN`。

## 6. 打开的 PR

| PR | 内容 | 状态 |
| --- | --- | --- |
| #28 | QMT 实盘（探测、一致性核对、模拟盘、实盘策略） | 草稿，实盘在用 |
| #32 | jq240 LightGBM 调参：IC 早停、排名标签、`--seed`、多 run 平均脚本、6 个候选配置；季度重训（`rolling.retrain_months`）和近期加权（`rolling.sample_half_life_years`）及 3 个配置 | 草稿，调参 3 步都跑完：2022 年以后没有提升，实盘不换；季度重训 + 近期加权 2026-11-11 开始在 2013 至 2021 年上比较（方案见 `docs/lgb-tuning.md`）；另含拥挤诊断脚本 `scripts/analysis/crowding_diagnostics.py` |
| #25 / #27 | GPT 做的 jkp406 / china469_pit | 草稿 |
| #26 | Codex 子智能体分工规则（文档） | 草稿 |
| #15 | RD-Agent 修复 | 用户要求先不合并 |

已合并的重要 PR：#29（2026-11-08，qlib 原生化：walk-forward 用 RollingGen + TrainerR + RollingEnsemble，模型写成
`src/rquant/workflow/workflows/` 下的 qrun YAML，报告用 PortAnaRecord、年化按 252 天；另有 DoubleEnsemble、qlib PyTorch 模型、
`report --signal-only`、`report --strategy enhanced-indexing` 和结构化风险模型）、#30（本文件）、#22（oap_jq_cnc、默认因子集 jq240）、#23（`scripts/analysis/window_metrics.py`，从已保存的 run 算任意时间段指标）、
#24（`AGENTS.md` 和 `docs/adding-factor-sets.md`）。

## 7. 其他结论

- RD-Agent 自动挖因子能跑通（WSL2 + 通义千问，PR #14 已合并），三次运行只重新发现了 Alpha158 类因子，暂停。
- FinRL 不建议用；以后要试强化学习，用 qlib.rl。
- 日内数据：Tushare `stk_mins` 足够做沪深 300 分钟级研究，还没开始。
- `report` 会覆盖原 run 目录；换参数重跑报告前先复制目录（如 `<run>-top30-drop3`）。

## 怎么维护

- 项目状态变化时（PR 合并或关闭、新的回测结论、实盘操作或参数改变、默认设置改变），完成任务的 agent 要在同一个 PR 里更新本文件，
  并改「最后更新」日期。
- 只写当前状态和结论，不写过程；旧内容过时就直接改掉或删掉。
- 每条结论写清楚口径（股票池、因子集、模型、组合、时间段）和 run id。
- 不写 token、账号、密码、资金账号或持仓明细。
