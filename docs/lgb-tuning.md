# jq240 LightGBM 调参（sh_main，1 日标签）

目标是让样本外表现真的变好，而不是把历史曲线调漂亮。所以所有比较都只用 **2013 至 2021 年**，2022 年以后的数据在
最后一步只看一次。2022 年以后的 Sharpe 标准误约 ±0.45，即使是 2013 至 2021 年，差距小于 0.1 左右也当作噪声。

## 为什么要调

内置 `lgb` 的参数是 Qlib 给 Alpha158 + 沪深 300 调的（L1 正则 206、L2 正则 581、210 个叶子、学习率 0.042），
`rolling.refit` 用验证段的 MSE 早停选轮数。在 jq240 + sh_main 上（run `20261101T210410Z-aada1bdd`）每年选出的轮数很不稳定：

| 预测年 | 2013 | 2014 | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 树数 | 441 | 415 | 229 | 292 | 741 | 617 | 201 | 260 | 144 | 22 | 150 | 95 | 26 | 8 |

2026 年的模型只有 8 棵树，平均每天 31% 的股票和别的股票同分（其他年份几乎没有）；实盘用的正是这个模型。
另外 `subsample: 0.8789` 没配 `bagging_freq`，LightGBM 实际上没有做行采样。

## 改了什么（代码）

- `rolling.early_stopping_metric: ic`：早停看验证段每天预测分数和标签的 Pearson 相关的均值，而不是 MSE。
  只支持省内存路径上的 `LGBModel`。
- 标签的截面标准化可以用 `CSRankNorm`（每天的百分位排名），涨跌停的极端收益不会主导训练。省内存路径和 Qlib
  `DatasetH` 逐位相同（`tests/test_models.py`）。
- `rquant walk-forward --seed N` 覆盖 `workflow.seed`。
- `scripts/analysis/blend_runs.py`：把几个 run 的预测按天 z-score 后平均，写成一个新 run，可以直接 `report`。

## 6 个候选配置（`config/workflows/`）

| 文件 | 树 | 正则 | 采样 | 学习率 / 最多轮数 / 早停 | 标签 |
| --- | --- | --- | --- | --- | --- |
| `jq240_lgb_base_ic.yaml` | 内置参数 | 内置参数 | 内置参数 | 0.042 / 1000 / 50 | z-score |
| `jq240_lgb_mid.yaml` | 63 叶、深 7、每叶至少 500 行 | L1 1、L2 50 | 列 0.6、行 0.8 每轮 | 0.02 / 2000 / 100 | z-score |
| `jq240_lgb_small.yaml` | 31 叶、深 6、每叶至少 1000 行 | L2 10 | 列 0.5、行 0.7 每轮 | 0.02 / 2000 / 100 | z-score |
| `*_rank.yaml` | 同上 | 同上 | 同上 | 同上 | 排名 |

全部用 IC 早停。这 6 个是事先定好的，不根据结果再加新配置；如果要加，先写进本文件再跑。

## 步骤和命令（Windows PowerShell）

每次运行前先看空闲内存，**至少 6 GB** 才开始；内存占用和现在的 LGB walk-forward 一样。

```powershell
(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB   # 空闲内存 GB
```

本 PR 合并前，用 worktree 跑它的代码（主 checkout 保持在 `main`）：

```powershell
cd D:\自己瞎搞\RQuant2.0
git fetch origin claude/project-thread-406dhr
git worktree add ..\RQuant2.0-tuning origin/claude/project-thread-406dhr
conda activate rquant
$env:PYTHONPATH = "D:\自己瞎搞\RQuant2.0-tuning\src"
$W = "D:\自己瞎搞\RQuant2.0-tuning\config\workflows"
$R = "D:\自己瞎搞\RQuant2.0"
```

### 第 1 步：6 个配置，只跑到 2021 年

```powershell
foreach ($name in "base_ic", "mid", "small", "base_ic_rank", "mid_rank", "small_rank") {
  python -m rquant --root $R --universe sh_main walk-forward --workflow "$W\jq240_lgb_$name.yaml" `
    --last-year 2021 --through 2026-10-21
}
```

每个 run 结束时在输出里给出 run id。逐个出报告（研究口径 top50、每天最多换 5 只）：

```powershell
python -m rquant --root $R --universe sh_main report <run_id> --topk 50 --n-drop 5 --rebalance-days 1
```

和基线对比 2013 至 2021 年（基线用已有的全期 run 截到 2021 年）：

```powershell
python D:\自己瞎搞\RQuant2.0-tuning\scripts\analysis\window_metrics.py $R\runs 2013-01-01 2021-12-31 `
  base=20261027T202921Z-41fb3dde-top50-drop5 base_ic=<run_id> mid=<run_id> small=<run_id> `
  base_ic_rank=<run_id> mid_rank=<run_id> small_rank=<run_id>
```

**选择规则（事先定好）**：2013 至 2021 年 Sharpe 最高的配置胜出；和第二名差距小于 0.1 时选报告里 Rank IC 更高的。
如果没有一个比基线高 0.1 以上，就保留内置参数，只做第 2 步的多种子平均。

预计每个配置 30 到 90 分钟（基线全期 14 年约 35 分钟；新配置学习率更低、轮数更多），6 个合计约半天，适合晚上跑。

### 第 2 步：胜出配置的多种子平均（仍只到 2021 年）

```powershell
foreach ($seed in 43, 44, 45, 46) {
  python -m rquant --root $R --universe sh_main walk-forward --workflow "$W\jq240_lgb_<胜出>.yaml" `
    --last-year 2021 --through 2026-10-21 --seed $seed
}
python D:\自己瞎搞\RQuant2.0-tuning\scripts\analysis\blend_runs.py $R\runs <胜出>-5seeds-2021 `
  <seed42 的 run_id> <seed43> <seed44> <seed45> <seed46>
python -m rquant --root $R --universe sh_main report <胜出>-5seeds-2021 --topk 50 --n-drop 5 --rebalance-days 1
```

5 个种子平均的 Sharpe 不低于单个种子时，最终配置用 5 个种子；否则用 1 个。

### 第 3 步：最终配置跑全期，只看一次 2022 年以后

```powershell
python -m rquant --root $R --universe sh_main walk-forward --workflow "$W\jq240_lgb_<胜出>.yaml" --through 2026-10-21
# 用 5 个种子时再跑 --seed 43..46，并用 blend_runs.py 平均
python -m rquant --root $R --universe sh_main report <run_id> --topk 50 --n-drop 5 --rebalance-days 1
python D:\自己瞎搞\RQuant2.0-tuning\scripts\analysis\window_metrics.py $R\runs 2022-01-01 `
  base=20261027T202921Z-41fb3dde-top50-drop5 new=<run_id>
```

`report` 会覆盖 run 目录里的回测结果；要出实盘口径（top30、每天最多换 3 只、20 万）的报告，先复制目录再跑。

## 第 1 步结果（2026-11-08）

sh_main、jq240、1 日标签，top50 每天最多换 5 只，2013-01-01 至 2021-12-31，净收益。基线是
`20261027T202921Z-41fb3dde-top50-drop5` 截到 2021 年（它的目录里没有 IC 文件）。

| 配置 | run id | 年化 | Sharpe | 比基线 | 最大回撤 | Rank IC |
| --- | --- | --- | --- | --- | --- | --- |
| `base_ic` | `20261108T092004Z-ac9bfa3f` | 49.3% | 1.481 | +0.097 | -46.3% | 0.090 |
| `mid_rank` | `20261108T110025Z-a188bc08` | 47.8% | 1.468 | +0.084 | -45.8% | 0.103 |
| `small_rank` | `20261108T113208Z-46930145` | 46.8% | 1.446 | +0.062 | -46.8% | 0.101 |
| 基线 | `20261027T202921Z-41fb3dde-top50-drop5` | 44.8% | 1.384 | | -46.4% | |
| `mid` | `20261108T094438Z-9104226d` | 45.6% | 1.369 | -0.015 | -46.3% | 0.085 |
| `base_ic_rank` | `20261108T102514Z-ae79fb73` | 43.2% | 1.367 | -0.017 | -45.1% | 0.103 |
| `small` | `20261108T100436Z-7f2b8d25` | 41.9% | 1.281 | -0.103 | -48.1% | 0.084 |

- 按规则没有配置胜出（最好的 `base_ic` 只高 0.097，不到 0.1），保留内置参数。
- 排名标签在三组参数上都把 Rank IC 从 0.084 至 0.090 提到 0.101 至 0.103，但 Sharpe 没有同样的提升。
- 换手（约 20.7%/天）、成本（约 4.2%/年）和回撤几乎不变。
- IC 早停选出的树数大多比 MSE 早停多（`base_ic` 2013 至 2021 年 195 到 877 棵，基线 144 到 741 棵）；`small_rank` 2018 年到了
  2000 轮上限。
- 每个配置 20 到 35 分钟；内存峰值工作集约 3.9 GB、提交内存约 5.9 GB（在 `training rows binned` 这一步），空闲 5 GB 就能跑。

## 实盘（QMT）影响

- 换参数和排名标签：只需要用新的 2026 年窗口模型重新导出 `qmt/models/*.json`（PR #28 的 `qmt/export_model.py`），
  numpy 打分代码不用改。行采样不影响打分。
- 多种子：要导出 5 个模型，QMT 里把 5 个模型的分数按天 z-score 后平均，需要在 PR #28 的打分代码里加一小段，
  并重新做一次一致性核对。
