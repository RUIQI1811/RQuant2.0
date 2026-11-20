# 因子集扩到沪深主板（`main` 股票池）

2026-11-18 用户提出：在 2022 年以后信号变弱的情况下，可选股票越多，头部选股越不靠运气，所以把股票池从沪市主板（`sh_main`，
60 开头，约 1800 只）扩到沪深主板（`main`，60 和 000/001/002/003 开头，约 3400 只）。创业板（300）和科创板（688）用户的账户
不能买，不包括。两个股票池的筛选规则相同：逐日剔除 ST、*ST、退市整理和上市不满 60 个交易日的股票（`src/rquant/universe.py`）。

## 比较方案

- 因子集 jq240，模型内置 `lgb`，1 日标签，每年重训，训练 3 年，和 `sh_main` 基线（`20261027T202921Z-41fb3dde`）完全相同，
  只换股票池。
- 研究口径 top50 每天最多换 5 只（100 万），实盘口径 20 万 top30 每天最多换 3 只、2022 年起步、最低佣金 5 元。
- 看全期、2013 至 2021 年和 2022 年以后；Rank IC 按年比。2022 年以后的数据已经用过多次，结果只能作参考。
- 如果要换实盘，PTrade 的打分和下单代码要从「只看 60 开头」改成同时看深市主板，并重新做一次一致性核对。

## 内存

旧的建库代码一次把全部输入（148 个字段 × 日期 × 股票）放进内存，`sh_main` 约 5 GB，加上打包和补齐的复制约 10 GB，
`main` 估计要 12 到 16 GB，在 16 GB 的笔记本上建不了。现在 `KunQuantFactorEngine.build_from_canonical` 把每个输入字段、
它的打包和补齐副本以及财报衍生的输入都写进临时目录里的内存映射文件（`_DiskArrays`），由操作系统按需换入换出；
`main` 的 jq240 要约 27 GB 输出加最多约 34 GB 输入的临时磁盘。计算结果和放在内存里时逐位相同（`tests/test_jq240.py`、
`tests/test_tushare_tech.py`）。

2026-11-19 在 Windows 上第一次建 `main` jq240 时，进程私有内存仍有 6 到 7.7 GB，写临时文件的脏页又堆到 7.8 GB，
空闲内存掉到 0，被内存保护停掉。原因和改法：

- 行情：原来整张长表（所有股票、所有年份，约两千万行）读进来再逐字段 pivot。现在 `_CanonicalBars` 先只读日期、代码和
  股票池字段，确定股票和日期，再逐个年度文件只读股票池里的股票，直接写进（磁盘上的）`日期 × 股票` 数组。
- daily_basic：jq240、oap、barra_cne6、jkp153、china_chars、size_value_china 原来把所有年份拼起来再 pivot，现在用
  `rquant.factors.daily_panels.daily_basic_panels` 逐个文件填进数组。
- 写临时文件：每个输入写完、每组因子算完都立即落盘（`flush`），脏页不再堆在内存里；不需要补齐时财报衍生输入也写到磁盘，
  不再留在内存。
- 写结果：每年的分区按约 60 个交易日一块写成一个 row group，不再一次复制整年 233 列。

在 1000 天 × 3421 只股票的合成数据上，峰值私有内存从约 4.8 GB 降到约 2.2 GB；按真实规模估计 `main` jq240 峰值约 4 GB。
新旧读法的结果完全相同（`tests/test_lean_loaders.py` 和 pivot 写法逐值对比）。

改完后在 Windows 上仍被停掉。用 `scripts/analysis/profile_factor_build.py`（按阶段记录私有内存和最大的分配位置）查到：
Windows 上 pyarrow 默认的 mimalloc 内存池释放后不还给系统，读完行情和财报后多占约 2.5 GiB。现在建库开始时切换到系统内存池
（`_use_system_arrow_pool`）。另外 Windows 的「空闲内存」会被内存映射文件的缓存占满（看着只剩 0.1 到 0.3 GB），但这部分随时可以腾出来，
所以队列的内存保护改看提交内存的余量。

2026-11-19 实测（commit 8ff5ac0）：`main` jq240 建库 16 分钟（其中计算 9 分钟、写结果 7 分钟；以前 `sh_main` 用旧代码要 2 小时 45 分，
主要在换页），工作集峰值 8.1 GB、提交内存 10 GB，提交余量一直在 15 GB 以上；因子库约 12 GB，2026 年每天约 3000 只股票。
walk-forward 全期 74 分钟，工作集峰值 5.7 GB。其他因子集建库 1 到 10 分钟，oap 47 分钟（财报类因子单线程）。

## 结果（2026-11-19，LGB 1d，8 线程，数据到 2026-10-21）

用户 2026-11-19 决定只测 `sh_main` 上 2022 年以后 Sharpe 不低于 0.8 的 8 个因子集，外加按收益挑的 `oap_jq_cnc_ic`；
jq240 和 size_value_china 是全期回测，其余从 2020 年开始（`--first-year 2020`，每年模型只用前 3 年训练，所以 2020 年以后的结果和全期
回测相同）。研究口径 top50 每天换 5，实盘口径 20 万 top30 每天换 3、2022 年起步。对比的是同一因子集在 `sh_main` 上的同一时间段；
实盘口径对比现在实盘的 `sh_main` jq240（32.7%/1.12/-35.1%）。Windows 上的完整表在 `runs\main_universe\results.md`。报告目录是 `<run>-top50-drop5` 和 `<run>-live200k`。

Sharpe（`main` / `sh_main`）：

| 因子集 | run（main） | 研究 2020 以后 | 研究 2022 以后 | 实盘 2022 以后（main） |
| --- | --- | --- | --- | --- |
| size_value_china | `20261119T194901Z-2a761dc5` | 0.85 / 0.76 | 1.16 / 0.91 | 1.22（39.4%，回撤 -42.9%） |
| barra_cne6 | `20261119T224540Z-97ba3a99` | 0.94 / 0.99 | 1.07 / 0.86 | 1.13（36.3%，-44.7%） |
| oap | `20261119T205324Z-dc3bc9f6` | 1.03 / 1.08 | 1.07 / 0.90 | 1.04（31.3%，-46.1%） |
| jkp153 | `20261119T232631Z-b948f1d0` | 0.96 / 1.01 | 1.07 / 0.85 | 1.03（31.4%，-48.0%） |
| oap_jq_cnc | `20261119T221608Z-74957352` | 1.02 / 1.07 | 0.98 / 0.89 | 0.90（26.5%，-46.4%） |
| oap_jq_cnc_ic（样本内） | `20261120T001653Z-66527c8e` | 0.82 / 1.07* | 0.90 / 0.89* | 0.75（19.9%，-42.5%） |
| jq240 | `20261119T182443Z-30e72d4e` | 1.04 / 1.11 | 0.93 / 1.03 | 0.69（18.4%，-35.1%） |
| gtja191 | `20261119T211929Z-de4d6ebe` | 0.75 / 0.96 | 0.64 / 0.89 | 0.30（4.6%，-46.9%） |
| wq_alpha101 | `20261119T235329Z-d07b2107` | 0.59 / 0.87 | 0.43 / 0.83 | 0.27（3.8%，-43.3%） |

\* 对比的是 `sh_main` 上的 oap_jq_cnc。

- jq240 全期 49.9%/1.47 对 39.0%/1.26，2013 至 2021 年 1.75 对 1.38；Rank IC 每年都更高（2022 年以后 0.050 对 0.042），
  但 2022 年以后组合收益没有更好，实盘口径每年都输。持仓里深市股票约一半。
- 偏基本面和风格的因子集（size_value_china、barra_cne6、oap、jkp153）2022 年以后研究口径比 `sh_main` 高 0.17 到 0.25；
  纯量价的（jq240、gtja191、wq_alpha101）都变差。从 2020 年算，除了 size_value_china，`main` 都略差。
- 实盘口径只有 size_value_china（1.22）和 barra_cne6（1.13）高于现在实盘的 1.12，差距在误差内（Sharpe 标准误约 ±0.45），
  回撤都更大（-43% 到 -48% 对 -35%），又是从 9 个结果里挑最好的。size_value_china 的领先主要来自 2026 年（0.77 对 0.10）。
- **实盘不换。**

### oap_jq_cnc_ic（按收益挑，样本内）

用户 2026-11-19 要求按收益重新挑 oap_jq_cnc，并在卡片上选了用 2020-01-01 至 2026-10-21 的数据挑（和回测期重叠，明知结果偏高）。
候选是 oap、jq240、china_chars 在 `main` 上的 484 列；覆盖率 ≥ 0.8、日均 Rank IC 绝对值 ≥ 0.01 的有 187 个；按 |Rank IC| 从高到低、
相关性达到 0.8 就跳过，剩 78 个（jq240 43、oap 25、china_chars 10；门槛 0.02 剩 44、0.03 剩 23）。排在前面的是特质波动率、ATR、
成交额标准差、换手率这类低波动、低换手因子。代码 `scripts/remove_duplicates/select_by_rank_ic.py`、
`src/rquant/factors/libraries/oap_jq_cnc_ic.py`。即使样本内挑选，它也不如不看收益的 oap_jq_cnc。

## 命令（Windows，数据截止 2026-10-21）

```powershell
$env:PYTHONPATH = "D:\自己瞎搞\RQuant2.0-pr32\src"
$R = "D:\自己瞎搞\RQuant2.0"
# 1. 补深市主板的财报（要 Tushare token，约半小时），再重建参考数据
python -m rquant --root $R --universe main data sync-fundamentals --through 2026-10-21
python -m rquant --root $R --universe main data build-reference
# 2. 建库和检查（电脑空闲时）
python -m rquant --root $R --universe main factors build --factor-set jq240
python -m rquant --root $R --universe main factors validate --factor-set jq240
# 3. 回测
python -m rquant --root $R --universe main walk-forward --model lgb --horizon 1d --factor-set jq240 --through 2026-10-21
python -m rquant --root $R --universe main report <run_id> --topk 50 --n-drop 5 --rebalance-days 1
```

重建参考数据以后，`fundamentals.parquet` 多了深市主板的行，文件哈希变了，`sh_main` 上读财报的因子库（`jq240`、`oap`、
`china_chars`、`barra_cne6`、`jkp153`、`oap_jq_cnc`）在 `factors validate` 时会提示参考数据过期；它们沪市的行没有变，
回测和实盘不受影响，以后需要时再重建。旧的参考数据备份在 `data\canonical\reference\_backup_sh_main_20261118\`。
