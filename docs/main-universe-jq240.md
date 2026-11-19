# jq240 扩到沪深主板（`main` 股票池）

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
