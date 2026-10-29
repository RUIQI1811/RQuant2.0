# RQuant

RQuant 是一个个人、非商业用途的 A 股日频量化研究框架，使用：

- Tushare 作为唯一市场数据源；
- Qlib 管理数据集、模型、实验记录和组合回测；
- KunQuant 编译并计算全部十四个公式因子集；oap_jq_cnc 从已构建的源因子集选列。

完整工作流概览

一次完整研究按以下顺序执行：

```text
创建本地环境
  → 检查依赖与 Tushare 权限
  → 同步原始数据
  → 构建标准化数据和 Qlib provider
  → 构建并验证因子
  → 执行逐年滚动训练与样本外预测
  → 运行受约束的多头组合回测
  → 生成研究报告并审查运行产物
```

对应命令如下。第一次使用时不要一次性全部执行，请按本文各步骤检查上一步产物后再继续：

```bash
qlib_alpha158
wq_alpha101
gtja191
qlib_alpha360

rquant doctor
rquant data sync --through 2026-08-04
rquant data build-qlib
rquant factors build --factor-set qlib_alpha158
rquant factors build --factor-set wq_alpha101
rquant factors build --factor-set gtja191
rquant data sync-fundamentals --through 2026-08-04   # china_chars、jq240、oap、barra_cne6、jkp153 需要
rquant data build-reference
rquant factors build --factor-set china_chars
rquant factors build --factor-set size_value_china    # 只需 build-reference，不需要财报
rquant factors build --factor-set jq240
rquant factors build --factor-set oap
rquant factors build --factor-set barra_cne6
rquant factors build --factor-set tushare_tech        # 只用 OHLCV
rquant factors build --factor-set jkp153
rquant factors build --factor-set jkp406
rquant factors evaluate --factor-set qlib_alpha158 \
  --horizon 1d --start 2010-01-01 --end 2026-08-04
  
rquant walk-forward --model lgb --horizon 1d --factor-set jq240 \
  --first-year 2013 --last-year 2026 --through 2026-08-04
  
rquant report RUN_ID
```

数据同步、Qlib 构建、全量因子计算、滚动训练和回测都可能耗时较长。运行前应确认参数、剩余磁盘空间和输出位置。

## 1. 环境要求

需要：

- macOS 或兼容的类 Unix 环境；Windows 亦可运行（编译器见下一条）；
- Conda；
- Python 3.11；
- `clang++`（Windows 上为 MSVC `cl.exe`，即 Visual Studio Build Tools 的“使用 C++ 的桌面开发”工作负载）；
- 可用的 Tushare token，以及本项目所需接口权限。

项目只使用名为 `rquant` 的 Conda 环境，不要复用 `base`、`stocktrade`、旧版 RQuant 或其他项目的环境。

在项目根目录创建环境：

```bash
conda env create --name rquant --file environment.yml
conda activate rquant
```

`environment.yml` 负责安装 Python、pip 和 Apple Silicon 所需的 OpenMP；直接 Python 依赖及版本范围由
`pyproject.toml` 管理。

如果 `environment.yml` 已更新，使用以下命令更新现有环境：

```bash
conda env update --name rquant --file environment.yml --prune
conda activate rquant
```

确认解释器和可编辑安装确实来自当前工作区：

```bash
python --version
python -c "import os, sys, rquant; print(os.environ.get('CONDA_DEFAULT_ENV')); print(sys.executable); print(rquant.__file__)"
```

预期结果：

- Python 版本为 3.11；
- `CONDA_DEFAULT_ENV` 为 `rquant`；
- `sys.executable` 位于 Conda 的 `rquant` 环境中；
- `rquant.__file__` 位于当前项目的 `src/rquant/`。

## 2. 配置 Tushare

token 只放在环境变量中，不要写入源码、YAML、日志或提交记录：

```bash
export TUSHARE_TOKEN='你的 token'
```

执行系统检查：

```bash
rquant doctor
```

`doctor` 会检查：

- Python 是否为 3.11；
- `pyqlib==0.9.7`、`KunQuant==0.1.11`、`tushare==1.4.29`、`TA-Lib==0.8.2`；
- NumPy、Pandas、PyArrow、LightGBM 等依赖能否导入；
- KunQuant 所用的 C++ 编译器（macOS 为 `clang++`，Windows 为 `cl.exe`）；
- Tushare token；
- 2063 个当前可构建因子的固定目录（Alpha158、Alpha101、GTJA191、Alpha360、china_chars、size_value_china、talib、jq240、oap、
  barra_cne6、tushare_tech、jkp153、jkp406、china469_pit）；
- 默认配置和本地数据清单；
- Tushare 所需接口的实际访问权限。

只想跳过联网权限探测时可以执行：

```bash
rquant doctor --skip-permission-check
```

该选项只跳过接口探测，`TUSHARE_TOKEN` 本身仍是必需检查项。`doctor` 返回 `status: ok` 后再开始数据同步。

## 3. 检查默认配置

默认配置位于 `config/default.yaml`。重要参数包括：

| 配置                               |         默认值 | 含义                         |
| ---------------------------------- | -------------: | ---------------------------- |
| `data.formal_start`              | `2010-01-01` | 正式研究起点                 |
| `data.warmup_trading_days`       |        `300` | 正式研究前预热交易日数       |
| `data.through`                   |         `null` | `data sync` 与 `walk-forward` 未传 `--through` 时使用的截止日 |
| `data.universe`                  |     `sh_main` | 默认沪市主板（逐日剔除 ST、上市不足 60 个交易日）；也支持 `csi300`、`main` 或 `all` |
| `data.max_requests_per_minute`   |        `190` | 所有请求及重试共享的分钟限速 |
| `data.sync_workers`              |          `8` | 逐日分区并发请求线程数，用于掩盖网络延迟；总速率仍受上面的限速约束 |
| `data.fundamentals_start_year`   |       `2004` | `data sync-fundamentals` 抓取财报的最早公告年份 |
| `data.fundamentals_window_years` |          `6` | 财报按公告日分窗抓取的年数；已结束的窗口会缓存复用 |
| `factors.factor_set`             |      `jq240` | 默认因子集                   |
| `factors.workers`                |          `4` | KunQuant 执行线程数          |
| `factors.graph_outputs`          |         `20` | 每个 KunQuant 图最多编译的因子数；每个中间结果都要一块“日期 × 股票”的缓冲区，分图计算可大幅降低内存，结果逐位不变；只作用于严格浮点的因子集（qlib_alpha158/360、wq_alpha101 始终一张图）；`0` 表示整个因子集一张图 |
| `workflow.train_years`           |          `3` | 每个预测年使用的历史训练年数 |
| `workflow.validation_months`     |          `6` | 训练窗口最后几个月用于选择迭代数 |
| `workflow.first_prediction_year` |       `2013` | 第一个样本外预测年           |
| `backtest.initial_capital`       |    `1000000` | 初始资金，人民币             |
| `backtest.topk`                  |         `50` | 目标持股数量                 |
| `backtest.n_drop`                |         `50` | 每次调仓最多换出数量；等于 `topk` 时全量调整到最新前 `topk` 名 |
| `backtest.rebalance_days`        |          `5` | 每隔多少个交易日调仓一次（从第一个预测日起算），其余交易日不下单 |
| `backtest.by_horizon`            | 见下文 | 按训练标签覆盖公共组合参数；命令行参数优先 |
| `backtest.impact_cost`           |        `0.0` | Qlib 二次冲击成本系数，按成交额占当日成交额比例的平方计费 |

如需使用其他配置文件，所有命令都可以在主命令前指定：

```bash
rquant --config config/你的配置.yaml doctor
```

运行时会把完整配置及其指纹写入 stdout 返回的 `run_directory/run.json`，因此不要依赖未记录的临时参数。

## 4. 同步原始数据

指定一个包含式截止日期：

```bash
rquant data sync --through 2026-08-04
```

同步过程具有以下行为：

- 为 2010 年正式研究期自动准备 300 个交易日的预热历史；
- 获取全 A 股日线及所需参考数据，同时保留历史时点沪深 300 成分信息；
- 按 `through` 日期逐只留存全部股票的历史名称（`namechange`，用于逐日判断 ST）。全市场一次性分页查询的排序不稳定、会漏行，
  所以每只股票单独请求，约 6000 次，按默认限流约 30 分钟；
- 按 `through` 日期留存申万行业快照，分页获取历史已失效记录（`is_new=N`）和当前有效记录（`is_new=Y`）；标准数据只使用最新完整快照，以保持行业时点覆盖并避免当前记录被永久缓存；
- 每个分区都带有内容哈希和清单；
- 已完成且哈希有效的分区会作为缓存复用；
- 损坏或不完整的分区会重新获取；
- 开市日的 `daily`、`adj_factor`、`daily_basic`、`stk_limit` 必须返回数据；若返回空表（通常是当日数据尚未发布），同步会报错且不缓存该分区，稍后用相同参数重跑即可；旧版本缓存的空分区会自动重新获取；
- 停牌不单独拉取接口：停牌日在 Tushare 日线中没有记录，Qlib provider 中对应为 NaN 行情，回测视为不可交易；
- 所有请求和重试共同遵守每 60 秒 190 次的默认滑动窗口限制；
- 分区进度、缓存命中数和新获取数显示在 stderr；最终 JSON 结果输出到 stdout。

非交互任务可关闭进度条：

```bash
rquant data sync --through 2026-08-04 --no-progress
```

只有明确需要重新获取所有已验证分区时才使用：

```bash
rquant data sync --through 2026-08-04 --force
```

同步完成后检查：

```bash
python -m json.tool data/raw/sync_manifest.json
ls -td runs/* | head
```

必须确认：

- `data/raw/sync_manifest.json` 的 `status` 为 `complete`；
- `through` 等于期望的截止日期；
- `artifacts = cached_artifacts + fetched_artifacts`；
- 对应 `runs/<run-id>/run.json` 的 `status` 为 `complete`。

原始分区位于 `data/raw/<endpoint>/...`。不要手工修改这些文件；需要修复时应重新执行正式同步流程。

## 5. 构建标准化数据与 Qlib provider

原始同步清单完成后执行：

```bash
rquant data build-qlib
```

该命令依次完成三项工作：

1. 将不可变 Tushare 分区转换为 `data/canonical/` 下的标准日线契约；
2. 在 `data/canonical/reference/` 下构建扩展参考数据（见下文“参考数据”）；
3. 从标准数据构建 `data/qlib/` 下的 Qlib provider。

标准数据会处理：

- Tushare 成交量从“手”转换为“股”；
- 成交额从“千元”转换为“元”；
- 以同口径成交额除以成交量计算 VWAP；
- 原始价格、复权价格、成交量与复权因子；
- 停牌缺口、开盘涨跌停限制；
- 历史时点沪深 300 成分；
- 历史时点 ST 状态（`is_st`，没有 `namechange` 历史时为空）与上市交易日数（`listed_days`）；
- 历史时点行业分类和市值。

如果 Tushare 返回的 OHLC 违反 `low <= open/close <= high`，属于当日历史沪深 300
成分的记录会使构建硬失败；研究股票池之外的异常记录会从标准数据中隔离，原始值和原因保存在
`data/canonical/quality/ohlc_exclusions.parquet`，数量和指纹记录在标准数据清单的 `quality` 字段中。

构建完成后检查：

```bash
python -m json.tool data/canonical/manifest.json
python -m json.tool data/qlib/manifest.json
```

两个清单都必须满足：

- `status` 为 `complete`；
- 标准数据的 `raw_fingerprint` 与原始同步结果一致；
- Qlib 的 `canonical_fingerprint` 与标准数据一致；
- 起止日期、行数、证券数量和字段符合预期。

### 参考数据与财务报表

`data/canonical/reference/` 中除沪深 300 日线、行业和成分快照外，还有供 `china_chars` 使用的扩展参考数据。
它们不进入标准日线列，也不改变标准数据指纹，所以构建它们不会让已有因子库过期：

- `daily_basic/year=YYYY/data.parquet`：全 A 股的换手率、流通/总市值、估值和未复权收盘价，以及
  `turnover_total`（成交股数 / 总股本）；
- `market_factors.parquet`：全 A 股（不含北交所、上市不足 30 天的新股）按前一日流通市值加权的市场收益，以及每月初按
  前一日总市值与 1/PB 分组的 Fama-French 2x3 SMB、HML 日收益；
- `ch4_factors.parquet`：Liu-Stambaugh-Yuan（2019）CH-3/CH-4 日收益 `mkt`、`smb`、`vmg`、`pmo`、`smb4`，
  口径见下文“Size and Value in China 因子集 size_value_china”；
- `fundamentals.parquet`：利润表、资产负债表、现金流量表合并报表的首次披露值，每只股票每个报告期一行，
  `ann_date` 为三张表都已公告的日期；
- `manifest.json`：上述文件的哈希、所依据的原始同步指纹和财报同步指纹。

### 研究股票池

| 股票池   | 每个交易日的成员 |
| -------- | ---------------- |
| `csi300` | 当日可知的沪深 300 成分股 |
| `main`   | 沪市 60 开头、深市 00 开头的主板 A 股（含原中小板 002/003），剔除创业板 30、科创板 68、北交所和 B 股；当日名称为 ST、\*ST 或“退”（退市整理）的股票，以及上市后前 60 个交易日的次新股，当日也不在池内 |
| `sh_main` | `main` 中只保留沪市 60 开头的股票，ST 与次新股剔除规则相同 |
| `all`    | 全部 A 股 |

ST 状态按 Tushare `namechange` 的历史名称逐日判断（生效日即 `start_date`），上市天数按交易日历从 `list_date` 起数。
规则定义在 `src/rquant/universe.py`。股票池逐日决定因子库发布哪些行，因此只影响训练样本与可买入的股票；
时间序列指标仍使用每只股票进池前的历史。

除 `csi300` 外，每个股票池的因子库写在各自目录（`main` 为 `data/factors-main/`，`sh_main` 为 `data/factors-sh_main/`），与沪深 300 的
`data/factors/` 并存。临时切换股票池用全局参数 `--universe`，它覆盖 `data.universe`：

```bash
rquant --universe main factors build --factor-set qlib_alpha158
rquant --universe main walk-forward --model lgb --horizon 1d --factor-set qlib_alpha158 --through 2026-10-23
rquant report RUN_ID   # 自动按该滚动训练的股票池回测
```

`walk-forward` 会检查因子库的股票池与当前股票池一致；`report` 从滚动训练清单读取股票池，与配置无关。
`--model lgb` 的每个窗口只在内存里保留一份面板：处理器直接改写加载好的 float32 块，验证与测试行先复制出来，
训练行原地挪到缓冲区前部交给 LightGBM（列主序，免复制），分箱后即释放面板（`src/rquant/workflow/lgb_window.py`）。
得到的树和预测与经 Qlib `DataHandlerLP`/`DatasetH`/`LGBModel` 的原流程逐位相同（`tests/test_models.py`）。合成的 300 列
四年窗口上峰值从 6.1 GB 降到 3.1 GB；剩下的峰值在分箱那一刻：面板、验证与测试副本，加上 LightGBM 为确定分箱抽样的
20 万行（672 列时约 1.6 GB，用完即释放）。sh_main 上 672 列的最大窗口一份面板约 4.2 GB；每个窗口各阶段的进程内存
写在日志里（`rquant.workflow.lgb_window`，需要 psutil）。标签由 Qlib 的多进程池计算（Windows 上 5 个子进程合计约 6.3 GB），
因此先算标签、再分配面板，两者不会同时占用内存。`double-ensemble` 仍走 Qlib 流程。
`main` 与 `sh_main` 需要 ST 历史：在此功能之前同步的数据须再执行一次 `rquant data sync`（只新增逐只的 `namechange`）与
`rquant data build-qlib`，否则构建 `main` 因子库会报错提示。业绩基准仍是 `data.benchmark`（默认沪深 300）。

财报需要单独同步（按股票、按公告日窗口请求单股接口，不需要 `*_vip` 批量接口权限）：

```bash
rquant data sync-fundamentals --through 2026-08-04
rquant data build-reference
```

`sync-fundamentals` 只抓取股票池可能包含的股票（`csi300` 为全部历史成分，`main` 为全部主板股票，`all` 为全部 A 股），结果清单写在
`data/raw/fundamentals_manifest.json`，不改动 `data sync` 的清单。已结束的公告日窗口不会再变，再次同步只重抓
截至 `--through` 的最后一个窗口。某个窗口返回达到 100 行时，Tushare 可能截断了结果，会自动按年拆分重抓。
`build-reference` 只重建参考数据，要求原始同步与标准数据一致；原始数据变化后应改跑 `build-qlib`。

重建 Qlib provider 时采用临时目录和原子替换；已有 provider 会保存为带时间戳的
`data/qlib.previous.<时间>/` 备份。确认新 provider 正常后再考虑清理旧备份。

## 6. 查看、构建和验证因子

### 查看固定因子目录

```bash
rquant factors catalog
rquant factors catalog --factor-set qlib_alpha158
rquant factors catalog --factor-set wq_alpha101
rquant factors catalog --factor-set gtja191
rquant factors catalog --factor-set qlib_alpha360
rquant factors catalog --factor-set jq240
```

导出机器可读目录：

```bash
rquant factors catalog --factor-set jq240 --format json --output factor_catalog.json
rquant factors catalog --factor-set jq240 --format csv --output factor_catalog.csv
```

因子集：

- `qlib_alpha158`：`a158_001` 至 `a158_158`，共 158 列；
- `wq_alpha101`：`a101_001` 至 `a101_101`，共 101 列；
- `gtja191`：`gtja_001` 至 `gtja_191`（暂时排除 `gtja_030`），共 190 列；
- `qlib_alpha360`：`a360_001` 至 `a360_360`，共 360 列；它是 Qlib Alpha360 的六组归一化
  OHLC、VWAP 与成交量字段各取 60 个滞后期，而非 360 条手工设计的独立信号；
- `china_chars`：`cnc_001` 至 `cnc_109`，共 109 列中国 A 股股票特征（流动性、风险、动量、盈利、价值、投资等），
  见下文“中国 A 股股票特征 china_chars”；
- `size_value_china`：`svc_001` 至 `svc_021`，共 21 列，Liu-Stambaugh-Yuan“Size and Value in China”的
  规模、EP、异常换手率以及 CH-3/CH-4 因子载荷，见下文“Size and Value in China 因子集 size_value_china”；
- `talib`：`ta_001` 至 `ta_125`，共 125 列，TA-Lib 经典技术指标（64 列）与 K 线形态（61 列），见下文“TA-Lib 因子集 talib”。
- `oap_jq_cnc`：oap、jq240、china_chars 三个库的 484 个因子按截面秩相关 0.8 去重后剩下的 254 列，同类里按覆盖率留，
  不用任何收益信息，只用于 `sh_main`，成员锁定在 `src/rquant/factors/libraries/oap_jq_cnc.py`（见下文“合并去重因子集 oap_jq_cnc”）。
- `jq240`：`jq_001` 至 `jq_233`，共 233 列，按聚宽因子库公开定义重写的质量、基础科目、情绪、成长、风险、每股、技术、
  动量与风格因子，见下文“聚宽因子集 jq240”。
- `oap`：`oap_001` 至 `oap_142`，共 142 列，Open Source Asset Pricing（Chen-Zimmermann）212 个预测信号中能用 Tushare
  构造的部分，见下文“Open Source Asset Pricing 因子集 oap”。
- `barra_cne6`：`cne6_001` 至 `cne6_038`，共 38 列，按 MSCI Barra CNE6 公开定义重写的风格描述子（descriptor），
  见下文“Barra CNE6 因子集 barra_cne6”。
- `tushare_tech`：`tst_001` 至 `tst_075`，共 75 列，Tushare `stk_factor_pro` 接口的技术指标按通达信公式与 Tushare 参数重算，
  见下文“Tushare 技术因子集 tushare_tech”。
- `jkp153`：`jkp_001` 至 `jkp_144`，共 144 列，Jensen-Kelly-Pedersen 153 个全球因子特征中能用 Tushare 构建的部分，
  见下文“JKP 因子集 jkp153”。
- `jkp406`：`jkpx_001` 至 `jkpx_364`，共 364 列，JKP 扩展特征的 Tushare 日频版本，包含原 144 个定义及 220 个扩展定义；
  可核实清单、40 个排除项和官方名称数量差异见下文。
- `china469_pit`：`cnax_001` 至 `cnax_003`，按原文形成时间重写 Cop、Cla、Claq 的 Tushare 版本，见下文。

原先把 Alpha158 与 Alpha101 拼成 259 列的 `combined` 因子集已移除；`runs/` 下用它跑出的历史结果仍保留，
但当前代码不再构建或读取 `data/factors/combined`。

固定目录保留来源名称、公式标识、序号、最大回看期和实现方式，但模型特征只暴露规范列名。

### 因子库代码结构

所有可研究的因子公式由 RQuant 自己维护。Alpha158、Alpha101、Alpha360、GTJA191、china_chars、
size_value_china、talib、jq240 与 oap 全部使用 KunQuant 后端；它们共用目录、清单、年度 Parquet、验证器和下游加载契约。
原 Pandas 实现保留在 `tests/panel_reference/` 中作为独立数值对照。Python 负责读取、排列数据和按公告日发布财报结果；
财报公式在季度图中计算，日频公式在交易日图中计算，缺失的窗口、递推、回归和行业截面能力由 C++ 扩展算子提供：

```text
src/rquant/factors/
├── extensions/
│   ├── kunquant.py                      # KunQuant 缺失能力的最小扩展与适配
│   ├── kunquant_panel.py                # 与 Pandas 面板语义一致的窗口、递归和逐股 C++ 算子
│   ├── kunquant_jq240.py                # jq240 的停牌窗口与截面运算
│   ├── kunquant_oap.py                  # oap 的季度、月度和行业截面运算
│   ├── kunquant_oap_regression.py       # oap 奇异回归窗口的伪逆回退
│   └── ta_lib/ta_lib.hpp                # 编译到 KunQuant 图中的 TA-Lib C 函数
├── panel_operators.py                   # 独立数值参考及研究流程使用的 Pandas 算子
├── statements.py                        # 首次披露财报季度网格与公告日对齐
├── catalog.py                           # 动态目录、稳定命名和实现指纹
├── engine.py                            # 后端选择、执行和标准分区写入
├── selection.py                         # 按年度稳定性、覆盖率与相关性筛选因子
└── libraries/
    ├── base.py                          # 因子库接口与输入字段契约
    ├── registry.py                      # 因子库及组合因子集注册表
    ├── alpha158.py                      # 158 条本地 Alpha158 公式
    ├── alpha101.py                      # 101 条本地 Alpha101 公式
    ├── alpha360.py                      # 6 个归一化字段各 60 个滞后期的 Alpha360 图
    ├── gtja191.py                       # 190 条 GTJA191 KunQuant 公式与语义说明
    ├── oap_jq_cnc.py                    # oap_jq_cnc（oap、jq240、china_chars 只去重，按覆盖率）的锁定成员
    ├── china_chars.py                   # 109 个日频 point-in-time 中国 A 股股票特征
    ├── size_value_china.py              # 21 个 CH-3/CH-4 特征与因子载荷
    ├── talib_indicators.py              # 125 个 TA-Lib 技术指标与 K 线形态
    ├── jq240.py                         # 233 个按聚宽因子库定义重写的日频因子
    ├── oap.py                           # 142 个按 Open Source Asset Pricing 定义重写的日频信号
    ├── barra_cne6.py                    # 38 个 Barra CNE6 风格描述子
    ├── tushare_tech.py                  # 75 个 Tushare stk_factor_pro 技术指标
    ├── jkp153.py                        # 144 个 JKP 全球因子特征的日频版本
    └── custom.py                        # 自定义小型因子库构造器
```

Alpha158、Alpha101 与 Alpha360 的公式定义均保存在 RQuant 源码中，不从
`KunQuant.predefined.Alpha158/Alpha101` 导入。KunQuant 仅提供算子、图优化、C++ 代码生成和运行时；公式变更会
进入 RQuant 的实现指纹。Alpha158 与 Alpha101 的迁入公式基于 KunQuant 0.1.11 的 Apache-2.0 源码；
Alpha360 的迁入公式基于 Qlib 0.9.7 的 MIT 源码。两项来源与许可证文件路径均见
`THIRD_PARTY_NOTICES.md`。

因子公式优先直接组合 KunQuant 公共算子。只有 KunQuant 0.1.11 无法表达的能力才放入
`extensions/`；扩展覆盖行业截面、窗口、递推和逐股 C++ 计算，不复制
`OpBase`、优化 pass、代码生成器或运行时。

### 添加自己的因子库

迁移会改变各因子集的实现指纹，旧因子库需要按正式构建命令重建，再重建依赖它们的 oap_jq_cnc。
截断数据的数值对照测试使用临时目录，不替换已有因子库；完整历史数据上的性能和结果须另行验证。

在 `src/rquant/factors/libraries/` 中新建独立模块。小型因子库可直接使用 `CustomFactorLibrary`：

```python
from rquant.factors.libraries.custom import CustomFactor, CustomFactorLibrary


def price_spread(inputs):
    return inputs["close"] - inputs["open"]


LIBRARY = CustomFactorLibrary(
    family="custom_price",
    required_inputs=("open", "close"),
    factors=(
        CustomFactor(
            canonical_name="custom_price_001",
            source_name="price_spread",
            formula="close - open",
            max_lookback=1,
            builder=price_spread,
        ),
    ),
)
```

然后只在 `libraries/registry.py` 的 `_default_registry()` 中导入并注册：

```python
from rquant.factors.libraries.my_factors import LIBRARY

registry.register_library(LIBRARY)
```

注册后，因子库名称会自动成为 CLI 可选的 `--factor-set`，无需修改 `engine.py`、CLI 参数列表、数据加载器
或因子列前缀校验。如需把多个库组合成一个稳定因子集，再调用：

```python
registry.register_factor_set("research_v1", ("qlib_alpha158", "wq_alpha101", "custom_price"))
```

每个新库至少要补充：目录名称和顺序测试、小型数值基准测试、对应后端的执行测试，以及缺失输入测试。

### 构建 GTJA191

GTJA191 当前构建 190 条公式：186 条只依赖标准 OHLCV、成交额和 VWAP，另外 4 条需要指数序列：

- `gtja_075`、`gtja_149`、`gtja_181`、`gtja_182` 使用标准数据中的沪深 300 日线；构建器自动从
  `data/canonical/reference/index_daily.parquet` 选择 `000300.SH`，缺失时兼容 `399300.SZ`；
- `gtja_030` 的公式只保留在 Pandas 参考实现 `tests/panel_reference/gtja191.py` 中，因当前缺少可审计的逐日
  `mkt`、`smb`、`hml` 风格收益，暂时不注册到构建目录。

构建命令：

```bash
rquant factors build --factor-set gtja191
rquant factors validate --factor-set gtja191
```

结果写到 `data/factors/gtja191/year=YYYY/factors.parquet` 和
`data/factors/gtja191/manifest.json`；清单会记录沪深 300 文件哈希。

公式以国泰君安[《数量化专题之九：基于短周期价量特征的多因子选股体系》](https://guorn.com/static/upload/file/3/134065454575605.pdf)
附录为基准。原报告中
`Alpha159` 的 `HGIH` 按 `HIGH` 修正并保留其原始加权式；`Alpha181` 的分母 `SUM` 缺少窗口参数，
实现采用与分子一致的 20 日窗口，并在源码 `GTJA191_FORMULA_NOTES` 中固定这些解释。

GTJA191 由 KunQuant 编译执行，首次构建需要本机 C++ 编译器，编译耗时取决于平台与图规模。原 Pandas
实现保留在 `tests/panel_reference/gtja191.py`，测试逐格比对全部 190 条公式。为保证跨编译器、跨 CPU 结果一致：
截面 `rank` 和时序 `ts_rank` 先把输入保留 40 位有效二进制位并把绝对值小于 1e-12 的值置 0，使理论上相等的值
严格并列；该库编译时关闭 FMA 合并（GCC/Clang 的 `-ffp-contract=off`，MSVC 默认不合并）。

### 构建因子

Alpha360 是独立的因子集；先检查目录，再在明确需要新产物且已确认磁盘空间、无并发写入者后构建：

```bash
rquant factors catalog --factor-set qlib_alpha360
rquant factors build --factor-set qlib_alpha360
rquant factors validate --factor-set qlib_alpha360
```

完整构建会在完成 staging 后原子替换整个 `data/factors/qlib_alpha360/` 目录，而不是向既有年度分区追加；本文
不执行该构建。

如需进行限定日期的诊断构建，可以指定：

```bash
rquant factors build --factor-set qlib_alpha158 --start 2020-01-01 --end 2020-12-31
```

因子在股票池成员的全部历史行情上计算（包括入选沪深 300 之前和调出之后的日子），因此新进成分股在入选当天就有
完整的滚动窗口与滞后特征；写盘时只保留当日属于股票池的行。

因子结果按年份写入：

```text
data/factors/qlib_alpha158/year=YYYY/factors.parquet
data/factors/qlib_alpha158/manifest.json
```

限定日期构建不能代替完整研究所需的全历史因子。开始滚动训练前，要确认因子清单覆盖所有训练、验证、测试和标签所需日期。

### 合并去重因子集 oap_jq_cnc

`oap_jq_cnc` 是唯一的选择型因子集：不重算公式，从已构建的 `oap`、`jq240`、`china_chars` 复制所选列，因此必须先在
`sh_main` 上构建这三个库。候选是这三个 2020 年以后表现最好的基础库的全部 484 个因子（oap 142、jq240 233、
china_chars 109），只去掉重复：

- 相似度：两两因子的截面 Spearman 相关系数绝对值，在 sh_main 因子库上从 2008-05-12 到 2026-10-20 每 5 个交易日取
  一天（897 天）求平均（`scripts/remove_duplicates/select_remove_duplicates.py`，用
  `rquant.factors.selection.RankCorrelationAccumulator` 的矩阵乘法计算，与 pandas 的逐对 Spearman 一致）；
- 同类里留哪个：先留覆盖率高的（保留两位小数），再按因子库顺序，不用任何收益、IC 或做多端信息，所以 2020 年及
  以后对它仍是样本外；
- 阈值 0.8：与已保留因子的相关系数绝对值达到 0.8 就剔除，剩 254 个：oap 103、jq240 107、china_chars 44（0.7 时
  214 个，0.9 时 332 个）。

成员锁定在 `oap_jq_cnc.py`，由 `scripts/remove_duplicates/select_by_coverage.py` 生成。

```bash
rquant --universe sh_main factors build --factor-set oap_jq_cnc
rquant --universe sh_main factors validate --factor-set oap_jq_cnc
rquant --universe sh_main walk-forward --model lgb --horizon 1d --factor-set oap_jq_cnc
```

sh_main 上的 LGB 1d walk-forward（top 50 每 5 天调仓；年化 / 夏普 / 最大回撤）：

| 因子集 | run | 2013-2026 | 2020-2026 | 2022 年以后 | 2022 年以后去掉 2025 年的夏普 | 2020 年以来 / 2022 年以来累计 |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| oap_jq_cnc（254） | `20261028T191159Z-0a324a7c` | 30.5% / 1.06 / -42.6% | 19.3% / 0.79 / -39.5% | 15.3% / 0.65 / -39.5% | 0.33 | +217% / +92% |
| jq240 | `20261027T202921Z-41fb3dde` | 29.1% / 1.02 / -44.1% | 22.5% / 0.88 / -33.0% | 19.1% / 0.76 / -33.0% | 0.43 | +277% / +123% |
| oap | `20261027T185453Z-39c99759` | 23.2% / 0.87 / -50.0% | 22.5% / 0.89 / -42.6% | 18.9% / 0.76 / -42.6% | 0.42 | +277% / +122% |
| china_chars | `20261027T183236Z-2e1c2526` | 23.4% / 0.88 / -45.4% | 19.0% / 0.80 / -40.3% | 12.4% / 0.57 / -40.3% | 0.29 | +212% / +71% |

全期是做过的所有因子集里最好的（2018 年 +6.0%，另外三个都亏），但提升集中在 2013-2019 年；2020 年以后不如单独的 jq240。

成员保留源规范列名与原值，`direction` 是 sh_main 2020 年以后 RankIC 的符号，只作元数据，存储值不取负；树模型不受符号影响。
此前的 Hquality、recent、recent_sh_main、remove_duplicates 等按单因子效果或收益挑选的因子集已删除，需要时可从 git 历史找回。

### 中国 A 股股票特征 china_chars

`china_chars` 按 [Quantactix/ChinaAShareEquityCharacteristics](https://github.com/Quantactix/ChinaAShareEquityCharacteristics)
公布的特征定义（月频、基于 CSMAR，源自 Hou-Xue-Zhang 异象列表）用 Tushare 数据独立重写，没有使用其代码；
每个特征都改成在可知当天的日频 point-in-time 值。它依赖上节的参考数据和财报：

```bash
rquant data sync-fundamentals --through 2026-08-04
rquant data build-reference
rquant factors build --factor-set china_chars
rquant factors validate --factor-set china_chars
rquant walk-forward --model lgb --horizon 1d --factor-set china_chars
```

口径：

- 一个月按 21 个交易日、一个季度 63 个、一年 252 个；滚动回归、波动和偏度默认 120 日窗口、至少 40 个观测；
- 市场模型和三因子模型使用参考数据中的市场收益、SMB、HML，不扣无风险利率（只影响截距）；
- Dimson beta 的领先项只用到当日已知的市场收益，即窗口截至前一日；
- 财报取首次披露值；一个报告期在公告日（非交易日顺延到下一交易日）起可用，而且必须等所有更早的已披露报告期
  都已公告，因此滞后报告期不会比锚定报告期更新；利润表和现金流量表由累计值换算为单季值，估值比率用滚动四季；
- 停牌日不输出特征。

原始 118 个特征中，Tushare 当前权限拿不到或原实现未完成的 10 个未纳入：员工数（hn、lfe）、广告费（adm、gad）、
CH-4 因子载荷（mkt_beta、vmg_beta、smb_beta、pmo_beta）、q 因子残差矩（idvq、idsq）；原项目的 tail 未完成，也未纳入。
因子清单的 `reference_inputs` 记录所读参考文件的哈希。

### Size and Value in China 因子集 size_value_china

`size_value_china` 按 Liu、Stambaugh、Yuan（2019，JFE）“Size and Value in China”公布的 CH-3/CH-4 定义用 Tushare
独立实现。`rquant data build-reference` 先从全 A 股截面构建日度因子收益 `reference/ch4_factors.parquet`
（`src/rquant/data/ch4_factors.py`）：

- 每月第一个交易日用截至上一交易日（上月最后交易日）的数据分组，持有一个月；日收益按前一日总市值加权；
- 样本为代码以 60、00、30 开头的沪深主板和创业板股票，上市满 6 个月，过去 250 个交易日至少 120 天有交易、
  上个月至少 15 天有交易（春节所在月份全市场不足 20 个交易日时，改为至少该月交易日的 75%）；按总市值剔除最小的 30%（壳价值污染），剔除在交易过滤之前排序；
- 规模为总市值（含非流通股）；EP 为 `1 / pe_ttm`。论文用最近披露的年化扣非净利润，Tushare 全市场日频只有
  滚动四季净利润，所以用它代替；`pe_ttm` 对亏损公司为空，这些股票按论文对负 EP 的处理放入成长组；
- 异常换手率为近 20 日平均日换手（成交股数 / 总股本）除以近 250 日平均；
- 规模按中位数两分，EP 与异常换手率按剩余样本的 30%/70% 分位三分；`smb`、`vmg` 来自规模 x EP，`pmo` 在规模组内
  做多低异常换手（悲观）、做空高异常换手（乐观），`smb4` 是 EP 中性与换手中性 SMB 的平均（CH-4 的规模因子）；
  `mkt` 是剩余样本的市值加权收益，不扣一年期存款利率（只影响截距）。

个股列（日频、停牌日为空）：

- `size`、`ep`、`abturn`：上述三个分组变量本身；
- `ch3_*`：对 `mkt`、`smb`、`vmg` 的 120 日滚动回归载荷与残差波动；
- `ch4_*`：对 `mkt`、`smb4`、`vmg`、`pmo` 的 120 日载荷、残差波动、R²；`*_y` 为 250 日载荷、残差波动与截距；
- `ch4_resrev`：120 日拟合中最近 21 日残差的均值 / 标准差（残差反转）；`ch4_resmom`：250 日拟合中 t-249..t-21
  日残差的均值 / 标准差（残差动量）。

沪深 300 成分股基本都不在最小 30% 中，所以剔除只影响因子收益的构建，不影响个股样本。

```bash
rquant data build-reference
rquant factors build --factor-set size_value_china
rquant factors validate --factor-set size_value_china
rquant walk-forward --model lgb --horizon 1d --factor-set size_value_china
```

### TA-Lib 因子集 talib

`talib` 将仓库内的 TA-Lib 0.8.1 C 函数编译到 KunQuant 图中，逐只股票计算；
`TA-Lib==0.8.2` Python 包用于独立对照测试。定义锁定在 `src/rquant/factors/libraries/talib_indicators.py`：

- 只用 OHLCV，不需要参考数据；每只股票先去掉停牌和上市前的空行，再按自身连续交易日计算，所以 EMA、Wilder 平滑、
  希尔伯特变换等递归指标不会被停牌缺口重置；每只股票最初 `lookback` 个交易日（TA-Lib 无值，K 线形态会给占位 0）为空；
- 范围是经典 TA-Lib 0.4 的函数表，不含数学运算与数学变换；TA-Lib 0.6 以后新增的指标未纳入；
- 后复权价格的水平对每只股票是任意的，所以价格单位的输出都做了无量纲化：均线类（SMA、EMA、KAMA、MAMA、SAR、
  线性回归等）为 `指标 / close - 1`，价差类（MACD、DM、TRANGE、STDDEV、斜率、HT_PHASOR）为 `指标 / close`，
  布林带改为 %B 与带宽，OBV、AD 为 20 日累计线变化除以这 20 日成交量，ADOSC 除以 10 日均量；
  `tests/test_talib_indicators.py` 用价格乘 7、成交量除 7 的数据验证全部输出不变；
- 去掉了与另一个已发布输出成精确仿射关系的列（ROCP/ROCR/ROCR100、CMO = 2·RSI − 100、MA、TYPPRICE、WCLPRICE、VAR、
  ATR）和近似重复的列（MACDFIX、MACDEXT、SAREXT、MAVP、MOM、APO）；LINEARREG_ANGLE 依赖价格尺度、BETA 缺少市场输入，
  也未纳入；
- 61 个 K 线形态输出为 TA-Lib 原值：形态完成当天看涨 +100 / 看跌 -100（少数形态为 ±200），否则为 0。

因子清单的实现指纹包含公式模块和仓库内的 TA-Lib C 源码，修改这些实现会使旧的 `data/factors/talib` 失效。

```bash
rquant factors build --factor-set talib
rquant factors validate --factor-set talib
rquant walk-forward --model lgb --horizon 1d --factor-set talib
```

### 聚宽因子集 jq240

`jq240` 按聚宽（JoinQuant）因子库 `jqfactor` 公开的因子名称与定义独立重写，不使用聚宽代码或数据，只用 Tushare，
定义锁定在 `src/rquant/factors/libraries/jq240.py`，来源名就是聚宽因子名（如 `roe_ttm`、`VOL20`、`BIAS10`、`size`）。
九个类别：质量 66、基础科目及衍生 36、情绪 36、动量 34、每股 15、技术 15、风险 12、风格 10、成长 9。

- 财报取 `fundamentals.parquet` 的首次披露值，沿用 `china_chars` 的季度网格：报告期在公告日之后、且更早报告期都已
  公告后才可用；`_ttm` 为最近四个单季之和，聚宽按报告期给出的非 TTM 流量项取最近单季，资产负债项取最近报告期，
  “平均”余额为本期与上年同期的均值，同比增长除以基期绝对值；
- 价量指标在每只股票自己的交易日上计算（停牌行先剔除再写回为空），窗口数的是有行情的天数；
  均线、EMA、布林带、MACD 按聚宽口径除以收盘价，ATR 与多空力道也除以收盘价，`price_no_fq` 是 daily_basic 的不复权收盘价；
  成交量是标准数据的复权成交量，成交额为元，换手率为 daily_basic 的流通股换手率（%）；
- 风格因子按聚宽文档的 Barra CNE5 描述子计算（beta 半衰期 63、RSTR 504 日滞后 21 日、DASTD/CMRA/HSIGMA 等），
  截面标准化、排序和正交化只用当日的研究股票池成员（默认沪市主板），市场收益为参考数据中的全 A 市值加权收益；缺失的描述子按权重比例分摊；
- 未实现：`non_recurring_gain_loss` 与 `adjusted_profit_to_total_profit`（需要 `fina_indicator` 的扣非净利润），
  以及风格因子中依赖分析师预期的 EPIBS、EGRLF、EGRSF（其权重按比例分给其余描述子）。

`jq240` 需要的财报字段比 `china_chars` 多，`FUNDAMENTAL_ENDPOINT_FIELDS` 增加字段后旧的财报分区查询指纹会失效，需要
重新同步（`csi300` 股票池约一万个分区，按默认限流约一小时）并重建参考数据；`fundamentals.parquet` 的哈希随之改变，
已构建的 `china_chars` 也要重建。`tests/test_jq240.py` 用截断数据重算、停牌窗口与直接公式回归验证。

```bash
rquant data sync-fundamentals --through 2026-08-04
rquant data build-reference
rquant factors build --factor-set jq240
rquant factors validate --factor-set jq240
rquant walk-forward --model lgb --horizon 1d --factor-set jq240
```

### Open Source Asset Pricing 因子集 oap

`oap` 按 Open Source Asset Pricing（Chen and Zimmermann, “Open Source Cross-Sectional Asset Pricing”，openassetpricing.com）
`SignalDoc` 中 212 个 Predictor 信号的公开定义独立重写，不使用其代码或数据，只用 Tushare，定义锁定在
`src/rquant/factors/libraries/oap.py`，来源名就是 OAP 简称（如 `Mom12m`、`AssetGrowth`、`IdioVol3F`）。114 个 Placebo
信号不在其中。142 个已实现：价格 36、交易 11、会计 84（估值、盈利、投资与应计、R&D）、其他 4、事件 7（含
AnnouncementReturn）。

- 月度定义改为日频：一个月 21 个交易日，一年 252 个；需要月收益序列的信号（Beta、Coskewness、std_turn、VolumeTrend、
  ResidualMomentum、BetaTailRisk、VarCF 等）在以当日结尾、互不重叠的 21 日块上计算，因此每天更新；
- 财报沿用 `china_chars` 的季度网格（首次披露值、公告日后且更早报告期均已公告才可用）：Compustat 年度流量改为最近四个
  单季之和，资产负债项取最近报告期，“一年前”为四个季度前；Compustat 科目与 Tushare 字段的对应见 `ITEMS`；
- 市值为日度总市值；股利为 daily_basic 的 `dv_ttm` 乘总市值，过去 12 个月股利总额变化超过 5% 的日子视为一次派息变动，
  DivSeason/DivYieldST 以 11–12 个月前是否有变动判断下月是否派息；需要剔除送转的股数（ShareIss1Y/5Y、PS）用
  总股本 × 不复权价 / 复权价；
- 截面步骤（五分位标记、行业均值/中位数、截面回归、Herfindahl）只用当日研究股票池成员（默认沪市主板），行业为申万一级；
- 美股样本过滤（股价低于 5 美元、交易所、股票代码、SIC 金融业剔除等）不再使用；定义本身的限制保留（FirmAgeMom 只取最年轻
  五分之一，MomVol 只取换手率最高三分之一）；美国特有常数换成 A 股口径（法定税率 2008 年前 33%、之后 25%；ShareVol
  用沪深 300 换手率三分位代替美股 5%/10% 阈值）。

未实现的 70 个及原因记录在 `OAP_SKIPPED`：分析师预期与评级 20 个（IBES）、13F 机构持仓 8 个、个股期权 9 个，以及
专利、广告费、员工数、订单储备、养老金、分部数据、信用评级、融券余额、PIN、股份回购、递延收入（2018–2020 年预收款项
改为合同负债）、需要 8 年 R&D 的 RDAbility（Tushare 研发费用 2018 年起才单列）、11 年以上的季节性（行情从 2008 年开始）等。
R&D 类信号（RD、RDcap、SurpriseRD、RDIPO）在 2018 年以前为空。

`oap` 用到的现金流量表字段（吸收投资、取得借款、发行债券、偿还债务收到/支付的现金）和资产负债表字段（应交税费、递延所得税
负债）与 `jkp153` 相同，新增字段后需重新 `data sync-fundamentals` 与 `data build-reference`，已构建的
`china_chars`、`jq240` 也要重建。`tests/test_oap.py` 用截断数据重算、直接公式、派息事件与截面成分股回归验证。

```bash
rquant data sync-fundamentals --through 2026-08-04
rquant data build-reference
rquant factors build --factor-set oap
rquant factors validate --factor-set oap
rquant walk-forward --model lgb --horizon 1d --factor-set oap
```

### Barra CNE6 因子集 barra_cne6

`barra_cne6` 按 MSCI 公布的 Barra 中国股票模型（CNE6）描述子定义独立实现，不使用 MSCI 代码或数据，只用 Tushare，
定义锁定在 `src/rquant/factors/libraries/barra_cne6.py`。发布的是描述子本身，不合成风格因子，权重交给模型：

- 规模 LNCAP、MIDCAP；贝塔与残差波动 HBETA、HSIGMA、DASTD、CMRA；流动性 STOM、STOQ、STOA、ATVR；
  动量 STREV、SEASON、INDMOM、RSTR；长期反转 LTRSTR、LTHALPHA；估值与盈利收益率 BTOP、ETOP、CETOP、EM；
  杠杆 MLEV、BLEV、DTOA；盈利波动 VSAL、VERN、VFLO；盈利质量 ABS、ACF；盈利能力 ATO、GP、GPM、ROA；
  投资质量 AGRO、IGRO、CXGRO；成长 EGRO、SGRO；分红 DTOP；
- 收益为后复权收盘价的日收益，市场收益为参考数据中的全 A 市值加权收益，不扣无风险利率；指数加权窗口最新一行权重为 1，
  每过一个半衰期减半；带滞后的描述子（STREV 滞后 1-3、RSTR 11-21、LTRSTR/LTHALPHA 273-283）对各滞后取平均；
- 财报沿用 `china_chars` 的首次披露季度网格；比率用 TTM 流量，五年描述子用最近五个财年（Q4）值，至少需要三年；
- MIDCAP 与 INDMOM 两个截面描述子只用当日沪深 300 成分股做标准化、正交化和行业加权（行业为申万一级，权重为市值平方根）；
- 未实现：依赖分析师预期的 EPIBS、EGRLF、ETOPF_STD、DPIBS（Tushare 预期接口在当前账户只能每分钟调用一次）；
  ACF 用到 jq240 引入的投资活动现金流字段 `n_cashflow_inv_act`。

```bash
rquant factors build --factor-set barra_cne6
rquant factors validate --factor-set barra_cne6
rquant walk-forward --model lgb --horizon 1d --factor-set barra_cne6
```

### Tushare 技术因子集 tushare_tech

Tushare `stk_factor_pro` 每天给出约三十类通达信常用指标，但该接口需要更高积分，当前账户无权限。`tushare_tech` 按标准通达信
公式和 Tushare 文档里的参数在本地重算，不使用 Tushare 的结果，定义锁定在 `src/rquant/factors/libraries/tushare_tech.py`：

- 指标：ASI、ATR、BBI、BIAS、BOLL、BRAR、CCI、CR、DMA（dfma）、DMI、DPO、EMA（5 至 250 日）、EMV、EXPMA、KDJ、KTN、
  MA（5 至 250 日）、MACD、MASS、MFI、MTM、OBV、PSY、ROC、RSI、TAQ、TRIX、VR、WR、XSII，以及连涨、连跌天数和
  `topdays`、`lowdays`；
- 与 `talib` 一样逐只股票按自身交易日计算，停牌行先剔除再写回为空；`EMA(X,N)` 权重 2/(N+1)、`SMA(X,N,M)` 权重 M/N，
  都从首个有效值开始，每只股票最初 `lookback` 个交易日为空；
- 价格单位的输出无量纲化：均线、通道、布林带为 `线 / close - 1`，差值与区间（ATR、MACD、MTM、DPO、DMA、ASI）为
  `值 / close`，OBV 为 20 日变化除以这 20 日成交量；`tests/test_tushare_tech.py` 验证价格与成交量缩放不变；
- BBI 用标准的 3、6、12、24 日均线；去掉与另一输出成精确仿射关系的 `boll_mid`（即 MA20）以及 `xsii_td2`、`xsii_td4`。

```bash
rquant factors build --factor-set tushare_tech
rquant factors validate --factor-set tushare_tech
rquant walk-forward --model lgb --horizon 1d --factor-set tushare_tech
```

### JKP 因子集 jkp153

`jkp153` 按 Jensen、Kelly、Pedersen《Is There a Replication Crisis in Finance?》（Journal of Finance 2023）公开的 153 个
特征（作者 `factor_details` 表与公开代码的构造）独立重写为日频 point-in-time 版本，不使用上游代码或数据，只用 Tushare，
定义锁定在 `src/rquant/factors/libraries/jkp153.py`，来源名就是 JKP 特征名（如 `be_me`、`ret_12_1`、`ivol_ff3_21d`、`qmj`）：

- 月度窗口换成交易日：一个月 21 天、一年 252 天，`_21d`/`_126d`/`_252d`/`_1260d`、`ret_x_y`、`seas_*` 都按交易日或 21 日块计算；
  市场、SMB、HML 收益取参考数据中的全 A 市值加权序列；
- 财报沿用 `china_chars` 的首次披露季度网格；JKP 的年度 `_x` 科目用 TTM（按季度刷新），`q` 科目用最近单季，12 个月滞后为
  4 个季度；Compustat 科目与 Tushare 科目的对应见模块中的 `ITEMS`、`FLOWS`；中国现金流量表没有单列回购，按 0 处理；
  分红为 daily_basic 的 `dv_ttm` 乘总市值；
- 截面合成特征（mispricing、quality-minus-junk）只用当日沪深 300 成分股排序；
- 未实现 9 个：员工数相关的 `emp_gr1`、`sale_emp_gr1`，q 因子模型的 `ivol_hxz4_21d`、`iskew_hxz4_21d`，
  数据起点之前的 `seas_11_15*`、`seas_16_20*`，以及没有回购时与 `div12m_me` 相同的 `eqpo_me`；共 144 列 `jkp_001` 至 `jkp_144`。

`jkp153` 在 jq240 的财报字段之外还需要 `FUNDAMENTAL_ENDPOINT_FIELDS` 中标注为 jkp153 的 6 个字段（oap 也使用）（`taxes_payable`、
`defer_tax_liab`、`c_recp_cap_contrib`、`c_recp_borrow`、`proc_issue_bonds`、`c_prepay_amt_borr`），新增后资产负债表与
现金流量表分区需要重新同步（利润表分区可复用），并重建参考数据与 `china_chars`、`jq240`、`barra_cne6`。
`tests/test_jkp153.py` 用截断数据重算、直接公式与截面排序回归验证。

```bash
rquant data sync-fundamentals --through 2026-08-04
rquant data build-reference
rquant factors build --factor-set jkp153
rquant factors validate --factor-set jkp153
rquant walk-forward --model lgb --horizon 1d --factor-set jkp153
```

`barra_cne6`、`tushare_tech` 与 `jkp153` 与其他库一样在 KunQuant 中计算：财报公式在季度图中计算并按公告日发布，指数加权窗口、滚动回归、截面排名与标准化由逐股或逐日 C++ 算子完成，`tushare_tech` 的指标是逐股序列 C++ 内核；原 Pandas 实现保留在 `tests/panel_reference/` 中，测试逐列对齐。

### JKP 扩展因子集 jkp406

新增的 `jkp406` 独立注册，保留 `jkp153` 的 144 个定义、列名及历史结果契约。扩展库用 `jkpx_001` 至
`jkpx_364` 编号；扩展 220 个定义主要覆盖财报增长、资产变化、盈利与偿债比率、营运周期、估值及长短期收益。
全部通过 KunQuant 计算，不新增依赖、Tushare 接口或财报字段。已有参考数据若满足 `jkp153` 即可复用。
上市日文件始终记录实际读取哈希；参考清单若包含该文件则额外比对，兼容旧参考清单未列入该文件的格式。

清单锁定在 `src/rquant/factors/libraries/jkp406_catalog.py`，公式在 `jkp406.py`。
来源为 [JKP 官方说明](https://jkpfactors-data.s3.amazonaws.com/documents/Documentation.pdf)及
[官方源码版本 67174c7](https://github.com/bkelly-lab/ReplicationCrisis/tree/67174c7f170bf8b952b3c86876f187fd9970b5ba/GlobalFactors)，
并与官方 Python 版本 `c1762b9` 交叉核对。官方文档称 406 项，但该版本的明确导出清单为 315 个财报特征、
46 个月度市场特征、31 个日度市场特征、11 个组合特征，加上核心清单中的 `prc`，合计 404 个唯一名称。
`SOURCE_COUNT_DISCREPANCY=2` 记录未能对应到名称的差异；没有用额外变量凑满 406。

404 个具名项目中实现 364 个，另 40 个在 `JKP406_SKIPPED` 逐项记录原因，包括员工数、独立回购现金流、
年度/短窗口/特殊分红、特殊和非常损益、广告和人工费用、HXZ 模型收益，以及超出现有历史起点的 16–20 年季节性。
这些是当前数据接口的支持边界，单元测试不代表真实数据有完整覆盖或有预测收益。

口径沿用 `jkp153`：一个月 21 个交易日、一年 252 个交易日；财报取首次披露值，按真实公告日及连续可用季度发布；
年度流量用 TTM，年度滞后为四个季度；停牌日输出为空；截面排序使用引擎传入的当日研究股票池。
新增的长/短债务发行分别使用官方允许的长/短债务余额年度变化回退口径，不能解释为直接披露的发行现金流。
新增现金周转天数使用 365 个自然日和年初年末余额均值；`ivol_capm_60m` 用不重叠的 21 日收益估计，至少 36 个月有效值。
新增 `rvolhl_21d` 在计算高低价波动率之前校正隔夜跳空；保留原 `bidaskhl_21d` 定义。
继承的 144 个定义保留原有假设（包括 `dv_ttm` 分红近似及不单列回购），没有在这次扩展中重新定义。
其中 `eqnetis_gr1a`（`jkpx_064`）与 `eqis_gr1a`（`jkpx_071`）当前均以发行现金流代理，数值相同；官方允许在
回购项缺失时使用发行额，但本地缺独立回购数据时不能反映真实净发行，也不能把它们当作两条独立信息。
11–15 年同月与其他月份季节性追加为 `jkpx_363`、`jkpx_364`，前 362 个列名不变；它们要求最长 3780 个交易日回看，
本地历史后段才可能有值。新增列改变目录和实现指纹，旧版因子库、训练和回测保留为历史结果，需要新版本时重建。
测试参考保存在 `tests/panel_reference/jkp406.py`，按列比较数值及 NaN 掩码，并验证截断重算、停牌和比率边界。

```bash
rquant factors catalog --factor-set jkp406
rquant factors build --factor-set jkp406
rquant factors validate --factor-set jkp406
```

### China 469 形成时点因子 `china469_pit`

这三列来自 Li、Liu、Liu、Wei 的 [469 项异常补充材料](https://pubsonline.informs.org/doi/suppl/10.1287/mnsc.2023.4904/suppl_file/mnsc.2023.4904.sm1.pdf) A.4.18–20：
`cnax_001` Cop 使用年度（营业利润−净利润＋经营现金流）／当年总资产；`cnax_002` Cla 的分母为上年末总资产；
`cnax_003` Claq 使用单季分子和上一季末总资产。现有 JKP、OAP 的相关列采用不同利润重构或窗口，不能当成相同定义。

年度值在六月最后交易日收盘形成，只使用当时已公告的上一年年报，并固定到下一次形成；缺少指定年报或 Cla 所需的上年资产时清空，而非事后回填。
季度值在月初首个交易日形成，要求报告期末至少早四个自然月，且当期及所需前一季报告均在形成日之前公告；缺所需报告时回退到较早合格季度是本项目约定。
累计流量逐季转成单季值，分母为零、数据缺失或非有限行情时输出为空。形成日后发生的披露不回填。

财务字段使用 Tushare `operate_profit`、`n_income`、`n_cashflow_act`、`total_assets`，保留首次公告约束；尚未验证与 CSMAR 同值，不能声称逐值复现论文。
使用完整的 SSE 交易日历确定形成日，记录实际文件哈希；旧参考清单未列出日历哈希时，还需验证原始 Tushare 日历分区及原始同步指纹。
旧版同步总清单未保存各分区摘要列表：对比可以发现单侧篡改，但不能证明同步替换的原始分区和参考日历仍属于当时的原始快照。
月初信号在该日输出，项目下一交易日开盘成交会比月初开盘形成晚一个交易日；日历假定历史开闭市安排可用于确定形成边界。
三列分别以年度当期、年度前期、季度前期锚点发布，以小型测试核对公式、截断、晚公告、缺年、零分母、非有限行情及日历篡改；尚未进行全市场构建或生产回测。

### 验证因子

```bash
rquant factors validate --factor-set jq240
python -m json.tool data/factors/jq240/manifest.json
```

验证会检查所有分区的因子列是否完整、无重复、顺序固定，并拒绝暴露上游原始列名。清单中的
`canonical_fingerprint`、`catalog_fingerprint` 和 `compilation_fingerprint` 应与当前输入和实现一致。

### 评估因子有效性

验证通过后，可以在历史时点可知的沪深 300 股票池内批量评估单因子：

```bash
rquant factors evaluate \
  --factor-set jq240 \
  --horizon 1d \
  --start 2010-01-01 \
  --end 2026-08-03
```

默认取因子最高和最低各 20% 的股票。多头端超额定义为 `Top 分组收益 - 沪深 300 收益`，
空头端超额定义为 `沪深 300 收益 - Bottom 分组收益`；正值分别表示高因子组跑赢基准、低因子组跑输基准。
年度有效性要求该年有至少 20 个可评估交易日、每个所需口袋都有有效样本，且对应一端基准调整后的
等权口袋年度收益为正。
IC、Rank IC、ICIR 和 Rank ICIR 同时保留为诊断，但不单独决定年度有效性。`1d` 标签为下一交易日
开盘到再下一交易日开盘的收益，`5d` 标签为下一交易日开盘到第六个交易日开盘的收益，`20d` 标签为
下一交易日开盘到第 21 个交易日开盘的收益。三个期限都会在每个交易日产生评估结果；`5d` 和 `20d`
分别使用 5 个和 20 个等资金口袋错峰持有，每天只轮到一个口袋按当日因子排序换仓。每个口袋内部按
持有期复利不重叠收益，全部口袋的终值再等权合并，避免把重叠窗口当成同一笔资金重复复利。

输出写入该次 `runs/factors-evaluate/<run-id>/`：

```text
factor_evaluation.json
factor_daily.parquet
factor_summary.csv
annual_effectiveness.csv
```

`annual_effectiveness.csv` 的 `effective_side` 会标记为 `long`、`short`、`both`、`neither` 或
`insufficient_data`；`dominant_side` 表示两端均有效时贡献更大的一端。`factor_daily.parquet` 的
`pocket` 列记录每日轮值口袋，汇总文件中的 `pocket_count` 与 `required_pockets` 用于核对口袋覆盖。
多空端结果只用于因子诊断，最终交易结论仍以包含 A 股交易约束的 long-only 组合回测为准。

### 用 RD-Agent 自动挖因子

`rquant rdagent export` 把截至挖掘期末（默认 2021-12-31）的研究股票池日线（默认沪市主板）导出给微软 RD-Agent 的 `fin_factor`
循环，并附带把它的回测改成次日开盘口径的准备脚本。RD-Agent 用自己 test 段的结果挑因子，所以挖掘期之后的年份
只留给 walk-forward 做样本外检验。RD-Agent 只支持 Linux，需要在 WSL2 里单独安装，并准备 LLM API key；完整步骤见
[docs/rdagent.md](docs/rdagent.md)。

```bash
rquant rdagent export --mining-end 2021-12-31
```

## 7. 执行滚动训练与样本外预测

训练前必须确认：

- `data/qlib/manifest.json` 为 `complete`；
- 所选因子集的 `data/factors/<factor-set>/manifest.json` 为 `complete`；
- 两者覆盖完整训练区间；
- 磁盘空间足以保存逐年模型、预测和 MLflow 数据库；
- 当前没有另一个任务写入相同数据或运行目录。

建议显式指定所有日期，确保结果可复现：

```bash
rquant walk-forward \
  --model lgb \
  --horizon 1d \
  --factor-set jq240 \
  --first-year 2013 \
  --last-year 2026 \
  --through 2026-08-04
```

可选模型：

- `lgb`：LightGBM；
- `double-ensemble`：Qlib DoubleEnsemble。

可选预测期限：

- `1d`：下一交易日开盘到再下一交易日开盘的收益；
- `5d`：下一交易日开盘到第六个交易日开盘的收益，并按每 5 个交易日取一个预测锚点。

每个预测年使用此前 `workflow.train_years`（默认 3）个完整自然年：训练窗口最后 `workflow.validation_months`
（默认 6）个月用于模型选择，随后在整个训练窗口内重新拟合并预测下一年。最早的预测年为 `data.formal_start`
所在年份加 `train_years`。未传 `--through` 时使用 `data.through`；两者都没有时命令会报错，不会默认取当天日期。
训练边界会按预测期限清除 1 或 5 个交易日，避免标签跨越验证或测试边界。

命令完成后会在 stdout 返回 `run_id`。输出目录结构如下：

```text
runs/walk-forward/<run-id>/
├── run.json
├── run.log
├── mlflow.db
├── walk_forward.json
├── predictions.parquet
├── year=2013/
│   ├── model.pkl
│   ├── predictions.parquet
│   └── window.json
└── year=.../
```

检查训练结果：

```bash
python -m json.tool runs/walk-forward/RUN_ID/run.json
python -m json.tool runs/walk-forward/RUN_ID/walk_forward.json
```

必须确认：

- `run.json.status` 为 `complete`；
- 输入中的 Qlib 与因子指纹符合预期；
- `walk_forward.json.status` 为 `complete`；
- 每个目标年份都有 `window.json`、模型和预测文件；
- 合并后的 `predictions.parquet` 存在，且年度预测不重叠。

如果长任务失败，先查看 `run.json.error` 和 `run.log`，不要未经诊断直接重新训练。

## 8. 运行组合回测并生成报告

使用上一步得到的 `RUN_ID`：

```bash
rquant report RUN_ID
```

报告命令优先读取 `runs/walk-forward/RUN_ID/`；如果该目录不存在，会回退读取旧版
`runs/RUN_ID/`，因此已有滚动训练结果仍可使用。

报告命令使用该运行的样本外预测执行 A 股约束下的 long-only TopkDropout 回测：

- 初始资金 100 万元；
- 默认研究股票池为沪市主板 `sh_main`，逐日剔除 ST 和上市不足 60 个交易日的股票；报告沿用训练时的股票池；
- 目标持股 50 只，按训练清单中的标签自动选择调仓方式：`5d` 每 5 个交易日调整到最新前 50 名；`1d` 每日最多换出 5 只、补入排名靠前的股票；
  调仓日未能卖出的持仓（停牌、开盘跌停）保留到下一次调仓；
- 等权、多头；
- 信号后的下一可用开盘成交；
- 买入按 100 股整数单位；
- T+1；
- 停牌不可交易；
- 按方向执行开盘涨停不可买、开盘跌停不可卖；
- 佣金、最低佣金、印花税和可选的冲击成本（`backtest.impact_cost`）来自 `config/default.yaml`；
- 最后一个可用开盘尝试在相同约束下清仓，无法卖出的证券会明确保留在清单中。

`report RUN_ID` 自动读取该运行 `walk_forward.json` 中的 `horizon`，应用 `backtest.by_horizon` 对应方案。
优先级为命令行参数 > 标签方案 > 公共 `backtest` 配置；旧配置没有 `by_horizon` 时保持原参数。
更改默认配置不会重写已有报告，也不会把旧的沪深 300 预测变成沪市主板预测。
组合参数仍可在命令行临时覆盖，例如：

```bash
rquant report RUN_ID --n-drop 5 --rebalance-days 1
```

两种口径的 LGB walk-forward 对比（2013-2026，净收益，换手为年化双边成交额 / 账户市值）：

| 因子集 | 标签 | 组合 | 年化 | 夏普 | 最大回撤 | 年换手 | 年成本 | run |
|---|---|---|---:|---:|---:|---:|---:|---|
| Alpha158 | 1d | 每日最多换 5 只 | 19.5% | 0.86 | -48.5% | 51 | 3.8% | `20261023T094024Z-2c596001` |
| Alpha158 | 1d | 每 5 日全量换 top50 | 18.8% | 0.83 | -48.6% | 72 | 5.4% | `20261023T094024Z-2c596001-top50-every5` |
| Alpha158 | 5d | 每 5 日全量换 top50 | 14.8% | 0.71 | -49.3% | 59 | 4.4% | `20261024T200621Z-88ba61b4` |

`-top50-every5` 目录只复制了原 1d 运行的预测，用新组合重新回测。

`report` 自身也会创建 `runs/report/<report-run-id>/run.json`，记录所用的组合参数、预测文件哈希和 Qlib
指纹；失败时状态为 `failed`。回测产物仍写回滚动训练的运行目录，同一 `RUN_ID` 重跑会覆盖上一次的回测产物，
对应 walk-forward `run.json` 中的 `report_run_id` 指向最近一次报告运行：

```text
runs/walk-forward/<run-id>/
├── backtest.json
├── portfolio.parquet
├── positions.parquet
├── trades.parquet
├── trade_indicators.parquet
├── report.json
└── report.md
```

`report.json` 与 `report.md` 会同时包含按自然年汇总的稳定性表。策略年度收益按每日净收益
`return - cost` 复利计算，并与同期沪深 300 基准比较；表中还会列出年度超额收益、最大回撤、
日均换手、年度累计换手和现金交易成本。最后一个年份若只覆盖部分年度，应结合表中的起止日期解读。

检查：

```bash
python -m json.tool runs/walk-forward/RUN_ID/backtest.json
python -m json.tool runs/walk-forward/RUN_ID/report.json
sed -n '1,240p' runs/walk-forward/RUN_ID/report.md
```

重点核对：

- `backtest.json.status` 是否为 `complete`；
- 回测日期是否覆盖预期预测期；
- `terminal_liquidation.complete` 是否为 `true`；
- 如未完成期末清仓，检查 `unliquidated_instruments`；
- 收益、基准、成本、换手和交易数量是否合理；
- 年度收益是否稳定、是否持续跑赢基准，以及回撤、换手和成本是否集中在少数年份；
- `trades.parquet` 与 `positions.parquet` 是否能解释组合变化。

### 任意时间窗口的指标

`portfolio.parquet` 保存了每日的 `return`、`cost`、`turnover` 和基准 `bench`，因此任意起止日期的指标都可以直接从已有运行切片计算，不需要重新训练或回测：

```bash
python scripts/analysis/window_metrics.py runs 2019-01-01 2026-10-21 \
  jq240_5d=20261028T222815Z-b7072fd9 jq240_1d_drop5=20261027T202921Z-41fb3dde-top50-drop5
```

每个 `标签=目录` 指向 `runs/walk-forward/` 下已经运行过 `report` 的目录，结束日期可以省略。输出按夏普排序，包含年化、夏普、最大回撤、Calmar、累计收益、相对基准的年化超额和信息比率、日均换手和年化成本。口径与报告相同：每日净收益为 `return - cost`，夏普为日均收益 × 252 / (日收益标准差 × √252)。

## 9. 运行清单与状态判断

所有会改变数据或产生研究结果的主流程命令都会创建 `run.json` 和 `run.log`。三个研究命令使用分类目录，
其他命令和旧版运行仍使用根目录：

```text
runs/factors-evaluate/<run-id>/run.json
runs/walk-forward/<run-id>/run.json
runs/report/<run-id>/run.json
runs/<run-id>/run.json
```

查看最近运行：

```bash
find runs -name run.json -exec ls -t {} + | head
```

查看指定运行：

```bash
python -m json.tool runs/walk-forward/RUN_ID/run.json
tail -n 100 runs/walk-forward/RUN_ID/run.log
```

因子评估和报告分别把上述 `walk-forward` 换成 `factors-evaluate`、`report`；其他命令和旧版运行省略该分类层级。

`run.json` 的关键字段：

- `command`：实际命令和参数；
- `status`：`running`、`complete` 或 `failed`；
- `config` 与 `config_fingerprint`：本次运行配置；
- `inputs`：依赖版本、日期和上游指纹；
- `outputs`：正式结果摘要；
- `error`：失败异常。

判断成功时必须同时核对清单状态和约定产物。目录存在、日志停止增长或输出文件存在，都不能单独证明运行成功。

## 10. 测试与代码检查

测试不会主动启动全历史数据同步、全量因子构建或长时间模型训练。

先确认导入路径，再运行测试：

```bash
python -c "import sys, rquant; print(sys.executable); print(rquant.__file__)"
python -m pytest tests/test_cli.py -q
python -m pytest -q
python -m ruff check src tests
```

测试分为：

- CLI 和运行清单测试；
- Tushare 数据契约、限速、缓存和恢复测试；
- 标准数据与 Qlib provider 测试；
- 因子目录与 KunQuant 冒烟测试；
- 滚动窗口、模型和 A 股交易约束测试。

单元测试通过只证明小型夹具和代码路径通过，不能替代真实数据完整性、全历史因子、滚动训练和组合回测验证。

## 11. 研究默认口径

- 原始历史：2010 年前保留足够覆盖 300 个交易日预热的数据；
- 正式研究期：从 2010 年开始；
- 数据范围：保留全部 A 股日线；
- 研究股票池：历史时点可知的沪深 300（可切换为沪深主板 `main` 或全部 A 股 `all`）；
- 标签：下一开盘到再下一开盘的 1 日收益，或下一开盘到第六个开盘的 5 日收益；
- 滚动评估：前三个完整自然年训练，下一年样本外测试；
- 组合：long-only TopkDropout，持股 50，每 5 个交易日全量调整到最新前 50 名；
- 初始资金：人民币 1,000,000 元；
- 执行：下一开盘、100 股买入单位、T+1、停牌与开盘涨跌停限制；
- 期末：在最后可用开盘尝试受约束清仓，未能卖出的持仓不会被静默当作已卖出。

IC、分组收益和多空收益可以用于诊断因子，但最终可交易结论应以包含真实 A 股约束的多头组合结果为准。

## 12. 已知研究限制

- 这是研究模拟，不是实盘账户收益；
- 没有订单簿排队模型；
- 没有市场冲击或成交容量模型；
- 日线数据无法刻画开盘集合竞价的盘中成交优先级；
- 回测结果不能替代交易前风险评估。

## 13. 常见问题

### `ModuleNotFoundError: No module named 'rquant'`

先执行 `conda activate rquant`，再按第 1 节检查解释器和导入路径。如果仍无法导入，按
`environment.yml` 更新 `rquant` 环境，不要改用其他环境。

### `doctor` 报 Tushare 权限失败

核对 token 是否正确、账户积分和接口权限是否覆盖失败列表。不要因为单个接口失败就启动全量同步；先解决权限问题。

### 数据同步中断

查看对应 `run.json` 和 `run.log`。正常情况下可使用完全相同的 `--through` 参数重新执行，已通过哈希验证的分区会复用。
不要默认使用 `--force`。

### 提示上游清单不存在或不是 `complete`

按工作流顺序回到最近一个未完成步骤：

```text
data/raw/sync_manifest.json
  → data/canonical/manifest.json
  → data/qlib/manifest.json
  → data/factors/<factor-set>/manifest.json
  → runs/walk-forward/<run-id>/walk_forward.json
  → runs/walk-forward/<run-id>/backtest.json
```

不要手工把清单状态改成 `complete`。

### 因子验证失败

核对因子集、标准数据指纹、因子目录指纹、列数和列顺序。修复生成逻辑后重新构建；不要直接编辑 Parquet 或删除失败证据。

### 滚动训练失败

先检查命令参数、目标年份、`through` 日期、Qlib/因子清单指纹、交易日历覆盖范围、磁盘空间和 `run.json.error`。
昂贵任务失败后不要在未定位原因时反复重启。

## 完成检查表

- [ ] 已执行 `conda activate rquant`，且当前解释器来自该环境；
- [ ] `rquant doctor` 返回 `status: ok`；
- [ ] 原始数据同步清单为 `complete`，截止日期正确；
- [ ] 标准数据与 Qlib 清单为 `complete`，指纹衔接一致；
- [ ] 所选因子集已完整构建并通过验证；
- [ ] 滚动训练的年度窗口、模型和合并预测均完整；
- [ ] 回测及报告产物完整；
- [ ] 期末未清仓持仓已经核对；
- [ ] 所有研究结论都能追溯到对应 `run_id`、配置、输入指纹和产物。
