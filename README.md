# RQuant

RQuant 是一个个人、非商业用途的 A 股日频量化研究框架，使用：

- Tushare 作为唯一市场数据源；
- Qlib 管理数据集、模型、实验记录和组合回测；
- KunQuant 编译并计算 Alpha158 与 Alpha101 因子。

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
rquant doctor
rquant data sync --through 2026-08-04
rquant data build-qlib
rquant factors build --factor-set combined
rquant factors validate --factor-set combined
rquant walk-forward --model lgb --horizon 1d --factor-set combined \
  --first-year 2013 --last-year 2026 --through 2026-08-04
rquant report RUN_ID
```

数据同步、Qlib 构建、全量因子计算、滚动训练和回测都可能耗时较长。运行前应确认参数、剩余磁盘空间和输出位置。

## 1. 环境要求

需要：

- macOS 或兼容的类 Unix 环境；
- Conda；
- Python 3.11；
- `clang++`；
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
- `pyqlib==0.9.7`、`KunQuant==0.1.11`、`tushare==1.4.29`；
- NumPy、Pandas、PyArrow、LightGBM 等依赖能否导入；
- `clang++`；
- Tushare token；
- 259 个因子的固定目录；
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
| `data.universe`                  |     `csi300` | 历史时点沪深 300 股票池      |
| `data.max_requests_per_minute`   |        `190` | 所有请求及重试共享的分钟限速 |
| `factors.factor_set`             |   `combined` | 默认因子集                   |
| `factors.workers`                |          `4` | KunQuant 执行线程数          |
| `workflow.train_years`           |          `3` | 每个预测年使用的历史训练年数 |
| `workflow.first_prediction_year` |       `2013` | 第一个样本外预测年           |
| `backtest.initial_capital`       |    `1000000` | 初始资金，人民币             |
| `backtest.topk`                  |         `50` | 目标持股数量                 |
| `backtest.n_drop`                |          `5` | 每期最多换出数量             |

如需使用其他配置文件，所有命令都可以在主命令前指定：

```bash
rquant --config config/你的配置.yaml doctor
```

运行时会把完整配置及其指纹写入 `runs/<run-id>/run.json`，因此不要依赖未记录的临时参数。

## 4. 同步原始数据

指定一个包含式截止日期：

```bash
rquant data sync --through 2026-08-04
```

同步过程具有以下行为：

- 为 2010 年正式研究期自动准备 300 个交易日的预热历史；
- 获取全 A 股日线及所需参考数据，同时保留历史时点沪深 300 成分信息；
- 按 `through` 日期留存申万行业快照，分页获取历史已失效记录（`is_new=N`）和当前有效记录（`is_new=Y`）；标准数据只使用最新完整快照，以保持行业时点覆盖并避免当前记录被永久缓存；
- 每个分区都带有内容哈希和清单；
- 已完成且哈希有效的分区会作为缓存复用；
- 损坏或不完整的分区会重新获取；
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

该命令依次完成两项工作：

1. 将不可变 Tushare 分区转换为 `data/canonical/` 下的标准日线契约；
2. 从标准数据构建 `data/qlib/` 下的 Qlib provider。

标准数据会处理：

- Tushare 成交量从“手”转换为“股”；
- 成交额从“千元”转换为“元”；
- 以同口径成交额除以成交量计算 VWAP；
- 原始价格、复权价格、成交量与复权因子；
- 停牌缺口、开盘涨跌停限制；
- 历史时点沪深 300 成分；
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

重建 Qlib provider 时采用临时目录和原子替换；已有 provider 会保存为带时间戳的
`data/qlib.previous.<时间>/` 备份。确认新 provider 正常后再考虑清理旧备份。

## 6. 查看、构建和验证因子

### 查看固定因子目录

```bash
rquant factors catalog
rquant factors catalog --factor-set qlib_alpha158
rquant factors catalog --factor-set wq_alpha101
rquant factors catalog --factor-set combined
```

导出机器可读目录：

```bash
rquant factors catalog --factor-set combined --format json --output factor_catalog.json
rquant factors catalog --factor-set combined --format csv --output factor_catalog.csv
```

因子集：

- `qlib_alpha158`：`a158_001` 至 `a158_158`，共 158 列；
- `wq_alpha101`：`a101_001` 至 `a101_101`，共 101 列；
- `combined`：先排列 158 列 Alpha158，再排列 101 列 Alpha101，共 259 列。

固定目录保留来源名称、公式标识、序号、最大回看期和实现方式，但模型特征只暴露规范列名。

### 构建因子

完整研究通常构建组合因子集：

```bash
rquant factors build --factor-set combined
```

如需进行限定日期的诊断构建，可以指定：

```bash
rquant factors build --factor-set combined --start 2020-01-01 --end 2020-12-31
```

因子结果按年份写入：

```text
data/factors/combined/year=YYYY/factors.parquet
data/factors/combined/manifest.json
```

限定日期构建不能代替完整研究所需的全历史因子。开始滚动训练前，要确认因子清单覆盖所有训练、验证、测试和标签所需日期。

### 验证因子

```bash
rquant factors validate --factor-set combined
python -m json.tool data/factors/combined/manifest.json
```

验证会检查所有分区的因子列是否完整、无重复、顺序固定，并拒绝暴露上游原始列名。清单中的
`canonical_fingerprint`、`catalog_fingerprint` 和 `compilation_fingerprint` 应与当前输入和实现一致。

### 评估因子有效性

验证通过后，可以在历史时点可知的沪深 300 股票池内批量评估单因子：

```bash
rquant factors evaluate \
  --factor-set combined \
  --horizon 1d \
  --start 2010-01-01 \
  --end 2026-08-03
```

默认取因子最高和最低各 20% 的股票。多头端超额定义为 `Top 分组收益 - 沪深 300 收益`，
空头端超额定义为 `沪深 300 收益 - Bottom 分组收益`；正值分别表示高因子组跑赢基准、低因子组跑输基准。
年度有效性要求该年有至少 20 个可评估交易日，且对应一端基准调整后的复合年度收益为正。
IC、Rank IC、ICIR 和 Rank ICIR 同时保留为诊断，但不单独决定年度有效性。`1d` 标签为下一交易日
开盘到再下一交易日开盘的收益，`5d` 标签为下一交易日开盘到第六个交易日开盘的收益，`20d` 标签为
下一交易日开盘到第 21 个交易日开盘的收益。使用 `5d` 或 `20d` 标签时分别每 5 或 20 个交易日选取一个
非重叠评估锚点。

输出写入该次 `runs/<run-id>/`：

```text
factor_evaluation.json
factor_daily.parquet
factor_summary.csv
annual_effectiveness.csv
```

`annual_effectiveness.csv` 的 `effective_side` 会标记为 `long`、`short`、`both`、`neither` 或
`insufficient_data`；`dominant_side` 表示两端均有效时贡献更大的一端。多空端结果只用于因子诊断，
最终交易结论仍以包含 A 股交易约束的 long-only 组合回测为准。

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
  --factor-set combined \
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

每个预测年使用此前三个完整自然年：前一年的下半年用于模型选择，随后在三年训练窗口内重新拟合并预测下一年。
训练边界会按预测期限清除 1 或 5 个交易日，避免标签跨越验证或测试边界。

命令完成后会在 stdout 返回 `run_id`。输出目录结构如下：

```text
runs/<run-id>/
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
python -m json.tool runs/RUN_ID/run.json
python -m json.tool runs/RUN_ID/walk_forward.json
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

报告命令使用该运行的样本外预测执行 A 股约束下的 long-only TopkDropout 回测：

- 初始资金 100 万元；
- 目标持股 50 只，每期最多换出 5 只；
- 等权、多头；
- 信号后的下一可用开盘成交；
- 买入按 100 股整数单位；
- T+1；
- 停牌不可交易；
- 按方向执行开盘涨停不可买、开盘跌停不可卖；
- 佣金、最低佣金和印花税来自 `config/default.yaml`；
- 最后一个可用开盘尝试在相同约束下清仓，无法卖出的证券会明确保留在清单中。

报告会写回同一个运行目录：

```text
runs/<run-id>/
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
python -m json.tool runs/RUN_ID/backtest.json
python -m json.tool runs/RUN_ID/report.json
sed -n '1,240p' runs/RUN_ID/report.md
```

重点核对：

- `backtest.json.status` 是否为 `complete`；
- 回测日期是否覆盖预期预测期；
- `terminal_liquidation.complete` 是否为 `true`；
- 如未完成期末清仓，检查 `unliquidated_instruments`；
- 收益、基准、成本、换手和交易数量是否合理；
- 年度收益是否稳定、是否持续跑赢基准，以及回撤、换手和成本是否集中在少数年份；
- `trades.parquet` 与 `positions.parquet` 是否能解释组合变化。

## 9. 运行清单与状态判断

所有会改变数据或产生研究结果的主流程命令都会创建：

```text
runs/<run-id>/run.json
runs/<run-id>/run.log
```

查看最近运行：

```bash
ls -td runs/* | head
```

查看指定运行：

```bash
python -m json.tool runs/RUN_ID/run.json
tail -n 100 runs/RUN_ID/run.log
```

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
- 研究股票池：历史时点可知的沪深 300；
- 标签：下一开盘到再下一开盘的 1 日收益，或下一开盘到第六个开盘的 5 日收益；
- 滚动评估：前三个完整自然年训练，下一年样本外测试；
- 组合：long-only TopkDropout，持股 50，只换出 5；
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
  → runs/<run-id>/walk_forward.json
  → runs/<run-id>/backtest.json
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
