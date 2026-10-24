# RQuant

RQuant 是一个个人、非商业用途的 A 股日频量化研究框架，使用：

- Tushare 作为唯一市场数据源；
- Qlib 管理数据集、模型、实验记录和组合回测；
- KunQuant 编译并计算 Alpha158、Alpha101 与 Alpha360 因子；审计过的 Pandas 面板后端计算 GTJA191。

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
rquant factors build --factor-set Hquality
rquant factors validate --factor-set Hquality
rquant data sync-fundamentals --through 2026-08-04   # 仅 china_chars 需要
rquant data build-reference
rquant factors build --factor-set china_chars
rquant factors build --factor-set size_value_china    # 只需 build-reference，不需要财报
rquant factors evaluate --factor-set qlib_alpha158 \
  --horizon 1d --start 2010-01-01 --end 2026-08-04
  
rquant walk-forward --model lgb --horizon 1d --factor-set Hquality \
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
- `pyqlib==0.9.7`、`KunQuant==0.1.11`、`tushare==1.4.29`；
- NumPy、Pandas、PyArrow、LightGBM 等依赖能否导入；
- KunQuant 所用的 C++ 编译器（macOS 为 `clang++`，Windows 为 `cl.exe`）；
- Tushare token；
- 939 个当前可构建因子的固定目录（Alpha158、Alpha101、GTJA191、Alpha360、china_chars、size_value_china）；
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
| `data.universe`                  |     `csi300` | 研究股票池：`csi300`（历史时点沪深 300）或 `all` |
| `data.max_requests_per_minute`   |        `190` | 所有请求及重试共享的分钟限速 |
| `data.sync_workers`              |          `8` | 逐日分区并发请求线程数，用于掩盖网络延迟；总速率仍受上面的限速约束 |
| `data.fundamentals_start_year`   |       `2004` | `data sync-fundamentals` 抓取财报的最早公告年份 |
| `data.fundamentals_window_years` |          `6` | 财报按公告日分窗抓取的年数；已结束的窗口会缓存复用 |
| `factors.factor_set`             |   `Hquality` | 默认因子集                   |
| `factors.workers`                |          `4` | KunQuant 执行线程数          |
| `workflow.train_years`           |          `3` | 每个预测年使用的历史训练年数 |
| `workflow.validation_months`     |          `6` | 训练窗口最后几个月用于选择迭代数 |
| `workflow.first_prediction_year` |       `2013` | 第一个样本外预测年           |
| `backtest.initial_capital`       |    `1000000` | 初始资金，人民币             |
| `backtest.topk`                  |         `50` | 目标持股数量                 |
| `backtest.n_drop`                |          `5` | 每期最多换出数量             |
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

财报需要单独同步（按股票、按公告日窗口请求单股接口，不需要 `*_vip` 批量接口权限）：

```bash
rquant data sync-fundamentals --through 2026-08-04
rquant data build-reference
```

`sync-fundamentals` 只抓取股票池可能包含的股票（`csi300` 为全部历史成分，`all` 为全部 A 股），结果清单写在
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
rquant factors catalog --factor-set Hquality
```

导出机器可读目录：

```bash
rquant factors catalog --factor-set Hquality --format json --output factor_catalog.json
rquant factors catalog --factor-set Hquality --format csv --output factor_catalog.csv
```

因子集：

- `qlib_alpha158`：`a158_001` 至 `a158_158`，共 158 列；
- `wq_alpha101`：`a101_001` 至 `a101_101`，共 101 列；
- `gtja191`：`gtja_001` 至 `gtja_191`（暂时排除 `gtja_030`），共 190 列；
- `qlib_alpha360`：`a360_001` 至 `a360_360`，共 360 列；它是 Qlib Alpha360 的六组归一化
  OHLC、VWAP 与成交量字段各取 60 个滞后期，而非 360 条手工设计的独立信号；
- `Hquality`：从上述因子中按年度 RankIC 稳定性筛出并按相关性去重后的 137 列，来自 Alpha158、Alpha101 与 GTJA191（不含 Alpha360），
  成员、方向与筛选口径锁定在 `src/rquant/factors/libraries/hquality.py`（见下文“高质量因子集 Hquality”）；
- `china_chars`：`cnc_001` 至 `cnc_109`，共 109 列中国 A 股股票特征（流动性、风险、动量、盈利、价值、投资等），
  见下文“中国 A 股股票特征 china_chars”；
- `size_value_china`：`svc_001` 至 `svc_021`，共 21 列，Liu-Stambaugh-Yuan“Size and Value in China”的
  规模、EP、异常换手率以及 CH-3/CH-4 因子载荷，见下文“Size and Value in China 因子集 size_value_china”。

原先把 Alpha158 与 Alpha101 拼成 259 列的 `combined` 因子集已移除；`runs/` 下用它跑出的历史结果仍保留，
但当前代码不再构建或读取 `data/factors/combined`。

固定目录保留来源名称、公式标识、序号、最大回看期和实现方式，但模型特征只暴露规范列名。

### 因子库代码结构

所有可研究的因子公式由 RQuant 自己维护。Alpha158/Alpha101/Alpha360 使用 KunQuant 后端，GTJA191 使用
Pandas 宽表后端；它们共用目录、清单、年度 Parquet、验证器和下游加载契约：

```text
src/rquant/factors/
├── extensions/
│   └── kunquant.py                      # KunQuant 缺失能力的最小扩展与适配
├── panel_operators.py                   # 公共 Pandas 面板算子
├── catalog.py                           # 动态目录、稳定命名和实现指纹
├── engine.py                            # 后端选择、执行和标准分区写入
├── selection.py                         # 按年度稳定性、覆盖率与相关性筛选因子
└── libraries/
    ├── base.py                          # 因子库接口与输入字段契约
    ├── registry.py                      # 因子库及组合因子集注册表
    ├── alpha158.py                      # 158 条本地 Alpha158 公式
    ├── alpha101.py                      # 101 条本地 Alpha101 公式
    ├── alpha360.py                      # 6 个归一化字段各 60 个滞后期的 Alpha360 图
    ├── gtja191.py                       # 完整 GTJA191 提供者与公式语义说明
    ├── hquality.py                      # Hquality 选择型因子集的锁定成员与方向
    ├── china_chars.py                   # 109 个日频 point-in-time 中国 A 股股票特征
    ├── size_value_china.py              # 21 个 CH-3/CH-4 特征与因子载荷
    └── custom.py                        # 自定义小型因子库构造器
```

Alpha158、Alpha101 与 Alpha360 的公式定义均保存在 RQuant 源码中，不从
`KunQuant.predefined.Alpha158/Alpha101` 导入。KunQuant 仅提供算子、图优化、C++ 代码生成和运行时；公式变更会
进入 RQuant 的实现指纹。Alpha158 与 Alpha101 的迁入公式基于 KunQuant 0.1.11 的 Apache-2.0 源码；
Alpha360 的迁入公式基于 Qlib 0.9.7 的 MIT 源码。两项来源与许可证文件路径均见
`THIRD_PARTY_NOTICES.md`。

因子公式优先直接组合 KunQuant 公共算子。只有 KunQuant 0.1.11 无法表达的能力才放入
`extensions/kunquant.py`；当前仅包含 Alpha101 行业中性化所需的横截面扩展和轻量符号适配，不复制
`OpBase`、优化 pass、代码生成器或运行时。

### 添加自己的因子库

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
- `gtja_030` 的公式仍保留在源码中，但因当前缺少可审计的逐日 `mkt`、`smb`、`hml` 风格收益，暂时不注册到构建目录。

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

### 构建因子

Alpha360 是独立的因子集；先检查目录，再在明确需要新产物且已确认磁盘空间、无并发写入者后构建：

```bash
rquant factors catalog --factor-set qlib_alpha360
rquant factors build --factor-set qlib_alpha360
rquant factors validate --factor-set qlib_alpha360
```

完整构建会在完成 staging 后原子替换整个 `data/factors/qlib_alpha360/` 目录，而不是向既有年度分区追加；本文
不执行该构建。

### 高质量因子集 Hquality

`Hquality` 不重算公式，而是从已构建的 `qlib_alpha158`、`wq_alpha101`、`gtja191` 复制所选列，因此必须先构建
这三个源因子集。构建会核对每个源清单的状态、目录指纹、引擎指纹、标准数据指纹与分区哈希，任一过期即失败：

```bash
rquant factors build --factor-set Hquality
rquant factors validate --factor-set Hquality
rquant walk-forward --model lgb --horizon 1d --factor-set Hquality
```

筛选口径（锁定在 `hquality.py` 的 `criteria`，基于 2010-01-01 至 2026-10-19 的 1d 单因子评估）：

- 方向取全区间日均 Rank IC 的符号，年度统计都按该方向定向，稳定为负的因子以 `direction=-1` 反向保留；
- 只统计评估日数不少于 120 天的年份；
- 定向年度 Rank IC 为正的年份占比不低于 70%；
- 不要求年度多空收益为正：要求 75% 年份为正时只剩 43 个因子，LGB 回测年化 10.1%，明显低于 Alpha158 加 Alpha101 的 18.7%；
- 全区间 |Rank IC| 不低于 0.01；
- 排除全部 Alpha360：保留其 42 个滞后特征时回测年化 16.2%、夏普 0.74，去掉后为 19.1%、0.83；
- 覆盖率（各交易日有效观测数之和除以当日覆盖最好的因子观测数之和）不低于 80%；
- 去重：通过以上条件的 268 个因子按 |Rank IC| 从强到弱排序，若与已保留因子的日均截面 Spearman 相关系数（每隔
  5 个交易日取样）绝对值不低于 0.8 就剔除，最终保留 137 个；被剔除因子与其对应的保留因子记在 `DEDUPLICATED`。

成员保留源规范列名与原值，`direction=-1` 只标记反向使用，存储值不取负；树模型不受符号影响。

完整研究先构建各个源因子集，再构建 Hquality：

```bash
rquant factors build --factor-set qlib_alpha158
rquant factors build --factor-set Hquality
```

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
  上个月至少 15 天有交易；按总市值剔除最小的 30%（壳价值污染），剔除在交易过滤之前排序；
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

### 验证因子

```bash
rquant factors validate --factor-set Hquality
python -m json.tool data/factors/Hquality/manifest.json
```

验证会检查所有分区的因子列是否完整、无重复、顺序固定，并拒绝暴露上游原始列名。清单中的
`canonical_fingerprint`、`catalog_fingerprint` 和 `compilation_fingerprint` 应与当前输入和实现一致。

### 评估因子有效性

验证通过后，可以在历史时点可知的沪深 300 股票池内批量评估单因子：

```bash
rquant factors evaluate \
  --factor-set Hquality \
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
  --factor-set Hquality \
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
- 目标持股 50 只，每期最多换出 5 只；
- 等权、多头；
- 信号后的下一可用开盘成交；
- 买入按 100 股整数单位；
- T+1；
- 停牌不可交易；
- 按方向执行开盘涨停不可买、开盘跌停不可卖；
- 佣金、最低佣金、印花税和可选的冲击成本（`backtest.impact_cost`）来自 `config/default.yaml`；
- 最后一个可用开盘尝试在相同约束下清仓，无法卖出的证券会明确保留在清单中。

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
