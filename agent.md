# RQuant Agent Guide

本文件适用于当前目录及其所有子目录。若更深层目录存在更具体的 agent 指令，以更具体的指令为准。

## 1. 项目定位

RQuant 是个人、非商业用途的 A 股日频研究框架：

- Tushare 是唯一市场数据源；
- Qlib 用于数据集、模型、记录器与组合回测；
- KunQuant 用于编译和计算 Alpha158、Alpha101 因子；
- 当前工程独立于旧版 RQuant 与 StockTradebyZ，禁止从那些工程隐式导入、复制运行产物或混用环境；
- 当前工程不内嵌 Qlib 上游源码；运行时只使用 `pyproject.toml` 锁定并安装在 `rquant` 环境中的
  `pyqlib==0.9.7`。除非任务明确要求开发 Qlib 上游，否则不要在项目根目录克隆 Qlib，也不要
  将源码以 editable 方式安装到 `rquant` 环境。

实现、测试和文档必须保持以下研究口径：

- 正式研究从 2010 年开始，并保留至少 300 个交易日的预热数据；
- 股票池是历史时点可知的沪深 300 成分，不得使用当前成分回填历史；
- 1 日标签为下一开盘到再下一开盘的收益；5 日标签为下一开盘到第六个开盘的收益；
- 滚动训练使用此前三个完整自然年，测试下一年；
- 组合是 A 股约束下的 long-only TopkDropout，而不是可交易的多空组合；
- 交易约束包括次日开盘成交、100 股买入单位、T+1、停牌与涨跌停限制，以及未能在期末卖出的持仓披露。

## 2. 先确认工作区与解释器

每次任务开始先确认实际目录，不要根据旧会话或相似路径猜测：

```bash
pwd
realpath .
```

本项目只使用名为 `rquant` 的 Python 3.11 Conda 环境。不要使用系统 Python、`base`、`stocktrade` 或其他工程的环境。

执行 Python 命令前先验证：

```bash
conda activate rquant
python --version
python -c "import os, sys, rquant; print(os.environ.get('CONDA_DEFAULT_ENV')); print(sys.executable); print(rquant.__file__)"
```

`CONDA_DEFAULT_ENV` 必须为 `rquant`，`sys.executable` 必须来自该环境，`rquant.__file__` 必须指向当前工作区。若导入失败或指向另一个 checkout：

1. 先报告实际输出和相关 `.pth`/安装信息；
2. 不要用全局安装、切换环境或长期设置 `PYTHONPATH` 来掩盖问题；
3. 需要重建或更新 `rquant` 环境时，先说明影响并取得用户同意，再按 `README.md` 和 `environment.yml` 操作。

Tushare token 只能来自环境变量 `TUSHARE_TOKEN`。不得把 token 写入源码、配置、日志、测试夹具或回复。

## 3. 权威来源与目录职责

发生冲突时，按以下顺序核对事实：

1. 当前任务的用户要求；
2. 本文件及更深层 agent 指令；
3. 可执行源码和测试；
4. `config/default.yaml`、数据清单和运行清单；
5. `README.md`。

主要目录：

- `src/rquant/`：产品代码；
- `tests/`：快速、确定性的自动化测试；
- `config/default.yaml`：默认研究和执行参数；
- `data/raw/`：不可变的 Tushare 分区、分区级清单及总同步清单；
- `data/canonical/`：标准化、可审计的日线数据；
- `data/qlib/`：Qlib provider；
- `data/factors/`：按因子集和年份分区的因子结果；
- `data/cache/`：KunQuant 编译缓存；
- `runs/<run-id>/`：运行清单、日志、预测、回测与报告产物；
- Qlib 上游源码不属于本工程；需要查阅时使用独立于项目根目录的临时或同级 checkout。

不要手工编辑生成数据来“修好”结果。应修复生成逻辑，然后通过正式命令重建相应产物。

## 4. 工作方式

### 修改前

- 先阅读相关源码、测试、配置和已有清单；
- 检查工作区是否有同名文件、未完成运行或用户正在进行的修改；
- 当前项目根目录可能不是 Git 仓库，不得假设 `git status` 可用；若操作 `qlib/`，单独检查它自己的 Git 状态；
- 诊断失败运行时，优先查看 `runs/<run-id>/run.json` 与 `run.log`，同时核对命令参数、输入指纹、状态和输出文件；
- 只修改完成任务所需的最小范围，保留所有无关文件和用户产物。

### 修改时

- 保持 Python 3.11 兼容，遵循 `pyproject.toml` 中 Ruff 规则和 120 字符行宽；
- 新行为必须补充或更新测试，优先使用小型临时夹具，不依赖网络和全量本地数据；
- 路径必须经项目根目录解析，运行数据不得逃出项目根目录；
- 写入清单和关键产物应保持原子性；失败必须留下明确的 `failed` 状态与异常信息；
- 保留配置、数据、因子、编译和运行指纹，任何会改变结果的参数都必须进入可审计输入；
- 不吞掉异常，不用空结果冒充成功，不通过关闭警告来掩盖数值或数据问题。

### 修改后

- 先运行最小相关测试，再运行完整快速测试；
- 检查格式和静态规则；
- 报告实际运行过的命令、通过数量、跳过项和未验证范围；
- 不把单元测试通过表述为真实数据同步、全量因子构建、训练或回测已经验证。

## 5. 数据与研究契约

以下内容属于高风险语义，修改前必须先定位契约、调用链和回归测试：

- Tushare 单位换算：`vol` 为手、`amount` 为千元；标准化后 `raw_volume` 为股、`raw_amount` 为元；
- VWAP 必须由同口径的成交额除以成交量得到，不能用典型价格或 OHLC 均值替代；
- 复权价格、成交量和 `factor` 必须能恢复原始价格，不能混合不同复权基准；
- 停牌日要保留为不可交易缺口，不能用前值伪造正常行情；
- 指数成分、行业、涨跌停和其他横截面信息必须保持 point-in-time 语义；
- 标签、特征和股票池必须按同一交易日历对齐，并执行与预测期限相符的 purge；
- 因子列名和顺序是稳定契约：Alpha158 为 `a158_001` 至 `a158_158`，Alpha101 为 `a101_001` 至 `a101_101`；不得暴露上游原始名称；
- NaN、Inf、常数截面、零方差和极大有限值必须被显式处理，不能让相关系数或评估结果静默失真；
- IC、多空分组收益只用于因子诊断；最终可交易结论以受约束的 long-only 回测为准。

涉及上述语义的改动，至少需要一个能在改动前失败、改动后通过的聚焦回归测试。

### Tushare 同步契约

- `data sync` 的进度必须以真实分区完成数为准，并在 stderr 显示当前端点、分区、缓存命中数、新获取数、耗时、速率和 ETA；最终机器可读 JSON 保持只输出到 stdout；
- `--no-progress` 只能关闭终端进度显示，不得改变采集、缓存、清单或最终 JSON 的语义；
- 分区仅在 Parquet 与相邻清单同时存在、清单状态为 `complete`、查询指纹一致且文件 SHA-256 未变化时才可复用；损坏或不完整的缓存必须重新获取；
- 同步被中断或失败后，先检查已有 `run.json`、`run.log`、分区清单和 `data/raw/sync_manifest.json`。需要继续时使用相同的 `--through` 参数依靠缓存续跑，除非用户明确要求全量重抓，否则不得添加 `--force`；
- 一个 `TushareCollector` 的所有端点共用同一个 60 秒滑动窗口限流器；每次真实请求尝试都计入额度，包括提供方报错后的重试；
- `data.max_requests_per_minute` 必须是正整数。默认值 `190` 是项目级保守保护，不代表所有 Tushare 账户和端点都允许 190 次/分钟；遇到端点专有限额时应采用更严格限制；
- 修改采集阶段、缓存判定、重试、限流或进度输出时，必须同步更新 CLI、配置、README 和聚焦测试，并保持 stdout/stderr 契约可回归验证。

## 6. 命令与验证层级

安全的只读或轻量检查：

```bash
conda activate rquant
rquant doctor --skip-permission-check
rquant factors catalog
python -m pytest tests/test_cli.py -q
python -m ruff check src tests
```

完整快速测试：

```bash
conda activate rquant
python -m pytest -q
```

如果解释器或 `rquant` 导入来源检查未通过，不要把后续测试失败归因于业务代码，也不要声称完成验证。

以下命令可能访问网络、处理全量数据或长时间占用资源，除非用户明确要求启动，否则只给出准确命令、预计影响和产物位置，不要自行执行：

```bash
conda activate rquant
rquant doctor
rquant data sync --through YYYY-MM-DD
rquant data build-qlib
rquant factors build --factor-set combined
rquant factors validate --factor-set combined
rquant walk-forward --model lgb --horizon 1d --factor-set combined
rquant report RUN_ID
```

获得授权启动长任务后：

- 使用 CLI，而不是临时脚本直接调用内部 runner；
- 明确记录完整参数与目标日期；
- 续跑数据同步时优先复用已验证分区，并分别核对 `cached_artifacts` 与 `fetched_artifacts`，不得把进度条到达 100% 单独当作完成证据；
- 通过 `runs/<run-id>/run.json` 的 `status`、`inputs`、`outputs` 和 `error` 判断结果；
- 同时核对 `run.log` 和约定产物，不能仅凭目录存在判断成功；
- 失败后先诊断已有运行，不要未经确认直接重新执行昂贵任务。

## 7. 安全边界

- 不删除或覆盖 `data/`、`runs/`、模型、缓存或用户文件；需要清理时先列出精确目标，并优先采用可恢复方式；
- 不执行 `git reset --hard`、批量 checkout、递归删除或其他破坏性命令；
- 不修改无关配置来让测试通过；
- 不联网查询或上传本地数据、token、持仓、日志和运行产物；
- 不新增依赖，除非现有依赖无法合理完成任务，并先说明必要性与锁定方式；
- 不在未经授权的情况下发布、提交、推送或创建 PR。

## 8. 交付标准

最终回复应简洁且可审计，至少说明：

- 修改了什么及原因；
- 关键文件的准确路径；
- 实际执行的验证及结果；
- 未执行的长任务或真实数据验证；
- 任何已知风险、失败清单或下一步所需授权。

证据不足时明确写“未验证”，不要用推测补齐结论。
