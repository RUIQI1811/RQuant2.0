# 新增因子集接入说明（给外部 coding agent）

本文件写给 GPT、Codex 等外部 coding agent：把网上公开的一套因子（论文附录、因子库文档、开源项目的因子定义）
按本仓库的规矩实现成一个新的、可构建、可回测的因子集。通用规则见根目录 `agent.md`，本文件只补充“新增因子集”
这一类任务的具体做法；两者冲突时以 `agent.md` 为准。

## 0. 一句话版本

在新分支上新增 `src/rquant/factors/libraries/<name>.py`（KunQuant 实现）、在注册表注册、写 `tests/test_<name>.py`、
更新目录计数和文档，让 `ruff` 和 `pytest` 全部通过，开 draft PR 交给用户审。真实数据的构建和回测由用户在自己的
Windows 机器上跑，你只负责把命令写清楚。

## 1. 分支与 PR 规则（必须遵守）

- 从最新 `main` 切新分支，例如 `feat/factors-<name>`；一个 PR 只做一个因子集。
- **不得推送 `main`，不得合并 PR，不得改写已有分支历史**。PR 由用户审阅、在 Windows 上验证后自己合并。
- PR 开成 draft。PR 描述必须包含：来源与版本、实现/跳过数量、跳过清单摘要、改动文件、你实际跑过的测试命令和
  结果、用户需要在 Windows 上执行的完整命令（见第 7 节），以及会因本 PR 失效、需要重建的已有因子库（如果有）。
- 不要在同一个 PR 里顺手改无关代码、重排已有因子、改默认配置或改其他因子集的列。

## 2. 硬性约束

1. **只用 KunQuant 实现**。`backend` 必须是默认的 `"kunquant"`；不要新增 `PanelFactorLibrary`（Pandas 后端），
   也不要用 `KunQuant.predefined` 里的现成公式。公式由本仓库自己维护。
2. **独立重写，标明出处**。按公开的定义重写，不复制 GPL 或许可不明的代码。模块 docstring 写清楚：来源（论文/
   网站/仓库 URL）、版本或访问日期、日频改造方式、与原定义的差异。每个因子在目录表里保留原名（`source_name`）
   和一句话定义。
3. **做不出来的因子单独列出并写原因**。照 `oap.py` 的 `OAP_SKIPPED`：一个 `<NAME>_SKIPPED: dict[str, str]`，
   键是源因子名，值是原因（例如“需要分析师预期，Tushare 当前积分拿不到”）。实现的加跳过的必须恰好等于来源的
   完整清单，并用测试锁住这个数。
4. **不能有前视**。任何因子在第 t 行只能用到 t 日收盘及以前已知的信息：行情、指数成分、行业、市场因子收益、
   财报（按首次披露、公告日起可用）。测试必须用“截断数据重算”验证（第 6 节）。
5. **不能用收益挑因子**。新因子集是一个完整的公式库，不要按 IC、回测收益筛选成员（用户已删除所有按收益选出来的
   因子集）。原定义中本来就有的去重或冗余剔除可以保留，并写明原因。
6. **不新增依赖**。确实需要时先在 PR 里说明，并锁定版本。
7. **不碰 token、数据和运行产物**。`TUSHARE_TOKEN` 只来自环境变量；不要提交 `data/`、`runs/` 下的任何东西。

## 3. 现有可参考的实现

先读 `src/rquant/factors/libraries/base.py`（`FactorLibrary`、`FactorSpec` 的契约）和
`src/rquant/factors/libraries/registry.py`，再按新因子集需要的数据挑一个最接近的模板：

| 需要的数据 | 参考实现 | 要点 |
|---|---|---|
| 只要 OHLCV 的技术指标 | `tushare_tech.py`、`talib_indicators.py` | `quote_fields` 让时序窗口跨过停牌；`cpp_prelude` + `SeriesKernelOp`（`kunquant_panel.series_kernel`）逐股跑 C++；价格单位的输出除以 close 无量纲化 |
| 价量公式（Alpha 风格表达式） | `alpha158.py`、`alpha101.py`、`gtja191.py` | 直接组合 KunQuant 算子与 `extensions/kunquant_panel.py` 中的窗口、排名算子 |
| daily_basic、市场因子、财报、截面排序 | `jq240.py`（最推荐的模板）、`oap.py`、`china_chars.py`、`jkp153.py`、`barra_cne6.py` | `load_reference_inputs` 读参考数据并记录文件哈希；`statement_fields`/`statement_outputs`/`build_statements` 在季度×股票的财报图里算，再按公告日发布成日频输入；截面运算只用 `universe` 成员 |
| CH-3/CH-4 等风格因子载荷 | `size_value_china.py` | 读 `reference/ch4_factors.parquet` |

扩展机制（都在 `src/rquant/factors/extensions/`）：

- `kunquant_panel.py`：带 Pandas 语义的窗口/递推算子、`cpp_state`（逐股带状态的 C++ 步进）、
  `series_kernel`（`SeriesKernelOp`，逐股一次读完整条序列的 C++ 内核）、`scalar_map`（逐元素调用 C 库函数）。
- `kunquant_oap.py`、`kunquant_jq240.py`、`kunquant_oap_regression.py`：截面排名/分组/回归、滚动回归等。
- 只有现有算子确实表达不了时才新增扩展模块，命名 `extensions/kunquant_<name>.py`，并在因子库模块里以
  `from rquant.factors.extensions import kunquant_<name> as ...` 整模块导入——这样 `implementation_fingerprint`
  会自动包含它的源码。若用到了其他模块里的源码（如 `rquant.factors.statements`、`FUNDAMENTAL_ENDPOINT_FIELDS`），
  照 `jq240.py` 覆盖 `implementation_fingerprint` 把它们也放进指纹。

## 4. 可用数据

全部来自 Tushare，研究从 2010 年开始，带 300 个交易日预热。

- **标准日线**（`CANONICAL_INPUT_FIELDS`，在 `required_inputs` 里声明）：`open high low close volume amount vwap`
  （后复权，成交量为股、成交额为元，VWAP = 成交额/成交量）、`cap`（总市值）、`sector`/`industry`/`subindustry`
  （申万一、二、三级行业编码）。停牌日为缺失值。
- **参考数据** `data/canonical/reference/`（由 `rquant data build-reference` 生成，带 `manifest.json`）：
  - `daily_basic/year=YYYY/data.parquet`：全 A 股的 `close turnover_rate turnover_rate_f pe_ttm pb ps_ttm dv_ttm
    total_share float_share free_share total_mv circ_mv`（见 `rquant/data/reference.py` 的 `DAILY_BASIC_FIELDS`）；
  - `market_factors.parquet`：全 A 市值加权市场收益 `mkt` 和 `smb`、`hml`；
  - `ch4_factors.parquet`：Liu-Stambaugh-Yuan CH-3/CH-4 日收益；
  - `fundamentals.parquet`：每只股票每个报告期一行首次披露的利润表、资产负债表、现金流量表，字段见
    `src/rquant/data/contracts.py` 的 `FUNDAMENTAL_ENDPOINT_FIELDS`；
  - `stock_basic.parquet`（上市日等）、`index_daily.parquet`（沪深 300 指数日线）。
- **股票池**：引擎把当日研究股票池成员作为参考输入 `universe` 传入（`csi300` 或 `sh_main` 等）。截面运算
  （排名、标准化、分组中位数、截面回归）只能用当日 `universe == 1` 的股票；在 `reference_inputs` 里声明 `universe`。

**新增财报字段**：在 `FUNDAMENTAL_ENDPOINT_FIELDS` 对应报表里追加，并像现有代码一样加注释
`# <name> (...) items.`；只追加，不删、不改已有字段。代价是用户要重新 `data sync-fundamentals` 与
`data build-reference`，并重建所有读财报的因子集（`china_chars`、`jq240`、`oap`、`barra_cne6`、`jkp153`、`jkp406`、`china469_pit`，
以及 sh_main 上的 `oap_jq_cnc`）。能不加就不加；加了必须在 PR 描述里写明。

**新增 Tushare 接口**（例如分析师预期、北向持股、龙虎榜）：需要改采集器、契约和账户积分，属于大改动。默认不做，
把依赖它的因子放进 `<NAME>_SKIPPED`，在 PR 里列出“如果接入某接口可以多实现哪些因子”，由用户决定。

## 5. 实现步骤清单

以下 `<name>` 是因子集名（小写蛇形，如 `xyz123`），`<prefix>` 是列名前缀。已占用的前缀：`a158_ a101_ gtja_ a360_
cnc_ svc_ ta_ jq_ oap_ cne6_ tst_ jkp_ jkpx_ cnax_`，新前缀不能和它们冲突。

1. **因子库模块** `src/rquant/factors/libraries/<name>.py`：
   - 模块 docstring：来源、版本/日期、独立重写声明、日频改造规则、与原定义的差异；
   - 目录表：按稳定顺序的元组，每项至少有（源名、类别、一句话定义、lookback），参考 `JQ240`、`OAP`、`TUSHARE_TECH`；
   - `<NAME>_SKIPPED`：未实现因子及原因；
   - 库类继承 `FactorLibrary`：`family = "<name>"`、`required_inputs`、需要时 `reference_inputs`、`quote_fields`、
     `cpp_prelude`、`strict_float`、`compile_split_source`（因子很多时，参考 `oap.py` 的 64）；
   - `specs` 返回 `FactorSpec(canonical_name=f"<prefix>{ordinal:03d}", source_name=..., family=self.family,
     ordinal=..., formula=..., max_lookback=..., implementation="rquant.factors.libraries.<name>....",
     catalog_version=1)`，列名从 001 连续编号；
   - `build(inputs)` 返回 `{source_name: KunQuant 表达式}`，键必须覆盖全部 `specs`（由 `statement_factors` 直接发布的除外）。
2. **注册**：在 `registry.py` 的 `_default_registry()` 里 import 并 `registry.register_library(<Name>Library())`，
   放在 `JKP406Library()` 之后、`register_factor_selection(OAP_JQ_CNC_SELECTION)` 之前。CLI 的 `--factor-set`
   选项、目录和引擎都由注册表派生，不要在别处硬编码新名字。
3. **目录计数**：总数当前是 2078，新增 N 列后改成 2078+N。用 `rg -n 2078 src tests agent.md README.md`
   找全，目前在：`src/rquant/cli.py`（doctor 检查）、`tests/test_catalog.py`、`tests/test_cli.py`、
   `tests/test_china_chars.py`、`tests/test_jq240.py`、`tests/test_size_value_china.py`、
   `tests/test_talib_indicators.py`、`agent.md`、`README.md`。另外 `tests/test_factor_libraries.py` 的
   `test_default_registry_keeps_locked_factor_sets_and_input_contracts` 要把新名字加进 `factor_sets()` 元组
   （在 `jkp406` 和 `oap_jq_cnc` 之间），并加一行它的 `required_inputs` 断言。
4. **测试** `tests/test_<name>.py`：见第 6 节。
5. **文档**：
   - `agent.md` 第 5 节加一条新因子集说明（列名范围、来源、读哪些数据、截面口径、回归测试文件），更新第 5、6 节
     的计数列表，第 6 节长任务命令列表里加上 build/validate；
   - `README.md`：doctor 一节的因子总数和列表、因子集列表、目录树注释，并仿照“JKP 因子集 jkp153”加一节；
   - `THIRD_PARTY_NOTICES.md`：加一节写来源和“只用了定义、未使用其代码或数据”；若确实迁入了带许可证的代码，
     把许可证全文放进 `licenses/` 并写明修改边界。

## 6. 测试要求

CI（`.github/workflows/tests.yml`）在 Ubuntu + Python 3.11 + clang 上跑 `python -m ruff check src tests` 和
`python -m pytest -q`，两者都必须通过。KunQuant 计算测试用 `@unittest.skipUnless(KUNQUANT_READY, ...)`
（KunQuant 和 C++ 编译器都在时才跑），CI 上会真的编译执行，所以不能靠跳过过关。参考 `tests/test_tushare_tech.py`
（纯价量）和 `tests/test_jq240.py`、`tests/test_oap.py`（带参考数据和财报）。

至少包含：

1. **目录测试**：列数、首尾列名（`<prefix>001`…）、源名不重复、实现与跳过不相交且合起来等于来源总数、`backend == "kunquant"`。
2. **合成数据计算**：在小型随机面板（几只到几十只股票、几百到两千个交易日，含停牌段、晚上市、涨跌停日）上
   用 `KunQuantFactorEngine(tmp, FactorBuildConfig(factor_set="<name>", workers=2))` 编译运行。每个因子形状对齐、
   至少有有限值、没有 Inf，停牌日不发布值。
3. **无前视**：把输入截断到某个日期 t（财报按 `ann_date <= t` 截断）重算，t 及以前的值必须和全量结果一致；
   至少取两个截断点，逐因子比较。
4. **直接公式核对**：挑一批有代表性的因子，用 NumPy/Pandas 按原定义直接算几个格子，与 KunQuant 输出对比。
5. 按情况补充：价格/成交量缩放不变性（技术指标）、停牌跨窗口、截面只用 `universe` 成员（非成员的值变化不影响
   成员的截面因子）、财报在公告日当天出现、`test_statement_lines_are_synced`（用到的财报字段都在
   `FUNDAMENTAL_ENDPOINT_FIELDS` 里）、`load_reference_inputs` 校验文件哈希。

测试只用临时目录和合成数据，不联网、不读 `data/`。

## 7. 本地验证（你能跑的）与交接（用户来跑的）

### 你在云端能跑的

需要 Python 3.11 和 clang（或 g++）。不要用其他 Python 版本：

```bash
python3.11 -m venv .venv && . .venv/bin/activate
python -m pip install -e ".[dev]"
python -m ruff check src tests
python -m pytest tests/test_<name>.py -q
python -m pytest tests/test_catalog.py tests/test_cli.py tests/test_factor_libraries.py -q
python -m pytest -q
rquant factors catalog --factor-set <name>
python -c "from rquant.factors.catalog import get_catalog; c=get_catalog(); print(len(c.specs)); print({s: len(c.select(s)) for s in c.factor_sets()})"
```

最后一条的输出应为总数 2078+N，且原有各因子集的列数不变：`qlib_alpha158=158`、`wq_alpha101=101`、`gtja191=190`、
`qlib_alpha360=360`、`china_chars=109`、`size_value_china=21`、`talib=125`、`jq240=233`、`oap=142`、`barra_cne6=38`、
`tushare_tech=75`、`jkp153=144`、`jkp406=364`、`china469_pit=18`、`oap_jq_cnc=254`。

你没有 Tushare token，也没有本地数据，**不要**运行 `rquant data ...`、`rquant factors build/validate`、
`rquant walk-forward`、`rquant report`，也不要在 PR 里声称真实数据构建或回测已经验证。

### 用户在 Windows 上跑的（写进 PR 描述）

用户的计算机器是家里的 Windows（PowerShell，conda 环境 `rquant`，项目在 `D:\自己瞎搞\RQuant2.0`，编译器为
VS 2022 Build Tools 的 `cl.exe`，`factors.workers` 保持默认 4）。PR 描述里给出下面这组命令，把 `<branch>`、
`<name>` 换成实际值；若新增了财报字段，在最前面加上同步与重建参考数据的两行，并列出需要重建的其他因子集。

```powershell
conda activate rquant
cd D:\自己瞎搞\RQuant2.0
git fetch origin
git checkout <branch>
python -m pip install -e ".[dev]"   # 仅当 pyproject.toml 有变化

# 仅当新增了 FUNDAMENTAL_ENDPOINT_FIELDS 字段：
# rquant data sync-fundamentals --through 2026-10-21
# rquant data build-reference

rquant doctor --skip-permission-check
rquant factors build --factor-set <name>
rquant factors validate --factor-set <name>
rquant --universe sh_main factors build --factor-set <name>
rquant --universe sh_main factors validate --factor-set <name>

# 回测：sh_main、LightGBM、1 日标签，数据截止日与 jq240 基准一致
rquant --universe sh_main walk-forward --model lgb --horizon 1d --factor-set <name> --through 2026-10-21
# 1d 默认组合：top 50，每日最多换出 5 只
rquant --universe sh_main report <RUN_ID>
# 5d 标签另行训练，默认每 5 个交易日调整到最新前 50 名
rquant --universe sh_main walk-forward --model lgb --horizon 5d --factor-set <name> --through 2026-10-21
rquant --universe sh_main report <RUN_ID_5D>
```

说明：

- csi300 的因子库写在 `data/factors/<name>/`，sh_main 的写在 `data/factors-sh_main/<name>/`。每次 `factors build`
  会原子替换同名整个目录，需要的空闲磁盘至少是输出体积的两倍加 1 GiB。
- 每条命令的 stdout 是 JSON，`run_directory/run.json` 的 `status` 为 `complete` 才算成功；失败先看同目录的 `run.log`。
- `walk-forward` 的 stdout 里有 `run_id`，回测结果在 `runs/walk-forward/<RUN_ID>/report.md`。

### 和 jq240 比什么

jq240 是当前默认、也是表现最好的基础因子集，sh_main、LGB 1d、top 50 的基准（walk-forward 运行
`20261027T202921Z-41fb3dde`，数据到 2026-10-21）：

| 口径 | 2013–2026 Sharpe | 2020–2026 Sharpe | 2022 年以后 年化 / Sharpe / 最大回撤 |
|---|---|---|---|
| 每 5 天全换（历史 1d 标签运行） | 1.02 | 0.88 | 19.1% / 0.76 / -33.0% |
| 每日换 5 只 | 1.26 | 1.11 | 28.8% / 1.03 / -35.5% |

新因子集要报告同样的指标，外加预测 RankIC。基础因子库没有用收益挑过，2022 年以后是真正的样本外区间，
比较以这一段为主。这些数字只用于对照，不是合并门槛；合不合并由用户决定。

## 8. 完成标准

- [ ] 新模块、注册、扩展（如有）全部是 KunQuant 实现，来源和日频改造写在 docstring 里；
- [ ] 实现 + 跳过 = 来源全集，跳过的每条都有原因；
- [ ] 目录计数、`factor_sets()` 元组、`agent.md`、`README.md`、`THIRD_PARTY_NOTICES.md` 已同步；
- [ ] `tests/test_<name>.py` 含目录、计算、无前视、直接公式测试；
- [ ] `python -m ruff check src tests` 和 `python -m pytest -q` 在 Python 3.11 下全部通过（写明通过数和跳过数）；
- [ ] draft PR 已开，描述包含第 1 节列出的内容和第 7 节的 Windows 命令；没有推送 `main`、没有合并。

## 9. 用户交给 agent 的任务模板

```text
请在 RUIQI1811/RQuant2.0 里新增一个因子集，来源：<论文/网站/仓库 URL>。
因子集名 <name>，列名前缀 <prefix>。先读 AGENTS.md、agent.md 和 docs/adding-factor-sets.md，严格按其中规则实现：
从最新 main 切新分支，KunQuant 实现，做不出来的因子写进 <NAME>_SKIPPED 并注明原因，补齐测试和文档，
Python 3.11 下 ruff 和 pytest 全部通过后开 draft PR。不要推送 main，不要合并，不要运行需要数据或 token 的命令；
PR 描述里写清楚我要在 Windows 上执行的构建和回测命令。
```
