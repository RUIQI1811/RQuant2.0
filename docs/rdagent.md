# 用 RD-Agent 自动挖因子

[RD-Agent](https://github.com/microsoft/RD-Agent)（微软，MIT 许可）的 `fin_factor` 循环让 LLM 提出因子假设、写成
pandas 代码、在 Qlib 上回测，再根据结果决定保留哪些因子、下一轮怎么改。RQuant 不把 RD-Agent 装进 `rquant` 环境，
只负责两件事：把数据按挖掘窗口截断后导出给它，以及把它的回测口径改成 RQuant 的口径。挖到的因子最终要回到
RQuant 的 walk-forward 上，在 RD-Agent 从未见过的年份里检验。

## 1. 口径与防泄漏

RD-Agent 只用固定的 train/valid/test 三段，并且**用 test 段的 IC 和超额收益决定保留哪些因子**，所以它的 test
段等于选因子用的数据。因此：

- `rquant rdagent export --mining-end 2021-12-31`（默认）只导出 2021-12-31 及以前的数据；
- 挖掘窗口默认划分为 train 2010–2016、valid 2017–2018、test 2019–2021（test 取最后三年，valid 取再往前两年，
  train 从 `data.formal_start` 开始），写在 bundle 的 `rdagent.env` 里；
- 2022 年以后只留给 RQuant 的 walk-forward；拿挖到的因子报告结果时，只有 2022 年以后的年份算样本外。

导出的股票是挖掘窗口内任意时点的沪深 300 成分（历史时点可知），每只股票保留它在导出范围内的全部行情，
这样滚动窗口有预热数据。字段是 `$open $close $high $low $volume $amount $vwap`：价格为后复权，每只股票首日
收盘价归一为 1；`$amount` 是成交额（元，不复权），`$volume * $vwap = $amount`。

RD-Agent 自带的数据还有复权因子 `$factor`，它的数据说明把它写成 "factor value of the stock"。第一次 16 轮试跑中，
LLM 有一半轮次把它当成基本面数据来构造因子，挖出来的东西只是在测除权事件，所以导出时去掉了它，并用 bundle 里的
`README.md` 替换 RD-Agent 的数据说明（它会原样进入写因子的提示词）。

`rdagent_setup.py` 会改 RD-Agent 1.0.0 的三个 Qlib 因子模板：

| 项目 | RD-Agent 默认 | 改为 |
| --- | --- | --- |
| 标签 | `Ref($close,-2)/Ref($close,-1)-1` | `Ref($open,-2)/Ref($open,-1)-1`（下一开盘到再下一开盘） |
| 成交价 | close | open |
| 买入成本 | 0.05% | 0.03%（佣金） |
| 卖出成本 | 0.15% | 0.13%（佣金 + 0.1% 印花税） |
| LightGBM 线程 | 20 | `--threads`，默认 4 |

模型仍是 RD-Agent 自带的 LightGBM 和 qlib 的 TopkDropout（持股 50、每天最多换 5 只），基线因子是 qlib 自带的
Alpha158 表达式。它只用来给候选因子排序，不替代 RQuant 的组合回测。

## 2. 在 Windows 上准备 WSL2

RD-Agent 只支持 Linux。以下步骤只需要做一次。

1. 管理员 PowerShell：`wsl --install -d Ubuntu-22.04`，重启后按提示建立 Linux 用户。
2. 限制 WSL 内存，避免和 Windows 抢满 16 GB。新建 `%UserProfile%\.wslconfig`：

   ```ini
   [wsl2]
   memory=10GB
   swap=8GB
   processors=4
   ```

   保存后在 PowerShell 执行 `wsl --shutdown` 生效。
3. 在 Ubuntu 里安装 Miniconda（RD-Agent 用 conda 创建回测环境）：

   ```bash
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3
   ~/miniconda3/bin/conda init bash && exec bash
   conda create -y -n rdagent python=3.10
   conda activate rdagent
   pip install rdagent==1.0.0
   pip install -U litellm
   ```

   第一次回测时 RD-Agent 会自动再建一个 `rdagent4qlib` 环境（从 GitHub 安装固定版本的 qlib，并装 torch，约 2–3 GB），
   需要能访问 GitHub 和 PyPI。

## 3. 导出数据（Windows，rquant 环境）

```powershell
conda activate rquant
rquant data build-qlib          # 确保 data/qlib 是最新的
rquant rdagent export           # 写入 data/rdagent/
```

`data/rdagent/` 包括：

```text
factor_source_data/daily_pv.parquet         全量：挖掘窗口内的沪深 300 历史成分
factor_source_data_debug/daily_pv.parquet   调试：最后两年、前 100 只
README.md                                   给 LLM 的数据说明（替换 RD-Agent 自带的）
rdagent.env                                 .env 模板（千问 / 百炼，API key 留空）
rdagent_setup.py                            在 WSL 里运行的准备脚本
manifest.json                               窗口、行数、哈希和 Qlib provider 指纹
```

## 4. 准备 RD-Agent 工作目录（WSL，rdagent 环境）

RD-Agent 的回测模板写死读 `~/.qlib/qlib_data/cn_data`。把 RQuant 的 Qlib provider 复制过去（不要直接读
`/mnt/d`，跨文件系统读取很慢）：

```bash
conda activate rdagent
mkdir -p ~/.qlib/qlib_data
rsync -a --delete "/mnt/d/自己瞎搞/RQuant2.0/data/qlib/" ~/.qlib/qlib_data/cn_data/
python "/mnt/d/自己瞎搞/RQuant2.0/data/rdagent/rdagent_setup.py" \
  --bundle "/mnt/d/自己瞎搞/RQuant2.0/data/rdagent" \
  --workdir ~/rdagent-work
```

脚本会把 Parquet 转成 RD-Agent 读取的 `git_ignore_folder/factor_implementation_source_data*/daily_pv.h5`，改好上面的
模板（原文件备份为 `*.yaml.orig`，重复运行总是从备份重新改）、给 RD-Agent 的 embedding 接口打分批补丁，并在工作目录写入 `.env`（已存在则不覆盖）。

编辑 `~/rdagent-work/.env`，填 `DASHSCOPE_API_KEY`：在阿里云百炼控制台（中国内地地域）开通服务并创建 API Key。
同一个 key 同时用于 chat 模型 `qwen-plus` 和 embedding 模型 `text-embedding-v4`，都走 LiteLLM 自带的 DashScope
接口，需要较新的 LiteLLM（准备脚本会检查，不满足时提示 `pip install -U litellm`）。

- `qwen-plus` 上下文 128k，够用且便宜；想要更强可以改成 `qwen3-max`。不要用 `qwen-max`：它只有 30k 上下文，
  装不下 RD-Agent 的提示词；也不要用 QwQ 等推理模型（输出 `<think>`，慢且贵，要用就得再加 `REASONING_THINK_RM=True`）；
- `EMBEDDING_MAX_STR_NUM=10`：百炼的 embedding 每次最多 10 条。RD-Agent 1.0.0 的 LiteLLM 接口其实不读这个设置，
  会把所有文本一次发出，知识库超过 10 条后（约第 5 轮）报 `batch size is invalid`；准备脚本会给它的
  `rdagent/oai/backend/litellm.py` 打补丁按这个值分批（原文件备份为 `litellm.py.orig`）；
- `.env` 里还留了 DeepSeek（chat）+ 硅基流动（embedding）的注释配置，想换时注释掉千问那几行即可。

## 5. 运行

先确认 key 和模型能用（发一次 JSON 模式的 chat 和一次 embedding，走的是 RD-Agent 自己的 LLM 接口）：

```bash
python "/mnt/d/自己瞎搞/RQuant2.0/data/rdagent/rdagent_setup.py" --workdir ~/rdagent-work --check-llm
```

不要用 `rdagent health_check` 检查：RD-Agent 1.0.0 的这个命令只认 OpenAI 和 DeepSeek 的 key，遇到千问配置会报
`UnboundLocalError: local variable 'embedding_model' referenced before assignment`，并不代表配置有错。

两行都显示 OK 后开始挖：

```bash
cd ~/rdagent-work
rdagent fin_factor --loop-n 20
```

每一轮会提出一组因子、写代码、算因子值，再跑一次 Alpha158 + 新因子的 LightGBM 回测；论文里 44 轮约 12 小时。
可以另开一个终端 `rdagent ui --port 19899 --log-dir log/` 在浏览器里看进度。中断后用 `rdagent fin_factor --path log/<目录>`
从断点继续。

产出在 `~/rdagent-work/git_ignore_folder/RD-Agent_workspace/`（每个因子的 `factor.py`）和 `log/`。把整个工作目录
留着，下一步由 RQuant 把被保留的因子收编成独立因子集，在 2022 年以后的年份上用 walk-forward 检验。
