# AGENTS.md

Codex 等 coding agent 会自动读取本文件。本仓库的完整 agent 规则在 `agent.md`，开始任何任务前先读它。

- 新增因子集（把论文或网上公开的因子库实现进来）：按 `docs/adding-factor-sets.md` 做。
- 只在新分支上改，开 draft PR 交给用户审阅；不得推送 `main`，不得合并 PR，不得改写已有分支历史。
- 所有因子库都用 KunQuant 实现，并通过 `src/rquant/factors/libraries/registry.py` 注册。
- 提交前必须在 Python 3.11 下通过 `python -m ruff check src tests` 和 `python -m pytest -q`（CI 同样检查这两项）。
- 没有 Tushare token 和本地数据时，不要运行 `rquant data ...`、`rquant factors build/validate`、`rquant walk-forward`、
  `rquant report`；全量构建和回测由用户在自己的 Windows 机器上执行，你只在 PR 里写清楚命令。
