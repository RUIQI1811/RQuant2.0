# RQuant Subagent Rules Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans for this single documentation/configuration task and one fresh final review. The user authorized applying the recommendations from the preceding comparison; no additional design approval is needed. Steps use checkboxes for tracking.

**Goal:** Make the existing subagent workflow reusable and specific to RQuant research contracts.

**Architecture:** Keep Superpowers as the execution workflow. Add six project-local role definitions and a shared policy loaded through AGENTS.md. Use explicit role instructions as a fallback when named dispatch is unavailable.

**Tech Stack:** Markdown and Codex TOML; no dependency or product-code changes.

**Spec:** The user's approved recommendations in this chat; the resulting policy is docs/subagents.md.

## Global Constraints

- Preserve all existing working-tree and staged changes; commit only the files listed below on codex/rquant-subagent-rules.
- Do not set global configuration, fixed models, per-role sandbox/MCP overrides, or change research defaults.
- Keep write tasks serial and reviews independent; project and user authorization govern production operations.
- Use current source/config/contracts as truth, with specific RQuant time-availability and research-validity checks.

## Review Focus

- Named dispatch may be unavailable: the explicit-prompt fallback must remain usable and honestly labeled.
- Read-only role instructions must not claim operating-system permission isolation.
- Dirty/staged user changes must not disappear from relevant review evidence or enter this commit.
- Test evidence for another code version must not be reused as verification of the changed version.
- Missing independent or quantitative review must not be reported as a passing gate.

## Task 1: Project policy and six roles

**Files:** Modify AGENTS.md; create docs/subagents.md and .codex/agents/rquant_{explorer,researcher,worker,tester,reviewer,quant_reviewer}.toml.

- [x] Create a fresh branch without discarding existing changes.
- [x] Add task boundaries, six-item dispatch contracts, evidence/status reporting, recovery and RQuant review rules.
- [x] Add six roles using only name, description and developer_instructions; connect the policy from AGENTS.md.
- [x] Parse role TOML, check required fields/names/references and inspect the task-only diff.
- [x] Run Python 3.11 Ruff and pytest as required by AGENTS.md; record exact scope and results.
- [x] Obtain one independent review and check representative dispatch/review scenarios; resolve material findings.
- Delivery: commit only this task, push the new branch and open a draft PR against main; never merge. Publication status is recorded in the PR and final response.

## Validation record

- The existing rquant Conda Python 3.11.17 imports rquant from this checkout.
- All six TOML roles parse with unique names, required fields and valid shared-policy references; whitespace checks pass.
- `python -m ruff check src tests` passes; `python -m pytest -q` exits 0 with one skip and one MLflow/SQLAlchemy deprecation warning.
- The local suite ran on the current checkout, including the user's pre-existing source changes; it is not a clean-commit CI result.
- One independent reviewer checked the policy, all roles and five representative scope/evidence/production-operation scenarios without material findings.
- Final clarification: already authorized in-scope interface/dependency work does not require repeat confirmation; only scope expansion returns to the parent.
- Native named-role loading, model routing and long-term behavioral reliability remain unverified.
