# experiment1 server workspace

Read AGENTS.md and docs/guides/CODE_VERSIONING.md before changing files. Preserve per-task Git commits, exact scientific settings and immutable experiment source snapshots. Do not upload secrets or large model/data files to Git.

This Google Cloud workspace is a development and research copy. GPU training and large artifacts remain on Inspire. Migrating this repository does NOT authorize new training, restarting monitors, modifying running jobs, or taking over another chat's active experiments. Begin with discussion/read-only inspection; the user will explicitly hand over job control. Do not execute old submission scripts just because a report says pending.

Start research handoff with docs/guides/CLAUDE_RESEARCH_HANDOFF_20261117_SESSION_WISE.md and docs/guides/CLAUDE_RESEARCH_HANDOFF_20261117_SESSION_WISE_SOURCES.json. It connects the original H0/H1 stability and multi-seed evidence, continuous diffusion/RQ design and diagnostics, CF and AR controls, and the latest L5O wave. It also flags stale progress and overstrong claims in older summaries. Then read experiments/09_kuairec_listwise/l5o_stage0_20261117/HANDOFF.md, WAVE_REPORT.md, SASREC_CALIBRATION.md. These contain dated snapshots, not live state. Distinguish accepted results, TIGER interim observations and hypotheses.

Broader new-paradigm context: docs/guides/NEW_CHAT_HANDOFF_20261114_NEW_PARADIGM.md, then the referenced older handoffs and current reports. Preserve negative results and evidence boundaries. Old README or handoff completion statements cover only their original scope.

Repository root is /home/nancy/projects/experiment1 on this server. Windows C:/Users/nancy/Documents/experiment1 references identify the original workspace: resolve repository-relative suffixes here without rewriting immutable reports. /inspire/... refers to Inspire shared storage, NOT a local Google Cloud mount. runtime_env.sh is an Inspire-specific helper, not a Google Cloud environment installer.

Code and small reports belong here. Large checkpoints, datasets and predictions remain at recorded remote paths; local migration does not prove their remote completeness. Do not delete originals. Install/authenticate Inspire separately only when requested; do not borrow another Linux user's credentials. Do not run full training/inference to validate a repository copy.

Google Cloud access and code-sync procedure: docs/guides/GCP_INSPIRE_WORKFLOW.md. Before any work that needs Inspire, run `python3 scripts/ops/gcp/inspire_access.py ensure` once (on demand, no background monitoring); if it reports a step that needs the user (manual VPN login, OAuth device authorization), tell the user that step instead of retrying blindly. The user authorized configuring this server's Inspire access, but not altering existing jobs or submitting test jobs. CLI installation is not proof of VPN/OAuth readiness; check the current setup report and live read-only status. Retain the pre-migration GOAL_TRACKING_STATE.json modification. Never blanket-stage a migrated worktree or transfer secrets through Git.

## Conversation log (auto-saved Markdown, every turn)

Every conversation turn is recorded under `docs/discussions/`, not only research analysis. Local slash commands (e.g. `/model`, `/effort`) are not turns.

- One file per day: `docs/discussions/YYYY-MM-DD.md` (Asia/Shanghai date of the question). At most 20 turns per file; turn 21 onward continues in `YYYY-MM-DD_2.md`, then `_3`, and so on. A turn crossing midnight belongs to the date it was asked.
- Each turn: `## 第 N 轮 · HH:MM`, then `### 问` with the user's message verbatim (unchanged text in a blockquote), then `### 答` with the reply exactly as given in chat (heading levels lowered to fit). Detailed analysis written in that turn goes in `### 附` sections after the answer, keeping full explanations, tables and formulas. Standard Markdown tables; inline math `$...$`, display math `$$...$$`; never wrap the whole document in a code block.
- File header: date, turn range, created and last-updated time, and a turn table with a status per turn: 讨论中 / 已确定方案 / 阶段结果 / 已验收结果 / 操作问答 / 工作流配置. A new idea is not an approved experiment; a stage result is not final acceptance.
- Append only. Never rewrite earlier turns. When a conclusion changes, explain in the new turn what changed and why, and add a one-line `> 后续补注（第 N 轮添加）：...` under the earlier turn. Never edit original experiment reports or acceptance materials to tidy notes.
- Write the turn to the file before replying; the chat reply matches the recorded answer (a commit hash, created only after committing, may be added in chat). If the write failed, say so; never claim it was saved.
- Links are kept as written. Repository files mentioned in a recorded answer are written as relative Markdown links from the log file (e.g. `[REPORT.md](../../experiments/.../REPORT.md)`) so they open in preview; Inspire paths (`/inspire/...`) stay in code format. Each turn and appendix gets an explicit `<a id="turn-N">` / `<a id="turn-N-app…">` anchor for in-file links. Check links after restructuring files.
- `docs/discussions/INDEX.md` stays short: date, file link, turn range, main topics, last updated. Read only the current day's file or the turn the user refers to, not all history.
- No passwords, tokens, cookies or other credentials: replace such a fragment with `[已删除凭据]` and note it; otherwise keep the user's words unchanged.
- Commit logs with the complete task they belong to, per `docs/guides/CODE_VERSIONING.md`, using explicit paths; not one commit per reply. Pure conversation turns go with the next commit. Never mix with other tasks' changes.
