# Code version management

The user explicitly requests incremental Git commits after each completed code task.

- Inspect status first; keep other tasks' changes intact. Stage explicit paths or hunks, never a blanket `git add .`.
- Commit one coherent change with its configuration, existing relevant tests and necessary design/run documentation. Report the commit hash and validation boundary.
- Separate model changes, execution optimizations, evaluation diagnostics and experiment reports. Do not represent an unrun design as implemented or an ongoing run as accepted.
- Do not commit credentials, checkpoints, datasets, large prediction files, downloaded archives, personal drafts or transient platform responses. Keep artifact locations and checksums in small manifests instead.
- Preserve immutable experiment code snapshots and exact source hashes when available. When retrospectively archiving current files, state that the commit was made now; do not invent intermediate versions or backdate commits.
- Do not change model settings, start/resume experiments, delete artifacts, rewrite existing Git history or push to a remote merely to perform version management.
- Use the existing relevant checks; version management alone does not require rerunning training or inference. Keep active experiment files intact.

See `docs/guides/CODE_VERSIONING.md` for the workflow and the retrospective index.

Conversation log: every conversation turn is recorded verbatim (user question, then the reply) in one Markdown file per day under `docs/discussions/` (max 20 turns per file, index `docs/discussions/INDEX.md`); rules in `CLAUDE.md` section "Conversation log". Logs never replace or edit original experiment reports.
