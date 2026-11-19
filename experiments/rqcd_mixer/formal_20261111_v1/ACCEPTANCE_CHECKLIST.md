# Final acceptance — verified 2026-11-12

Authoritative closure: FINAL_ACCEPTANCE.json. Research report: report/REPORT.md.

| Requirement | Verified evidence |
|---|---|
| Three independent single-H100 arms | status_B*.json: one GPU each, SUCCEEDED; instances_B*.json: separate nodes, instance_success |
| Shared settings; only eval31 change | CHANGE.json; before/current config comparison in close_acceptance.py; checkpoint config/runtime/source checks |
| Original common initialization | audit_checkpoints.py: initial.pt SHA, prepared manifest, source binding |
| 10000 item epochs, 110000 updates | pretrain_recovery_audit.json: actual last/selected checkpoint and optimizer clocks |
| Earliest minimum collision selection | All 600 diagnostic nodes; selected state and receipt agree |
| R35/T300; frozen Encoder, preserved Decoder | All 93 checkpoints: counters and frozen tensors checked against selected pretrain state |
| 31 immutable checkpoints per arm | checkpoint_audit.json: sizes, SHA, data/optimizer clocks, RNG and recovery fields |
| Full validation/test | Every validation count71657; best/final test count53194; checkpoint, noise and inference settings match |
| Validation-selected best | Recomputed argmax: B0 R25, B1 R28, B2 R24 |
| Loss and encoding diagnostics | 19 plots visually inspected; distinct solid colors; per-layer CSV included |
| Best/final and full-node comparisons | comparison.csv, validation_nodes.csv, Chinese REPORT.md; single-seed and causal limits stated |
| Recovery assets retained | 509 remote entries plus9 external inputs; additional PT files read and finite |
| Exact GPU recovery scope | Accepted benchmark next-R recovery reused; no repeated training/pilot |
| Local small artifacts | 24 report files match remote SHA; no local .pt in this formal directory |
| Resource closure | All GPU jobs/instances terminal; cpu_final_status.json STOPPED |

Remote models, predictions and logs remain intact. Final closure receipt is local; reports, manifest and state audits were synchronized before CPU shutdown. Pretraining VQ-loss growth and low absolute recommendation scores remain research limitations.
