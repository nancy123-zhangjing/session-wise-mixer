# Five-arm execution record

Scientific source: `b66a50f`; protocol: `docs/research/session_wise_ar_mixer/FIVE_ARM_DESIGN_20261113.md`.

`worker.py` is an orchestration-only addition, deployed as `delivery/worker.py` below the new remote root. It does not change model configuration or submit jobs itself.

- `prepare-tiger` / `prepare-diger`: build one common parent per family on the CPU preparation channel before GPU allocation; validates an existing completed parent instead of recreating it.
- `AR-T0`, `AR-T1`, `AR-D0`, `AR-D1`, `CF-M0`: explicit pretraining, recommendation training, best/final testing, one process on one assigned GPU.
- Existing valid `last.pt` selects the existing engine's resume path; no automatic removal of running locks or resetting incomplete runs.
- `snapshot`: reads small progress receipts only.
- Training/test completion is recorded with acceptance still pending; it is not final Goal acceptance.

Local validation: Python compilation and mocked routing of all five arms through 15 phases passed; no real data/model was executed by this check. GPU numerical checks, deployment verification, submission and final acceptance remain to be performed. Current state is in `DEPLOYMENT_PROGRESS.json`, not inferred from this document.

The existing `env-zzl` CPU channel was STOPPED before this task; this task requested its start. Close it when deployment/acceptance consumers no longer need it, after verifying ownership/current consumers. Preserve all old experiment data.

## 阶段曲线（仅读取已归档指标）

使用包含 matplotlib 的本地环境运行 `plot_interim.py`。输出 `interim_curves/cf_validation.png`、`ar_validation.png` 和输入SHA清单 `MANIFEST.json`。脚本检查完整验证样本数71657、AR指标与checkpoint的SHA对应关系；不运行训练或推理，不代表最终恢复验收。CF仅画同轮可比区间，AR-T1没有推荐验证结果所以不画零分。不同方法的epoch不是相同计算预算。首次生成已检查两图可读，数据为CF至R22、T0至第3轮、D0/D1各第1轮。
