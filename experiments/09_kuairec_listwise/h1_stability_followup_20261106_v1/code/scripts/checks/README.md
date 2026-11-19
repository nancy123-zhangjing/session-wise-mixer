# checks：脚本清单

用于preflight、短跑结果核验、Tokenizer产物校验及按身份查找checkpoint。脚本可能需要完整训练依赖/服务器资产；本机AST通过不代表这些检查已执行。

所有命令从项目根目录执行，例如脚本路径写成 `scripts/checks/文件名`。历史配方内部的旧源码路径保留原样，不手工改历史锁。

推荐反馈短对照：[diger_feedback_pilot_protocol.py](diger_feedback_pilot_protocol.py)。本地 `--plan-only` 只核对设计；完整资产检查须在具有初始化权重和数据的服务器执行，见 [实验说明](../../experiments/05_crosspse_joint/pilots/feedback_update/README.md)。

- [check_diff_crosspse_diger_smoke.py](check_diff_crosspse_diger_smoke.py)
- [check_diff_crosspse_frozen_smoke.py](check_diff_crosspse_frozen_smoke.py)
- [check_diff_refinedpse_frozen_smoke.py](check_diff_refinedpse_frozen_smoke.py)
- [check_diff_refinedpse_joint_smoke.py](check_diff_refinedpse_joint_smoke.py)
- [find_diff_checkpoint.py](find_diff_checkpoint.py)
- [preflight_diff_crosspse_diger.py](preflight_diff_crosspse_diger.py)
- [preflight_diff_crosspse_diger_v2.py](preflight_diff_crosspse_diger_v2.py)
- [preflight_diff_crosspse_frozen.py](preflight_diff_crosspse_frozen.py)
- [preflight_diff_crosspse_joint.py](preflight_diff_crosspse_joint.py)
- [preflight_diff_formal.py](preflight_diff_formal.py)
- [preflight_diff_refinedpse_frozen.py](preflight_diff_refinedpse_frozen.py)
- [preflight_diff_refinedpse_joint.py](preflight_diff_refinedpse_joint.py)
- [preflight_formal.py](preflight_formal.py)
- [verify_diff_crosspse_artifact.py](verify_diff_crosspse_artifact.py)
- [verify_diff_rqvae_artifact.py](verify_diff_rqvae_artifact.py)

返回 [脚本总入口](../README.md) · [项目地图](../../docs/guides/PROJECT_MAP.md)
