# train：脚本清单

用于开发层的AR / DiffGRM训练、profile、smoke。参数不因归类改变。部分旧脚本复用历史run_id或自动查找产物；新实验须先审核配置和输入，使用唯一run_id。复现已归档结果优先使用repro配方。

所有命令从项目根目录执行，例如脚本路径写成 `scripts/train/文件名`。历史配方内部的旧源码路径保留原样，不手工改历史锁。

新增短因果对照：[run_diger_noise_pilot.py](run_diger_noise_pilot.py)；共同起点、三臂预算和后台运行方式见 [DIGER 噪声更新短对照](../../experiments/05_crosspse_joint/pilots/noise_update/README.md)。这不是旧版本正式训练的替换入口。

下一轮推荐反馈对照：[run_diger_feedback_pilot.py](run_diger_feedback_pilot.py)；冻结 / 重构更新 / 重构＋推荐反馈，固定更新后再恢复 Decoder，见 [协议和运行顺序](../../experiments/05_crosspse_joint/pilots/feedback_update/README.md)。使用独立新输出，不覆盖噪声对照。

- [profile_formal.sh](profile_formal.sh)
- [run_ar_internal_study.sh](run_ar_internal_study.sh)
- [run_ar_smoke.ps1](run_ar_smoke.ps1)
- [run_ar_smoke.sh](run_ar_smoke.sh)
- [run_ar_smoke_v2.ps1](run_ar_smoke_v2.ps1)
- [run_ar_smoke_v2.sh](run_ar_smoke_v2.sh)
- [run_diff_crosspse_diger.sh](run_diff_crosspse_diger.sh)
- [run_diff_crosspse_diger_v2.sh](run_diff_crosspse_diger_v2.sh)
- [run_diff_crosspse_frozen.sh](run_diff_crosspse_frozen.sh)
- [run_diff_crosspse_joint.sh](run_diff_crosspse_joint.sh)
- [run_diff_crosspse_v1.sh](run_diff_crosspse_v1.sh)
- [run_diff_crosspse_v2.sh](run_diff_crosspse_v2.sh)
- [run_diff_crosspse_v3.sh](run_diff_crosspse_v3.sh)
- [run_diff_crosspse_v3_catalog.sh](run_diff_crosspse_v3_catalog.sh)
- [run_diff_formal.sh](run_diff_formal.sh)
- [run_diff_pse_refinement.sh](run_diff_pse_refinement.sh)
- [run_diff_refinedpse_frozen.sh](run_diff_refinedpse_frozen.sh)
- [run_diff_refinedpse_joint.sh](run_diff_refinedpse_joint.sh)
- [run_diff_refinedpse_v1.sh](run_diff_refinedpse_v1.sh)
- [run_diff_refinedpse_v2_catalog.sh](run_diff_refinedpse_v2_catalog.sh)
- [run_diff_rqkmeans_v6.sh](run_diff_rqkmeans_v6.sh)
- [run_diff_rqvae_v6.sh](run_diff_rqvae_v6.sh)
- [run_formal_pair.sh](run_formal_pair.sh)

返回 [脚本总入口](../README.md) · [项目地图](../../docs/guides/PROJECT_MAP.md)
