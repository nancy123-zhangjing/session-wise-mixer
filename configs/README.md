# 配置文件索引

配置文件保留原路径，因为运行脚本直接引用它们。下面按实验阶段分类，避免看到多个 `v1/v2/v6` 时混淆。

## AR-GRM

| 用途 | 配置 |
|---|---|
| 正式公共设置 | `ar_formal_common.yaml` |
| 3×256 tokenizer 对照 | `ar_formal_pse.yaml`、`ar_formal_rqkmeans.yaml`、`ar_formal_rqvae.yaml` |
| RQ-KMeans 容量消融 | `ar_formal_rqkmeans_128_256_512.yaml`、`ar_formal_rqkmeans_512_256_128.yaml` |
| 早期冒烟测试 | `ar_smoke_6gb.yaml`、`ar_smoke_v2_unique.yaml` |

## DiffGRM tokenizer 对照

| 用途 | 配置 |
|---|---|
| 公共设置 / PSE | `diff_formal_common.yaml`、`diff_formal_pse.yaml` |
| 当前复现基线 | `diff_formal_repo_v6.yaml` |
| RQ-KMeans / RQ-VAE | `diff_formal_rqkmeans_v6.yaml`、`diff_formal_rqvae_v6.yaml` |
| 历史数值稳定性试验 | `diff_formal_bf16_v3.yaml`、`diff_formal_bf16_v4.yaml`、`diff_formal_repo_v5.yaml` |
| 早期 profile / smoke | `diff_profile.yaml`、`diff_smoke.yaml`、`diff_diger_smoke.yaml`、`diff_joint_smoke.yaml` |

## PSE 改进

| 用途 | 配置 | 状态 |
|---|---|---|
| 早期 Cross-PSE | `diff_formal_crosspse_v1.yaml`、`v2.yaml`、`v3.yaml` | 历史试验 |
| 最终离线 Cross-PSE | `diff_formal_crosspse_v3_catalog.yaml` | 阶段 04 使用 |
| 早期 Refined-PSE | `diff_formal_refinedpse_v1.yaml` | 历史试验 |
| 最终离线 Refined-PSE | `diff_formal_refinedpse_v2_catalog.yaml` | 阶段 04 使用 |
| 间接 SIA 联合训练 | `diff_formal_crosspse_joint_sia.yaml` | 联合训练 pilot |
| DIGER ST 联合训练 v1 | `diff_formal_crosspse_joint_diger.yaml` | fast anneal 历史结果 |
| DIGER ST 联合训练 v2 | `diff_formal_crosspse_joint_diger_v2.yaml` | 当前主要结果 |
| Frozen Cross-PSE 对照 | `diff_formal_crosspse_frozen_decoder_v1.yaml` | 下一组严格 Decoder-only 对照 |
| Joint Refined-PSE 对照 | `diff_formal_refinedpse_joint_diger_v1.yaml` | 同 DIGER 协议、关闭跨码本 mixer |
| Frozen Refined-PSE 对照 | `diff_formal_refinedpse_frozen_decoder_v1.yaml` | 同 Refined 初始化、仅训练 Decoder |

Frozen 对照的短测试覆盖文件是 `diff_frozen_crosspse_smoke.yaml` 和 `diff_frozen_refinedpse_smoke.yaml`。它们只用于验证梯度与 SID 冻结，不进入正式结果。

不要直接覆盖已有配置；下一轮 matched control 应新建配置，并在 `run_id` 中明确写出 `frozen_decoder_continuation` 或 `joint_refinedpse`。
