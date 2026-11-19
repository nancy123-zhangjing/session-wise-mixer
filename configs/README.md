# 配置文件索引

34 份 YAML 已按共享设置和 01–05 阶段分目录存放，文件名、内容、参数和加载顺序未变。B2c又将脚本按用途归类：启动脚本的子目录改变，命令参数不变。阶段目录表示研究归属，**不表示里面每个历史版本都应重新运行**。

**复现已归档结果时，先看 `experiments/对应阶段/reproducibility/recipes/`。** 配方记录配置覆盖顺序和 seed，实际读取的是证据副本；这里的可编辑配置用于后续研究，不自动替换历史配方。详见 [复现入口](../reproducibility/README.md)。

| 目录 | 配置数 | 用途 |
|---|---:|---|
| [shared/](shared/) | 5 | AR/Diff 公共参数、Diff v6 协议、通用短跑覆盖 |
| [01_ar_baseline_3x256/](01_ar_baseline_3x256/) | 4 | AR 两种 RQ 基线、早期 smoke |
| [02_ar_capacity_ablation/](02_ar_capacity_ablation/) | 3 | AR 的 PSE 与容量消融 |
| [03_diffgrm_tokenizer_comparison/](03_diffgrm_tokenizer_comparison/) | 6 | Diff tokenizer 对照及早期数值设置 |
| [04_crosspse_offline/](04_crosspse_offline/) | 6 | 离线 Refined/Cross-PSE 的演进 |
| [05_crosspse_joint/](05_crosspse_joint/) | 10 | 联合训练、冻结对照和专用 smoke |
| [diagnostics/](diagnostics/) | 1 | 只读 SID 一致性诊断 JSON，不参与训练 YAML 层叠 |

不复制共享文件到多个阶段。下表文件名均相对于对应目录；跨阶段复用不代表新增独立实验。

## 共享设置：`shared/`

| 配置 | 用途 |
|---|---|
| `ar_formal_common.yaml` | AR 正式实验共用参数 |
| `diff_formal_common.yaml` | Diff 正式实验共用参数 |
| `diff_formal_repo_v6.yaml` | 03–05 共用 v6 训练协议，不是单独一组实验 |
| `diff_smoke.yaml`、`diff_profile.yaml` | 通用短跑覆盖；不代表正式训练设置 |

## AR-GRM

| 用途 | 配置 |
|---|---|
| 01 正式 3×256 基线 | `01_ar_baseline_3x256/ar_formal_rqkmeans.yaml`、`ar_formal_rqvae.yaml` |
| 01 早期 smoke | `01_ar_baseline_3x256/ar_smoke_6gb.yaml`、`ar_smoke_v2_unique.yaml` |
| 02 PSE 对照 | `02_ar_capacity_ablation/ar_formal_pse.yaml` |
| 02 RQ-KMeans 容量消融 | `02_ar_capacity_ablation/ar_formal_rqkmeans_128_256_512.yaml`、`ar_formal_rqkmeans_512_256_128.yaml` |

## DiffGRM tokenizer 对照：`03_diffgrm_tokenizer_comparison/`

| 用途 | 配置 |
|---|---|
| PSE | `diff_formal_pse.yaml` |
| RQ-KMeans / RQ-VAE | `diff_formal_rqkmeans_v6.yaml`、`diff_formal_rqvae_v6.yaml` |
| 历史数值稳定性试验 | `diff_formal_bf16_v3.yaml`、`diff_formal_bf16_v4.yaml`、`diff_formal_repo_v5.yaml` |

## 离线 PSE 改进：`04_crosspse_offline/`

| 用途 | 配置 | 状态 |
|---|---|---|
| 早期 Cross-PSE | `diff_formal_crosspse_v1.yaml`、`diff_formal_crosspse_v2.yaml`、`diff_formal_crosspse_v3.yaml` | 历史试验 |
| 最终离线 Cross-PSE | `diff_formal_crosspse_v3_catalog.yaml` | 阶段 04 使用 |
| 早期 Refined-PSE | `diff_formal_refinedpse_v1.yaml` | 历史试验 |
| 最终离线 Refined-PSE | `diff_formal_refinedpse_v2_catalog.yaml` | 阶段 04 使用 |

## 联合训练与冻结对照：`05_crosspse_joint/`

| 用途 | 配置 | 状态 |
|---|---|---|
| 间接 SIA 联合训练 | `diff_formal_crosspse_joint_sia.yaml` | 联合训练 pilot |
| DIGER ST 联合训练 v1 | `diff_formal_crosspse_joint_diger.yaml` | fast anneal 历史结果 |
| DIGER ST 联合训练 v2 | `diff_formal_crosspse_joint_diger_v2.yaml` | 05 主对照之一，不表示收益已被证实 |
| Frozen Cross-PSE 对照 | `diff_formal_crosspse_frozen_decoder_v1.yaml` | 已完成，05 阶段 Decoder-only 对照 |
| Joint Refined-PSE 对照 | `diff_formal_refinedpse_joint_diger_v1.yaml` | 同 DIGER 协议、关闭跨码本 mixer |
| Frozen Refined-PSE 对照 | `diff_formal_refinedpse_frozen_decoder_v1.yaml` | 同 Refined 初始化、仅训练 Decoder |

本目录另有 `diff_joint_smoke.yaml`、`diff_diger_smoke.yaml`、`diff_frozen_crosspse_smoke.yaml`、`diff_frozen_refinedpse_smoke.yaml`，仅用于短测，不进入正式结果。

新增 [pilots/diger_noise_pilot_seed2024.json](05_crosspse_joint/pilots/diger_noise_pilot_seed2024.json) 固定共享起点的三组噪声更新短对照。它由独立 runner 读取，不作为旧训练入口的 YAML overlay；设置和启动规则见 [实验说明](../experiments/05_crosspse_joint/pilots/noise_update/README.md)。

新增 [pilots/diger_feedback_pilot_seed2024.json](05_crosspse_joint/pilots/diger_feedback_pilot_seed2024.json) 拆分重构更新与推荐反馈；三组仍保留同一 Cross Mixer。它使用新的 feedback 入口和输出，旧 noise JSON 不变，见 [实验说明](../experiments/05_crosspse_joint/pilots/feedback_update/README.md)。

## 使用与复现边界

- **加载顺序按启动脚本，不按文件名排序。** 例如 Frozen Cross 仍是：`shared/common → shared/repo_v6 → 05/joint_diger_v2 → 05/frozen_control → 05/smoke（仅短测）`。后项覆盖前项。
- YAML 内的 `../../data`、`../../outputs` 仍以运行时 `external/DiffGRM` 工作目录为基准，不应因 YAML 搬家而改写。
- 训练shell/PowerShell入口已在 `scripts/train/`，准备工具在 `scripts/prepare/`，Attention入口在 `scripts/analysis/`，详见 [脚本索引](../scripts/README.md)。手动传 `--config configs/旧文件名.yaml` 时要改用本页的新地址；旧根目录没有第二套配置副本。
- 历史 metadata、证据、配方及源码 tar 内的旧路径保持原样；隔离复现使用锁定快照，不依赖本次可编辑配置的新位置。
- [B2b 迁移记录](../reproducibility/layout_migrations/20261008_b2b_configs/README.md) 保存每个文件的旧→新路径、哈希和调用顺序验证。

以后做新实验，新建配置并使用唯一 `run_id`，不要覆盖已归档实验的设置或输出。只有新的研究问题才新增阶段目录；重复 seed 仍属于同一阶段。
