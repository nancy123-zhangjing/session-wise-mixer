# Semantic ID × 生成式推荐实验项目

这个目录用于研究一个核心问题：**Semantic ID 的构造方式与各位置的信息分配，会怎样影响 AR 与扩散式推荐模型的解码难度和最终推荐效果。**

## 从哪里开始看

跨对话接手先读 [项目交接摘要](docs/guides/PROJECT_HANDOFF.md)。当前09为KuaiRec五商品列表生成：数据已准备，J/H两种DiffGRM列表结构与Full Mixer开/关的交替联合训练代码已实现，尚未启动正式训练/GPU。见[实现入口](docs/research/onerec_listwise/IMPLEMENTATION.md)、[参数来源](docs/research/onerec_listwise/IMPLEMENTATION_REFERENCES.md)。原08单目标历史侧交互已按要求退役，保留[清理记录](experiments/08_session_codeword/RETIREMENT_20261026.md)；不能混称为当前session-wise任务。下文较早阶段状态不代表当前待办。

| 想看什么 | 入口 |
|---|---|
| 已整理好的阶段报告 | [`experiments/README.md`](experiments/README.md) |
| 当前联合训练结果 | [`experiments/05_crosspse_joint/README.md`](experiments/05_crosspse_joint/README.md) |
| 迁移后的当前验收状态与运行目录 | [2026-10-09 迁移收尾](docs/guides/EXPERIMENT_WORKFLOW.md#六2026-10-09-迁移收尾与验收范围) |
| 全部实验身份与基线复用关系 | [`experiments/experiment_registry.json`](experiments/experiment_registry.json) |
| 01–04 的资产恢复、源码锁定和复现操作 | [`reproducibility/README.md`](reproducibility/README.md) |
| 以后怎样建实验、存结果和做 Git 版本管理 | [`docs/guides/EXPERIMENT_WORKFLOW.md`](docs/guides/EXPERIMENT_WORKFLOW.md) |
| 项目地图：每类文件应该放哪里 | [PROJECT_MAP.md](docs/guides/PROJECT_MAP.md) |
| 本次完整归类与校验记录 | [B2c 脚本归类记录](reproducibility/layout_migrations/20261008_b2c_scripts/README.md) |
| 当前日常复现、资产位置及旧导入的分工 | [B1 工作流](reproducibility/WORKFLOW_B1.md) |
| 配置文件说明 | [`configs/README.md`](configs/README.md) |
| 运行和分析脚本说明 | [`scripts/README.md`](scripts/README.md) |
| 测试分类与本地回归入口 | [`tests/README.md`](tests/README.md) |
| 技术说明与历史记录 | [`docs/README.md`](docs/README.md) |
| 原始实验输出 | `outputs/` |
| 历史证据包和旧导出 | [`archive/README.md`](archive/README.md) |

## 研究推进顺序

| 阶段 | 内容 | 当前状态 | 主要入口 |
|---:|---|---|---|
| 01 | AR-GRM：RQ-KMeans 与 RQ-VAE 的 `3×256` 基线 | 已完成，3 seeds | [`experiments/01_ar_baseline_3x256/`](experiments/01_ar_baseline_3x256/) |
| 02 | AR-GRM：PSE 与 `128×256×512`、`256×256×256`、`512×256×128` 容量消融 | 已完成，seed 2024 | [`experiments/02_ar_capacity_ablation/`](experiments/02_ar_capacity_ablation/) |
| 03 | DiffGRM：PSE、RQ-KMeans、RQ-VAE 的 `4×256` 对照 | 已完成，seed 2024 | [`experiments/03_diffgrm_tokenizer_comparison/`](experiments/03_diffgrm_tokenizer_comparison/) |
| 04 | Refined-PSE / Cross-PSE：离线训练码本后固定 SID | 已完成，结果不够理想 | [`experiments/04_crosspse_offline/`](experiments/04_crosspse_offline/) |
| 05 | PSE × DiffGRM：联合训练与 Frozen 对照 | 四组主对照、两版历史尝试已归档；四组输入已校验，Cross Joint v2 GPU 短测通过；正式重训未验收 | [`experiments/05_crosspse_joint/`](experiments/05_crosspse_joint/) |

## 目录职责

```text
experiment1/
├── experiments/  # 01–05 实验包：报告、证据副本、结果表、复现配方
├── outputs/      # 训练/分析脚本产生的原始结果；不要随意改名
├── reproducibility/ # 01–04 共用源码快照、迁移资产清单、原环境记录
├── configs/      # shared/ 共用设置 + 01–05 阶段配置；保留历史试验
├── scripts/      # repro/train/checks/prepare/tokenizers/analysis/reports/audit/legacy
├── tests/        # 标准库安全回归与模型测试；依赖分别说明
├── docs/
│   ├── guides/  # 当前工作流与重构清单
│   ├── methods/ # 方法及实验协议说明
│   └── history/ # 旧版本和排错记录，保留原文
├── external/     # 固定版本的 DiffGRM / GenRec 上游代码
├── data/         # 数据、embedding 和本地缓存
├── patches/      # 对上游仓库的补丁
├── archive/      # 证据压缩包、旧报告导出和可恢复的废弃副本
└── requirements* # Windows / Linux 环境依赖
```

## 文件管理约定

- `outputs/` 是原始证据层：脚本和 metadata 会引用其中的路径，因此不重命名内部目录。
- `experiments/` 按研究顺序编号；01–05 均按实验包组织。01–04 共用根目录 `reproducibility/` 的代码快照，05 保持已经封存的独立复现入口。
- `archive/` 是历史层：旧压缩包、重复报告和临时文件放在这里，不再混入当前结果。
- 新实验使用明确的 `run_id`，不要覆盖已有配置或结果；先记录代码 commit、完整配置与初始化 hash。确有新研究问题时才新建 `06_...` 阶段，同阶段的重复 seed 不另起阶段。
- `.runtime/` 是本地工具缓存，不属于实验。它目前被 Windows 权限锁定，已加入忽略列表；关闭相关 IDE/Codex 进程后可手动删除。

### `outputs/` 内部内容

| 子目录 | 内容 | 是否直接拿去汇报 |
|---|---|---|
| `attention/` | AR 和 DiffGRM 的 Attention 原始 CSV、运行信息及中间图 | 一般不直接使用，由 `experiments/` 中报告引用整理后的图 |
| `figures/` | 各次分析脚本生成的原图 | 需要核对细节时使用 |
| `reports/` | 每个 run 的 `per_seed.csv`、summary、evaluation history 等机器汇总 | 是报告数值的主要证据来源 |
| `run_metadata/` | run ID、seed、checkpoint、SID hash、训练耗时和测试指标 | 用于确认实验身份与可复现性 |
| `tokenizers/` | tokenizer 的 summary、geometry、training 记录和冻结 SID 信息 | 用于分析碰撞、利用率、重构和 SID 变化 |

`data/` 在当前本地副本中没有正式数据文件；服务器上的原始数据与 embedding 不应为了写报告重复复制到这里。`external/` 保存上游代码，普通整理结果时不要修改或移动。

2026-10-07 首次归档保留了旧文件；随后按批次迁移实验包、配置和脚本，未改训练算法。2026-10-09 服务器运行资产已独立迁入新目录，旧服务器目录仅改名为退役备份，没有删除。旧 README 已备份到各阶段的 `reproducibility/previous_README.md`；旧 PDF 和图片保留，但以更新后的 README 和可核验数据为准。历史临时目录里也可能有唯一证据，先不要按名字清理。

归档基线已保存于 `32f3b4e`。A批已将实验包由 `output/` 整体迁至 `experiments/`；原始证据、历史 README 中的旧路径保留，当前入口和包定位已更新。`outputs/` 原始结果目录未移动。

B1进一步分开了日常复现和一次性历史导入：两个日常 `build` 已退役，旧资料导入放在 `scripts/legacy/`。大资产位置使用本机配置，身份仍按原SHA256锁定；实验设置和结果没有变。旧资料保留，脚本入口现已按用途分类。

已提交的回退点：A=`737bebf`，B1=`5f4f433`，B2a=`07d6a83`，B2b=`3d41254`，B2c=`fd5faee`；其后跨平台字节保护与运行路径兼容分别为 `5cc5ef3`、`594b6ad`。B2c 将全部75个根层脚本按职责归类，不改科学证据及历史源码。详见 [项目地图](docs/guides/PROJECT_MAP.md) 和 [B2c记录](reproducibility/layout_migrations/20261008_b2c_scripts/README.md)。私人草稿、授权截图仅加入忽略规则，未删除或搬移。

检查整理后的日常入口（不启动训练、不恢复大文件）：

```powershell
python tests/run_repro_checks.py
python tests/test_runtime_paths.py
```

旧 A/B1/B2a/B2b/B2c 布局审计和收据对应各自整理时点；B2c 的路径变更限定审计不作为 `594b6ad` 及后续研究改动的日常总验收。当前运行校验使用上述两条命令，迁移检查见最新收据；不改写旧收据让新版本通过。历史快照、实验参数与原始结果不因此更新。

## 当前最重要的下一步

05 阶段的 Joint/Frozen Cross 与 Joint/Frozen Refined 四组已完成，不需要再把它们列为待补实验。目前仍不足以证明码本交互或 Tokenizer 联合更新带来稳定收益：Cross v2 最佳点尚未更新 Tokenizer，Refined 的指标升降混合。

迁移运行链路已完成有限范围验收，见 [迁移收尾](docs/guides/EXPERIMENT_WORKFLOW.md#六2026-10-09-迁移收尾与验收范围)。不必为整理项目重跑全部历史实验；下一步先确定更严格的初始化、Decoder 更新量和选模协议对照，再建立新实验配置与运行入口。旧阶段报告和 CHECKS 中的状态对应其注明的整理日期，当前迁移状态以本页和 [05 复现手册](experiments/05_crosspse_joint/reproducibility/README.md) 为准。短测只验证运行链路，不增加新的科学结论。
