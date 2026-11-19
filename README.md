# Semantic ID × 生成式推荐实验项目

这个目录用于研究一个核心问题：**Semantic ID 的构造方式与各位置的信息分配，会怎样影响 AR 与扩散式推荐模型的解码难度和最终推荐效果。**

## 从哪里开始看

| 想看什么 | 入口 |
|---|---|
| 已整理好的阶段报告 | [`output/README.md`](output/README.md) |
| 当前联合训练结果 | [`output/05_crosspse_joint/README.md`](output/05_crosspse_joint/README.md) |
| 配置文件说明 | [`configs/README.md`](configs/README.md) |
| 运行和分析脚本说明 | [`scripts/README.md`](scripts/README.md) |
| 技术说明与历史记录 | [`docs/README.md`](docs/README.md) |
| 原始实验输出 | `outputs/` |
| 历史证据包和旧导出 | [`archive/README.md`](archive/README.md) |

## 研究推进顺序

| 阶段 | 内容 | 当前状态 | 主要入口 |
|---:|---|---|---|
| 01 | AR-GRM：RQ-KMeans 与 RQ-VAE 的 `3×256` 基线 | 已完成，3 seeds | [`output/01_ar_baseline_3x256/`](output/01_ar_baseline_3x256/) |
| 02 | AR-GRM：PSE 与 `128×256×512`、`256×256×256`、`512×256×128` 容量消融 | 已完成，seed 2024 | [`output/02_ar_capacity_ablation/`](output/02_ar_capacity_ablation/) |
| 03 | DiffGRM：PSE、RQ-KMeans、RQ-VAE 的 `4×256` 对照 | 已完成，seed 2024 | [`output/03_diffgrm_tokenizer_comparison/`](output/03_diffgrm_tokenizer_comparison/) |
| 04 | Refined-PSE / Cross-PSE：离线训练码本后固定 SID | 已完成，结果不够理想 | [`output/04_crosspse_offline/`](output/04_crosspse_offline/) |
| 05 | Cross-PSE × DiffGRM：DIGER-style 梯度桥接与交替联合训练 | 已完成一轮，存在正向信号，仍需对照 | [`output/05_crosspse_joint/`](output/05_crosspse_joint/) |

## 目录职责

```text
experiment1/
├── output/       # 人工整理后的阶段报告；平时汇报优先看这里
├── outputs/      # 训练/分析脚本产生的原始结果；不要随意改名
├── configs/      # 所有实验配置，含正式配置和历史试验配置
├── scripts/      # 数据准备、训练、检查、统计和画图入口
├── docs/         # pipeline 说明、方案设计和历史排错记录
├── external/     # 固定版本的 DiffGRM / GenRec 上游代码
├── data/         # 数据、embedding 和本地缓存
├── patches/      # 对上游仓库的补丁
├── archive/      # 证据压缩包、旧报告导出和可恢复的废弃副本
└── requirements* # Windows / Linux 环境依赖
```

## 文件管理约定

- `outputs/` 是原始证据层：脚本和 metadata 会引用其中的路径，因此不重命名内部目录。
- `output/` 是汇报层：只放 README、表格和必要图片，按研究顺序编号。
- `archive/` 是历史层：旧压缩包、重复报告和临时文件放在这里，不再混入当前结果。
- 新实验使用明确的 `run_id`，不要覆盖已有配置或结果；新阶段报告继续采用 `06_...` 的编号方式。
- `.runtime/` 是本地工具缓存，不属于实验。它目前被 Windows 权限锁定，已加入忽略列表；关闭相关 IDE/Codex 进程后可手动删除。

### `outputs/` 内部内容

| 子目录 | 内容 | 是否直接拿去汇报 |
|---|---|---|
| `attention/` | AR 和 DiffGRM 的 Attention 原始 CSV、运行信息及中间图 | 一般不直接使用，由 `output/` 中报告引用整理后的图 |
| `figures/` | 各次分析脚本生成的原图 | 需要核对细节时使用 |
| `reports/` | 每个 run 的 `per_seed.csv`、summary、evaluation history 等机器汇总 | 是报告数值的主要证据来源 |
| `run_metadata/` | run ID、seed、checkpoint、SID hash、训练耗时和测试指标 | 用于确认实验身份与可复现性 |
| `tokenizers/` | tokenizer 的 summary、geometry、training 记录和冻结 SID 信息 | 用于分析碰撞、利用率、重构和 SID 变化 |

`data/` 在当前本地副本中没有正式数据文件；服务器上的原始数据与 embedding 不应为了写报告重复复制到这里。`external/` 保存上游代码，普通整理结果时不要修改或移动。

## 当前最重要的下一步

联合 Cross-PSE 的结果比静态 PSE 基线更好，但目前还不能把增益唯一归因于“码本交互”。下一步先补两个 matched controls：

1. **冻结 Cross-PSE，只继续训练 Decoder**：保持相同的 recommender 更新预算，排除“只是 Decoder 多训练了一段”的解释。
2. **Joint Refined-PSE（关闭 mixer）**：使用相同交替训练与梯度桥接，只去掉跨码本交互，隔离 mixer 本身的作用。

在这两个对照完成前，暂时没有必要扩展到更多随机种子。
