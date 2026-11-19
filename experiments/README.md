# 阶段报告索引

`experiments/` 按研究推进顺序编号。01–05 统一包含：研究问题、实验设置、逐位置结果、推荐结果、必要的 Tokenizer / Attention 分析、原始证据副本和复现配方。原始运行目录、历史压缩包和旧报告均保留。

| 编号 | 报告 | 里面放了什么 |
|---:|---|---|
| 01 | [`01_ar_baseline_3x256/`](01_ar_baseline_3x256/) | RQ-KMeans / RQ-VAE 的 3 seeds AR 基线；逐位置 SFT 指标、推荐指标、SID 统计，以及补充 Attention 分析和 PDF 版本 |
| 02 | [`02_ar_capacity_ablation/`](02_ar_capacity_ablation/) | PSE、RQ-VAE 与三种 RQ-KMeans 容量布局的 seed 2024 内部对照；逐位置 NLL/accuracy、推荐结果、碰撞率、码本统计与 AR Attention |
| 03 | [`03_diffgrm_tokenizer_comparison/`](03_diffgrm_tokenizer_comparison/) | PSE、RQ-KMeans、RQ-VAE + DiffGRM；逐位置 NLL、CPD 解码顺序、推荐指标与 Attention 图 |
| 04 | [`04_crosspse_offline/`](04_crosspse_offline/) | Original PSE、Refined-PSE、Cross-PSE 的离线 tokenizer 对照；本次单 seed 未观察到交互增益，不把原因写成已证明 |
| 05 | [`05_crosspse_joint/`](05_crosspse_joint/) | Joint/Frozen Cross 与 Joint/Frozen Refined 四组主对照，SIA/v1 历史尝试；统一结果、原始证据、固定代码/输入、校验及重跑说明 |

## 后续阶段入口

| 编号 | 入口 | 研究内容 |
|---:|---|---|
| 06 | [连续检索基线](06_continuous_baseline/README.md) | SASRec/eSASRec；商品级检索协议与03–05的SID级指标分开解读 |
| 07 | [商品内交互](07_item_local_attention/README.md) | 商品内局部注意力、交替联合训练及后续验证诊断，完成范围以该阶段收据为准 |
| 08 | [已退役：单目标历史侧码字交互](08_session_codeword/RETIREMENT_20261026.md) | 用户确认目标为OneRec列表生成；原08不对题，专属代码/模型/备份已删除，历史结果保留；新列表方案尚未训练 |

近期进度与已验证交付统一从[项目交接摘要](../docs/guides/PROJECT_HANDOFF.md)进入。下面01–05的汇报顺序、运行数量及目录约定保留其历史整理范围，不把后续实验混入旧计数。

建议汇报时按 `01 → 02 → 03 → 04 → 05` 讲：先说明现象，再说明由现象引出的 Cross-PSE 方法，最后报告联合训练的初步结果。

## 实验之间怎样对应

| 阶段 | 新运行 | 复用基线 | 统计口径 |
|---|---:|---|---|
| 01 | 6：两种方法 × 3 seeds | 无 | AR，item-level |
| 02 | 3：PSE + 两种非均匀容量 | 01 的 RQ-KMeans / RQ-VAE，均取 seed 2024 | AR，item-level，五组比较 |
| 03 | 3：三种 Tokenizer | 无 | DiffGRM，SID-level，seed 2024 |
| 04 | 2：Refined / Cross 离线训练 | 03 的正式 PSE v6 | DiffGRM，SID-level，seed 2024 |
| 05 | 6：四组主对照 + SIA / DIGER v1 | 04 提供初始化，不是额外的结果行 | DiffGRM，SID-level，seed 2024 |

完整身份索引是 [experiment_registry.json](experiment_registry.json)。01–04 共 **14 次独立正式运行、17 个阶段结果条目**；不要把复用行重复计数，也不要把 AR / DiffGRM 混成同一张性能榜。

## 每个实验包都怎么看

```text
阶段目录/
├── README.md                 # 当前报告：问题 → 设置 → 结果 → 限制
├── HISTORY.md                # 版本演进、pilot 与被替代方案
├── runs.csv                  # 实验身份、证据与配方入口
├── results.csv               # 每个 run 的推荐结果
├── summary.csv               # 01–04：按方法汇总；单 seed 不填标准差
├── evidence/                 # 原始文件逐字节副本，不手工改数字
├── derived/                  # 01–02：从原控制台安全解析的标量曲线
├── report_assets/ 或 figures/ # 保留的报告图
└── reproducibility/          # 配方、资产 hash、代码版本、验收状态、旧 README
```

重算表格、恢复依赖和隔离重跑见 [01–04 复现入口](../reproducibility/README.md)；05 继续使用 [自己的入口](05_crosspse_joint/reproducibility/README.md)。本地整理通过不代表 GPU 复现已经通过。
