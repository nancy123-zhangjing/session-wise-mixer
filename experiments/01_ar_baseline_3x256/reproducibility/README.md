# 01｜AR 三 seed 基线复现

操作与所有公共路径见 [项目复现手册](../../../reproducibility/README.md)。本目录不是独立完整训练仓库，需一并保留根目录 `reproducibility/` 和 `scripts/repro/project_repro.py`、`project_replay.py`、`stage05_repro.py`。

## 绑定范围

- RQ-KMeans / RQ-VAE × seed 2024、2025、2026，共六次独立运行。
- 三位语义码，每位256；另加一位消歧 suffix。评估是严格 item-level。
- 推荐模型随机初始化；**不从本组最终 checkpoint 续训**。
- 每个 seed 的历史 SID、PCA 缓存、共享 embedding、item mapping 和 RQ-KMeans 量化产物均独立绑定，不混用 seed2024 的缓存。
- RQ-VAE 的离线权重属于 Tokenizer 产物；本入口重放下游推荐训练，不再次拟合 Tokenizer。

## 文件职责

| 文件 | 用途 |
|---|---|
| [recipes/](recipes/) | 六个实验的配置顺序、数据与初始化绑定 |
| [assets.lock.json](assets.lock.json) | 必需输入、最佳 checkpoint、量化产物和原 console 的迁移包成员 / SHA256 |
| [code_versions.json](code_versions.json) | 固定重放源码，历史 runtime commit 未记录 |
| [evidence_manifest.json](evidence_manifest.json) | 原始证据副本与来源 hash |
| [analysis_checks.json](analysis_checks.json) | 从九份 AR 正式日志中对应的六份核对 NDCG@10 最佳轮数，验证位置记录 |
| [package_integrity.json](package_integrity.json) | 配方、锁、辅助代码及派生表完整性 |
| [previous_README.md](previous_README.md) | 整理前报告原文，不覆盖 |

训练总 loss 和 validation history 在 [../derived/](../derived/)，由日志解析，不冒充原生 CSV。逐位置诊断和 Attention 在 `../evidence/`。逐 seed 结果分别保留，三 seed 汇总使用样本标准差。

```bash
python scripts/repro/project_repro.py verify --stage 01
python scripts/repro/project_repro.py analyze --stage 01
python scripts/repro/project_repro.py plan --stage 01 --experiment rqk_seed2024
```

执行训练前先按公共手册恢复并检查输入。旧 AR trainer 没有 batch 限额，所以隔离入口不支持 AR smoke；不能把全量一轮声称为两 batch 短测。formal 入口与最佳权重重评均**尚未 GPU 验收**。原始完整参数记录不全，现在的源代码快照不等于证明当时的执行版本。
