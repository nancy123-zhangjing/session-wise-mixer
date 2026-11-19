# 03｜DiffGRM 三 Tokenizer 复现

完整操作见 [项目复现手册](../../../reproducibility/README.md)。三组均4×256、seed2024、SID-level，Decoder随机初始化，不与AR的item-level结果混排。

## 精确身份

| 实验 ID | 原 checkpoint 目录时间戳 |
|---|---|
| `pse_seed2024` | `AmazonReviews2014_Aug-11-2026_14-09-45` |
| `rqk_seed2024` | `AmazonReviews2014_Aug-12-2026_06-37-17` |
| `rqvae_seed2024` | `AmazonReviews2014_Aug-12-2026_12-04-25` |

PSE必须是正式v6，不能换成同日 `10-58-33` 的v5。即使测试数字相同，也不是同次运行。训练配置按 recipe 顺序覆盖，warmup、累计梯度和早停不能只读取 common YAML 猜测。

## 保留了什么

- [recipes/](recipes/)：三组的历史SID、数据缓存、配置及随机初始化定义。
- [assets.lock.json](assets.lock.json)：数据/embedding/SID/可用Tokenizer产物/最终最佳权重的SHA256。
- [evidence_manifest.json](evidence_manifest.json)：原始metadata、位置/顺序/标量曲线、Attention及报告副本；history保留pilot材料。
- [analysis_checks.json](analysis_checks.json)：验证weighted score选出的最佳轮数，以及train/test四位置数据。
- [code_versions.json](code_versions.json)：当前绑定源码，非冒称历史runtime commit。
- [previous_README.md](previous_README.md)：未覆盖的旧报告。

PSE早期metadata没有码本利用率字段；补充统计来自04正式Tokenizer summary中的 `base_pse`，副本见 `../evidence/supplementary/`，来源单独注明。Attention是完整测试集的greedy beam1路径，不是beam256正式搜索的每条分支。

```bash
python scripts/repro/project_repro.py verify --stage 03
python scripts/repro/project_repro.py analyze --stage 03
python scripts/repro/project_repro.py run --stage 03 --experiment pse_seed2024 --mode smoke
```

上面的run只显示计划。恢复并校验输入、激活GPU环境后加 `--execute` 才运行。当前只有本地证据/代码/测试检查；GPU smoke、正式重训、独立checkpoint重评**未验收**。重新拟合Tokenizer需另建实验，本入口保持历史SID不变。
