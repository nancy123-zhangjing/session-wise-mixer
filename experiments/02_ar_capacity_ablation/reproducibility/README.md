# 02｜AR 五组内部对照复现

公共操作见 [项目复现手册](../../../reproducibility/README.md)。这里绑定五个结果条目，但只有三次新增训练。

| 实验 ID | 来源 |
|---|---|
| `pse_seed2024` | 本阶段新增：PSE 3×256 |
| `rqk_128_256_512_seed2024` | 本阶段新增：RQ-KMeans 128×256×512 |
| `rqk_512_256_128_seed2024` | 本阶段新增：RQ-KMeans 512×256×128 |
| `rqk_seed2024` | 精确复用01的 RQ-KMeans seed2024 |
| `rqvae_seed2024` | 精确复用01的 RQ-VAE seed2024 |

全部是严格 item-level AR。三位语义码之外仍有第四位 suffix。三种 RQ-KMeans 布局的组合容量相等，但非均匀组码字总数及输出层大小不同；不要宣称参数量完全一致。PSE、RQ-VAE 和 RQ-KMeans 的预处理也不是同一套。

## 核验入口

- [recipes/](recipes/)：配置覆盖顺序、历史 SID、共享 embedding / PCA、随机 Decoder 初始化、基线引用。
- [assets.lock.json](assets.lock.json)：准确的缓存和最佳权重，不从“最新文件”推断身份。
- [evidence_manifest.json](evidence_manifest.json)：原始证据路径及逐文件 hash。
- [code_versions.json](code_versions.json)：固定重放源码；与根目录共享快照。
- [analysis_checks.json](analysis_checks.json)：NDCG@10 最佳轮数、四个位置结果核对。
- [previous_README.md](previous_README.md)：旧报告保留。原 PDF/图（如有）未改写。

原 `ar_internal_study/per_seed.csv` 包含额外 seed 行，因此本阶段按 **seed + 精确 metadata 身份** 选五组，不把其他 seed 混入单 seed 表。`derived/` 是日志解析结果，不是原始文件。

```bash
python scripts/repro/project_repro.py verify --stage 02
python scripts/repro/project_repro.py analyze --stage 02
python scripts/repro/project_repro.py plan --stage 02 --experiment rqk_128_256_512_seed2024
```

当前已完成证据与资产绑定、结果重算校验；GPU正式重训、checkpoint重评未执行。AR旧 trainer 不支持 batch 截断，本次没有修改它，因此没有可靠两 batch smoke。固定 SID 重训下游 ≠ 重新量化。完整恢复/运行说明在公共手册，不能只复制这一份 README 到服务器。
