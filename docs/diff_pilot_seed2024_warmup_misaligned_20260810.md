# DiffGRM Seed 2024 Pilot（Warmup 调度不匹配，排除出正式汇总）

## 身份

- 原 run ID：`diff_formal_pse_4x256`
- Seed：2024
- 运行日期：2026-08-10
- 状态：Pipeline 验证成功，但不进入正式三 seed 统计
- 原因：physical batch 64、gradient accumulation 16，使 effective batch 为 1024；100 epochs 约有 10,400 个 optimizer updates，而 warmup 为 10,000 steps，约 96% 的训练处于 warmup。最佳验证结果出现在最后一轮，尚未形成合理的完整调度。

## 已取得的结果

| 项目 | 数值 |
| --- | ---: |
| Best epoch | 100 |
| Best validation weighted score | 0.0501 |
| Test HitRate@5 | 0.049904 |
| Test HitRate@10 | 0.073917 |
| Test NDCG@5 | 0.035168 |
| Test NDCG@10 | 0.042881 |
| Test legal@10 | 1.000000 |
| Test duplicate@10 | 0.002267 |
| Fit time | 3.61 hours |

## Test fully-masked position diagnostics

| Position | NLL | Accuracy |
| --- | ---: | ---: |
| SID0 | 8.4139 | 0.0049 |
| SID1 | 8.3573 | 0.0055 |
| SID2 | 8.3065 | 0.0063 |
| SID3 | 8.2684 | 0.0048 |

## Test greedy CPD mean decode rank

| Position | Mean rank |
| --- | ---: |
| SID0 | 2.456 |
| SID1 | 2.526 |
| SID2 | 2.506 |
| SID3 | 2.513 |

## PSE SID collision

- Items：12,101
- Unique full SIDs：10,656
- Collided items：2,172（17.9489%）
- Colliding groups：727
- Largest collision group：40

## 归档位置

服务器运行 `bash scripts/run_diff_formal.sh archive-pilot 2024 python` 后，完整证据复制至：

```text
outputs/archive/diff_pilot_seed2024_warmup_misaligned_20260810/
```

归档包含 console log、run metadata、position/order CSV、checkpoint 和带 SHA-256 的 `archive_manifest.json`。原始文件不会被移动或删除。
