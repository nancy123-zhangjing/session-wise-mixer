# RQCD-H 交接（2026-11-05）

> 新窗口从这里开始即可。不需要前面更早的 H0/H1/Freeze/Decay 上下文；如需回溯看 `docs/research/onerec_listwise/` 下协议与验证文档。

## 1. 这次讨论确定了什么

用户提出的新架构（区别于已做的离散 SID DiffGRM）：

> **固定 RQ-VAE tokenizer → 扩散模型逐件生成连续潜向量 → 目录约束残差量化成 SID → 落地商品 → 作为下一件的前缀条件**，session-wise 生成 KuaiRec 五商品列表。

已定稿并锁定的实验协议：`docs/research/onerec_listwise/RQ_CONTINUOUS_SESSION_BASELINE_PROTOCOL_20261104.md`

核心设定：
- 数据：KuaiRec 五商品列表（沿用 Stage 09 的 `kuairec_listwise_v1/run1`：709,443 训练 / 71,657 验证 / 53,194 测试 / 10,728 商品）
- RQ-VAE：4层×256码字，latent dim 32，输入 256 维 PCA 内容特征（BGE-M3 → PCA），预训练 60,000 更新后**全冻结**
- 推荐器：1层历史 Encoder + 4层去噪 Decoder，hidden 256，BF16，1000 扩散步 cosine，x0 预测，MSE；推理 50 步 DDIM、eta=0、逐件顺序生成 5 件
- 训练：LR 1e-4、有效 batch 1024、warmup 500、35 epoch / 24,255 updates、11 个锁定验证点按 NDCG@5 选 best、保留 R35
- 评估：每样本单份生成（无 beam/重排），评估 batch 1024（H100 实测提速 ~30×且数学等价）

## 2. 代码在哪

- 实现：`scripts/rqcd/`（common/model/tokenizer/catalog/data/training/evaluation/run/preflight/analysis）
- 配置：`configs/09_kuairec_listwise/rqcd_h_v1.json`
- 测试：`tests/test_rqcd.py`（19 项）+ 复用 listwise metrics/decoding 测试
- 运行说明：`scripts/rqcd/README.md`
- H100 实测报告：`docs/research/onerec_listwise/RQCD_IMPLEMENTATION_VERIFICATION_20261105.md`

## 3. 实验结果（seed 2024，已完成，负结果）

Job `nancy-rqcd-h-train-20261105-r3`，H100，SUCCEEDED，约 2.1 GPU 小时。

| 检查点 | val NDCG@5 | test NDCG@5 | test HR@5 |
|---|---|---|---|
| best = R08 | 0.000694 | 0.000580 | 0.00293 |
| R35 | 0.000626 | 0.000598 | 0.00303 |

随机基线 recall@5 ≈ 0.00047 → **接近随机**。

关键诊断（比指标本身更重要）：
- `raw_valid` ≈ 0.25%：生成向量自由量化出的 SID 几乎从不命中真实商品，目录约束量化承担几乎全部合法性
- `changed` ≈ 99.75%
- completion=1.0、0 重复、0 非法 → 管道正确，瓶颈是**向量生成质量**（训练 MSE 1.91→0.438 走平），不是量化环节

收据与备份：
- `experiments/rqcd_h_baseline/rqcd_h_v1_seed2024_20261105/CLOSURE.json`（完整验收收据）
- 本地备份：`experiments/rqcd_h_baseline/rqcd_h_v1_seed2024_20261105/backup/rqcd_h_backup_20261105.tar.gz`（346MB，SHA `e5237f71…`，已做独立恢复验证）
- 远端运行根：`/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_rqcd/rqcd_h_v1_seed2024_20261105`（含全部 35 epoch checkpoint + 4106 文件 MANIFEST）
- 注意：远端 `analysis.py` 被修过一个 int64 序列化 bug（本地 commit `ec72dc7`），与封存 source binding 有一处哈希差异，仅影响事后分析

## 4. 平台/运维要点（下次跑还会用到）

- Inspire CLI 必须 `NO_PROXY=sii.edu.cn,*.sii.edu.cn`（环境变量代理会导致 ECONNRESET）
- env-zzl 是 `CPU资源空间` 的 notebook（exec/scp 都走它）；GPU notebook `nancy-rqcd-h100-20261105-r1` 已 STOPPED
- Job 创建参数：`--group "开发区-H100-cuda12.8版本-183核" -q "1,20,200" -i ngc-pytorch:25.02-cuda12.8.0-py3 --max-time 4`
- 远端 Python：`/inspire/hdd/project/project-public/<inspire-account>/nancy/conda-envs/grm-exp/bin/python`
- 脚本入口 `delivery/run_rqcd_h.sh` 已含断点续跑逻辑（有 `last.pt` 自动 `--resume`）
- 坑：`output_lock` 拒绝已存在的输出目录；`nvidia-smi | head` 会 SIGPIPE

## 5. 未决问题 / 可能的下一步（供讨论，不是已批准计划）

- 同一批生成向量做"直接最近邻检索 vs SID 落地"对照 → 定位是扩散器问题还是量化接口问题
- 扩散器容量/条件不足？（MSE 走平、raw_valid 极低）
- RQ z 空间是为重建优化的，对生成任务可能不是好目标空间
- 是否值得联合训练 RQ（协议明确留作第二阶段，本次未做）
- 五向量联合扩散（方案 B）未尝试
- 不要自动重跑或加参数搜索；任何新实验先讨论
