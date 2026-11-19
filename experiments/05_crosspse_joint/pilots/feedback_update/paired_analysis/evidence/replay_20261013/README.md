# 2026-10-13 配对重放：证据索引

这里只保留本轮轻量原样收据、配对统计与案例核验；不混入旧实验封存证据，不含完整权重。主要说明见 [RESULTS.md](../../RESULTS.md)。

| 文件 | 含义 |
|---|---|
| `summary.json` | 固定三组 `decoder_3` 的原样重放、指标逐项误差、输入/代码哈希与模型未变检查 |
| `paired_summary.json` | 原 float32 分数聚合、共享分组及碰撞分层的所有配对统计 |
| `COMPLETE.json` | 正式业务完成及上述两个 JSON 的 SHA256 |
| `preflight_passed.json` | 实际 SSD 推理代码 `45ffc26`、59 项预检、存储写入核验；不是 HDD 同步证明 |
| `formal_started.json` / `formal_finished.json` | 实际启动命令、单 H100、应用退出码与业务成功标记 |
| `collection_receipt.json` | 仅轻量结果包的 SHA256、大小及不含权重的声明 |
| `local_verification.json` | 下载包核验、三份预测校验、本地独立重算与 GPU JSON 完全一致 |
| `hit_case_audit.json` | 从全部逐样本预测选取 55 个主比较 HR@10 得失案例，含目标编码/碰撞/排名/源哈希 |
| `audit_hit_cases.py` | 此固定运行的案例核验源码原字节，含期望计数与四个目标案例断言；使用显式输入/新输出路径重放 |

原始压缩预测及 `catalog_ids.json` 留在：

- 服务器：`/inspire/ssd/project/project-public/<inspire-account>/nancy/experiment1_analysis/outputs/analysis/stage05_feedback/paired_formal_20261013_1123/`。
- 本地：`outputs/pilots_delivery/feedback_paired_20261013_1123/formal_local/paired_formal_20261013_1123/`。
- 本地已核验下载包：`outputs/pilots_delivery/feedback_paired_20261013_1123/formal_proof.tar.gz`，SHA256 `14fcd2b6038a037977b1554ad30f23c796aee6683947f9afc38748968ae87483`。

原模型仍在 HDD 原训练输出中。上述包、收据及案例 JSON **不是权重备份**，仅凭它们不能恢复训练。

从项目根目录独立重算（以下新输出名称须不存在）：

```powershell
python -B scripts/analysis/feedback_paired_analysis.py --input-dir outputs/pilots_delivery/feedback_paired_20261013_1123/formal_local/paired_formal_20261013_1123 --output-dir outputs/pilots_delivery/feedback_paired_20261013_1123/regroup_NEW --expected-count 22363
python -B experiments/05_crosspse_joint/pilots/feedback_update/paired_analysis/evidence/replay_20261013/audit_hit_cases.py --formal-dir outputs/pilots_delivery/feedback_paired_20261013_1123/formal_local/paired_formal_20261013_1123 --output outputs/pilots_delivery/feedback_paired_20261013_1123/hit_cases_NEW.json
```

案例审计含生成时间，重算后的完整文件 SHA256 不必相同；应比较源输入哈希、检查字段及全部案例内容。不要覆盖本目录的原记录，也不再次启动训练。
