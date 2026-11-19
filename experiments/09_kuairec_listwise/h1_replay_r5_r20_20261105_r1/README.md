# H1 原协议 R5→R20 重放与历史节点保存

用户授权：从原 H1 R5 完整状态，以完全相同方法与参数训练到 R20，
保留所有中间验证节点；核对通过后直接提交单 H100，无需再次询问。
用户另外授权：训练结束、确认原结果一致之后，清理本次实验冗余文件。

- 原始父文件：`goal_20261027_r1/runs/H1/initial.pt`，R5/T0。
- 父 SHA256：`991352d4b48998427d44397411eb4178678487de21f1e08e5cd9d39240a06fc1`。
- 直接导入原 `goal_20261027_r1/code` 下、父 checkpoint 绑定的冻结源码。
  不改模型、损失、学习率、优化器、训练循环或数据顺序，不添加门控。
- 使用父文件完全相同的配置：seed2024，BF16，microbatch256×累积4，
  有效 batch1024，40T+3R，FP32/beam32/batch32 完整验证。
- **仍构建原 35R/400T 日程**，不改 cycles；仅在 R20 验证完成、
  checkpoint 发布后通过外部保存钩子停止。源代码文件未修改。
- R8/T40、R11/T80、R14/T120、R17/T160、R20/T200 均保留完整模型、
  Tokenizer、两个优化器、scheduler、RNG、进度和 SID。
- 原 100-update 滚动 last 保存机制保持不变。每个验证节点另有不可覆盖的
  `checkpoints/Rxx_Txxx.pt` 及 SHA 收据，不再只依赖 best/last。
- 不评估 test，不增加 seed/对照，不继续到 R23。

远端根：
`/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/h1_replay_r5_r20_20261105_r1`

产物位于 `runs/H1-Replay-R20/`。大文件保留远端，不自动下载/重新打包到本地。
本地仅保留入口、配置说明、提交/预检收据和指标摘要。

验收：逐节点比较完整验证 summary 和解压后的逐样本 JSON；
R20 另外逐字段、逐张量核对原 best 的模型、Tokenizer、优化器、调度器、
随机状态及计数。所有节点在 GPU 上用原严格 restore 实际加载并回读状态。
准确性判断为精确相等，不用任意容差把不一致归为成功。
若有差异，保存证据并报告，不自动重跑或删除。

清理 allowlist 仅为本次新 run 中的 `best.pt`、`last.pt`：
在 `exact_replay_verified=true` 后，且 SHA 与已保留某节点 checkpoint 完全一致
才删除。保留各节点 checkpoint、完整验证、日志、源码引用、验收清单；
原实验和其他实验目录完全不在删除范围。

预检分 CPU 和 GPU：CPU 不宣称恢复 CUDA RNG；GPU 正式训练前再次检查
完整状态及 CUDA RNG 精确恢复。实际长期重放是否逐字节一致以最终收据为准。
