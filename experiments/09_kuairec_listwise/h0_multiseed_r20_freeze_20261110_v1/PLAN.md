# H0 两个新 seed 的 R20 冻结对照

用户在 2026-11-10 授权补跑 seed2025、2026 的 H0 Freeze。

从 `multiseed_freeze_20261109_v1/seed{seed}_H0/checkpoints/R20_T200.pt` 分叉，恢复模型、整个 Tokenizer、R/T optimizer、scheduler 和 RNG。启动时核对 checkpoint SHA、绑定及 R20/T200 时钟，逐项验证恢复无变化。沿用既有单卡训练入口，仅修改分叉初始化和 H0 的冻结条件。原实验与原代码均保留。

R20 后跳过全部 T 更新及 SID 刷新，只执行 R21–R35。Mixer 始终关闭，整个 Tokenizer/SID 与 T optimizer 保持 R20 指纹。保留原 35R/400T 计划编号、推荐器学习率日程、每阶段随机种子与样本顺序；不重启 optimizer、不缩短 scheduler。终点 R35/T200，新增 R 更新 10,395 次。单卡 H100，BF16 训练、FP32 推理、global batch1024/microbatch256、beam32、eval batch32。

验证节点 R23、26、29、32、35；复用 R5–R20 验证记录，不重新推理。主比较固定 R35，辅助比较全部候选 R5–R35 上按验证 NDCG@5 选择的 best。两个父组的 R20 均为截至分叉时最佳，初始 best 因此继承 R20。只测试 best 和 R35；若相同则复用。

对照一：同 seed H0 Freeze 与 H0 持续交替，测量 R20 后冻结的效果。对照二：同 seed H0 Freeze 与 H1 Freeze，统一冻结日程比较是否使用 Mixer 的完整训练方案。不能据此把差异定位到某个具体机制。历史 seed2024 H0 Freeze 已有，不重复训练。

每组独立一张卡，12小时作业上限。大型 checkpoint 和预测留远端；本地保存代码、小指标及 SHA 收据。不自动重提失败任务，不删除父实验。验收核对父状态、更新数、冻结指纹、最佳点选择、完整验证/测试及 GPU 释放；已有四组最终逐样本审计未成功回传，仍单独保留未验收状态。
