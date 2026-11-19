# 两个新 seed 的完整方案对照

用户在 2026-11-09 授权：seed 2025、2026 各一组 H0 持续交替至 R35/T400、一组原 H1 在 R20/T200 冻结整个 Tokenizer 和 SID 后训练推荐器至 R35。共四张独立 H100。不是 H0 Freeze 对照，不把收益单独归因于 Mixer。

## 固定协议

只改变随机种子和 H1 R20 后跳过 T 更新/目录刷新。只读复用已冻结的历史源、数据与 PSE 初始资产，不使用新模型改动。保留中心编号标记、Full Mixer、原 35R/400T 学习率及 Gumbel 日程、BF16 训练/FP32 推理、global batch=1024、microbatch=256、eval batch=32、beam=32。

每组从对应 seed 独立初始化，前 R5 均禁用 Mixer，与原暖身一致。同 seed 两张卡各自执行确定性暖身，在 R5 比较模型、Tokenizer、两个优化器、scheduler、RNG 指纹；一致才作为成对对照。这样四卡均执行有效训练，不占卡等待另一组暖身。不同 seed 不从 seed2024 的 checkpoint 继续。

验证 R5、R8、R11、R14、R17、R20、R23、R26、R29、R32、R35，保存每个验证节点完整 checkpoint。H1 在 R20 记录整个 alignment 及 T optimizer 指纹，后续验证/阶段边界检查完全不变。T 步数为200，R步数为24255；H0 T步数400。R scheduler 不重启、不缩短；冻结不重新计算SID。

主比较为固定 R35 测试 NDCG@5；辅助比较为按验证 NDCG@5 选出的 best（相等取较早），统一候选集 R5 至 R35。两者均报告 HR@5、Recall@5。测试只运行 best 和 R35，若同节点则复用，不根据测试挑点。

最终验收：四个终态、GPU释放、完整训练时钟、R5配对、冻结指纹、数据/源码绑定、逐样本测试指标独立复核。合并历史seed2024的原H0与H1 Freeze，报告每seed成对差、三seed均值及样本标准差；不把3个seed夸大为确定的显著性证据。按用户成组的配对bootstrap只作固定模型的样本不确定性补充，不能替代跨seed结果。

大型checkpoint和预测只保留共享盘，本地保存源码、指标、SHA和收据。每作业24小时硬上限，实际完成自动释放；不自动重提失败作业，不恢复旧监控。

## 操作

`python control.py prepare` 准备和检查，`python control.py submit 2025 H0` 等四组分别提交，`python control.py status` 查询当前状态。入口 experiment.py 支持显式 --resume，但需先查明失败及已有结果；不自动覆盖已有测试。
