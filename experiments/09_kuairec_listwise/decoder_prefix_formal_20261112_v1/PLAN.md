# 四组 Decoder 前缀正式实验：先到 R20

用户于2026-11-12选择四组先训练到R20，查看结果再决定。每组独立一张H100，无DDP，最多同时四张；每个作业12小时运行上限。四组为H0-A/H1-A（非对称前缀注意力）、H0-B/H1-B（独立前缀交叉注意力）。

seed2024，microbatch256，有效batch1024，BF16。沿用已验收两批执行优化；模型/损失/优化器/OCN/Gumbel/dropout/数据顺序不改。完整R35/T400日程用于学习率、DIGER退火，外层在R20/T200完成验证并保存完整状态后停止，不进入T201，不把日程压缩到R20。

A/B分别按新结构从新初始化暖身R5。同方案H0/H1的暖身均不启用Mixer，独立暖身结果在R5逐字段指纹核对；之后H0不启用Mixer，H1启用。旧H1 R5和短测权重均不混用。新结构的训练轨迹不要求与旧结构逐位相同。

复用原数据及PSE资产，静态商品ID缓存先在CPU通过原collator校验生成一次，再供四卡只读使用；缓存不含SID、学习表示或随机样本。数据/资产绑定与缓存SHA一致才载入。

直接复用已验收run.train训练循环，外层publish_checkpoint钩子在R5/R8/R11/R14/R17/R20保存不可覆盖checkpoint。R20之后仅补测按验证选出的best和固定R20测试集（beam32、FP32推理），不按测试集选点。每100次更新保留原断点，异常不另建重复作业。

大型checkpoint/预测留远端；本地只留代码、配置、指标和小收据。作业终态自动释放GPU。启动验收看真实训练日志及checkpoint计数，不能把Job CREATED当作开跑。

目录：/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/decoder_prefix_formal_20261112_v1。每组warmup、train、test_best、test_R20独立，父实验资产保持只读。
