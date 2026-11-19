# 阶段08设置与执行链路复审

日期：2026-10-24。请求：再次确认当前设置是否合理、有无错误。本次仅本地阅读、代码修正及合成CPU检查，未申请GPU、运行真实数据训练或访问远程平台。

## 结论

当前设置适合作为既定暖启动上的首轮探索，不需要因此更换checkpoint、重新量化、关闭三臂共有Full Mixer或立刻扩大超参搜索。此前55项通过只证明当时测试覆盖的路径，并不排除未测边界；本次发现并修正以下问题。

| 发现 | 修正及验证 |
|---|---|
| 含0号PAD的history_item_ids在缺history_mask时，原历史Encoder仍会读取PAD；独立模型fixture复现有效位置最大差0.5253553 | prepare_joint_batch自动由ids推导mask；显式mask校验形状并与非PAD相交。修后仅IDs与显式mask的隐藏状态、实际生成逐位相同 |
| 恢复身份未绑定根仓库实际读取的三层YAML；norm_eps等非Tensor设置可能改变而权重相同 | SESSION_IDENTITY增加三个配置文件SHA和稳定解析后模型设置。真实context构造探针验证YAML指纹或norm_eps漂移均拒绝继续 |
| CLI deadline未接入trainer验证batch的检查，整库beam256验证可能跨过进程截止 | 适配原trainer的每batch截止hook，启动/每batch/完成均检查，异常恢复原hook。只是协作式进程边界，不保证平台累计分配上限 |
| set_recommender_cache若被外部BF16上下文包裹，非零Mixer可能产生不同缓存精度；当前runner正常调用不受影响 | 缓存构建也关闭外层autocast；激活非零Mixer后缓存、实时读取与BF16环境重建逐位一致 |
| 旧候选计划按NDCG封存best，新实现按weighted且包含R0 | 旧DESIGN/EXPERIMENT_PLAN添加明确过期提示；当前README统一为固定R35 NDCG主比较，含R0的weighted-best仅作次要结果 |

## 保留设置的理由

- **共同起点：** pre-05 Aug28 DiffGRM与05共享E0，三臂共有tensor相同；完整新增F也共同保存，baseline仅禁用并从R优化器排除。现有04/05资产元数据及锁显示同一SID语义，旧入口还要求离线权重全部相等、重基前后SID一致；本轮真实资产尚未重新加载验证。
- **完整配方：** 35R/3640次R更新、60次T更新、相同数据与Gumbel流、全局Full Mixer统一开。它回答在该配方上新增session分支的增量，不证明Full本身有效。
- **结构：** 64维物理码字、4头、独立双轴、FFN128、一轮、零初始化64→256读出。scale1/dropout0/原R学习率是待验证初值，初态相等不保证后续残差幅度安全；先看现有诊断，不先扩大搜索。
- **梯度：** R更新推荐器/W/F；T冻结它们的参数，但保留新分支输入回C的导数。读出初始为0、no_context时间轴只有自身key时，部分核心梯度为0可能是正常情况。
- **指标：** 三臂统一验证beam256、同样候选处理；旧beam32数字不能直接充当baseline。主要看axial对no_context并结合baseline、碰撞和额外时延；有信号后再多seed。

同一快照内canonical SID不随session变化，但T后SID会刷新，三臂可逐渐不同。因此比较识别的是上下文分支及其经T反馈的总效应，不是固定Tokenizer上的纯表示差异。数据继续采用训练早期前缀、val/test目标前最近50件，不能称长序列全滑窗。

## 实际验收与剩余范围

[CPU_ACCEPTANCE.json](CPU_ACCEPTANCE.json)：8套检查共62项，61通过、1项Linux文件锁检查在Windows跳过；收据绑定代码/YAML/测试SHA、逐套日志、Python与根/引擎HEAD，测试期间源字节未变。修改尚在工作树，不能仅凭HEAD复现。

新启动探针执行真实legacy/Stage08 context_for、真实F和CrossPSE重基；替换的是Pipeline、目录与大依赖。它证明配置进入构造、共同state保存和身份拒绝逻辑，不等同于完整真实Beauty Pipeline。

仍须在独立执行副本验证真实资产、完整Beauty初始化和初态指标、H100 BF16所有应更新参数、实际吞吐/显存、Linux锁与恢复。GPU作业还需平台硬截止、预算及分配时长核对；本次没有确认这些，不把CPU通过写成“可以保证无错”或推荐有效。
