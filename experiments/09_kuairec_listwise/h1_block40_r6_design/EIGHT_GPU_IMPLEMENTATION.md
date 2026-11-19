# E 八卡同步数据并行实现

用户已选择八卡共同训练同一个E；这里只实现和测试代码，未申请资源或运行正式数据。旧A/D源码及验收产物未修改。

## 用法与范围

入口：scripts/listwise_diffgrm/block40_distributed_run.py。Linux/NCCL用torchrun启动8个进程，支持单机8卡，也可由torchrun提供跨节点rank环境；全局world_size必须8。

示例（路径需使用实际已验收的远端路径）：

    torchrun --standalone --nproc_per_node=8 -m scripts.listwise_diffgrm.block40_distributed_run --parent <原H1_initial.pt> --dataset <数据目录> --assets <PSE目录> --output <新的E输出目录> --preflight --max-updates 2

正式执行必须提供--acceptance，不能使用--preflight或--max-updates。恢复用--resume <last.pt>，输出目录保持相同；只支持同world_size精确恢复。运行锁需先核验旧进程/作业结束后人工处理，不能盲删。

## 实现

- 8个副本各处理128条，全局batch1024；使用显式梯度SUM同步，所有rank同步后裁剪、执行相同AdamW更新。这是同步数据并行，未使用DDP包装器，避免历史forward_loss与T阶段输入梯度路径绕过DDP钩子。
- 原单卡256×4变为8×128×1；每rank局部推荐损失按实际条数/全局条数缩放，短尾batch不填充不丢样本。
- T辅助项按全局batch目标去重，每rank计算相同全局项并除以8，再SUM。因此不改变唯一目标集合或平衡项定义；该部分重复计算不能期待八倍加速。
- 未使用的参数保持grad=None；R/T仍分别更新原两套优化器。SID只在40T末由rank0刷新并广播。
- R5/T0父状态、总35R学习率、400T噪声时钟沿用原绑定。新日程5×(40T+6R)，R样本和seed按原epoch映射，T复用原前五个40T块样本。
- 每rank保存Python/NumPy/CPU torch及本卡CUDA随机状态；checkpoint同时绑定world_size、独立随机策略和源文件。只有rank0写文件。周期边界36状态，滚动last和原子保存。
- 训练生产不可变checkpoint及evaluation_requests，完整验证10节点、子集诊断10节点；不在训练进程中运行验证。TRAINING_COMPLETE明确为evaluation_pending，不冒充科学实验验收完成。

## 与单卡A/D比较的新增边界

样本序列、全局batch、损失定义及时钟可保持；随机张量不与历史单卡逐位一致。新策略为原phase seed加rank偏移，Gumbel/dropout/扩散随机采样及并行浮点归约都会导致数值路径差异。因此原方案“严格配对T随机流”对八卡版不再成立。结果需标注E同时改变更新安排和执行后端；CPU测试不能证明真实BF16八卡与单卡科学等价。

## 本地验证与GPU门禁

tests/test_block40_distributed.py使用8个真实CPU/Gloo进程：检验全局1024、835短尾和3条含空rank批次；非线性全局唯一目标辅助项；双优化器权重和Adam状态对单进程参考；checkpoint收集/广播、各rank RNG恢复及后续更新；日程和无重复无遗漏分片。

这是小型可解析损失fixture，并非真实DiffGRM/H100测试。正式GPU验收还必须核验真实模型FP32控制随机输入的梯度参考、BF16差异、R/T隔离、rank状态一致、40T刷新、完整断点续跑及短尾。不得为通过验收临时放宽容差。评估消费器、平台作业预算封装和最终实验验收仍需在启动实施阶段接入新checkpoint格式。

正式入口拒绝缺少GPU验收的启动。receipt需status=passed_e_8gpu_preflight、当前execution_binding以及global_aux_verified/gradient_reference_verified/rank_state_equal/rng_resume_verified/short_batch_verified五项真实通过证据；不要手填通过来绕过检查。

预算需另行核定：8卡1小时=8 GPU小时；原24 GPU小时若不扩展，只相当于最多3小时八卡占用（还须扣短测/评估/释放开销）。本次未更改旧Goal或申请额度。
