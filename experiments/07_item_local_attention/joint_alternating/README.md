# 商品内注意力的交替联合训练

最终交付（2026-10-20）：两组均完成45个逻辑轮、验证选模及唯一正式测试，最佳轮次均为25，全部GPU已释放。B相对A的测试NDCG@10提高0.425%、HR@10提高0.279%，未达预设3%初筛；累计8个作业使用7.428056 GPU小时。真实成对审计、335件备份文件校验、14份本地权重/状态回读及曲线检查均已通过，见[结果报告](REPORT.md)和[总验收索引](FINAL_ACCEPTANCE_20261020_r3.json)。本地备份在`outputs/stage07_item_local_joint/delivery_20261020_final_r3/`；13件原始输入的身份及远端逐件校验已保存，原始数据/cache不包含在包内。

冻结训练根版本`842ff2a`、引擎`1c87925`。成对审计使用[r3后处理工具](postprocess_tools_r3/README.md)：同一种TF32点积输入舍入方式在CPU精确匹配四份checkpoint全部193616码位，原FP32/FP64差异和失败收据保留；不声称追溯了历史CUDA标志。训练代码、权重、SID和测试结果均未为通过检查而修改。原[r2交付步骤](postprocess_tools_r2/POSTPROCESS_RUNBOOK.md)仅作历史操作参考，第3步须使用r3审计入口。

本轮要回答：**在商品码本已经接受推荐反馈的情况下，历史商品的4个码位在压缩前增加局部注意力，能否进一步改善推荐？**这检验的是局部交互的增益，不是连续向量生成，也不是再跑一轮全局码本Mixer。

| 组别 | 交替联合训练 | 全局Mixer | 商品内局部注意力 |
| --- | --- | --- | --- |
| control | 开启 | 关闭 | 关闭 |
| item_local | 开启 | 关闭 | 开启 |

两组共用05 `refined_joint`锁定的离线Refined-PSE v2和配套离线DiffGRM暖启动，seed2024相同；共同基础参数、随机状态、初始SID已在短测及两组实际正式初始化中分别核对一致，见[正式起点比较](FORMAL_PAIR_INITIALIZATION_MATCH_R2.json)。03固定PSE结果只作历史背景，主要比较是本轮修复后、同协议的新两组。

代码差异追溯见[根仓库差异索引](ROOT_CODE_DIFF.json)及[引擎绑定](ENGINE_BINDING.json)。根补丁保存2fc97b6至842ff2a的运行/检查代码变化；三段引擎补丁覆盖局部模块、OCN修复及交替训练支持。两组正式运行使用同一套冻结代码，实验变量仍是局部注意力开关。

局部注意力位于历史SID查表后、`item_mlp`压缩前。输入`[B,L,4,256]`，把`B×L`当批次，只在同一商品的4个码位间做1层、4头残差注意力，再恢复原形状。新增263,680个参数，dropout为0，输出投影零初始化。训练、验证、生成以及Tokenizer的可微历史路径都经过同一模块。

推荐器阶段冻结Tokenizer，训练推荐器和局部注意力；Tokenizer阶段冻结推荐器及局部注意力的参数，但保留计算图，让推荐损失通过DIGER-ST更新基础码本，再刷新SID。文本特征、OPQ和4×256码本容量固定。采用05 `refined_joint`完整交替配方：先5轮推荐器，再循环1个Tokenizer阶段＋3轮推荐器，最多45个逻辑轮；每个Tokenizer阶段96个batch、6次更新，推荐器每轮完整1652个batch、104次更新。batch64、累积16、BF16，val batch8。

这里没有使用**05最新共享起点四组短对照**的24次Tokenizer更新＋3轮恢复协议，也不将早期05长期四组结果混作本轮两组结果。

选模用验证集`0.8×NDCG@10＋0.2×HR@10`，包括合法的epoch0候选；每完成5轮推荐器验证一次，patience5。验证beam32，最终测试beam256。测试前重载最佳权重和对应SID，每组测试只执行一次。主指标是SID-level，不能直接与06商品级指标排榜。预先约定的探索信号是item_local的NDCG@10相对control至少提高3%，且HR@10不下降；单seed结果不等同于统计显著性。

已完成的检查及证据：

- [CPU核心检查](cpu_acceptance_20261020/ACCEPTANCE.json)：20项，覆盖更新/冻结、OCN/BF16梯度及阶段恢复。
- [r2真实H100短测](SMOKE_R2_PROCESS_RECEIPT.json)：两组均完成rec→tok→rec，核对推荐梯度到码本、局部模块更新、SID映射、保存及恢复后的实际更新；不评估测试集。
- [正式control首次交替检查](first_tokenizer_review_r2_20261020/TIMING_AND_ALTERNATION_REVIEW.json)：第6轮Tokenizer完成6次更新，推荐CE到码本梯度非零，冻结参数未变；第7轮恢复推荐器更新。它证明链路生效，不证明推荐质量改善。
- [正式item_local首次交替检查](progress_snapshots/item_local_after_first_alternation_20261020.json)：第6轮完成6次码本更新，首batch的独立推荐损失梯度探针为0.3262925，1个码本参数张量改变、102个冻结参数张量未变，正式SID刷新及最近邻/映射检查通过。实际码本更新使用复合损失，不能把全部变化归因于推荐损失。第7轮随后完成104次推荐器更新，90个活跃参数张量改变、13个冻结参数张量未变，并使用第6轮刷新后的SID；epoch5验证发生在首次码本更新前，不归因于联合码本更新。
- 两组均已完整结束：[control完成证据](control_final_r2_20261020/LIVE_COMPLETION_OBSERVATION.json)、[item_local完成证据](progress_snapshots/item_local_formal_completed_20261020.json)、[真实成对产物审计](JOINT_PAIR_AUDIT_20261020_r3.json)。两组各35轮推荐器、10个Tokenizer阶段，共3640次推荐器更新和60次码本更新，最佳均为epoch25，最终各测试一次。

资源上限沿用用户后续批准的累计28 GPU小时，单次1张H100。全部8个作业从Scheduled到Pod删除累计26741秒，即7.428056 GPU小时，旧失败/中止/短测均已计入，见[最终分配账本](ALLOCATION_FINAL_R2.json)。item_local在20:23:56释放，全部本Goal GPU已释放。每个正式Job运行一个完整模型，没有按轮次拆分规避回收，没有向冻结远端副本同步另一任务的本地改动；GPU时长不等于已核实结算金额。

历史失败单独保留：旧固定SID训练受OCN/BF16梯度问题影响，没有接续其84轮；用户随后要求改为交替联合训练。joint r1短测在模型构建前因导入冲突失败，102秒分配已计入账本；修复后使用新r2副本并重新通过短测。原始日志、旧副本和收据保留，不作为正式结果。

交付文件：[结果与限制](REPORT.md)、[下载收据](DOWNLOAD_COMPLETION_20261020_final_r3.json)、[实际checkpoint回读](LOCAL_CHECKPOINT_READBACK_20261020_final_r3.json)、[训练/验证曲线](curves_20261020_final_r3/joint_training_curves.png)、[总验收索引](FINAL_ACCEPTANCE_20261020_r3.json)。本地CPU回读不等于完整异地训练恢复；本轮没有追加seed、调参或重复测试。

入口为`scripts/train/run_item_local_joint.py`，配置为`configs/07_item_local_attention/joint_alternating_seed2024.json`。`prepare`在CPU核验和复制13件锁定输入；`smoke`使用独立短测目录；`formal`执行完整协议。本轮已经结束，不重复提交。以下CPU准备命令仅作恢复参考：


```text
python -B scripts/train/run_item_local_joint.py --mode prepare --variant control --config configs/07_item_local_attention/joint_alternating_seed2024.json --project-root <历史项目> --engine-root <新引擎> --output <本轮根>/runs/stage07_item_local_joint_control_seed2024_20261020_r1
```

另一组将`--variant`改成`item_local`，输出名改成`stage07_item_local_joint_item_local_seed2024_20261020_r1`。GPU入口还要求完整代码提交、引擎提交、实际剩余预算和两组短验收；正式训练必须使用完整模型作业的实测预算/平台规则检查证据。源码中的断点恢复只接受本轮自己的`last.pt`，不会接续旧07的84轮。
