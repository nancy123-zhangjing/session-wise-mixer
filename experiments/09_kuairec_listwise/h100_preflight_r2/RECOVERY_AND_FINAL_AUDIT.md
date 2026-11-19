# 正式执行的恢复与最终验收

2026-10-28更新。用户已批准累计75 GPU小时，见`../preparation_20261027/BUDGET_APPROVAL.json`。J/H共同暖身均已完成、释放、CPU验收并在本地独立恢复；四个正式分支已于10月27日23:46:38启动于同一个两卡Job `nancy-stage09-four-arm-20261027-r1`。GPU0运行J0/H0，GPU1运行J1/H1，四进程独立，无DDP。当前进度以[执行记录](../EXECUTION.md)和实际平台状态为准，不重提已运行任务。

启动前7个已结束Job累计13535 GPU秒，当前正式两卡占用另计。平台上限24小时，最终评估另预留8 GPU小时，包含已用共预留59.759722小时。实际费用按最终分配至释放计时，未核实结算。平台仍适用GPU利用率低于40%持续3小时回收规则；短测和启动窗口不能证明整个长任务合规。

## 启动和恢复

命令模板见 `FORMAL_COMMANDS.md`，首次暖身和正式分支均已实际执行，不重新运行首次启动命令。四次ready和两对initial完整状态验收见`../formal_execution_20261027/FORMAL_READY_*.json`及`FORMAL_INITIAL_PAIR_*.json`。J0/J1继承J完整暖身，H0/H1继承H完整暖身；不是四组共用同一个暖身。正式调度脚本拒绝已有run目录且不自动resume，以下恢复命令只供确认旧作业终态、释放及剩余预算后使用。

发生抢占、超时或观察连接中断时：

1. 先查原Job的权威状态和资源释放事件。观察超时不是训练停止，不直接新建同名实验。
2. 旧Job确认终止后，检查其输出和`.running`所有者；归档遗留锁，再恢复。禁止只凭锁文件判断活跃状态。
3. 优先同目录完整`last.pt`。若尚未发布last，可显式恢复该目录完整`initial.pt`；不要使用`.tmp`或重新从暖身分叉覆盖已有实验。若没有有效checkpoint，保留失败现场后另行处理。
4. 恢复前按实际分配时间扣减累计预算，并预留所有仍在运行Job的资源上限。平台Job超时只是后备保护，不能代替全Goal累计计时。

沿用 `FORMAL_COMMANDS.md` 的R、PY、CFG、ARCH、ARM变量：

```bash
# 暖身恢复，输出目录保持原样
"$PY" -u -B -m scripts.listwise_diffgrm.run \
  --config "$CFG" --architecture "$ARCH" --seed 2024 \
  warmup --device cuda:0 --precision bf16 \
  --resume "$R/runs/warmup_$ARCH/last.pt" \
  --output "$R/runs/warmup_$ARCH"

# 分支恢复，不再传 --shared-warmup
"$PY" -u -B -m scripts.listwise_diffgrm.run \
  --config "$CFG" --architecture "$ARCH" --seed 2024 \
  train --device cuda:0 --precision bf16 --arm "$ARM" \
  --resume "$R/runs/$ARM/last.pt" --output "$R/runs/$ARM"
```

程序没有SIGTERM临时保存处理；只承诺已完成原子替换的checkpoint有效。R每100次更新和阶段边界保存，硬停最多需要重放100次R；每段T只有40次，最坏重放整段40T。完整验证前已保存R尾，因此验证中断时重跑完整验证，不重复该R尾。重试目录为`validation_Rxx_retryN`，不完整验证不能参与best选择。日志可能含恢复重放，不能按日志行数直接当作实际有效更新数。

## 四组完成后，先冻结best再测试

只读CPU工具`delivery/formal_final_acceptance.py`已部署并回读SHA `d5b981a6077dead491289965d559a235082d63e6bf73fbca4cd4a42896f09935`，尚未对真实最终模型执行。它复用锁定的分支验收工具，检查四组完整状态和所有验证记录，一次输出四份best身份快照；调用方在确认原Job结束后保存该输出。身份快照不提供文件系统锁，测试前后仍核对checkpoint SHA。最终测试模板为`../formal_execution_20261027/FINAL_EVAL_JOB_TEMPLATES.json`；其中best/冻结清单/最终时限占位必须由真实完成证据填充，不能现在提交。

- 每份暖身3465次R、T=0；每个分支最终24255次R、400次T，等于35R/400T。实际总计算的90090次R更新已经去掉重复计算暖身。
- 检查`complete.json`、完整`last.pt`中的游标、优化器时钟、scheduler、T退火时钟和SID；单独的complete文件不够。
- 从last的`progress.validations`读取真正接受的验证记录，核对完整71657样本、beam32、item级`list/ndcg@5`、平局取较早。共享R5是两分支共同候选；另有R8、11、14、17、20、23、26、29、32、35。
- 固定R35比较读取已接受的R35验证，不把best当作R35。曲线应按phase/update与恢复attempt辨别重放，不机械累加日志。
- 在首次测试之前保存一份四组best冻结清单：四臂文件SHA/bytes、训练完成证据SHA、best epoch和R/T计数、选模来源、执行代码/配置/数据/共同资产绑定，以及固定1024验证子集SHA。
- 所有四组best都完成冻结后，才执行4次全测试；每次评估前后核对相同checkpoint SHA。训练完成状态、命令、输出路径及文件SHA必须一并绑定，不能仅凭评估summary证明用了同一个best。
- 评估仅接受R阶段完成且update=0的checkpoint；不使用中间T状态、T结束边界、R尾尚未完成验证的游标或preflight checkpoint。

## 独立指标复算与配对分析

每臂全测试须有53194条样本，每个敏感性评估须有固定1024条验证样本。对12份最终评估（4个test、8个subset）逐一运行独立复算，绑定原split并核对真实item、用户、标签、冷目标分组及保存指标：

```bash
python -B -m scripts.listwise_diffgrm.analysis recompute \
  --predictions "$EVAL/per_sample.jsonl.gz" --dataset "$DATA" \
  --output "$AUDIT/metric_recompute.json"

python -B -m scripts.listwise_diffgrm.analysis compare \
  --baseline "$J0_EVAL/per_sample.jsonl.gz" \
  --treatment "$J1_EVAL/per_sample.jsonl.gz" \
  --repeats 10000 --seed 2024 --output "$AUDIT/J1_minus_J0.json"
```

H配对同理。`EVAL/DATA/AUDIT`等需指向实际回收或独立分析副本。分析工具采用经过验收的新提交并记录版本，不热更正在运行的冻结训练副本。

- 单列表只报@5；聚合另报HR/Recall/NDCG@10/20/50及覆盖率，不能混称单列表质量。
- 报告请求平均、用户平均、含任意冷目标与全目标已见分组。用户bootstrap共10000次，重采样用户并带上其全部请求；不能当成多个训练seed。
- `compare`要求同split、同beam、同架构、同seed、同样本和评价协议。分别在beam32和256内求J1−J0/H1−H0，再比较增益是否依赖beam；不能把beam32和256直接传给配对compare。
- best/R35、训练曲线、时间和资源汇总由上述训练证据额外整理，现有指标compare不自动证明这些项目。

## 最终交付的证明范围

源码版本边界：训练binding会哈希`scripts/listwise_diffgrm/*.py`，也包含`analysis.py`。本次新增的冷分组独立验收只在下载结果后的离线分析副本使用，不能覆盖远端已通过H100验收的`1c423662d9a25a926328ee6466c30c0f2857b4f6`源码。正式checkpoint的代表恢复也应使用该冻结bundle恢复出的源码；分析版本另行记录，避免仅分析脚本变化导致checkpoint绑定不匹配。

| 要求 | 必需证据 |
|---|---|
| 起点及控制变量 | 两份暖身、四份initial完整状态；对应配对模型/PSE/优化器/scheduler/RNG/SID一致，Mixer开关差异单列 |
| 实际训练量 | 四份last、complete、阶段与时钟审计、恢复attempt记录 |
| 选模和最终结果 | 全验证记录、四best冻结清单、4全测试和8验证子集评估、逐样本复算及用户bootstrap |
| 可恢复 | 2暖身及4组initial/best/last、对应SID、共享资产和代码配置；远端来源/本地目的地、逐文件及整包SHA、独立目录恢复与代表checkpoint CPU前向 |
| 成本与资源 | 所有任务的Scheduled至Pod删除计时、失败/闲置计入、最终无活跃GPU；无结算明细则只报报价估算 |
| 文字交付 | REPORT.md、SUMMARY.json、HANDOFF.md、PROJECT_HANDOFF更新及最终commit；明确单seed和碰撞等证据限制 |

当前准备资产、短验收、两份5R共同暖身及其本地独立恢复已完成；四分支正在训练，正式结果和最终模型交付尚未完成。不得用短验收、暖身或本清单替代最终完成证据；训练/代表checkpoint恢复继续锁定`1c423662d9a25a926328ee6466c30c0f2857b4f6`，本地`83a1b9c`新增analysis仅供离线结果复算。
