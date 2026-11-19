# H1 训练稳定性四组续训，版本 v1（2026-11-06）

本次交付代码和配置，不提交训练。四组共用一份单卡源码，各自输出到独立目录。
启智当前连接失败，两个真实父 checkpoint 的远端 SHA/完整状态尚待核对；
没有 PARENTS.lock.json 时入口明确拒绝训练。不要将本地小模型检查写成真实父文件或 GPU 验收通过。

## 实验定义

R 是推荐器 epoch，T 是 Tokenizer optimizer step。保持原 35R/400T 日程、
学习率、DIGER 噪声时钟、损失、优化器动量、seed2024、样本顺序和有效 batch1024（256×4）。
Tokenizer 包括基础码本与 Mixer；除完全冻结组外，基础码本仍正常训练。

| ID | 完整父状态 | 后续策略 | 新增预算 | 终点 |
|---|---|---|---|---|
| r26_hold | 原 IncrementLinear 的 R26/T280 | 固定使用 T280 最后一次实际使用的系数 0.0642713567839196 | 120T、9R | R35/T400 |
| r26_low | 同一个 R26/T280 | 从下一次 T281 起直接用 0.01 | 120T、9R | R35/T400 |
| release20_freeze | R14→R17 用0.1、R17后恢复1分支的 R20/T200 | 冻结整个 Tokenizer 与 SID，只训练 R | 0T、15R | R35/T200 |
| release20_linear | 同一个 Release1 R20/T200 | 未来新增 Mixer 修正系数 T201=0.1，逐步到 T400=0.01 | 200T、15R | R35/T400 |

R26 系数按旧公式 0.1−0.09×79/199 计算。它不是下一次 T281 原本会使用的
0.1−0.09×80/199≈0.0638190955。两组第一步采用不同系数，但分叉瞬间前向编码完全相同。

R26 两组继续执行三次“40次T更新，再训练3轮R”；Release20_linear执行五次；
Freeze保留同样相位编号、R样本种子与完整R学习率计划，跳过所有T更新，也不刷新SID。

## 门控语义：只改未来，不回撤过去

先从父 checkpoint 精确重建当前有效码本 C*，记基础码本 B* 和原始修正 M*。
新分支的有效码本为：

```text
C_k = C* + (B_k − B*) + accepted_delta_k
accepted_delta_0 = 0
accepted_delta_k = accepted_delta_(k−1) + alpha_k × (M_k − M_(k−1))
```

历史已接受的修正完整保留；修改 alpha 不重新缩放历史偏移、不突然解除旧门控。
门控只在真实 T optimizer.step 之后提交一次，所有 microbatch 使用同一份已提交状态。
梯度保持原 IncrementLinear 的 dC=dB+alpha*dM；不是学习率衰减，也不是 EMA。

冻结组不仅将Mixer系数置零，还完全跳过T优化器；基础码本、Mixer、SID及T优化器状态
必须逐字节保持不变。系数为零而基础码本继续训练，不算本实验的 Freeze。

## 公平比较和评价

- R26 两组与已完成 IncrementLinear 后半段比较：相同父状态、相同120T+9R，仅后续接受系数不同。
- Release20 两组比较相同起点的“完全冻结”和“继续更新并收紧”；T预算不同是干预本身，不能宣称同T预算对照。
- Release20_freeze与历史原H1 Freeze比较，可判断提前干预形成的整体状态是否带来续训收益；两者父模型的R和T都不同，不能把差异单独归因于Tokenizer。
- Release20_linear与历史原H1-R20 IncrementLinear比较，同样只解释整体训练路径差异。
- R26组完整验证节点：R26（复用父结果）、R29、R32、R35。
- Release20组：R20（复用父结果）、R23、R26、R29、R32、R35。
- best仅在各组预定完整验证节点按NDCG@5选择，同分取早；起点也参与。
- 保存best及固定R35的完整测试，若重合只评估一次。测试不参与选模。
- 保留原固定8192诊断子集：每次实际T块后和每轮R诊断；完整验证节点直接提取子集。
- 保存完整验证节点的不可覆盖checkpoint，以及initial与滚动last；不保存额外每轮大副本。
- 记录原始/接受修正、基础码本和有效码本位移、SID变化/碰撞/码字使用以及损失。
- 四组同为单seed，暂不证明跨seed稳定收益。

## 目录与版本

```text
configs/                    四组科学配置，不复制四套训练代码
code/                       共享只读基线快照 + 三个新 recovery_stability*.py 模块
INHERITED_SOURCE.json        旧文件原始哈希，要求全部保持一致
SOURCE_MANIFEST.json         v1全部代码、配置、运行说明和子集的SHA锁
INPUTS.json                 原数据、资产及两个父运行目录
PARENTS.lock.json            真实远端文件核对后生成，目前未生成
lock_parents.py              CPU只读核验父记录、状态、来源、校验和；不推理
job.py                      单组执行入口，默认只打印方案
tests/test_followup.py       小模型CPU回归检查
runs/<arm>/                 四组各自的训练、checkpoint和评估输出（未来远端）
jobs/<arm>/                 独立日志、退出码和时长（未来远端）
```

旧实验代码和输出不修改。新模块复用原训练器、优化器更新、采样、解码、指标和恢复校验。
父状态恢复后只正规化已完成R边界游标，不重复跑父节点。恢复检查绑定arm、父SHA、
源码、数据、门控锚点、累积接受量和系数时钟，防止跨组误续训。
后续若改科学设置，创建v2目录，不在运行中的v1里改文件。

## 将来的运行方式（本次未执行训练）

CPU文件入口先核对一次两个父文件：

```text
python -B lock_parents.py
```

只预览一组（无GPU、无训练、无提交）：

```text
python -B job.py --arm r26_hold
```

未来在一张GPU的作业内明确执行指定一组：

```text
python -B job.py --arm r26_hold --execute
```

其余arm为r26_low、release20_freeze、release20_linear。job.py不创建平台作业。
先跑train，再test，再完整checkpoint恢复；三个阶段的收据不替代独立结果复算和资源收尾。
续跑需先确认旧作业终止，再显式加--resume；代码不自动删除运行锁或重试提交。
大checkpoint和逐样本预测保留远端；本地只保存代码、摘要和校验收据。

## 当前未完成项

1. 网络恢复后执行父文件只读核对，生成PARENTS.lock.json；尤其不能用原H1 R26或其他分支替代。
2. 真实父模型的CUDA编码/SID连续性由训练入口在第一步更新前自动检查。
3. 尚未提交、分配GPU、运行这四组实验。本地检查范围见VERIFICATION.md。
