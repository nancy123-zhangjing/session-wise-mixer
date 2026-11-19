# 原 H1 历史验证节点 checkpoint 核对

2026-11-05。只读检查；没有重跑训练、下载 checkpoint 或解包大型归档。

## 结论

在下列已检查范围内，没有找到原 H1 的 R17/T160 完整 checkpoint。
原正式运行保留下来的三个 checkpoint 是 R5/T0、R20/T200、R35/T400。
R8/R11/R14/R17 等验证目录保存的是指标和逐样本预测，不是模型状态。

远端正式运行根：
`/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/goal_20261027_r1/`

| 路径（相对正式运行根） | R updates | T updates | R epoch | progress.phase |
|---|---:|---:|---:|---:|
| `runs/H1/initial.pt` | 3465 | 0 | 5 | 5 |
| `runs/H1/best.pt` | 13860 | 200 | 20 | 25 |
| `runs/H1/last.pt` | 24255 | 400 | 35 | 45 |

以上三个远端文件均通过共享 CPU Python 实际 `torch.load(map_location="cpu")` 读取。
均包含 model、alignment、r_optimizer、t_optimizer、scheduler、progress、rng。
这是反序列化、字段及计数核对，不是本次进行了 GPU 下一步精确恢复测试。
R20 还具有此前的本地 CPU 模型/优化器加载验收收据：
`outputs/stage09_restore/goal_20261027_r1/acceptance/H1_CPU_RESTORE_20261028.json`。

## 保存逻辑和归档证据

冻结训练源码
`outputs/stage09_restore/goal_20261027_r1/code/scripts/listwise_diffgrm/run.py`
中的 `publish_checkpoint`（80–89 行）在成绩改善时覆盖 `best.pt`，
每个阶段结束覆盖 `last.pt`。训练循环也会在中途及验证前更新 `last.pt`。
没有按验证轮次永久保存 `R17.pt` 等文件的策略。

实际枚举本地、远端
`delivery/stage09_run_H1_r1.tar` 的成员，模型文件均只有：

- `runs/H1/best.pt`
- `runs/H1/initial.pt`
- `runs/H1/last.pt`

外置清单 `STAGE09_RUN_H1_R1.json` 同样只有这三个模型文件。
R17 的两个归档成员是
`runs/H1/validation_R17/summary.json` 和
`runs/H1/validation_R17/per_sample.jsonl.gz`。
另外核对本地 warmup_H、operations、evaluations 归档；后两者无 checkpoint。

## 搜索范围及容易混淆的文件

- 本地 outputs、experiments 中的 Stage09 checkpoint 文件名及原正式运行恢复目录。
- 远端个人共享盘 experiment1_stage09 全部运行目录的模型/归档文件清单
  （跳过代码、输入数据、资产目录）。
- 原正式运行的 preflight H1 文件：实读计数仅为 R updates=1或2、
  T updates=2，且 `preflight_only=true`，不是历史正式验证节点。
- 本地 Freeze A/B 导出 tar 和最终 Freeze tar 的 checkpoint 成员。
- 项目盘 `/inspire/hdd/project/project-public/<inspire-account>/nancy/`
  下已有文件中的 checkpoint 路径；排除环境、缓存和 Git 内部目录。
  没有发现 Stage09/原 H1/R17 的候选模型文件。

找到了其他实验的 R17 文件，但它们不能替代原 H1 R17：

| 分支 | 文件 | 原因 |
|---|---|---|
| H1-Spread200-R35 | `phase20_R_R17_T080.pt` | T 预算及此前交替日程不同 |
| H1-Block40-R6-R35 | `phase18_R_R17_T080.pt` | T 预算及此前交替日程不同 |
| H0-Freeze20-R35 | `phase20_R_R17_T160.pt` | H0 重放分支，无 H1 的 Mixer |

Spread200/Block40 也有名称中 R17 的后续 T 阶段文件，不能误认为 R17/T160。
其他后期干预的 R23 等节点也不属于原 H1 训练轨迹。

## 对 R17→R20 新实验的影响

按现存原 H1 文件，已确认的较早起点是 R5/T0。
要建立原协议下的 R17 分叉点，需要从该起点重放至 R17（12R、160T），
核对历史节点结果/协议后，另存完整 R17 状态。重放是否能够逐字节重现
原轨迹需要验证，不能提前保证；模型也不能由验证预测或从 R20 逆推恢复。
本次没有执行重放，也没有提交任何新 GPU 任务。

这个结论限于列出的已检查位置与归档，不声明排除了用户另行保存的、
未登记或不可访问的外部副本。
