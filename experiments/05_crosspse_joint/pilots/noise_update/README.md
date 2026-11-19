# DIGER 噪声更新短对照

这次先回答一个问题：Tokenizer 更新时几乎随机的临时 SID，是否让更新偏离了正式无噪声编码？先做一次更新和短恢复，不重跑四组长期实验，也不据此选择最终测试集最好的超参数。

**启动前验收（2026-10-12）：** 18 份输入检查、40 项新测试、78 项复现回归检查，以及 A/B/C CPU 和单 H100 小流程通过。B 无实际加噪，C 的 16 批小测试均已加噪。[验收材料](preparation/README.md) 独立留档；[GPU 验收](preparation/gpu_smoke_20261012_1530/README.md) 已完成。用户随后已授权提交、同步并启动正式短对照；实际执行版本和运行状态以新作业的 `outputs/pilots_delivery/<唯一名称>/launch_receipt.json`、Job 日志和输出收据为准，不能以这些历史准备收据代替。

## 三组分别改变什么

所有组都从 Cross v2 的第 5 轮最佳权重开始；该快照还没有更新过 Tokenizer。Decoder、码本、Mixer 和正式 SID 的起点完全相同。

| 组 | 开始时的干预 | 随后的 Decoder 训练 | 想检验什么 |
|---|---|---|---|
| A | 不更新 Tokenizer | 冻结 Tokenizer，训练 3 轮 | 只训练 Decoder 的参考轨迹 |
| B | 无 Gumbel 噪声，更新一次 Tokenizer phase | 冻结更新后的 Tokenizer，训练相同 3 轮 | 去掉临时随机编码后，更新能否留下有用的正式 SID |
| C | 使用原 v2 第一次 phase 的噪声配置，并确保每批实际加噪，更新相同一次 | 冻结更新后的 Tokenizer，训练相同 3 轮 | 与 B 比较，观察噪声的影响 |

B 和 C 保留 hard-forward/soft-backward 推荐梯度，以及原有重构、均衡和锚定损失。不是把 B 改成只做重构。每个 Tokenizer phase 是 96 个 batch，梯度累积 16 次，对应 6 次优化器更新。

C 延用原来的 60 次 Tokenizer 更新退火跨度，只走第 0 至第 5 步。不能把 1.0 到 0.2 压缩到这 6 步，否则同时改变了噪声强度和退火速度。

2026-10-12 复核发现：旧引擎的 guided 打分会递归切换 `eval/train`，不保留“Decoder 为 eval、Tokenizer 为 train”的混合模式。旧 trainer 每阶段只设置一次模式，后续 batch 可能因此跳过 Gumbel；日志中的 `noise_scale` 只反映调度值，不能证明实际加噪。因此，前面的固定快照一致性统计描述的是**显式加噪时的编码差异**，不是历史每批实际执行情况的证明。新入口每批恢复模式，并检查 history/target 的分配与加噪调用数。C 是原配置被完整施加的干预，不是精确重现旧执行；历史源码与证据不改写。

## 保持一致的设置

- 数据、商品目录、训练样本、验证集、4×256 SID、网络结构和推荐损失沿用现有 Cross v2。
- Tokenizer 学习率沿用码本 `1e-5`、Mixer `1e-4`；推荐反馈权重 1，均衡和锚定权重均为 0.01。
- 分配距离温度 0.1，反向 softmax 温度 1.0。
- 验证使用原来的 SID-level 协议和验证 beam，不扩大到 item-level；这次不跑 test。
- 各臂训练顺序和 Decoder 随机种子按恢复轮数对齐；Gumbel 采样不消耗 Decoder 的随机流。B 的实际加噪调用数应为 0，C 的正式一次 phase 应为 192（96 批 × history/target）。
- 不早停、不以不同最佳轮数作为主要对比。比较更新前、更新后立即、恢复 1/2/3 轮的完整轨迹。

历史权重未保存完整优化器和随机状态，所以这是“共同权重起点、重新建立优化器”的新实验，不是精确续训。三组 Decoder 统一使用固定学习率 `1e-4`，不继承 500 步 warmup：短实验总共约 312 次 Decoder 更新，继承该调度会使整个短实验都处在 warmup。这个学习率是保守的短对照起点，不宣称已经调优，结论只适用于这次统一预算。

## 文件和运行顺序

| 内容 | 位置 |
|---|---|
| 独立训练入口 | `scripts/train/run_diger_noise_pilot.py` |
| 固定参数和共同权重绑定 | `configs/05_crosspse_joint/pilots/diger_noise_pilot_seed2024.json` |
| 输入哈希与预算检查 | `scripts/checks/diger_noise_pilot_protocol.py` |
| 标准库测试 | `tests/test_diger_noise_pilot.py` |
| 新输出 | `outputs/pilots/stage05_noise/<唯一名称>/` |

历史训练入口、`evidence/`、权重和复现锁不改动。先在本地检查，再同步审查后的代码到新服务器主目录；不要把 `/tmp` 中的临时测试副本当成正式代码同步完成。

项目根目录先做 CPU 检查：

```bash
python -B tests/test_diger_noise_pilot.py
python -B scripts/checks/diger_noise_pilot_protocol.py --plan-only
```

具有完整数据和权重的服务器主目录继续检查：

```bash
source runtime_env.sh
python -B scripts/checks/diger_noise_pilot_protocol.py
```

然后先做单 GPU smoke：

```bash
python -u -B scripts/train/run_diger_noise_pilot.py \
  --arm all --smoke \
  --output outputs/pilots/stage05_noise/noise_smoke_20261012
```

Smoke 使用缩小的 batch、每阶段 16 个 batch、1 个 Decoder epoch 和 2 个验证 batch，只检验流程、梯度路由、共享初始化与保存，不分析这些推荐分数。

Smoke 通过、输出空间充足后，使用另一唯一目录正式运行短对照：

```bash
python -u -B scripts/train/run_diger_noise_pilot.py \
  --arm all \
  --output outputs/pilots/stage05_noise/noise_pilot_20261012
```

目录必须不存在。时间以 GPU smoke 的实测速度估计，不沿用前面 45 轮训练的耗时。三个臂可以在同一个单 GPU 作业里顺序运行。

每个臂保存初始、干预后和每轮恢复的权重与 SID；汇总保存验证指标、优化器更新数、梯度、SID 变化、验证历史/目标覆盖和源码/输入哈希。初始快照不能被最后状态覆盖。

## 怎么解读

- B 恢复后持续超过 A，且有正式 SID 变化：支持当前无噪声更新有用，但仍是单 seed 的短实验观察。
- B 比 C 好：支持当前噪声设置不利于这次更新；不能推广成所有 Gumbel 方法都无效。
- B 更新后掉分，恢复后只是回到 A：更像 Decoder 在适应编码扰动，没有证据证明 Tokenizer 带来额外收益。
- B/C 都几乎不改变正式 SID：这次干预太小，不能直接判定交互机制无效。
- 即使 B 有改善，仍需要后续“去掉推荐反馈”和“共同起点去掉 Mixer”的对照，才能分别归因于推荐梯度和码本交互。

## GPU 作业与本地断开

代码检查和小型流程测试可以用 CPU。完整推荐更新与验证适合 GPU；一张卡即可。正式运行推荐提交为启智平台的 GPU Job，不在网页终端前台长期训练。

Job 在服务器运行。确认任务真正开始训练、日志和权重写入项目共享盘后，可以关网页、断 VPN、关闭本地电脑。重新连接只用于查看和管理；不要点停止或删除远端任务。

2026 年 10 月 12 日重新读取到的 `分布式训练空间` 策略：GPU Job 利用率低于 40% 持续 3 小时会被回收；GPU Notebook 利用率低于 15% 持续 3 小时，或运行超过 18 小时会被回收。策略可能变化，每次提交前重新查询。保持本地在线不能阻止平台回收。

CPU 准备阶段没有申请 GPU。2026-10-12 在用户明确请求后提交了一个单 H100、最多 15 分钟的 smoke Job，现已成功结束；没有启停他人的开发机、修改共享环境或自动 commit。正式短对照需另行确认代码提交/同步、输出空间及当前资源条件。
