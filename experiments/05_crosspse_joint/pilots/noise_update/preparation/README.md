# 短对照准备与验收

这里保留 2026-10-12 正式启动前的准备证据：输入检查、标准库测试、三组 CPU 小流程及单 H100 GPU smoke 均通过。**不能从 smoke 推断推荐效果。** 准备时服务器主目录尚未完整同步，CPU 测试使用独立 `/tmp` 副本，GPU 测试使用共享盘中 SHA 绑定的独立交付副本。随后提交、同步和启动的版本/状态以新作业启动收据为准；不回写这些原始准备记录。

## 已核对

- 18 份权重、数据、配置、历史锁与 metadata 输入通过核验；最终测试后再次核验，未改变。
- 三组初始完整模型的 tensor-state SHA256 相同，且共同快照的 Tokenizer 更新数为 0。
- 三组恢复训练的数据顺序 SHA256 相同；每组都训练 16 个 microbatch、1 次 Decoder optimizer update。
- B/C 各进行 16 个 Tokenizer microbatch、1 次 optimizer update。Tokenizer 阶段 Decoder 梯度为 0，恢复阶段 Tokenizer 梯度为 0。
- 新的预算、输入绑定、模式保护测试共 40 项，在本地和远端模型 Python 3.10 环境通过；原六套复现回归检查共 78 项通过。

| CPU smoke 组别 | history/target 分配调用 | 分配时 Tokenizer 为 train | 实际加噪调用 |
|---|---:|---:|---:|
| A：不更新 | 0 | 0 | 0 |
| B：无噪声更新 | 32 | 32 | 0 |
| C：原噪声配置更新 | 32 | 32 | 32 |

最后一列依据已锁定的 assignment 实现的加噪条件统计，而不是单看 `noise_scale`。正式 B/C 各 96 批时，调用数应为 192，B 加噪为 0，C 加噪为 192。模式意外改变或调用数不符会直接报错，不会继续产生“完成”收据。

完整 CPU 小流程耗时约 37.8 秒。这里只使用每组 1 轮、16 个 batch、每 batch 2 个样本和 2 个验证 batch；**不展示或分析这些 HR/NDCG，不能据此估算完整 GPU 验证耗时或证明优化有效。**

## 保留的材料

- [input_preflight.json](input_preflight.json)：运行前输入检查。
- [execution.json](execution.json)：版本、测试、交付与证据哈希，以及未完成项。
- [CPU 流程汇总](cpu_smoke_20261012/result/summary.json)：梯度、数据顺序、模式计数、快照哈希与输入后验。
- [CPU 执行来源](cpu_smoke_20261012/result/provenance.json)：实际脚本路径/哈希、引擎版本与环境。
- [运行日志](cpu_smoke_20261012/smoke.log)、[40 项测试日志](cpu_smoke_20261012/tests.log)。
- [GPU 验收与作业记录](gpu_smoke_20261012_1530/README.md)：已结束的单 H100 smoke，以及下载并核验的日志、JSON 和只读验收脚本。上述 `execution.json` 保留 CPU 准备时的历史状态；最新 GPU 状态以该子目录收据为准。

下载的 18 KB `proof.tar.gz` 只包含日志和 JSON。CPU smoke 的临时模型权重没有下载，它们不是需要保留的正式科研结果；正式 GPU 实验则必须把权重、SID 和训练状态保存在项目共享盘。

## 还需要完成

1. 审查并提交本次代码、参数和说明，将相同版本同步到服务器主目录；不能把当前 `/tmp` 测试副本当作正式运行版本。
2. GPU smoke 已通过。正式短对照前仍需检查其完整输出空间和当时的资源/预算条件；此次 1 GiB 分配探针只证明 smoke 有足够空间，不证明不限容量。没有启停其他人的开发机。
3. 用户确认后，在单个后台 Job 中顺序跑正式 A/B/C。使用新的唯一输出目录；不以测试集选超参数。

平台 Job 不依赖本地电脑、浏览器或 VPN 持续在线，但应确认业务日志正在推进且产物写入共享盘后再断开。本地在线不能阻止平台回收或低优先级抢占。
