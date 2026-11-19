# 推荐反馈对照：准备与启动记录

2026-10-12：服务器真实张量检查、绑定输入核验、三组单 H100 GPU smoke 均通过。正式后台 Job 已提交并实际启动，尚无最终科研结果。

## 固定研究问题与运行版本

三组保留相同 Cross Mixer，从同一完整模型权重和正式 SID 开始：`frozen` 不更新 Tokenizer；`cross_recon` 只用重构等目标更新；`cross_rec` 额外加入推荐反馈。正式更新组各 24 次 optimizer update，然后三组都固定 Tokenizer、恢复 3 个完整 Decoder epoch，只评估验证集，不按各组最佳轮数比较。

- 实际运行代码：`299e5fa2a25793fcb7be094fbbb602d41f71f546`；后续归档文档提交不改变该运行版本。
- DiffGRM 引擎：`39e94e6153c97e59877ba02d4ef1d9a202ed5384`；未修改引擎。
- 服务器主目录：`/inspire/hdd/project/project-public/<inspire-account>/nancy/experiment1_reorganized`。
- 同步方式：增量 Git bundle，服务器先核对干净代码树，再快进；不逐文件覆盖源码。

## 已通过的门

- 本地协议 16 项、运行器 14 项、旧噪声协议 40 项、复现安全回归 78 项通过。本地缺少 Torch 的 5 项及 Windows 符号链接 1 项跳过，随后服务器运行器 **20 项全部通过，无跳过**。
- 18 份数据、权重、metadata、锁及配置输入均通过哈希核验；GPU smoke 结束后再次验证，未改变。
- GPU smoke 三组共享完整模型起点、SID、码本和配对数据顺序。各组恢复 1 次 Decoder optimizer update；两个更新组各 1 次 Tokenizer update。
- smoke 中 `cross_recon` 原始推荐梯度 norm 为 1.396850，但实际加权推荐梯度为 **0**；`cross_rec` 实际加权推荐梯度为 **1.396850**。两组都能产生原始梯度不意味着两组都使用推荐反馈。
- Tokenizer 阶段冻结 Decoder；恢复阶段冻结 Tokenizer。probe 不更改随机状态、模型/缓存或正式 mapping，probe 的新 SID 与最终刷新 SID 对齐。
- smoke 前后分别通过 2 GiB 实际分配及 16 MiB 写入/fsync 检查，无配额错误。检查证明当时可写，不代表已知精确剩余配额或不限容量。

smoke 只用每 batch 2 个样本和 2 个验证 batch，不能用其 HR/NDCG 推断优化收益。轻量材料不包含完整模型、优化器、SID/码本数组，不能单独恢复训练；全量快照留在服务器独立输出。

## 作业与产物

| 内容 | 位置 / 标识 |
|---|---|
| 已成功的 smoke Job | `nancy-stage05-feedback-smoke-20261012-2150` |
| 正式 Job | `nancy-stage05-feedback-formal-20261012-2150` |
| Workspace / Project | 分布式训练空间 / 公共科研项目 |
| 资源 / 镜像 | 单 H100 80GB，20 CPU，200 GiB 内存，4 GiB shm；`pytorch:25.06-py3:25.06` |
| 正式输出（相对服务器主目录） | `outputs/pilots/stage05_feedback/feedback_pilot_20261012_2150/` |
| 同步、应用日志和启动收据 | `outputs/pilots_delivery/feedback_20261012_2150/` |
| 本地原始轻量 smoke 产物 | [result/summary.json](result/summary.json)、[result/provenance.json](result/provenance.json)、[result/COMPLETE.json](result/COMPLETE.json) |
| GPU 验收门 / CPU 输入收据 | [smoke_gate.json](smoke_gate.json)、[preflight_passed.json](preflight_passed.json)、[bound_inputs.json](bound_inputs.json) |

正式 Job 设置最长运行 4 小时保护，不是预计必定跑 4 小时；关闭自动失败重启和飞书通知。后台 Job 不依赖本地电脑、浏览器或 VPN 持续在线，但平台调度/回收规则仍适用。没有停止其他人的开发机或作业，也没有覆盖历史实验。

## 后续读取结果

```powershell
inspire job status nancy-stage05-feedback-formal-20261012-2150 --workspace 分布式训练空间 --account sii-shared
inspire job logs nancy-stage05-feedback-formal-20261012-2150 --workspace 分布式训练空间 --account sii-shared --tail 30
```

完成后同时核对平台 `SUCCEEDED`、输出 `COMPLETE.json`、`summary.json` 的三组清单和全量验证覆盖。主要比较相同 Decoder 恢复第 3 轮及完整轨迹，再结合第 6/24 次更新的正式 SID 变化及验证历史/目标覆盖；不能把 smoke、非零梯度或更大的 SID 变化率当作推荐提升。
