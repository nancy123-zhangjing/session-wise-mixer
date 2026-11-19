# 单 H100 GPU smoke：已通过

2026-10-12 在用户请求下提交后台 Job `nancy-stage05-noise-smoke-20261012-1530`。平台终态为 `SUCCEEDED`，三组保存了 `COMPLETE.json`，并独立重读输入和产物哈希通过验收。**这只证明 GPU 流程跑通，不证明 Tokenizer 改善了推荐。正式短对照没有启动。**

## 实际运行和验收

- GPU：一张 NVIDIA H100 80GB；Python 3.10.20、PyTorch 2.6.0+cu118。Decoder 用 BF16，Tokenizer 用 FP32 并关闭 TF32。
- 三组从同一完整模型状态开始，初始正式 SID 未改变，恢复阶段数据顺序、随机种子和优化器更新预算一致。
- A 不更新 Tokenizer；B/C 各做 16 个 microbatch、1 次 Tokenizer optimizer update。三组随后各做 16 个 Decoder microbatch、1 次 update；只验证 2 个 batch。
- B 的 32 次 history/target 分配没有加噪，C 的 32 次分配均进入实际加噪分支。调度跨度仍为 60，不压缩到小测试预算。
- 更新侧梯度有限且大于 0，冻结侧梯度为 0。恢复期间 SID 和交互后码本文件未变；完整 Tokenizer state 冻结由锁定 runner 的硬断言保证，独立标准库验收不反序列化 `.pt`。
- 18 份绑定输入再次重读 SHA256、大小通过；9 份快照中的模型、训练状态、SID、码本文件哈希全部通过。
- 输出约 368 MiB。runner 记录的流程耗时约 38.6 秒，不含此前部分模块导入；平台提交到结束约 3 分 22 秒，不能用于外推完整验证耗时。

## 结果在哪里

服务器完整结果（权重、SID 和优化器/RNG 状态均保留）：

```text
/inspire/hdd/project/project-public/<inspire-account>/nancy/experiment1_reorganized/outputs/pilots/stage05_noise/gpu_smoke_20261012_1530
```

独立执行代码与共享日志：

```text
/inspire/hdd/project/project-public/<inspire-account>/nancy/experiment1_reorganized/outputs/pilots_delivery/gpu_smoke_source_20261012_1530
```

本地下载的轻量证明材料不含 `.pt` 和 `.npy`，不能单独恢复训练：

- [验收 JSON](proof/acceptance.json)、[完整汇总](proof/summary.json)、[完成收据](proof/COMPLETE.json)。
- [执行版本与环境](proof/provenance.json)、[18 份输入后验](proof/post_input_verification.json)。
- [原始共享日志](proof/gpu_smoke.log)、[资源与交付记录](execution.json)。
- [只读验收脚本](verify_artifacts.py)：在有完整结果的服务器上重新核验。

`proof.tar.gz` 的 SHA256 为 `7f7b510eeb86dbdff83be0100969530d3cf2b47791ba0487070cccec0e2b23f6`；本地下载后再次核验 archive 和 summary 均通过。

本次执行的是未提交但逐文件锁定哈希的独立副本；远端根仓库仍为 `594b6add85c0a72731890fa45fa67d414fa358a2`，DiffGRM 引擎为 `39e94e6153c97e59877ba02d4ef1d9a202ed5384`。**不表示主目录已同步到本地最新源码，也不是历史精确续训。**

## 下一步

先审查、提交新训练入口/参数/说明并同步同一版本，再确认正式输出空间，使用另一个唯一目录跑正式验证集短对照。不能删除 `--smoke` 后复用本次输出目录。没有自动提交、推送或启动正式训练。

已结束的 Job 不再需要本地电脑、浏览器或 VPN 保持连接。未来 Job 也应先确认业务日志和共享产物正在推进再断开；本地在线不能阻止平台回收。
