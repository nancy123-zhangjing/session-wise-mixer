# 四个 Decoder 前缀版本：第一批执行优化（2026-11-12）

本次仅优化执行方式，未申请 GPU、未启动正式训练。适用于 H0-A、H1-A、H0-B、H1-B。
四份 h[01]_prefix_{asymmetric,cross_attention}.json 配置统一开启 `execution.optimized_io=true`；
其他配置默认关闭。关闭同一开关即可使用参考执行路径做对照，不能跨配置直接绕过 checkpoint binding 恢复检查。

## 已实现

1. 启动时分块运行原始数据 collator 的全部校验，将真实历史商品 ID、历史 mask、目标商品 ID 预制为连续 CPU Tensor。时间戳和冷启动等分析字段不再每次训练反复构造，验证仍保留全部字段。
2. 按原 phase_indices 批量取行，保持顺序、重复样本、尾 batch、microbatch 划分和权重不变。不缓存 SID、Gumbel assignment、embedding 或其他可训练状态。
3. 单生产线程提前准备下一个有效 batch；CUDA 上使用锁页 staging buffer、独立 H2D stream 和 event 依赖，consumer record_stream 管理显存生命周期。线程不调用模型、Tokenizer 或任何随机采样。预取队列只提前一个有效 batch；结束/异常时收尾线程和传输流。CPU 上同样预取，不使用 CUDA。
4. microbatch loss/辅助指标先 detach 暂存，在更新结束集中取回。将原标量无损提升至 float64 后传输，仍按原顺序在 Python 中做相同加权累计，不以 GPU reduction 替换原统计。保留每个 microbatch 的 loss 有限性检查与梯度有限性检查。
5. 原 collator 完整验证后的内部 batch 不再重复执行目标 ID 正数检查。普通外部 batch 仍检查；动态 SID 范围/形状检查保留。
6. runner 的训练日志改为常驻缓冲句柄，每个检查点、相位边界及退出时刷新。每步仍记录，记录字段及数值不变；非正常强杀时尚未刷新、且未进入检查点的日志尾部可能丢失，最近检查点的日志在保存前已刷新。
7. 原始评估数据及 Arrow 列按 split 复用，每次检查绑定文件大小/mtime；有变化则重新加载并进行原哈希校验。模型输出、当前 SID/碰撞目录不缓存，仍按每个节点重建。独立 evaluate 命令不额外预制训练 Tensor。
8. 复用静态四级遮码深度和前缀注意力遮罩。缓存按形状/设备区分，不含参数、梯度或 RNG，不进入 state_dict。没有复用具有随机行为的模型前向。

## 明确保留

- 模型结构、参数初始化、SID 表示、注意力计算与顺序、OCN 训练/评估切换。
- Dropout/Gumbel 调用顺序、seed 和 RNG 状态；不改变确定性、TF32、训练/推理精度。
- 第一批执行优化时保留了当时的 microbatch=16；用户随后要求对齐旧实验，四份配置现已统一为 effective batch=1024、microbatch=256（每次更新累积 4 次）。此次对齐未改 loss 归一化、R/T 更新预算和学习率。
- 验证节点、beam、样本顺序、去重/碰撞、平局规则、best 选择不变。
- checkpoint 节点、内容、恢复校验；没有修改旧实验快照或结果。
- 未做 fused attention/Adam、torch.compile、多卡、异步验证、减少验证等变化。

## 入口与测试

- `scripts/listwise_diffgrm/execution.py`：训练 Tensor 缓存、限界预取、日志句柄。
- `run.py`：四份配置共用的 build_context / training_updates / train / evaluate。
- `training.py`：延迟取回指标、已验证静态输入路径。
- `model.py`：仅增加执行常量缓存，不新增模型权重。
- `tests/test_listwise_execution.py`：真实小型 backbone、Dropout=0.1、BF16、Gumbel T 阶段的逐位对照。

运行本地小数据一致性检查：

```powershell
python -X utf8 -m unittest tests.test_listwise_execution -v
```

测试覆盖四组合的每步 loss、模型及 Tokenizer 梯度、参数、优化器、scheduler、RNG；
相同数据顺序及尾批；T 后恢复；beam 验证逐样本输出；完整 runner 的共同预热、H0/H1 分叉、日志、保存与完成后恢复。
CPU 可用时执行 CPU 对照；CUDA 对照在没有 GPU 时显式 skipped，不能据此宣称 H100 数值一致或给出提速倍数。
最终运行记录及源码哈希见 `EXECUTION_OPTIMIZATION_VERIFICATION_20261112.json`。

## 后续使用边界

配置对齐更新：四份新配置起初由 base.json 复制，microbatch=16；用户随后明确要求与旧实验一致，现已改为 256。四份配置的 training 字典已逐项核对，与 multiseed_freeze_20261109_v1/config_seed2025.json 完全相同。原优化验收 JSON 保留原时间点事实，不代表 microbatch=16 与 256 数值轨迹相同；新配置 SHA 及对齐结果见 DECODER_PREFIX_BATCH_ALIGNMENT_20261112.json。正式 GPU 执行和显存容量仍未验收。

正式 H100 运行前，仍需要在目标环境做短小的同状态 CUDA 对照（测试已提供），尤其检查异步传输依赖与 RNG/数值逐位一致性。
本次没有进行该硬件检查，也没有实际测得 H100 提速比例。不得将测试中的 toy 数据训练称为正式 KuaiRec 实验。
定制旧 experiment.py 若绕过 run.train，自行循环或写日志，不会自动获得全部预取/日志优化；这四个新版本应使用公共 runner，或在后续新日程 wrapper 中显式复用 training_updates 和 TrainingLog。
