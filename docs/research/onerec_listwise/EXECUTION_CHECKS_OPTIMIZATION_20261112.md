# Decoder 前缀执行优化第二批（2026-11-12）

四个 h0/h1 × asymmetric/cross_attention 配置启用 execution.optimized_model_checks=true。旧配置缺省 false；第一批 optimized_io 与静态数据复用保持不变。

本次仅三项：
1. 四码位、四深度的监督码位统计由 GPU sum().item() 改为 Python 整数 batch_size×10。不是修改 loss 的归约。
2. 每次 forward_loss 入口检查完整目标 SID 和历史 SID；内部仅对其切片/重复得到的前缀复用合法性结论。每个新 batch、每次 T 后仍检查当前 SID；独立推理/解码入口仍检查。shape 和 ST shape 检查保留，finally 恢复临时信任状态，检查结果不跨 forward 缓存。
3. OCN 全 MASK 输入按 batch、码位数量、device 缓存，只读复用。没有缓存 OCN 输出或带梯度结果，没有新增 state_dict 参数。

没有调整 batch、精度、R/T 日程、loss、优化器、dropout/Gumbel 次序、teacher/student 模式切换或 loss 有限性检查。

本地检查：python -m unittest tests.test_listwise_execution tests.test_listwise_decoder_prefix -q，15项，14通过、1项需 CUDA 的检查跳过。覆盖四版本随机 R/T 更新、状态/梯度/RNG、恢复、预测；新增非法目标/历史/独立前缀拒绝、异常后作用域清理、MASK缓存不变测试。

H100 短测目录：experiments/09_kuairec_listwise/decoder_prefix_h100_checks_20261112_v1。两条路径均启用第一批 optimized_io，只切换 optimized_model_checks；同卡顺序运行，每相位8步（首步预热、7步计时），完整真实模型和固定真实样本，另32样本验证与恢复下一步检查。正式训练未启动。GPU结论以该目录 COMPLETE.json、REPORT.md、FINAL_ACCEPTANCE.json 为准。
