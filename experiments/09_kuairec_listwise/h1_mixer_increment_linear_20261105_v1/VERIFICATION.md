# 本地代码验证

日期：2026-11-05。仅验证本次新版本；没有运行真实 KuaiRec 续训或提交 GPU 任务。

环境：现有 `.runtime/stage08_restore_env/Scripts/python.exe`，PyTorch 2.6.0+cpu。

命令（工作目录为本版本 code）：

```text
python -B -m unittest discover -s tests -p test_listwise_increment.py -v
python -B -m scripts.listwise_diffgrm.recovery_increment_run --help
```

5 项针对性测试全部通过，耗时 100.600 秒：

1. 200 个线性系数及完整增量累积公式；仅改变接受比例不回撤已接受的修正。
2. 合成 H1 R20/T200 父状态恢复后，码本起点、SID、两个优化器、
   推荐器调度和 RNG 保持一致；多次只读求值不推进门控。
3. 真实小模型 T 更新符合接受公式，基础码本和 Mixer 有梯度，
   推荐器参数不变；40T 后的 R 更新保持 Tokenizer 与门控状态不变。
   Tokenizer 学习率未改变。
4. checkpoint 实际写盘，在新建控制器中恢复后继续下一次 T 更新，
   损失和完整 checkpoint 指纹与不中断执行相同。拒绝缺失门控、
   篡改增量、锚点、时钟、调度、协议或学习率的状态。
5. 验证缓存绑定累计门控状态；验证指标平局时选择更早节点。

随后对最终源码单独核对了 T201=0.1、T400=0.01 的精确端点，
及全部 200 个系数严格递减；命令行入口导入成功。
训练 runner 与原 G01 runner 的差异仅为新控制器/调度引用、状态缓存绑定
及事件名称；日程、采样、batch、损失、评估点和选模逻辑未改。

`SOURCE_MANIFEST.json` 逐文件核对复用的脚本、模型、配置和输入字节，
记录新文件哈希。新 job 使用独立 `runs/IncrementLinear` 输出。

边界：测试使用合成数据、小模型和真实 Adam 更新，不是实际父 checkpoint、
全量数据或 H100 验收。未验证推荐性能改善；不声称新方案会优于 G01/Freeze。
