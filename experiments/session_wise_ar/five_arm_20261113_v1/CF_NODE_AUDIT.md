# CF 节点审计边界

`audit_cf_nodes.py --run <CF-M0/train>` 在共享盘CPU上顺序读取所有验证checkpoint，输出小型JSON路径/SHA清单，不复制大权重，不修改实验，不进行训练或推理。

检查范围：逐节点文件SHA与验证summary一致、config/runtime/binding一致、R/T计数、验证样本数、完整数据排列与随机流状态、优化器步数、关键模型与目录状态字段、31个验证节点齐备、最终点身份。

这不是所有节点的模型数值恢复测试，也未验证每个张量的数值或目录重建一致性。best/final自动测试调用原JointTrainer.restore，包含严格模型/优化器加载、目录重建与哈希检查和CUDA随机状态恢复；只有实际测试完成收据与身份核对通过后，才能引用相应GPU恢复证据。中间节点若需更强恢复证明，应单独记录其证据，不能以文件存在代替。

本次代码检查：Python语法解析通过，字段与实际state()/restore()实现核对；截至本文件创建尚未执行31节点审计，不宣称验收通过。

执行更新：31节点审计已在远端CPU完成，结果见CF_NODE_AUDIT_RESULT.json；best/final全测试也已完成并核对身份，见CF_TEST_COMPLETE.json。上述CPU与CUDA验证边界保持不变。
