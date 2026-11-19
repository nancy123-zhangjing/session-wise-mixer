# H0 / H1 Decoder 前缀双方案（2026-11-12）

状态：代码实现与本地 CPU 功能验证；未申请资源、未启动正式训练。没有改动已有实验结果和冻结源码。

## 四个版本

| 版本 | 配置文件（configs/09_kuairec_listwise/） | train 参数 | 含义 |
|---|---|---|---|
| H0-A | h0_prefix_asymmetric.json | --arm H0 | 无 Mixer，非对称自注意力 |
| H1-A | h1_prefix_asymmetric.json | --arm H1 | 原 Full Mixer，非对称自注意力 |
| H0-B | h0_prefix_cross_attention.json | --arm H0 | 无 Mixer，独立前缀交叉注意力 |
| H1-B | h1_prefix_cross_attention.json | --arm H1 | 原 Full Mixer，独立前缀交叉注意力 |

沿用原 runner：H0/H1 的 Mixer 开关由 `--arm` 控制，并非由配置文件名控制。
同一方案的 H0/H1 配置内容相同，因此可使用该方案自己的共同 warmup checkpoint 分叉。
所有配置从 base.json 继承，增加 model.h_prefix_mode 与 execution.optimized_io=true；microbatch 已按用户要求从 base 的 16 对齐到旧多 seed 实验的 256，有效 batch 仍为 1024。四份配置的 training 字典与旧多 seed 配置一致。prefix_mode 默认缺省值 history，原 H0/H1/J 行为保留。

## 共用表示与训练协议

- 真实历史仍经 SID embedding、商品 MLP、历史位置编码及原 Encoder；只编码真实历史。
- 每件已生成商品的四个 SID 使用原有分码位 embedding，再各自加同一个列表位置向量。位置表复用 item_position_embedding，不新增类型标记或商品 MLP。
- 当前商品仍是四个部分遮码 SID，只有当前四码位交给输出头和损失。训练用真实前缀，推理用各 beam 已生成的前缀。
- OCN 教师及四视图学生均读取相应前缀。保持原有教师 raw cross K/V、学生及推理 projected cross K/V 的历史协议，不在此次顺带修改。
- T 阶段前缀使用原 DIGER-ST assignment，同一 assignment 的梯度可从后续商品损失回传到码本/Mixer；R 阶段只更新推荐器。
- d=256，4 头，Decoder 4 层，FFN=1024，原 dropout/attention dropout/residual dropout=0.1。损失、学习率和优化器沿用原配置。
- 训练一次 forward 只编码一次真实历史，并保留梯度图；推理一次列表请求只编码一次，在 beam 筛选时按父分支重排历史表示。未实现 Decoder 前缀 K/V 缓存。

## A：非对称自注意力

输入为 [前缀商品四码位..., 当前商品四码位]。每层遮罩相同：

| Query / Key | 前缀 | 当前商品 |
|---|---|---|
| 前缀 | 全可见 | 不可见 |
| 当前商品 | 全可见 | 全可见 |

前缀内部双向，并非严格商品级因果。每层依次为非对称自注意力、原历史交叉注意力、原 FFN；前缀也参与后两者。最后只读取末尾四位置。无新增可学习参数。

## B：独立前缀交叉注意力

每层为当前四码位自注意力 → 新前缀交叉注意力 → 原历史交叉注意力 → 原 FFN。
前缀为原始 SID+列表位置表示，无独立前缀 Encoder。各层单独 LayerNorm 前缀并投影 K/V，当前隐藏向量经 LayerNorm 提供 Q。残差系数为 1，不设门控。
每层前缀注意力为 256 维、4 头、两个 LayerNorm，四层不共享参数。Attention 和残差 dropout 各 0.1。空前缀跳过新增分支。
新增模块使用原注意力类的初始化；构造时隔离 CPU RNG 消耗，使新增参数初始化不顺带改变后续 alignment 的初始随机流。
按现有 LayerNorm/投影实现，新增参数共 1,053,696。原联合 qkv 中的 Q 由当前商品使用，K/V 由前缀使用。

## 代码入口与使用边界

- scripts/listwise_diffgrm/model.py：输入嵌入、OCN/四视图、前缀训练、四位置输出。
- scripts/listwise_diffgrm/prefix_decoder.py：遮罩、复用原子层的 Decoder 运算、新增前缀注意力。
- scripts/listwise_diffgrm/decoding.py：历史缓存、beam 前缀传递与重排。
- scripts/listwise_diffgrm/runtime.py：配置校验。
- tests/test_listwise_decoder_prefix.py：真实 backbone 的小型 CPU 功能测试。

仅检查配置（不会训练）：

```powershell
python -m scripts.listwise_diffgrm.run --config configs/09_kuairec_listwise/h1_prefix_asymmetric.json inspect
```

正式训练继续用原 run.py 的 warmup / train / evaluate 入口，并传对应 --arm。
这四个配置继承原 base 的数据/资产路径以及 R35/T400 日程，尚未指定新的远端实验目录或 R20 冻结方案；本次只实现模型版本，没有擅自选择实验日程或启动训练。
原 checkpoint 的 code/config binding 不同，不能直接作为这些版本的普通 resume。B 还含新增参数。后续若选用旧 R5 共同起点，必须显式设计权重迁移及优化器状态处理，不能用 strict=False 偷跳恢复检查。

## 对照边界

A 和 B 输入信息、码位粒度一致，但 B 新增参数，且没有 A 那样的多层前缀上下文处理。因此两者是完整结构方案对比，不是完全等容量的单因素注意力消融。
已有最优点、seed 和冻结日程的实验结论不代表这两个新结构已经有收益。
