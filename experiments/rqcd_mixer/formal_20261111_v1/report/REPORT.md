# RQCD Mixer 三组正式实验总结（seed=2024）

三组均完成 R35/T300、31 次完整验证及 best/final 完整测试。best 仅按验证 NDCG@5 选择；不能用测试集重新选点。

## 设置与对照边界

B0 不使用 Mixer；B1 使用层内四头注意力；B2 使用按 RQ 层块的因果注意力。每组独占一张 H100，采用共同初始化与相同 seed。每组各自完成 10000 商品 epoch 的 RQ-VAE 预训练，按最低碰撞冗余率选点，再冻结 RQ Encoder，按 5R 预热＋30 次“10 次 T 更新→刷新目录→1 个 R epoch”完成联合训练。

R/T microbatch=256，累积4次，有效 batch=1024；评测 batch=256，bf16，固定20步 DDIM、单候选列表、固定初始噪声。验证节点为 R5 至 R35，共31次。验证/测试样本分别为71657/53194。此次未改变既定模型、损失或学习率，也没有重跑短测。

这一对照比较的是三条完整训练路线，包含 Mixer 对预训练及后续联合训练的共同影响。三组预训练选点和产生的潜空间不同，不能把差异全部归因于联合训练某一次交互操作。

## 最佳点与最终点

|组别|节点|R/T|验证 NDCG@5|测试 NDCG@5|测试 Recall@5|测试 HR@5|
|---|---|---|---:|---:|---:|---:|
|B0|best|25/200|0.001030544|0.000868859|0.000883558|0.004417791|
|B0|final|35/300|0.001001708|0.000802421|0.000815882|0.004079407|
|B1|best|28/230|0.001135971|0.000923472|0.000917397|0.004568184|
|B1|final|35/300|0.001116421|0.000932352|0.000928676|0.004624582|
|B2|best|24/190|0.000973441|0.000916586|0.000906117|0.004511787|
|B2|final|35/300|0.000874520|0.000828370|0.000812122|0.004041809|

全部逐轮验证结果见 [validation_nodes.csv](validation_nodes.csv)。

![验证曲线](validation_ndcg5.png)

## 已观察到什么

- best 对比：B1 的测试 NDCG@5 相对 B0 为 +6.29%。
- best 对比：B2 的测试 NDCG@5 相对 B0 为 +5.49%。
- final 对比：B1 的测试 NDCG@5 相对 B0 为 +16.19%。
- final 对比：B2 的测试 NDCG@5 相对 B0 为 +3.23%。

B1 的验证和测试方向较一致，且 R35 测试略高于其验证所选 best 的测试。B2 验证最佳值低于 B0，测试 best 却略高于 B0；这说明验证和测试排序并不完全一致，不能只看验证集就判定 B2 无效，也不能用测试集挑选有利节点。

这些数值仍来自单 seed，尚不能证明统计显著或稳定收益；绝对 NDCG@5 仍约为0.001，不能仅凭组间百分比就称为已经建立强基线。本次不额外运行随机/热门基线或新实验。

## 收敛与编码状态

R 阶段去噪损失整体下降、后期趋平；T 阶段潜空间重建及逐层 VQ 损失下降，去噪/对齐损失仍有波动。B1 的 R35 去噪损失并非三组最低，却具有最高测试 NDCG，说明连续向量误差与最终推荐命中并不等价。

预训练原始特征重建损失下降，但 VQ 损失上升，并带动总损失上升。不能把总损失曲线解释成良好收敛，也不能仅据此证明程序错误。当前按预先约定的碰撞指标选点；其与推荐效果的联系仍是待验证问题。

|组别|预训练选点 epoch|选点碰撞冗余率|T300碰撞冗余率|最后10T的SID变化比例|四层利用率|
|---|---:|---:|---:|---:|---|
|B0|8500|8.6782%|8.6969%|0.6432%|均100%|
|B1|2900|8.6503%|8.7621%|1.1559%|均100%|
|B2|6950|8.6223%|8.7808%|2.4049%|均100%|

碰撞冗余率为1−独立SID数/商品数；SID变化比例比较相邻两次目录刷新，不是累计变化率。利用率100%仅表示每个码字均有商品使用，不代表使用均匀或编码一定有效。

B2 的后层有效码本变化较大，后期变化幅度回升，同时验证结果波动；“跨层交互造成更大漂移、增加推荐器适配难度”是可检验的猜想，不是本次已证明的因果结论。B1/B2 的碰撞冗余率都略高于 B0，因此本次 Mixer 收益也不能简单解释成碰撞减少。

## 分开的损失图与诊断图

### R_total

![R_total](R_total.png)

### R_denoise

![R_denoise](R_denoise.png)

### R_align

![R_align](R_align.png)

### T_total

![T_total](T_total.png)

### T_denoise

![T_denoise](T_denoise.png)

### T_align

![T_align](T_align.png)

### T_latent

![T_latent](T_latent.png)

### T_vq

![T_vq](T_vq.png)

### pretrain_loss

![pretrain_loss](pretrain_loss.png)

### pretrain_reconstruction

![pretrain_reconstruction](pretrain_reconstruction.png)

### pretrain_vq

![pretrain_vq](pretrain_vq.png)

### encoding_sid_change

![encoding_sid_change](encoding_sid_change.png)

### encoding_collision_redundancy

![encoding_collision_redundancy](encoding_collision_redundancy.png)

### encoding_collision_items

![encoding_collision_items](encoding_collision_items.png)

### utilization

![utilization](encoding/utilization.png)

### perplexity

![perplexity](encoding/perplexity.png)

### correction_relative_rms

![correction_relative_rms](encoding/correction_relative_rms.png)

### effective_relative_change

![effective_relative_change](encoding/effective_relative_change.png)

修正/基础 RMS 是有效修正相对基础码本的幅度，不是每次优化器更新的幅度；逐层有效码本变化图才反映相邻 T 块之间的变化。19张图已进行视觉检查，三组采用不同颜色的实线；利用率曲线完全重叠是数据相等。

## 执行效率和验收

每组预训练约12–19分钟；R/T实际训练合计约79–80分钟，31次验证合计约119–123分钟，best/final测试合计约6分钟。该实验的主要耗时是逐件、多步扩散生成的完整验证，不能把全部用时当成反向传播训练耗时。峰值已分配显存约4.86GiB；显存小本身不等于GPU利用率低。

93个不可覆盖验证checkpoint已核对大小、SHA、模型/优化器计数、数据流及随机状态字段；RQ Encoder和保留的RQ Decoder在联合训练中逐节点与预训练选点一致。共同初始化、源码绑定、配置、推理噪声及验证/测试checkpoint身份均通过核验。精确GPU下一步恢复沿用已验收短测证据；本次进行了状态完整性审计，没有重复训练恢复测试。

全部大权重、预测文件和原始日志保留远端；本地仅保留源码、小摘要、图表和SHA清单。资源关闭与最终验收状态以同目录上级 FINAL_ACCEPTANCE.json 为准。

## 下一步建议（未启动）

优先保留B1为有希望的交互方向，但先查清预训练VQ曲线和生成向量落地质量，再决定多seed复现。B2可进一步研究后层变化与适配的关系；不要同时改交互结构、监督信号和量化方式，否则无法识别收益来源。
