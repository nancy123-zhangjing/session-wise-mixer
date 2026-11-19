# H0/H1 L5O适配与H100短测边界

数据与指标见 `L5O_DATA_PROTOCOL_20261117.md`；本配置为 `configs/09_kuairec_listwise/l5o.json`。

## 保持的模型与训练逻辑

原H架构：商品SID embedding经MLP压缩，历史Encoder重新编码已生成前缀；OCN四视图、四层Decoder、四套256中心PSE、DIGER-ST梯度、H1全码本Mixer均不变。不引入此前Decoder-prefix变体。

R阶段仅更新推荐器，T阶段仅更新基础码本和H1 Mixer。原学习率、AdamW参数、辅助损失、Gumbel日程、bf16推荐器/float32量化、beam32和碰撞处理保留。确定性算法开启，TF32关闭。

旧训练集709443条，effective batch1024，ceil(709443/1024)=693更新/轮。新R单位固定693步，R35共24255步；5R暖身+10×(40T+3R)，共400T。新R单位不等于遍历一遍新数据：每单位按独立配对种子无放回抽693×1024条，完整batch，比旧末批多189条/单位；锁定的是优化器步数而不是精确曝光数。

每个R单位验证并保留完整不可覆盖checkpoint；最佳按验证EV-NDCG@5选。没有自动冻结或自动提交下一阶段。

## 执行优化

| 项 | 参考路径 | 候选提速路径 |
|---|---|---|
| 模型/数据/有效batch | 同一初始化、history128、1024 | 相同 |
| microbatch | 256×4累积 | 1024×1 |
| 数据 | 原Arrow行collator | 分块向量校验后int32 ID+bool mask常驻GPU，取batch才转int64；不缓存SID或可学习表示 |
| 有限性检查 | 每microbatch同步 | GPU聚合loss/梯度检查，每次update判断一次，异常在optimizer前终止 |
| AdamW | 原实现 | fused=True |
| 解码 | beam32，逐query GPU前沿筛选 | 同beam规则，一次传输整个batch的分数，CPU执行原相同筛选/去重/可行性和tie-break逻辑 |
| 验证batch | 4 | 测128/256，最终按实测选择共同值 |
| compile | 关闭 | 独立有界reduce-overhead候选，未完成数值/恢复验收不启用 |

采用原数据collator与向量缓存逐字段对照；原CPU随机R/T更新与状态精确一致。变更microbatch和fused后的H100比对允许数值/RNG执行差异，不声称逐位一致。

## 短测验收预案

单H100依次H0/H1×参考/候选路径，各200更新（80R→40T→80R），共同初值/数据顺序。记录各阶段秒/update、显存、GPU利用率和loss。阶段均值loss相对差≤5%作为短测实用门槛，不能据此证明长期统计等价；各阶段候选须快于参考。

同一候选checkpoint做4/128/256验证batch对比，固定256条验证样本与beam32，确保256档实际使用满批；报告完全一致列表比例（接受门槛99%，差异须说明）。这仅验证执行优化，不作模型效果结论。保存标记preflight_only的完整状态，恢复后下一次GPU更新要求状态和RNG精确一致。compile候选限240秒尝试，失败或证据不足则关闭；不改变H0/H1共同配置。

本地已通过的检查是CPU正确性，H100数值、恢复、吞吐与最终共同执行配置必须在短测报告中另行验收。静态PSE资产只在核对item_map/item_text不变后另建绑定清单，旧资产不修改、不重新拟合PCA/PSE。
