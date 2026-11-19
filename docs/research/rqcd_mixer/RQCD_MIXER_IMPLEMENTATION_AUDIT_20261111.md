# RQCD Mixer v1：实现一致性审查

审查对象：scripts/rqcd_mixer/、configs/rqcd_mixer/、tests/test_rqcd_mixer.py。
依据：RQCD_MIXER_FULL_DESIGN_20261111.md及其设计一致性审查。设计文件SHA256仍为
fdb73fe54c9dba4a27fc7d0d56cf049a5f23e9b6888408e12ea261a8be51f61d。

本次授权范围是实现与本地检查。未申请GPU、连接启智、运行真实训练或评测，也未恢复旧监控。
本文件区分“已实现并做CPU检查”和“尚待H100检查”，不把代码交付写成实验成功。

## 1. 方案与代码逐项核对

| 设计要求 | 实现位置 | 核对／检查 |
|---|---|---|
| 沿用数据划分、PCA256归一化、50历史／5目标；不做OPQ | data.py、复用rqcd/data.py | 内容SHA、目录映射、样本数、时间顺序、PAD／唯一目标校验 |
| B0/B1/B2独立命名空间，首轮seed2024 | v1.json、run.py | 正式入口不接受不同科学配置；旧训练源码不修改 |
| 三组复制同一Enc/Dec／基础码本／推荐器初值；B1/B2复制Mixer初值 | run.prepare | 合成三组流程逐张量核对；首批K-means独立随机流 |
| RQ MLP及4×256×32真残差量化 | tokenizer.py | 隐藏ReLU、末层线性；四层硬选择及残差；无BN／额外dropout |
| 层内四头，或按层块因果四头 | CodebookMixer | 层内与密集块对角参考输出／梯度一致；因果方向和同层中心置换检查 |
| 无中心编号；LN后加层标记；输出零初始化 | CodebookMixer | 初始有效码本等于基础码本；两组共享参数结构 |
| 当前层RMS/第一层RMS，ratio停止梯度，无0.1门控 | CodebookMixer | 固定偏置修正验证尺度比例；记录W/U/Δ/C RMS及Δ/W |
| 10000商品epoch、11更新/epoch、100步K-means、50epoch验证 | v1.json、Pretrainer | 预算核算110000更新；预训练尾批保留 |
| ETEGRec式预训练STE及四层VQ均值 | RQTokenizer.pretrain_forward | 重建梯度到Enc/Dec；VQ到码本／Mixer；内部β=.25 |
| 最低碰撞严格改善选点；另存best_loss和last | run.pretrain | 合成完整预训练／导出；best_loss按样本加权的epoch总损失 |
| z、μ、单一s、各层a固定，train目标频次统计 | fixed_statistics、Catalog | FP64统计与显式全中心距离参考一致；不逐维标准化 |
| 四张独立E表；50件→200token | Recommender.lookup/encode_history | 真实50历史形状检查；相同编号不同层参数不同 |
| T5 Encoder2、Decoder4；width256、heads4、FFN1024 | model.py | 架构配置核对；与已安装Transformers4.53.2随机T5逐参数对齐 |
| T5Norm、32桶、max128、相对偏置共享、QK scale=1 | model.py | 官方T5前向及输入梯度数值对照，距离桶逐项一致 |
| 五个前缀视图；长度1/5/9/13/17；无未来目标 | training_prediction | 五视图与逐件参考前向／梯度一致；改未来前缀不影响早位置 |
| 32→256→32、时间MLP、x0预测、1000等级／20步DDIM | Recommender、Diffusion | 时间频率／余弦公式审查；DDIM20次、端点1000→1→0 |
| 历史固定、实际落地商品进入前缀 | generate | 一次历史编码；追加的是落地SID；不同batch划分合成输出一致 |
| cache不保留带噪query | DecodeCache | 五种前缀长度、多噪声等级与未缓存参考一致；缓存不被query修改 |
| T内部带噪硬选／软梯度；推荐输入cached-hard | joint_quantize | 输入前向与缓存E完全一致；单独去噪损失能回传码本／Mixer |
| 残差层间不额外detach；噪声按商品出现位置抽取 | joint_quantize、JointTrainer.draws | 代码路径审查；整有效batch先生成噪声，五视图复用 |
| KL为sg(P)||Q，同真实SID前缀，无候选mask／Gumbel | alignment_loss | 显式公式数值对照；目标无梯度，预测及当前中心有梯度 |
| R两项，T四项，latent/VQ只对5B目标 | recommendation_losses | 参数冻结／更新隔离；R保持对齐的预测梯度 |
| 5R预热＋30×(10T→刷新→1R) | JointTrainer.advance | 缩小日程端到端计数检查；正式R35/T300、R20/T150 |
| 有效batch1024、R尾批835、每有效batch一步 | JointTrainer.update | 3条尾批拆分与未拆分梯度／loss对照；按m/N累积后裁剪 |
| R/T优化器持续各自计数；Rwarm500＋cosine | common.py、training.py | 端点、最终步数24255、T噪声1→.2检查 |
| T冻结推荐器且eval但保留输入梯度；RQEnc/Dec冻结 | JointTrainer._set_mode | R/T参数改动隔离及去噪桥接测试 |
| 合法RQ前缀、只排除已输出商品、同SID不同商品可用 | Catalog.land | 标量参考对照、全碰撞桶消歧、原商品ID平局规则 |
| 桶内按生成z与商品原始z距离消歧 | Catalog.land | 不用码本重建误差或全目录ANN替代 |
| 11个完整验证点；best和R35完整测试 | run.train/test | 合成三个节点／三组全流程，测试身份与选点绑定 |
| 固定噪声只依赖版本/seed/split/sample/j | evaluation_noise | 调换样本顺序／batch不改噪声；噪声和逐样本输出校验SHA |
| 每正式节点完整保存；部分T块能精确恢复 | JointTrainer.state/restore | 含dropout的CPU逐张量恢复；旧SID快照保留；游标／优化器钟检查 |
| 中断验证复用准确模型节点 | run.train、evaluate | 人工中断合成验证再恢复；不可变节点SHA保持一致 |
| 完整指标、loss及码本状态输出 | events、summary、report | 三组生成6行best/最终点表；loss分别成图；报告不运行模型 |

## 2. 提速调整的边界

这些是执行组织变化，未改变训练目标或预算：

| 优化 | 保持的等价条件 | 当前证据 |
|---|---|---|
| B1四个小注意力块 | 同参数、同层标记、块对角可见性 | CPU输出和梯度与密集参考一致 |
| 五视图一起处理 | query位置、真实前缀、相对偏置和因果mask不变 | CPU逐件前向／梯度对照 |
| Cross-attention折叠query维 | 各view使用相同历史K/V，各query仍分别softmax | CPU参考一致；CUDA kernel待profile |
| 历史／干净前缀缓存 | eval，前缀不看query，历史不更新 | CPU五前缀、多噪声等级参考一致 |
| 距离GEMM与批量落地 | FP32距离、同候选、同平局规则 | CPU逐样本标量落地对照 |
| 固定统计量代数化 | 对全部中心距离的均值，不是最近中心 | FP64显式距离参考一致 |
| 梯度累积 | 同有效batch、m/N权重，一次clip/step | dropout关闭时梯度一致；不声称不同划分的dropout逐位相同 |
| 混合精度／fused优化器 | BF16仅推荐器计算、主权重FP32、量化FP32 | 实现已接入；CPU检查不能代替H100数值验收 |

没有用减少验证量、减少DDIM步骤、删除loss、增大有效batch、降低精度到FP8、放开TF32或改模型层数来换取速度。

## 3. 本地验收记录与尚未验证的部分

测试入口：tests/test_rqcd_mixer.py；所有数据由测试临时构造。
本次最终CPU验收：28项通过，耗时14.585秒；无失败或跳过。
最终结果、命令、包版本、源文件哈希见 implementation_20261111/LOCAL_ACCEPTANCE.json，
逐项日志见 implementation_20261111/unittest.log。

检查覆盖：模型数学／官方T5参考、梯度、缓存、落地、恢复以及三组缩小流程。
测试中实际50件历史的形状为[1,200,256]，输出为[1,5,32]；缩小完整流程不代表真实模型已经训练。

以下尚未执行：

- H100的BF16前向／梯度与FP32误差检查、fused AdamW与CUDA确定性行为。
- 三组共同microbatch选择、实际attention kernel和完整有效更新吞吐。
- 真实数据的预训练收敛、推荐指标、全验证耗时和资源验收。

benchmark入口已准备好，但本次未执行。micro_r、micro_t、evaluation_batch保留null，
正式训练会拒绝这些未选定的值。未来只需一次合并短测，数值与三组共同候选都通过后才写runtime.locked.json。

## 4. 一致性结论

按上述逐项核对，核心模型、损失、冻结规则、日程和评测设置与完整设计保持一致。
本次补全的是独立模块组织、文件格式、错误收据、测试夹具、缓存及batch化实现；没有更换科学方案。
未修改设计正文、早期H0/H1源码快照、历史结果或飞书文档。
