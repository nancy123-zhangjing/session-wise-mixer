# RQCD-H 代码实现与本地验收

日期：2026-11-05  
对应协议：`kuairec_rqcd_h_v1`  
结论：**新模型、RQ预训练、训练/验证/测试入口已实现；本地41项测试及完整网络的CPU合成预检通过。尚未在H100运行，尚未进行真实数据训练或正式评估。**

## 1. 交付入口

- [代码与命令说明](../../../scripts/rqcd/README.md)
- [唯一baseline配置](../../../configs/09_kuairec_listwise/rqcd_h_v1.json)
- [实验协议](RQ_CONTINUOUS_SESSION_BASELINE_PROTOCOL_20261104.md)
- [机器验收收据](../../../outputs/rqcd_verification_20261105_r1/VERIFIED.json)
- [完整网络CPU预检](../../../outputs/rqcd_verification_20261105_r1/full_model_cpu/PREFLIGHT.json)
- [新模型测试](../../../tests/test_rqcd.py)

只在新目录`scripts/rqcd/`实现，不修改原DiffGRM和GenRec源码，不改原实验权重、报告、数据或验收收据。没有启动Goal、恢复监控、申请/重启远端资源，也没有执行旧训练、推理或统计。

## 2. 已实现的功能

| 环节 | 实现 |
|---|---|
| RQ资产 | 使用既有PCA256内容；前三层STE、末层Sinkhorn；有界K-means；内容验证选模、健康门槛；固定潜向量/SID/码本导出 |
| 连续模型 | 1层历史Encoder、4层去噪Decoder；外层五件前缀条件、内层连续扩散；只预测量化前潜向量 |
| 训练 | 全五位置等权MSE；BF16路径；实际尾组加权累积；35R更新时钟；独立τ/ε RNG |
| 解码 | 50步DDIM；目录约束逐层量化；重建误差/原始ID消歧；列表去重、不屏蔽历史 |
| 诊断 | 同轨迹无约束SID有效率、目录约束改写率、残差和桶大小；固定8192清单的teacher-forced MSE |
| 评估 | 完整验证11节点；best/R35分开；封存后测试；逐样本商品指标和预测输出 |
| 恢复 | 模型/优化器/更新游标/LR时钟/RNG/资产与源码绑定；验证前边界保存；拒绝身份不匹配 |
| 后处理 | 保存预测的用户聚类bootstrap、固定时间分段及推荐集中度；不加载模型 |

历史逐样本文件格式可能不同。配对分析入口目前要求明确的统一`sample_id/user_id/target/metrics`格式；若直接读取历史格式，须先做经过核对的适配。没有为此重算旧统计，也没有声称已复现所有历史冷/热门分析。

## 3. H100方向的性能改动

1. **五个训练视图合批。** 用一个Decoder批次处理全部位置，仍保持每个位置独立的因果前缀；通过数值与梯度参考测试。
2. **历史计算复用。** 历史Encoder每个列表批次只运行一次；每层cross-attention K/V先在原批次计算，再扩展至五视图，减少重复投影。
3. **SDPA与BF16。** 注意力使用PyTorch SDPA，CUDA训练使用BF16 autocast、FP32参数/MSE/优化器状态；不启用TF32或FP8。
4. **融合优化器。** CUDA启用fused AdamW；CPU保留参考实现。
5. **数据张量缓存。** Arrow列一次性校验并打包；H100默认将紧凑历史/目标ID表驻留GPU，训练热路径不做逐行Python拼装。CPU缓存模式支持固定内存及non-blocking传输。
6. **推理复用K/V。** 历史K/V在五件商品共250个去噪步中复用；没有改变采样等级或输出精度。
7. **GPU目录操作。** 合法码字掩码与残差距离批量计算，避免每件商品逐目录Python扫描。

**这些是代码路径优化，不是已经测得的H100加速倍数。** 严格确定性和padding mask可能使某个PyTorch版本选择math或其他融合attention backend；没有强行禁用确定性来追求FlashAttention。

新增`python -m scripts.rqcd.preflight --device cuda:0 --require-h100 ...`有界验收入口，会记录实际选中的attention/optimizer算子、峰值显存和更新吞吐。它不申请GPU资源。

本版继续遵守协议：不启用`torch.compile`、FP8、CFG、EMA、额外排序器，也不自动扩大microbatch或缩减采样步数。128条列表的microbatch尚未在H100上测得最优。

## 4. 实际执行的测试与证据

使用已有本地环境`.runtime/stage08_restore_env`，PyTorch **2.6.0+cpu**、NumPy **2.2.6**、PyArrow **25.0.0**；没有安装或升级依赖。

| 测试 | 结果 | 证明范围 |
|---|---:|---|
| `tests/test_rqcd.py` | 19/19通过 | 新模型数值、梯度、资产、恢复和合成完整流程 |
| `tests/test_listwise_metrics.py` | 7/7通过 | 复用的真实商品指标、集合相关性、PAD/重复口径 |
| `tests/test_listwise_decoding.py` | 15/15通过 | 历史目录/碰撞规则回归；不是新模型H100推理 |
| 完整架构CPU合成预检 | 通过 | hidden256、1层Encoder/4层Decoder、1000级/50步采样、训练更新、恢复和完整五商品生成 |
| 新Python源码解析、`git diff --check` | 通过 | 语法和差异格式；不替代运行测试 |

合计**41项单元/集成测试**，另有完整架构合成预检。日志路径和SHA、源码/配置SHA均写入机器收据；最初的CPU预检副本保留，正式引用本页列出的最终收据。

19项新测试包含：

- 五视图批量版与逐位置参考版的输出、参数梯度一致；
- 当前干净目标/未来标签不泄漏；PAD历史不影响输出；
- 生成的历史K/V缓存与不缓存结果一致；
- cosine截断β日程、50步索引及oracle DDIM恢复；
- 目录真实潜向量的SID回读，GPU式批量目录算法与标量参考一致；
- 合法、互异的五商品生成；非有限输出立即失败；
- 评估噪声不受batch顺序影响；
- 尾组加权梯度与整组平均一致；
- 含dropout、优化器、τ/ε随机状态的下一更新精确恢复；
- RQ中断后恢复与连续运行的最终参数一致；
- RQ训练/eval分配差异、损失和残差梯度口径；
- 训练目标频次加权缩放及逆变换；
- 合成Parquet→RQ训练/导出→推荐训练/验证→封存测试完整流程；
- 冻结资产篡改、真实历史PAD、未来历史和配对身份错误被拒绝；
- bootstrap按样本加权，而非用户等权；
- CPU BF16 attention前后向有限。**最后这一项不等于CUDA BF16验收。**

最初直接执行旧指标测试时，旧测试文件未自行加入项目根路径，报`No module named scripts`；显式设置项目根`PYTHONPATH`后，7项指标与15项目录测试全部通过。最终日志来自正确的运行环境，不把入口配置错误当成模型故障。

## 5. 尚未验证的部分

- **H100/CUDA实测未完成。** 本地只有CPU版Torch；此前Inspire只读Notebook列表查询返回`APIError code 13`，未取得可用远端实例。没有通过重启历史资源绕过该问题。
- 本地完整网络预检使用8条合成列表和512件合成目录，不是有效batch1024、全目录10728的H100吞吐测量。
- 未执行真实KuaiRec资产启动核验、新RQ拟合、35R训练或完整推荐评估；因此无推荐效果、收敛或资源成本结论。
- 测试证明范围限定为对应环境和案例。正式GPU执行前仍应运行有界H100预检，确认BF16/fused optimizer、确定性kernel和显存预算。

本次交付完成的是**独立代码实现与本地必要测试**，不是新baseline实验结果。
