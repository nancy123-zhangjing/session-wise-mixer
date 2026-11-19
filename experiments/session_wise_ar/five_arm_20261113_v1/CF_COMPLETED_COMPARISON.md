# CF无Mixer训练及测试结果

状态：训练和best/final全测试完成；五组总Goal未完成。单seed，不作显著性或跨方法因果结论。

| 节点 | 无Mixer验证 | 有Mixer验证 | 无Mixer测试 | 有Mixer测试 | 测试相对变化 |
|---|---:|---:|---:|---:|---:|
| best R9 | 0.006130 | 0.006534 | 0.008246 | 0.008855 | -6.88% |
| final R35 | 0.005713 | 0.005518 | 0.006534 | 0.006676 | -2.12% |

两组都按验证集选R9为best。Mixer最佳点优势延续到测试集；无Mixer在R35的验证优势没有延续到测试。当前不能据后期验证曲线宣称去除Mixer更优。

完成R35/T300、24255次推荐更新，31次完整验证（每次71657条），best/final各53194条测试。节点权重SHA、配置绑定、优化器/数据计数和状态字段检查31/31通过。自动测试对best/final实际执行原JointTrainer.restore；测试summary与选点SHA一致。中间节点仅完成CPU结构/哈希检查，未逐个做CUDA数值恢复，后续总验收须保留该边界。

资源：nancy-five-cf-m0-1113-v1平台SUCCEEDED，唯一worker为instance_success，无运行中实例。大权重仍在原共享盘，未删除。原始平台回执在raw/CF_FINAL_JOB_STATE.json、raw/CF_FINAL_INSTANCES.json。

证据：CF_TEST_COMPLETE.json、CF_NODE_AUDIT_RESULT.json、CF_INTERIM_COMPARISON.json、CF_M1_REFERENCE_SUMMARY.json。

## 配置来源复核

CF_SOURCE_PARENT_AUDIT.json记录实际远端只读复核：原CF路由verify_parent通过，源码清单、共同PREPARED/initial、教师与验证/测试噪声SHA匹配；两组config.json/runtime.json相等，绑定中的prepared、teacher、target_mode、data、source相等。初始化SHA为5e1a9634a2de982f5205dad5e588764dac95c713ed50dd7ecd718433a20af701。本次未更换任何资产或重跑模型。

## 训练曲线与末端编码状态

完整曲线在cf_completed_curves/，来源为CF_FINAL_CURVE_DATA.json和CF_M1_CURVE_DATA.json；plot_cf_completed.py仅绘制已有日志。每个loss分开绘图、两组均实线。10张图覆盖推荐器与Tokenizer损失，以及SID变动和碰撞状态；图源SHA见MANIFEST.json。

| R35/T300状态 | 无Mixer | 有Mixer |
|---|---:|---:|
| 推荐器末轮总loss | 0.800312 | 0.877185 |
| Tokenizer最后10次更新平均总loss | 0.758776 | 0.834976 |
| 最后10次T更新SID变化商品比例 | 2.1066% | 18.6801% |
| 碰撞商品比例 | 17.9344% | 18.1208% |
| 最大碰撞桶 | 1029 | 782 |

无Mixer loss较低、SID变动较小，但best和final测试都较低。不能把loss较低或编码变动较少直接解释成推荐更好。两组碰撞比例接近，但桶分布不同；这里只报告日志观察，不证明Mixer通过何种机制提升效果。CF使用连续向量在碰撞桶中判别，与AR的256容量后缀不同，最大桶超过256不表示同一种执行失败。

训练相位日志累计：无Mixer R=4996.10秒、T=58.46秒；有Mixer R=4854.66秒、T=59.20秒。这不是总作业时长，不含验证、预训练、保存和等待。
