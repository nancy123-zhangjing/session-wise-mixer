# 06：连续向量推荐强基线

本阶段为新研究入口，与05早期长期四组、最新共享起点Mixer短四组分别记录。05已完成的运行不作为待重跑任务。目标是建立充分训练的连续商品向量检索基线，再为给定Decoder学习适配编码提供可复用起点。

## 2026-10-16 最终测试结果

本轮预先封存的验证选模选择SASRec＋FullCE；测试集6项均值也均高于eSASRec。结果为5个新seed（3031–3035）的均值±样本标准差（ddof=1），每个seed均使用最佳验证checkpoint；测试未参与模型或配置选择。

评估口径：Beauty，22,363条测试样本，12,101件真实商品全库排名，排除PAD，不屏蔽历史商品，不抽样负例。这里是item-level；历史03–05主要为SID-level，暂不直接并榜。

| Test指标 | SASRec＋FullCE | eSASRec |
|---|---:|---:|
| HR@5 | 0.049439 ± 0.001248 | 0.046881 ± 0.000926 |
| HR@10 | 0.072280 ± 0.001294 | 0.064920 ± 0.001759 |
| HR@20 | 0.103179 ± 0.000622 | 0.088459 ± 0.001884 |
| NDCG@5 | 0.034171 ± 0.000893 | 0.033389 ± 0.000705 |
| NDCG@10 | 0.041515 ± 0.000845 | 0.039181 ± 0.000795 |
| NDCG@20 | 0.049302 ± 0.000714 | 0.045109 ± 0.000780 |

相对eSASRec，SASRec＋FullCE的平均HR@10高11.34%，NDCG@10高5.96%。本次验证集预选配置分别为SASRec13与eSASRec19；结论限于当前数据、协议与固定搜索预算。

| 最终配置 | SASRec＋FullCE | eSASRec |
|---|---:|---:|
| 维度 / 层 / 头 | 256 / 1 / 1 | 512 / 1 / 4 |
| 参数量 | 3,900,672 | 10,945,536 |
| Dropout / 学习率 | 0.3 / 0.0001 | 0.3 / 0.001 |
| Batch | 256 | 128 |
| 训练损失 | 全库交叉熵 | sampled softmax，负样本1024 |

两者共享最大历史50、FFN倍率4、Adam betas=(0.9,0.98)、梯度裁剪1、常数学习率、最多300轮、patience50。两套完整基线的容量、结构和训练损失同时不同，不能将本结果单独归因于其中一个因素。

完整[13项测试原始分数](execution_acceptance/final_delivery_20261016/test_results.csv)、[5seed均值/标准差与旧协议配对汇总](execution_acceptance/final_delivery_20261016/summary.json)、[本地独立重算](execution_acceptance/final_delivery_20261016/LOCAL_FINAL_METRIC_AUDIT.json)、[服务器逐样本验收收据](execution_acceptance/final_delivery_20261016/DELIVERY_PASS.json)均已保存。89项有效训练、13项测试逐样本重算与10份正式模型/商品向量导出已通过，另见[最终测速原始样本重算](execution_acceptance/final_delivery_20261016/FINAL_BENCHMARK_AUDIT.json)和[正式导出逐张量/逐向量核对](execution_acceptance/final_delivery_20261016/FINAL_EXPORT_AUDIT.json)。[完整本地备份](execution_acceptance/final_delivery_20261016/FINAL_BACKUP_LOCAL_AUDIT.json)和[代表状态独立副本CPU恢复检查](execution_acceptance/final_delivery_20261016/LOCAL_RECOVERY_TRANSACTION_AUDIT.json)也已PASS；本地产物索引已验证，实验和备份交付均已验收。用户已[接受实际GPU时长×费率的费用交付](execution_acceptance/final_delivery_20261016/FEE_ACCEPTANCE_20261016.json)，保留实际结算未核实说明；本轮全部验收通过，交付提交`aec2307`（未push）；Goal工具状态以当前任务状态为准。

完整实验意图、训练/选模设置、旧协议配对、训练曲线、最终测速、资源开销和结论边界见[06正式报告](REPORT.md)；[验证选择与时序独立审计](execution_acceptance/final_delivery_20261016/FINAL_SELECTION_AND_ORDER_AUDIT.json)覆盖全部89项原始曲线，并核对训练完成、验证封存、测试和最终测速的先后顺序。

本轮[最终交付总验收](execution_acceptance/final_delivery_20261016/FINAL_DELIVERY_STATUS.json)为`ALL_REQUIRED_EVIDENCE_PASS_WITH_USER_ACCEPTED_FEE_BOUNDARY`，交付提交`aec2307`（未push），详[Git交付核查](execution_acceptance/final_delivery_20261016/FINAL_GIT_DELIVERY_AUDIT.json)。当前无需再执行本Goal；下一研究为尚未实施的固定推荐器编码对照，需要另行确定实验范围。Goal工具状态以当前任务状态为准。

## 当前完成范围：2026-10-16

- **规定实验已完成**：64搜索、12复验、10正式、3旧协议配对，共89/89成功；[全部89项/740个元数据文件](execution_acceptance/completed_metadata_precision_v2_batch20/AUDIT.json)与[178份best/last完整状态](execution_acceptance/checkpoint_payload_precision_v2/LOCAL_RECEIPT_AUDIT_22.json)通过核对。修复后训练冻结为`bbaa45b`/b240，恢复supervisor为`37ef406`/578f；旧数值故障版本28项不进入最终结果。
- **最终评估和导出已核验**：验证预选SASRec＋FullCE；13项测试逐样本重算、10份正式权重/商品向量重载，以及固定seed3031的12组/2,400次最终测速均PASS。当前是连续查询向量→全库真实商品检索基线，尚未实施给定白盒Decoder学习适配Encoder或列表生成。
- **GPU已释放**：最后作业`nancy-stage06-20261016-8h100-supervisor-v3`于15:14:19 SUCCEEDED、15:14:40 Pod删除，剩余节点0，见[平台终态](execution_acceptance/resource_accounting_20261016/SUPERVISOR_V3_TERMINAL_PLATFORM_20261016.json)。不重提GPU，不影响既有非Goal Notebook或CPU桥接。
- **完整本地备份与代表状态恢复已核验**：9分卷/1,300文件、14,409,004,551解压字节全部按大小/SHA256验证，178份best/last摘要与独立ledger交叉一致；412份可用文件已安全提取到`outputs/stage06_continuous/final_delivery_20261016/verified/`，包括10exports、13逐样本结果、锁定输入/27份源码及代表SASRec13/3031完整状态。`recoverable_primary/`独立副本已用原消费者加载model、Adam、scheduler和CPU RNG；CUDA RNG仅字节保全/校验、未执行，保留COMPLETE且未启动训练。其余seed完整状态仍在已验证分卷中。
- **本地复用与已验收交付**：[已验证本地产物索引](execution_acceptance/final_delivery_20261016/LOCAL_ARTIFACT_INDEX.json)列出10exports、13逐样本结果、178份checkpoint分卷成员及SHA、首选SASRec13/3031和`recoverable_primary/`，以其中本地路径为准。本轮全部验收通过，交付提交`aec2307`（未push），当前无需再执行本Goal；Goal工具状态以当前任务状态为准。五次Goal作业分配33.2203 GPU小时、平台runtime32.9347 GPU小时，按实测费率折算同值点券。用户已回复“接受，以此完成Goal”，[确认收据](execution_acceptance/final_delivery_20261016/FEE_ACCEPTANCE_20261016.json)接受此计算口径作为费用交付，费用不再是阻塞；金额仍非已确认平台结算，`actual_goal_settlement_verified=false`，详[资源与费用审计](execution_acceptance/resource_accounting_20261016/FINAL_RESOURCE_AUDIT.json)。

## 早期开发与运行快照：11:54及以前（历史）

以下保留第一阶段与早期执行的原始证据和异常过程；“正在运行”“测试锁定”“尚无正式结果”等只描述对应时点，均已被上方最终结果取代，不作为恢复或重提任务的指令。

11:54：修复版本提交`bbaa45b`通过远端36项CPU gate、增强GPU gate与初始独占测速，新8H100 Job `nancy-stage06-20261016-8h100-precision-v2` 正RUNNING并执行原89项队列。4项梯度对照、35轮回归、8卡补位及2,400次原始测速已[独立核对](execution_acceptance/gpu_precision_v2_20261016_01/GPU_GATE_AUDIT.json)；[首个完整checkpoint审计](execution_acceptance/gpu_precision_v2_20261016_01/FIRST_COMPLETED_CHECKPOINT_AUDIT.json)通过，不代表全部89项完成。测试仍锁定，旧版本结果不混用。见[新提交收据](execution_acceptance/PLATFORM_SUBMISSION_20261016_PRECISION_V2.json)。下文是此前时点记录。

11:38更新：CUDA FP32 Math注意力修复已通过有界梯度校准、514步重放、35轮压力检查及本地73项CPU gate。新源码仍需服务器CPU验收和实际提交版本的增强GPU gate，再从头运行原89项清单；旧28项及消耗单独保留，不混入新选模。两个单卡诊断已结束释放。详见[故障与修复](NUMERICAL_FAILURE.md)。

第一阶段已完成**本地数据准备、协议实现与CPU验收**，原始证据见 [ACCEPTANCE.md](ACCEPTANCE.md)。后续Goal已获得代码提交、远端同步及单节点8H100运行授权。

截至2026-10-16，SASRec＋FullCE和eSASRec、训练/恢复、全库评估、导出、8worker队列及交付审计入口已实现。当前源码通过72项本地CPU测试、零跳过；两模型各完成真实数据3次CPU更新、22,363条完整验证及导出重载检查。远端35项模型/队列/交付CPU测试及全部缓存核验通过。这些是接口验收，不是正式效果结果。

执行版本`4d7c4662c077ca3aaa9d78363e1ee09d8bb34432`通过初始GPU smoke和测速，但主8H100 Job于2026-10-16 11:08因eSASRec配置17数值异常停止，平台FAILED且8卡已释放。已有28项技术完成、1项失败、7项中断、28项待启动，数值有效性待审；Top-3和最终测试均未执行。独立1卡诊断在另一张H100复现原BF16路径异常，修复仍需验证，见[故障及重跑边界](NUMERICAL_FAILURE.md)。不能把此前smoke通过等同于长程稳定。Decoder适配编码仍未实施。[提交收据](execution_acceptance/PLATFORM_SUBMISSION_20261016.json)、[原GPU gate](execution_acceptance/gpu_20261016_01/GPU_GATE_AUDIT.json)。

当时证据：[CPU gate](execution_acceptance/local_cpu_20261016_01/CPU_GATE.json)、[真实数据CPU集成](execution_acceptance/local_cpu_20261016_01/cpu_integration.json)、[实现状态](IMPLEMENTATION_STATUS.md)、[模型来源和适配](SOURCES.md)。

## 输入与身份（第一阶段准备协议，沿用至最终实验）

原始输入沿用项目的Amazon Reviews 2014 Beauty，保留原用户/商品编号和序列顺序。

| 文件 | SHA256 |
|---|---|
| `all_item_seqs.json` | `43476d3f941f103d32ae58eb87da79c8894d76e3fb64da57f8e2d18c0c51d1b1` |
| `id_mapping.json` | `4d88fede709699e3415962419728b07dc8600140ebc000484dd371c81f66a0be` |

权威身份来自既有 [01资产锁](../01_ar_baseline_3x256/reproducibility/assets.lock.json) 中对应的两项；这两项也与最新05运行的本地收据一致。不改写任何历史锁。仅从本地迁移备份读取两个精确列名的普通文件；不调用整包解压、不重新下载或拟合数据、不改变商品映射。

第一阶段此处只验两份输入的字节身份，未重新计算4.6GB迁移包的整体哈希；原始输入副本在每次新运行的 `inputs/` 下，当时尚未在线核验远端资产。后续Goal已完成远端锁定输入/缓存的实际验收，并将来源绑定至89项训练和最终结果；不重复生成或覆盖原数据。

## 两套协议

原序列记为 `s`，以下位置均为**从0开始的原始交互位置**。

| 协议 | 训练目标位置 `t` | 历史 | 预期训练目标数 |
|---|---|---|---:|
| `full_train_prefix_v1` | `1 <= t <= len(s)-3` | `s[max(0,t-50):t]` | 131,413 |
| `legacy_prefix_v1` | `2 <= t <= min(len(s)-3,50)` | `s[:t]` | 105,668 |

两者都以 `s[t]` 为唯一监督商品。旧协议的50同时限制目标位置，不将其悄悄改成完整滚动窗口。验证为历史 `s[:-2][-50:]`、目标 `s[-2]`；测试为历史 `s[:-1][-50:]`、目标 `s[-1]`。两套协议的验证/测试数据必须逐字节相同，各22,363条。

泄漏判断依据用户和交互位置，不依据商品值交集：重复购买的商品可同时出现在训练、历史或留出目标中；测试历史允许包含该用户验证位置的交互。训练历史与目标均必须严格早于最后两个留出位置。

完整协议多25,745个训练目标，因此后续收益不能全部归因于模型结构或编码方式。评估划分沿用当前leave-one-out基准，不把它描述为全局时间切分，也不据此宣称线上效果。

## 输出格式与后续加载约定

每个新运行使用 `outputs/stage06_continuous/<唯一run_id>/`，拒绝覆盖已有目录。

```text
inputs/{all_item_seqs,id_mapping}.json
resolved_protocol.json
catalog_item_ids.json
statistics.json
full_train_prefix_v1/{train,validation,test}.jsonl
legacy_prefix_v1/{train,validation,test}.jsonl
manifest.json
provenance.json
PREPARED.json
```

JSONL字段为 `user_id`、`target_position`、`history_start`、`history`、`target`。按原用户数字ID、再按目标位置排序，UTF-8、LF、固定键序；`history` 不含PAD，最长50。批处理阶段再左补PAD=0，真实商品ID保持1至12,101，候选表排除PAD，不额外屏蔽历史商品。禁止把这批已展开前缀再次用于重复的全位置监督。

`manifest.json`覆盖两份输入、完整配置、候选表、统计和六个划分文件的SHA256、大小及行数。`provenance.json`记录本地环境、实际命令、代码哈希、根Git状态和来源；时间、绝对路径和环境信息不参与跨运行数据内容一致性比较。

`PREPARED.json`仅表示生成完成，**不表示独立验收通过**。独立验证器从原始序列逐条核查完整协议；旧协议通过AST执行项目原 `_sliding_train` 方法作为参照，不导入Torch/HuggingFace数据集。

入口：[准备脚本](../../scripts/prepare/prepare_continuous_data.py)、[只读验证器](../../scripts/checks/verify_continuous_data.py)、[协议配置](../../configs/06_continuous_baseline/data_protocols.json)、[CPU测试](../../tests/test_continuous_data.py)。

## 后续入口

完整本地备份与代表状态独立副本CPU恢复检查已通过；模型与恢复状态按[已验证本地产物索引](execution_acceptance/final_delivery_20261016/LOCAL_ARTIFACT_INDEX.json)复用；费用计算交付已获用户[接受](execution_acceptance/final_delivery_20261016/FEE_ACCEPTANCE_20261016.json)，仍保留实际结算未核实说明；本轮全部验收通过并已提交`aec2307`（未push），当前无需再执行本Goal，工具状态以当前任务状态为准。下一研究为尚未实施的固定推荐器编码对照，需要另行确定实验范围；不重复下载或重新训练。89项训练、测试和最终GPU测速已经结束，不按早期快照重跑训练或重提GPU。详细结果见[正式报告](REPORT.md)，当前交付状态以[实施状态](IMPLEMENTATION_STATUS.md)和[项目交接](../../docs/guides/PROJECT_HANDOFF.md)顶部为准。历史03–05的SID-level成绩不能与本阶段item-level结果直接混排；Decoder适配编码仍为后续独立研究，本Goal不执行。
