# KuaiRec列表生成代码入口

2026-10-28更新。两种列表结构及全局Full Mixer开/关对照均已实现，四组全部使用交替联合训练。真实BGE/PSE资产、H100正确性与效率验收、两份共同5R暖身及本地独立恢复已完成；四组正式分支已启动，尚无最终对照结果。当前Job、更新计数、GPU布局及预算统一见[09执行记录](../../../experiments/09_kuairec_listwise/EXECUTION.md)，不根据下方复现示例重复启动。

远端执行根为`/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/goal_20261027_r1/`；仅部署09新资产，未迁移原项目。科学源码继续冻结在`1c423662d9a25a926328ee6466c30c0f2857b4f6`。正式资源方案为2张H100、每卡两个独立模型，无DDP；用户已批准累计75 GPU小时，准备/失败/闲置均计入。

共同资产见[PSE清单](../../../experiments/09_kuairec_listwise/assets_preparation_20261027/PSE_MANIFEST.json)及[独立审计](../../../experiments/09_kuairec_listwise/assets_preparation_20261027/PSE_ASSET_AUDIT.json)：10728件、码本`[4,256,64]`，9513个唯一SID、1505件位于碰撞桶（14.0287%），最大桶84件。四组共享同一内容向量、变换、E0、SID0和商品级评价协议；碰撞是解释结果时需保留的限制。

[R1](../../../experiments/09_kuairec_listwise/h100_preflight_r1/report.json)、[R2](../../../experiments/09_kuairec_listwise/h100_preflight_r2/report.json)已覆盖真实BF16梯度、R/T参数隔离、完整恢复、共同初始化、SID刷新及无目标列表生成。正式运行采用[R2配置](../../../experiments/09_kuairec_listwise/preparation_20261027/runtime_efficiency_r2.json)：微批256、有效batch1024、推理batch32，SHA `d8e3a40f9c7ac446dbca6f4e8b01811ee4598c0d4d07e18310f524afda9bd2bb`。正式训练不复用短验收状态。

两份暖身的71657条完整验证还已[独立复算](../../../experiments/09_kuairec_listwise/warmup_metric_recompute_20261027/RECEIPT.json)，样本、真实标签、用户及冷目标标记与源数据一致；这不替代四组最终测试和最终模型交付。

| 组 | 列表模型 | Full Mixer | PSE训练 |
|---|---|---|---|
| J0 / J1 | 五商品20码共同去噪 | 关 / 开 | R/T交替更新 |
| H0 / H1 | 外层逐商品，内层四码DiffGRM | 关 / 开 | R/T交替更新 |

配置：[base.json](../../../configs/09_kuairec_listwise/base.json)。逐项来源：[IMPLEMENTATION_REFERENCES.md](IMPLEMENTATION_REFERENCES.md)。其中1层Encoder/4层Decoder是论文及作者Beauty运行配方；列表位置、J四视图、全列表beam32和训练预算属于本轮适配，未证明最优。

## 文件分工

- `scripts/listwise_diffgrm/model.py`：直接复用原骨干；J/H forward、OCN、ST输入及soft-label CE。H仅使用此前目标前缀，生成API没有target参数。
- `assets.py / encode_text.py / prepare_assets.py / alignment.py`：固定版本中文向量、PCA/OPQ-IVF1残差PSE、E0和旧联合模块对接。
- `data.py`：读取已验收Parquet、保留item ID，在每步查询当前SID。
- `decoding.py / metrics.py`：全局列表beam、任意码位目录约束、SID碰撞消歧和真实商品级多相关指标。
- `training.py / run.py`：共享暖身、成对分叉、R/T优化器与调度、完整状态保存/恢复、验证选best、独立测试。

原`external/DiffGRM`代码和01–07结果未改动。新增适配代码在主仓库，运行绑定记录两边的源码hash；不是只记录根Git commit却遗漏引擎版本。

## 后续使用顺序

以下是复现入口说明。已有真实资产、已完成暖身和正在运行的正式分支，执行前先核对现有Job和目录，不重新编码、拟合或重提首次训练。下方路径为本地布局示例，远端实际路径由DEPLOYMENT及本轮配置绑定。

1. 只读检查配置：

```powershell
python -m scripts.listwise_diffgrm.run inspect
```

2. 编码文本：固定使用已核验的BGE-M3 revision和本地模型目录，默认CPU，未隐式选择CUDA。输出含输入文本快照、向量、原始item顺序、截断统计和hash；中断后以完全相同设置加`--resume`。

```powershell
python -m scripts.listwise_diffgrm.encode_text --dataset outputs/kuairec_listwise_v1/run1 --output outputs/kuairec_bge_m3_v1 --revision 5617a9f61b028005a4858fdac845db406aefb181 --model-path outputs/stage09_model_cache/bge-m3/5617a9f61b028005a4858fdac845db406aefb181 --model-manifest experiments/09_kuairec_listwise/assets_preparation_20261027/BGE_M3_DOWNLOAD_VERIFIED.json
```

3. 同一次拟合全目录静态内容资产，四组共同复用。数据和模型目录不进入Git。

```powershell
python -m scripts.listwise_diffgrm.prepare_assets --dataset outputs/kuairec_listwise_v1/run1 --embeddings outputs/kuairec_bge_m3_v1/dense_embeddings.npy --embedding-item-ids outputs/kuairec_bge_m3_v1/original_item_ids.npy --embedding-metadata outputs/kuairec_bge_m3_v1/metadata.json --output outputs/kuairec_listwise_assets_v1
```

4. H100 BF16验收与测速已完成，microbatch、有效batch、beam及预算已经锁定。35R/400T是本轮固定训练预算，尚不能据此断言收敛；实际启动记录见09执行记录。

5. 同架构同seed只训练一次共享5R暖身，然后分为±Mixer两个分支。示例以H、seed2024说明；J更换`--architecture`与组名，各自建立共享暖身，不拿H暖身权重当J共起点。

```powershell
python -m scripts.listwise_diffgrm.run --architecture H --seed 2024 warmup --device cuda:0 --precision bf16 --output outputs/listwise/H2024_warmup
python -m scripts.listwise_diffgrm.run --architecture H --seed 2024 train --arm H0 --shared-warmup outputs/listwise/H2024_warmup/last.pt --device cuda:0 --precision bf16 --output outputs/listwise/H0_2024
python -m scripts.listwise_diffgrm.run --architecture H --seed 2024 train --arm H1 --shared-warmup outputs/listwise/H2024_warmup/last.pt --device cuda:0 --precision bf16 --output outputs/listwise/H1_2024
```

恢复同一分支使用同配置、同目录加`--resume .../last.pt`，不能把另一个分支当resume。断点含模型、PSE/当前SID、两个Adam、R学习率时钟、T退火时钟、RNG、phase/update游标。数据顺序独立按phase确定，尾批不丢弃。每100次更新及阶段结束保存；突然中断可能重做上个断点之后的更新，`attempts.jsonl`区分尝试，断点计数是权威记录。`.running`独占锁避免同目录双跑，异常进程遗留锁需要先确认进程已结束再处理。

6. 只按完整验证集单列表NDCG@5选best，平局较早者；正式测试使用该best且beam32。固定R35状态保留在last。用同一best做验证beam256敏感性时可显式`--beam 256 --limit N`，它不重训、不回写best，结果标明是否全验证集。

```powershell
python -m scripts.listwise_diffgrm.run --architecture H --seed 2024 evaluate --checkpoint outputs/listwise/H1_2024/best.pt --split test --device cuda:0 --precision bf16 --output outputs/listwise/H1_2024_test
```

`--precision`用于恢复训练配置绑定；当前推荐解码统一FP32，TF32关闭并记录。推理的20个逻辑填码阶段不是一次前向；J历史编码只做一次，H每换商品前缀重算一次。

## 评估文件怎样读

- `list/ndcg@5`、`list/recall@5`、`list/hr@5`：最高分单份五件列表，分母按五个真实相关商品计算。
- `aggregate/*@10/20/50`：最多32份完整列表按分数展开去重后的候选指标，与单列表质量分开。
- 同时保存有效完整列表率、缺项/重复/非法率、候选覆盖、用户平均与冷目标子组。
- `per_sample.jsonl.gz`保存真实标签、候选item列表、score、beam/complete mask及各指标，可独立复算。一个SID不按整bucket算命中，历史重看商品保留。
- 本轮置信区间来自同一训练seed下的用户配对bootstrap；它不能替代多seed稳定性证据。

## 本轮验收边界

**55项CPU检查全部通过**，覆盖K=1原模型兼容、四层Decoder BF16梯度、前缀防泄漏、共同E0、推荐梯度到码本、两阶段实际权重更新、真实噪声更新的精确恢复、全局beam/目录碰撞/指标，以及两路线完整微型暖身→两分支→恢复→测试流程。正式验证集只读检查确认10728件商品、71657条样本及历史50/目标5接口。[机器验收收据](../../../experiments/09_kuairec_listwise/code_acceptance_20261027/CPU_ACCEPTANCE.json)、[完整CPU日志](../../../experiments/09_kuairec_listwise/code_acceptance_20261027/cpu_tests.log)。

后续91项CPU回归、独立源码解包回归、远端Linux回归、14项预算工具检查和两轮四臂H100验收已通过。两份5R共同暖身已完成，四个正式分支由对应暖身分叉，四次ready及两对initial完整状态一致性通过；正式训练仍在运行，尚不能产生Mixer收益结论。当前状态见09执行记录。训练/代表checkpoint恢复仍锁定`1c423662d9a25a926328ee6466c30c0f2857b4f6`；本地新版analysis只用于离线复算，不同步改变训练源码binding。
