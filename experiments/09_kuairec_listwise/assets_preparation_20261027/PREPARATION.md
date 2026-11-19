# Stage09 资产准备记录

2026-10-27。此目录只记录共同资产准备，不是四组推荐训练结果。

## 最新状态

正式共同暖身已于北京时间20:35左右RUNNING：`nancy-stage09-warmup-j-20261027-r1`在`qb-prod-gpu002`、平台上限2.15小时，`nancy-stage09-warmup-h-20261027-r1`在`qb-prod-gpu146`、上限2.4小时；各1张H100、无DDP。见`../formal_execution_20261027/STATUS_START.json`及同目录`J_CREATE.json`、`H_CREATE.json`、`j_LOG_START.json`、`h_LOG_START.json`。TRAINING_START.log快照已确认J完成113次R更新、H完成59次R更新，均在epoch1、T=0；尚无完整暖身结果；四分支尚未启动。

累计75 GPU小时已获批准，见[授权收据](../preparation_20261027/BUDGET_APPROVAL.json)和[资源计划](../h100_preflight_r2/APPROVED_RESOURCE_PLAN.json)；J/H暖身已提交且RUNNING；TRAINING_START.log快照已确认J完成113次R更新、H完成59次R更新，均在epoch1、T=0；尚无完整暖身结果。

BGE远端四分片、整tar和11个dense文件已全部回读校验PASS，见`BGE_REMOTE_TRANSFER_VERIFIED.json`（SHA256：`65918bcdcce487cc895b4c0b3d68c9ae8c375a03b39ce63368038dc315685137`）。远端32条真实文本CPU probe编码耗时5.4594659秒、5.86138件/秒，带30%余量的CPU全量估算40.4383分钟，见`ENCODER_CPU_PROBE.json`及`../preparation_20261027/REMOTE_ENCODER_CPU_PROBE.log`。

内容编码Job `nancy-stage09-bge-encode-20261027-r1`已SUCCEEDED，日志确认10728件、exit0，输出位于本轮远端根`assets/bge_m3_cuda_v1/`。节点`qb-prod-gpu050`、1张H100 80GB，仅冻结BGE做向量推理，不是推荐器训练。平台Scheduled北京时间2026-10-27 18:02:14至SuccessfulDeletePod 18:04:29共135秒；后续H100验收R1占266秒、R2占255秒，三个准备Job均已释放。已终态的准备作业合计656 GPU秒（0.182222 GPU小时），准备阶段原7200秒上限下剩6544秒（历史）。最新已批准累计75 GPU小时（270000秒），包含已终态作业656秒，269344秒为两暖身分配前余额；当前新增占用和实际余额待平台终态事件核算。编码状态、事件及日志见`../preparation_20261027/BGE_ENCODER_JOB_STATUS.json`、`BGE_ENCODER_JOB_EVENTS.json`及`BGE_ENCODER_JOB_LOGS.json`，验收事件见同目录`H100_PREFLIGHT_JOB_EVENTS.json`及`H100_PREFLIGHT_R2_JOB_EVENTS.json`。

CPU `prepare_assets`及`asset_diagnostics`已完成并通过工程审计，见`PSE_MANIFEST.json`及`PSE_ASSET_AUDIT.json`：`toy=false`、10728件、码本`[4,256,64]`，数值有限，最近邻SID逐项一致，重构误差独立复算一致。9513个唯一SID；1505件商品位于多商品碰撞桶（14.0287%），共290个碰撞桶、最大桶84件。这是商品可区分性的限制，不是工程失败；资产审计通过不等于推荐质量通过。

R1/R2四臂及supervisor均passed，见`../h100_preflight_r1/report.json`、`supervisor.json`及`../h100_preflight_r2/report.json`、`supervisor.json`，覆盖真实BF16梯度、R/T隔离、恢复、共同初始化、SID刷新及无目标五商品生成。R1的`METRIC_RECOMPUTE.json`仅为每臂4条beam2样本的商品指标独立复算，不是正式推荐质量。R2使用`../preparation_20261027/runtime_efficiency_r2.json`，微批256、有效batch1024、推理batch32；科学模型、beam、评价定义不变，正式训练不能接短验收状态。

共同生成资产18文件、94,105,600字节tar已在`outputs/stage09_restore/goal_20261027_r1/`本地独立恢复PASS，见`GENERATED_ASSETS_LOCAL_RESTORE.json`（SHA256：`3843450250a3a164acb0bb446a288653334855233875b8582ec07a3e3a81a992`）；逐文件hash、NPY/Torch一致、最近邻SID及误差独立复算通过，不含正式模型。

`../h100_preflight_r2/BUDGET_ESTIMATE.json`估后续基础58.1855 GPU小时，含25%余量和已用量累计72.9141 GPU小时。已批准2卡J/H共同暖身、最多4卡四个独立分支，每模型单卡、无DDP，后续墙钟约15.868/19.835小时（基础/含余量，不含排队）；按1点/GPU小时仅作未结算估算。用户已批准累计75 GPU小时上限，J/H暖身已提交且RUNNING；TRAINING_START.log快照已确认J完成113次R更新、H完成59次R更新，均在epoch1、T=0；尚无完整暖身结果。远端训练及代表checkpoint恢复锁定`1c423662d9a25a926328ee6466c30c0f2857b4f6`；本地`83a1b9c`新增analysis仅离线使用，不改变远端训练源码binding。

## 已核实

- 公开模型固定为 `BAAI/bge-m3@5617a9f61b028005a4858fdac845db406aefb181`。模型卡及API来自 Hugging Face 官方仓库；manifest绑定实际Git/LFS文件hash。
- dense所需11件文件合计 **2,293,331,623 bytes**，不下载ONNX、稀疏头或ColBERT头。`pytorch_model.bin`为2,271,145,830 bytes，其SHA256为 `b5e0ce3470abf5ef3831aa1bd5553b486803e83251590ab7ff35a117cf6aad38`。
- 本地模型目录：`outputs/stage09_model_cache/bge-m3/5617a9f61b028005a4858fdac845db406aefb181/`。11件文件已完整下载并通过Git/LFS校验，完整证明为`BGE_M3_DOWNLOAD_VERIFIED.json`；原下载日志保留在`outputs/stage09_asset_preparation_20261027/model_download.log`。
- `BGE_M3_DOWNLOAD_MANIFEST.json`是早期版本和小配置取证；下载完成应引用后续`BGE_M3_DOWNLOAD_VERIFIED.json`，不能把下载成功等同于远端同步、编码或PSE拟合成功。
- 独立安装 `faiss-cpu==1.11.0.post1` 到`.runtime/stage09_dependencies`，使用`--target --no-deps`；未升级现有环境。模块报告版本为1.11.0，兼容现有Python3.12.14、NumPy2.2.6。见`DEPENDENCIES.json`。
- 真实CPU小矩阵检查：1024个合成32维向量先PCA16，再真实`OPQ4,IVF1,PQ4x8`；新入口与原`fit_original_pse`的残差、中心、SID、粗中心逐位完全相同，6.94秒完成。见`FAISS_CPU_SMOKE.json`。小样本不足FAISS建议的9984件产生警告是预期现象；正式10728件超过此建议。**这不是正式KuaiRec拟合。**
- 真实tokenizer及10件非权重文件已核验，见`TOKENIZER_CPU_SMOKE.json`。正式目录10728件只做token计数：4件空文本、3件超过512，均值52.94，p50/p95/p99为47/111/151，最长1252。见`CATALOG_TOKEN_LENGTH_AUDIT.json`。文本模板未改变，未据此删商品。

## 后续命令

所有命令在项目根目录执行。以下保留本地复现入口；当前远端全量编码、CPU共同PCA/PSE和独立审计均已完成，不应据此重复启动相同工作。

Windows独立CPU环境：

```powershell
$env:PYTHONPATH = (Resolve-Path .runtime/stage09_dependencies).Path
```

1. 模型已下载。若运行主机满足内存门槛，先对32条真实文本做4线程CPU测速。只读模型及文本，输出在独立诊断目录；该命令不会自动启动全量编码。本机最近可用内存仅1.33GiB，未执行该探针，见`LOCAL_CPU_MEMORY_GATE.json`；以下是本地路径示例，远端须按部署收据绑定实际资产路径。

```powershell
.runtime/stage08_restore_env/Scripts/python.exe -m scripts.listwise_diffgrm.encoder_cpu_probe --dataset outputs/kuairec_listwise_v1/run1 --model-path outputs/stage09_model_cache/bge-m3/5617a9f61b028005a4858fdac845db406aefb181 --manifest experiments/09_kuairec_listwise/assets_preparation_20261027/BGE_M3_DOWNLOAD_VERIFIED.json --output outputs/stage09_asset_preparation_20261027/encoder_cpu_probe_r1
```

测速包含模型加载时间和按实际文本token长度分层选择的32件，给出含30%余量的全量估算。加载前至少需要6 GiB可用内存；本机内存在准备过程中一度低于此值，执行时需重新检查。

2. 若CPU预计不超过90分钟、内存合理且代码已锁定，再启动一次全目录编码。只有首次运行使用新输出目录；中断恢复在完全相同设置后追加`--resume`。

```powershell
.runtime/stage08_restore_env/Scripts/python.exe -m scripts.listwise_diffgrm.encode_text --dataset outputs/kuairec_listwise_v1/run1 --output outputs/kuairec_bge_m3_v1 --revision 5617a9f61b028005a4858fdac845db406aefb181 --model-path outputs/stage09_model_cache/bge-m3/5617a9f61b028005a4858fdac845db406aefb181 --model-manifest experiments/09_kuairec_listwise/assets_preparation_20261027/BGE_M3_DOWNLOAD_VERIFIED.json --device cpu --num-threads 4 --batch-size 8 --chunk-size 128
```

本地snapshot及manifest的revision和全部实际文件hash均会核验，包含权重；记录加载路径但不把机器绝对目录放入语义fingerprint，因此可校验迁移。

3. CPU拟合一次共同PCA/PSE，四组共用。

```powershell
.runtime/stage08_restore_env/Scripts/python.exe -m scripts.listwise_diffgrm.prepare_assets --dataset outputs/kuairec_listwise_v1/run1 --embeddings outputs/kuairec_bge_m3_v1/dense_embeddings.npy --embedding-item-ids outputs/kuairec_bge_m3_v1/original_item_ids.npy --embedding-metadata outputs/kuairec_bge_m3_v1/metadata.json --output outputs/kuairec_listwise_assets_v1 --seed 2024 --omp-threads 4
```

冻结PCA256白化、后L2归一化、OPQ旋转、IVF1粗中心与E0。seed2024用于PCA/初始Mixer；FAISS聚类沿用旧factory默认随机设置，不声称全部聚类随机种子为2024。

4. 独立重算量化和碰撞诊断。

```powershell
.runtime/stage08_restore_env/Scripts/python.exe -m scripts.listwise_diffgrm.asset_diagnostics --assets outputs/kuairec_listwise_assets_v1 --dataset outputs/kuairec_listwise_v1/run1 --output experiments/09_kuairec_listwise/assets_preparation_20261027/PSE_ASSET_AUDIT.json
```

报告包括PCA保留/丢弃方差、归一化dense向量的逆投影误差、OPQ残差重构误差和cosine、逐码位使用率/熵、完整SID与前缀碰撞、行为训练见过/未见过商品分组。解释方差衡量数值方差保留，不能直接当作语义或推荐质量保留比例。

## 当前边界

已通过项目包括本地及远端完整BGE权重校验、CPU FAISS兼容性、合成矩阵语义对照、真实目录token长度检查、远端32文本CPU probe，以及10728件正式PCA/PSE资产和独立审计。两轮真实H100四臂验收及共同资产本地独立恢复也已通过，三个GPU Job均释放。累计75 GPU小时预算已批准，J/H暖身已提交且RUNNING；TRAINING_START.log快照已确认J完成113次R更新、H完成59次R更新，均在epoch1、T=0；尚无完整暖身结果；工程验收、微型指标复算不能替代正式推荐质量结果。上方本地命令仅为复现入口，不能据此重复启动已有工作。

## 远端写入范围更新

原主项目写入报配额错误且清理06/07后仍未恢复。用户最新同意本轮09使用以下独立global_user目录，覆盖此前仅写原项目的限制；仅部署09新资产，不迁移原项目：

```text
/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/goal_20261027_r1
```

该新根于2026-10-27 08:35:18 UTC通过1MiB写入、fsync及回读SHA，见`../preparation_20261027/GLOBAL_USER_WRITE_PROBE.json`。661708800字节输入包和BGE权重均已完成远端部署与校验；单H100内容编码已实际读写新根，完整Goal容量仍待验收。CPU Notebook `nancy-stage09-prep-20261027-r1`只是本轮准备入口，不访问或改动`env-zzl`。

后续源码副本、模型、数据、依赖安装、缓存及临时输出均放在上述本轮目录内；已有环境只读复用，并关闭向其写入字节码缓存。路径决策不改变KuaiRec划分、固定模型revision或短验收GPU预算。旧传输manifest/归档保持原字节，由新的DEPLOYMENT收据绑定实际部署路径；不把1MiB探针当作全部资产或完整Goal容量验收。
