# 五组实验实现记录

设计依据：`FIVE_ARM_DESIGN_20261113.md`。本轮实现和CPU合成检查，不启动真实实验。

## CF-M0

`scripts.session_wise_ar.cf`仅路由到冻结CF引擎的`--arm B0`，旧科学配置、模型代码与运行配置不改。入口核对原源码、共同prepared/initial、教师和验证/测试噪声SHA，输出必须独立于共同prepared目录。

先查看命令（不训练）：

```powershell
python -m scripts.session_wise_ar.cf plan --dataset DATASET --prepared ORIGINAL_CF_PREPARED --output NEW_CF_M0_ROOT
```

未来经授权可将`plan`替换为`pretrain`、`train`、`test`，各阶段显式调用，不自动申请资源、不重训教师。`--resume`只用于已有阶段的完整断点。

已验证边界：CF路由、禁止覆盖共同父目录、冻结源码逐文件SHA。没有访问远端prepared或执行真实预训练。
## AR 模型与数据适配

`scripts/session_wise_ar/{config,data,model,upstream}.py` 和 `configs/session_wise_ar/` 固定设计文档哈希、DIGER提交及逐文件SHA。DIGER模型/量化器由隔离模块加载原文件；外部源码不改写。缺少源码时用 `python -m scripts.session_wise_ar.upstream --fetch` 下载固定版本，下载不运行实验。

每个session展开五个任务；真实历史与已知列表前缀紧凑追加到Encoder，当前商品内四码位自回归。TIGER独立全局词表、用户hash、4+4层T5；DIGER保留原6+6层T5及分码位embedding/输出头。TIGER无已核实完整官方训练入口，本实现是论文配置适配，不宣称官方逐字复现。

CPU人工张量检查覆盖前缀不泄漏、两种头的前向/逐码位推理、TIGER的128宽度与6×64注意力投影。该检查不代表真实数据/GPU验收。

## Tokenizer 与 Mixer 接入

`tokenizer.py`：TIGER使用设计中明确的维度求和/三层求和RQ损失；DIGER直接调用固定源码的预训练和FrQUD量化器。通过PyTorch参数化向原量化器提供有效中心，原源码每处 `embedding.weight` 和中心查表均看到同一修正结果。RQ初始化在未参数化的原模块中完成，避免k-means写入临时有效中心。

层内Mixer直接复用当前B1类：共享Q/K/V、4头、无编号标记、按层RMS比例缩放、零初始化输出。共同RQ父状态和推荐器父状态分开保存，Mixer使用独立随机流。

CPU人工张量已核对零Mixer修正时两组前向一致；推荐损失单独反传能更新DIGER基础中心和Mixer输出，RQ Encoder保持冻结。联合损失保留原semantic+suffix、latent重建及VQ三部分；不添加CF落地损失。

## 阶段入口与恢复

`run.py`提供 `plan / prepare / pretrain / train / test` 五个显式阶段，没有平台提交或自动启动功能。`runtime.py`保存模型、优化器、学习率调度器、随机状态、数据排列/游标、SID缓存、FrQUD频率EMA和验证选点状态。验证前先固化checkpoint；中断恢复复用同一checkpoint和完成的预测文件，不重复刷新SID。预训练与推荐阶段均拒绝覆盖数据或父资产目录。

以下仅为未来运行命令模板，路径应替换为当次独立远端目录。当前没有执行这些真实数据命令。

```powershell
# 只查看预算，不加载数据或申请资源
python -m scripts.session_wise_ar.run --arm AR-T0 plan
python -m scripts.session_wise_ar.run --arm AR-D0 plan

# 一系列只创建一个共同父状态；T0/T1共享T_PARENT，D0/D1共享D_PARENT
python -m scripts.session_wise_ar.run --arm AR-T0 prepare --dataset DATA --content PCA256.npy --output T_PARENT
python -m scripts.session_wise_ar.run --arm AR-D0 prepare --dataset DATA --content PCA256.npy --output D_PARENT

# 例：AR-D1；其他AR组替换arm及独立输出路径
python -m scripts.session_wise_ar.run --arm AR-D1 pretrain --dataset DATA --content PCA256.npy --prepared D_PARENT --output D1_PRE
python -m scripts.session_wise_ar.run --arm AR-D1 train --dataset DATA --content PCA256.npy --prepared D_PARENT --pretrain D1_PRE --output D1_TRAIN
python -m scripts.session_wise_ar.run --arm AR-D1 test --dataset DATA --content PCA256.npy --prepared D_PARENT --pretrain D1_PRE --training D1_TRAIN --output D1_TEST
```

`pretrain/train --resume`只恢复对应阶段的 `last.pt`，配置、来源、父状态和执行环境必须一致。`.running`是单写者标记；异常断电留下标记时须先确认原进程确已退出再处理，程序不擅自解除活跃锁。`test`对best/实际final评分，同节点复用预测。不要从旧完整checkpoint中摘权重冒充本次共同初始化。

## 配置落实与必要补全

| 项目 | 实现 |
|---|---|
| AR输入 | 现有PCA256内容；3层RQ＋消歧suffix；50历史＋最多4前缀 |
| batch/epoch | 256个下一商品任务；每session逻辑展开5项；尾batch保留；四AR共用任务排列规则 |
| TIGER | RQ Adagrad 0.4、20,000目录epoch选最终点；4+4层T5；Adafactor外部0.01/平方根衰减；200,000推荐更新 |
| DIGER | 原预训练10,000目录epoch、最低碰撞率；6+6层T5；逐batch双AdamW；120轮上限、patience15；不套CF日程 |
| DIGER预训练裁剪 | 原 `scripts/rqvae/trainer.py` 的norm clip1一并保留（设计未另列此默认） |
| Mixer | 仅层内4头；TIGER推荐期冻结、DIGER推荐期更新；所有量化/刷新路径都用有效中心 |
| AR推理 | 每件内部beam20；固定4码位，不增加EOS损失；原头归一化后合法路径过滤；每件提交后重新编码历史＋前缀 |
| CF-M0 | 只选择原引擎B0；不改教师、损失、日程、噪声、有效batch或BF16执行配置 |
| 选点 | 每完整session遍历全验证；TIGER额外最终部分遍历；验证NDCG@5择优，同分保留较早；测试不参与选点 |

TIGER完整官方训练入口仍未核实：MLP初始化采用现有Xavier-uniform/零bias，k-means固定random_state42/n_init10；HF Adafactor是已明确的候选补全，不声称与T5X优化器逐位一致。DIGER除会话任务、输入维度、共同seed、Mixer及共同列表评测外复用固定源码与核心配置；预训练无约束k-means，保留原初始化实现。

## 效率与验证边界

训练只展开索引，不复制五份数据；历史打包一次，批量查表；DIGER每batch去重后仅量化出现的商品；Mixer每次量化只计算一份有效码本。推理用目录树和张量beam，不逐用户扫描全目录。当前商品最多4个码位，保留原DIGER完整前缀解码，不额外引入尚未核验的KV缓存。AR先FP32、关闭TF32；CF保持已验收执行设置。

依赖候选见 `configs/session_wise_ar/requirements.txt`（原DIGER依赖版本；另加Parquet读取库）。本地实际检查为Python3.11、PyTorch2.6.0+cpu；Transformers4.53.2及隔离安装的原指定4.57.1均做CPU检查，未修改旧环境。尚未验证原torch2.5.1/CUDA环境，也未测H100吞吐。因此不提供未经实测的训练耗时，不把CPU逐位恢复写成GPU数值验收。

CPU检查命令：

```powershell
$env:PYTHONPATH = "$PWD\.runtime\session_ar_deps;$PWD"
& .runtime/stage08_restore_env/Scripts/python.exe -m unittest discover -s tests -p 'test_session_ar_*.py' -v
```

合成端到端检查覆盖四AR组的预训练、训练、验证选点、最终测试；检查在epoch边界、验证写出后中断的精确恢复。人工小目录的k-means可能报告聚类数少于请求数，这是小型fixture现象，不能当作真实目录诊断结论。CF只复用既有检查并验证路由/来源，无真实数据推理、训练或资源申请。

## 结果汇总入口

`python -m scripts.session_wise_ar.report --manifest RUNS.json --output REPORT_ROOT` 只读取已完成结果；可加 `--plots` 输出验证曲线和AR分项loss图（需要matplotlib）。`RUNS.json`是列表，每条含 `arm` 和 `train` 绝对路径；AR另给 `test`，可给 `pretrain`。CF直接读取原引擎的 `TRAINING_COMPLETE/BEST/TEST_COMPLETE`，不改原结果格式。

输出best/实际final的验证和测试NDCG@5、Recall@5、HR@5，全部验证节点、实际更新数、已记录的分阶段耗时；报告列明缺少的组，不把局部结果称作五组完成。检查组内共同父状态/教师、跨组数据划分，以及验证/测试checkpoint的SHA一致。CF只接受CF目标的B0/B1，拒绝用旧内容或hybrid结果冒充。

图横轴为session等价遍历量及已有日志中的阶段墙钟时间。阶段时间不含未记录的排队、空闲和部分I/O，不能直接称GPU计费时间；未来运行后的平台收据另行补充。参数总数表示checkpoint保留的模型参数，包含冻结模块；旧CF收据未记录的字段留空，不捏造。

本次检查汇总见 `CODE_REVIEW_20261113.md`。源码和参数已完成本地检查，不代表五组训练已完成。
