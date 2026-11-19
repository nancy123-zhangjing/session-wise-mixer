# RQCD Mixer v1：实现与运行入口

2026-11-13 新增两种生成目标及落地监督，见 [TARGETS_README.md](TARGETS_README.md)。
旧 v1 配置保留；下文是原实现说明，当前正式 v1 验证频率以配置中的 R5～R35 共31个节点为准。
新版本实现不代表已启动或完成新实验；历史已运行结果应查各实验最终报告。

实现依据：docs/research/rqcd_mixer/RQCD_MIXER_FULL_DESIGN_20261111.md。
设计 SHA256：fdb73fe54c9dba4a27fc7d0d56cf049a5f23e9b6888408e12ea261a8be51f61d。

本次只交付代码和本地合成数据检查，未申请 GPU、连接启智或启动真实实验。
这套实现与 scripts/rqcd/、scripts/listwise_diffgrm/ 的旧训练器分离；只复用已验证的数据装载和数据清单读取函数。

## 固定的科学设置

| 项目 | 实现 |
|---|---|
| B0 / B1 / B2 | 无 Mixer / 层内四头交互 / 按 RQ 层块因果四头交互 |
| Tokenizer | 256→512→256→128→32；4×256×32 残差码本；镜像 Decoder |
| Mixer | LN 后加层标记，无中心编号；零初始化输出；当前层 RMS / 第一层 RMS 校准 |
| 推荐器 | 四张独立 SID embedding；T5 Encoder 2 层、Decoder 4 层；宽256、4头、FFN1024 |
| 生成 | 预测32维干净量化前向量；1000噪声等级，推理20步 DDIM，逐件生成五件列表 |
| 预训练 | 10000商品epoch，选最低目录碰撞率；保存额外 best_loss 和最终状态 |
| 联合训练 | 冻结 RQ Encoder；不用 RQ Decoder 计算联合损失，但保留其预训练状态 |
| 日程 | 先5R，再30次“10T→刷新目录→1R”；最终 R35/T300 |
| 损失 | R：denoise + 0.0003 align；T：再加 latent + VQ，VQ内部β=0.25 |
| 数据和评测 | 原KuaiRec划分、50历史、5目标、有效batch1024；11次全验证；best和R35测试 |

完整数值以 configs/rqcd_mixer/v1.json 为准。入口拒绝传入不同科学配置；设计文件内容变化也会阻止运行。
测试里的缩小网络和合成数据仅由测试直接构造，不能通过正式 CLI 覆盖协议。

## 文件分工

| 文件 | 用途 |
|---|---|
| config.py / v1.json | 科学协议与执行参数分离、版本检查 |
| tokenizer.py | RQ预训练STE、联合训练软梯度桥接、两种Mixer、固定统计量 |
| model.py | T5语义、五视图训练、历史及干净前缀缓存、DDIM |
| losses.py | 四项损失、单向目标停止梯度KL |
| catalog.py | 目录版本、合法前缀树、多商品碰撞桶、固定噪声、五件列表生成 |
| data.py | 紧凑数据读取、独立R/T持续排列和游标 |
| training.py | 两套优化器、梯度累积、完整状态恢复、目录刷新 |
| evaluation.py | 完整验证／测试、逐样本输出、进度与完整性收据 |
| run.py | 显式prepare/pretrain/train/test入口，不含远端调度 |
| benchmark.py | 一次合并的CUDA数值与吞吐检查，选择三组共同执行参数 |
| report.py | 读取已有结果，产生best/最终点表、验证曲线、分开的loss曲线 |

## H100 上怎样提高效率

1. 将样本ID数组打包一次，默认放GPU；不每个batch读取Parquet或逐行collate。CPU模式沿用固定内存和非阻塞传输。
2. 每个microbatch的历史只编码一次。五个训练视图同时处理，query仍在0/4/8/12/16位置。交叉注意力把五视图合并到query维，复用历史K/V；送给SDPA的张量为四维，不复制五份历史。
3. B1将四层分别作为四个256-token注意力问题计算。B2保留层块因果掩码；没有宣称掩码本身能省掉所有被屏蔽块的运算。
4. 推理时历史Encoder和历史K/V只计算一次；干净前缀K/V跨20个去噪步骤复用。下一件只追加四个实际落地SID；不缓存带噪query。
5. 距离使用矩阵乘法，落地、去重、指标按batch计算，避免逐样本的CPU循环。预测输出由单独线程批量压缩写入。
6. 推荐器使用BF16 autocast，主权重和优化器为FP32；码本、Mixer、量化、KL、损失和DDIM保持FP32，统计量FP64。关闭TF32，不使用FP8。
7. 有效batch保持1024。micro_r、micro_t、evaluation_batch尚未实测，不填猜测值。测速为三组选择共同配置；不在正式训练中自动改变它们。

已实现的是这些执行路径，**尚无H100实测速度或显存结论**。相对位置偏置、梯度和确定性约束可能让SDPA使用math kernel；测速会保留实际profile，不能把“用了SDPA”写成“已启用FlashAttention”。首版不启用torch.compile。

## 使用方法（以下命令仅为说明，本次未执行真实流程）

从项目根目录执行。先查看不会训练的预算：

~~~powershell
python -m scripts.rqcd_mixer.run plan
~~~

正式运行时，在远端共享盘新建独立目录；dataset指向已验证的Stage09数据，content指向原PCA256归一化数组。不要使用OPQ数组。

~~~text
python -m scripts.rqcd_mixer.run prepare --dataset DATASET --content PCA256.npy --output SHARED/prepared
python -m scripts.rqcd_mixer.run benchmark --dataset DATASET --prepared SHARED/prepared --output SHARED/benchmark
~~~

prepare只建立三组共同初始化、K-means码本、固定验证／测试噪声和绑定清单。benchmark是显式的小批CUDA更新测试，参数全部丢弃，不产生正式checkpoint。它输出：

- BENCHMARK.json：三组R/T完整有效更新、推理、刷新耗时和显存；数值误差与阈值。
- R_trace.json / T_trace.json：完整更新的实际kernel与CPU/GPU时间。
- 单次保存恢复探针：比较下一次R更新的参数、优化器、数据顺序和CUDA随机状态。
- 只有三组共同候选可用、数值和恢复检查通过，才生成 runtime.locked.json。

短测的FP32/BF16相对L2误差门槛为：预测5%、loss3%、梯度10%；缓存与全前缀预测5%。
这是可见的工程检查阈值，不是论文保证或最终推荐效果结论；CPU上的FP32等价测试使用更严格的数值容差。

默认候选microbatch为64/128/256，可通过--microbatches指定其他执行候选；每项先预热一次，再测三次完整有效更新。只需一次合并验收，不为每组另开完整诊断实验。

然后每组使用同一份runtime.locked.json。下面以B1为例，B0/B2仅改变arm和输出目录：

~~~text
python -m scripts.rqcd_mixer.run pretrain --dataset DATASET --prepared SHARED/prepared --arm B1 --runtime SHARED/benchmark/runtime.locked.json --output SHARED/B1_rq
python -m scripts.rqcd_mixer.run train --dataset DATASET --prepared SHARED/prepared --pretrained SHARED/B1_rq --arm B1 --runtime SHARED/benchmark/runtime.locked.json --output SHARED/B1
python -m scripts.rqcd_mixer.run test --dataset DATASET --prepared SHARED/prepared --pretrained SHARED/B1_rq --arm B1 --runtime SHARED/benchmark/runtime.locked.json --run SHARED/B1
python -m scripts.rqcd_mixer.run report --runs SHARED/B0 SHARED/B1 SHARED/B2 --output SHARED/report
~~~

train结束不自动扩大预算或选测试集最优点；test只读取验证best和最终节点。二者为同一checkpoint时复用测试结果。report只读取已完成结果，不运行模型。

## 恢复与验收

- pretrain和train以--resume恢复对应输出目录的last.pt。源代码、设计、数据、预训练资产、执行配置和环境必须匹配；不能静默改seed或batch后续跑。
- 保存所有正式验证节点的完整checkpoint。中断于验证时，恢复已有不可变节点再验证，避免模型权重与报告标识错配。
- 部分T块恢复时保留旧的硬SID快照及最新可训练参数，完成该块以后才刷新目录。
- checkpoint包括两套优化器、R/T步数、数据排列／游标、各随机流、目录、固定统计量及绑定。校验优化器步数和游标一致。
- last.pt按临时文件原子替换。异常写ERROR.json，保留最后完整状态，不将半个梯度更新当作成功checkpoint。
- .running防止同目录并发写；如果作业被平台强杀，确认没有存活进程后才能移除该目录的残留锁。
- validation/Rxx/progress.json只报告已完成批次；summary.json及其哈希验收通过才代表全验证完成。
- 大权重和逐样本预测默认留远端。此入口不上传、下载、删除任何原实验。

## 本地验证与依赖

CPU合成验收，不用真实数据：

~~~powershell
.runtime/stage08_restore_env/Scripts/python.exe -m unittest discover -s tests -p test_rqcd_mixer.py -v
~~~

当前CPU参考环境：Python3.12.14、PyTorch2.6.0+cpu、NumPy2.2.6、PyArrow25.0.0、Matplotlib3.10.7。T5对照测试额外使用已安装的Transformers4.53.2，仅构造随机模块，不下载预训练权重。
远端CUDA依赖尚未安装或验证；requirements文件给出版本基准，CUDA wheel需按远端驱动选择。数值验收和吞吐结果以未来实际H100环境为准。

详细一致性检查见 docs/research/rqcd_mixer/RQCD_MIXER_IMPLEMENTATION_AUDIT_20261111.md。
