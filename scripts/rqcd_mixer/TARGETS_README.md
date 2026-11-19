# 纯协同 / 内容＋协同目标 v2

两种目标的实现协议见 `docs/research/rqcd_mixer/RQCD_TARGETS_DESIGN_20261113.md`。本次只实现和检查代码，未启动真实训练、验证或平台资源。

| 版本 | 科学配置 | RQ输入/预训练 | 联合推荐监督 |
|---|---|---|---|
| CF | configs/rqcd_mixer/v2_cf.json | 共用SASRec商品向量/整体尺度，重建协同向量＋VQ | denoise MSE＋0.1×RQ landing |
| Hybrid | configs/rqcd_mixer/v2_hybrid.json | 原内容输入，重建内容＋VQ＋0.001×协同对齐 | 同上 |

两组生成量化前32维 z，RQ Encoder 在联合训练中冻结；T只更新码本/Mixer，R只更新推荐器。RQ Decoder 和 Hybrid 对齐投影仅预训练使用。latent/VQ辅助项保持原权重1，旧KL不与新损失叠加。原网络、噪声、全局batch、学习率、R35/T300及31个验证节点不变。

## 实际入口（说明示例，本次未执行）

所有命令从项目根目录运行。`--config` 放在子命令前。`RUNTIME.json` 是显式给定的共同执行配置，字段沿用 runtime.example.json；不在代码内猜测/自动调整 microbatch。

先训练一次共用教师；两组必须使用同一个 `TEACHER/items.pt`：

```text
python -m scripts.rqcd_mixer.teacher --config configs/rqcd_mixer/v2_cf.json --runtime RUNTIME.json --dataset DATASET --output TEACHER
```

教师：256维、2层4头、FFN1024、dropout0.2，Adam lr1e-3、batch256、最多200轮、验证10轮不提升停止。训练样本展开为五个因果下一商品视图；只读取训练集更新，验证集第一件目标选best，不读取测试集。导出固定商品表、原ID和训练出现标记。`train_supported`表示有训练交互支持，未出现商品可能只有全目录交叉熵的负类梯度，不把这些行解释为可靠协同表示。使用 `--resume` 恢复输出目录中的 last.pt。

纯协同组（首轮两目标都建议 B1；未自动创建 B0/B2 的额外实验）：

```text
python -m scripts.rqcd_mixer.run --config configs/rqcd_mixer/v2_cf.json plan
python -m scripts.rqcd_mixer.run --config configs/rqcd_mixer/v2_cf.json prepare --runtime RUNTIME.json --dataset DATASET --content PCA256.npy --teacher TEACHER/items.pt --output CF_PREPARED
python -m scripts.rqcd_mixer.run --config configs/rqcd_mixer/v2_cf.json pretrain --runtime RUNTIME.json --dataset DATASET --prepared CF_PREPARED --arm B1 --output CF_RQ
python -m scripts.rqcd_mixer.run --config configs/rqcd_mixer/v2_cf.json train --runtime RUNTIME.json --dataset DATASET --prepared CF_PREPARED --pretrained CF_RQ --arm B1 --output CF_RUN
python -m scripts.rqcd_mixer.run --config configs/rqcd_mixer/v2_cf.json test --runtime RUNTIME.json --dataset DATASET --prepared CF_PREPARED --pretrained CF_RQ --arm B1 --run CF_RUN
```

Hybrid 使用 `v2_hybrid.json`，输出分别用 `HYBRID_PREPARED/HYBRID_RQ/HYBRID_RUN`，教师路径保持不变。CF准备入口仍要求原内容资产用于数据协议绑定，但内容不进入CF Encoder或重建目标。两组pretrain、train需要从各自新初始化开始，不能接旧v1 checkpoint。

已有 benchmark 子命令也支持v2，数值检查计算新落地损失；需要授权后显式调用，本次没有运行H100短测。不要将旧版吞吐或旧短测通过自动视为新损失GPU验证结果。

完成后可合并报告，不会把两个 B1 混在一起：

```text
python -m scripts.rqcd_mixer.run --config configs/rqcd_mixer/v2_cf.json report --runs CF_RUN HYBRID_RUN --output TARGET_COMPARISON
```

报告以 `cf/B1`、`hybrid/B1` 标识，拒绝把不同教师资产当作这两目标的共同教师对照。输出分开的denoise/path/bucket/landing/latent/VQ曲线和预训练CF对齐曲线。

## 关键实现

- teacher.py：共享协同教师、训练前缀、验证选点、完整状态恢复和目录导出。
- targets.py：教师/数据/原ID/SHA绑定；CF整体缩放；Hybrid同内容负例排除与冻结教师对齐。
- tokenizer.py、Pretrainer：保留原STE；对齐梯度到Encoder，码本/Mixer通过VQ训练；固定目录评估目标选best_objective。
- losses.py：逐层真实前缀，合法分支与列表去重，四层CE先求和再平均；桶内真实商品CE。单元素条件概率为1、该项损失为0。
- JointTrainer：动态记录新损失项；保留原梯度桥、T块内旧SID与块末刷新，支持部分T块恢复。
- run.py：prepared额外保存冻结目标输入；检查teacher和目标资产SHA；明确读取best_objective而非旧best_collision。

预训练对齐只在batch内计算；与当前商品内容完全相同的其他商品不做负例，但不会因此产生区分相同内容商品的新信息。旧内容限制仍应作为实验边界。

落地CE是训练时真实前缀条件下的代理监督，不是自由生成列表风险的精确表达。bucket尺度取 `2×32×固定潜空间scale²`，再乘温度0.1。合法树/统计量/教师z不反传，T阶段有效中心与预测向量可以反传。

## 本地检查

```text
.runtime/stage08_restore_env/Scripts/python.exe -X utf8 -B -m unittest discover -s tests -p test_rqcd_targets.py
.runtime/stage08_restore_env/Scripts/python.exe -X utf8 -B -m unittest discover -s tests -p test_rqcd_mixer.py
```

均使用小型合成数据；不需要真实数据、远端checkpoint或GPU。完整代码绑定会随实现更新改变；历史实验恢复应使用其原归档源码，不绕过绑定校验。
