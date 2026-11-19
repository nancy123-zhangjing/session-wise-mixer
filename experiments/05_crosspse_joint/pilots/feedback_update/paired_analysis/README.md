# 推荐反馈：逐样本配对诊断

2026-10-13 已完成三组完整验证集重放和本地独立重算，全部原指标精确复现。结果与解释见 [RESULTS.md](RESULTS.md)：目前没有额外净收益；目标子集的局部正差异不足以证明主动重编码有效。以下保留本轮固定分析方案和操作说明。

## 要回答的问题

三组完整训练已结束。相同 Decoder 恢复第 3 轮中，推荐反馈组没有超过仅重构更新组；但总体差异很小，不能只看平均分判断哪些样本受益或受损。

本次复用 `feedback_pilot_20261012_2150` 的三份 `decoder_3` 快照，只做推理和统计，不重新训练、不刷新 SID、不挑选其他轮数、不运行测试集。所有组保留 Cross Mixer，因此仍不判断 Mixer 的独立贡献。

## 固定分析方案

1. 加载原完整模型、官方 SID 和评估前 RNG；原样调用 `Trainer.evaluate`，保留诊断、生成和评分的执行顺序。
2. 总体各项指标须与原记录相差不超过 `1e-7`。若不复现，停止分组结论，保存差异供排查。
3. 验证三组的 22,363 个样本顺序、目标商品、实际截断历史完全相同；逐样本保存原 evaluator 的 float32 分数、官方目标 SID、未去重的 Top-10 预测和首次命中排名。
4. 取两个更新组相对共同起点发生正式 SID 变化的商品并集，所有模型按同一集合分组：目标变化、仅历史包含变化商品、两者都没有。
5. 主比较为 `cross_rec − cross_recon`，分别相对 Frozen 为辅助比较。统计新增/丢失/净命中及 HR/NDCG 差异，核验分组加权回到总体。
6. 另按共同起点 SID 碰撞，以及三组任一目标 SID 有碰撞的共同掩码分层。不从 SID 的任意代表商品反推真实目标。

分组是在观察范围内定位变化，不是新的随机化因果对照。编码没变的样本也可能受到全局 Decoder 参数变化影响。主指标仍是单 seed、SID-level 验证结果，不改称严格 item-level。

## 实现与保存

- 推理：`scripts/analysis/evaluate_feedback_paired.py`。
- 标准库重算与报告：`scripts/analysis/feedback_paired_analysis.py`。
- 测试：`tests/test_feedback_paired_evaluator.py`、`tests/test_feedback_paired_analysis.py`。
- 新输出：`outputs/analysis/stage05_feedback/<唯一名称>/`，不得覆盖旧目录。
- 可写数据缓存放在当前 Job 的临时目录，结束后释放；共享盘只保留轻量预测、核验和诊断，不重复保存权重。
- 启动与同步收据：`outputs/pilots_delivery/feedback_paired_<时间>/`。

目录规范化说明：本轮已启动的 `45ffc26` 版本实际执行 `scripts/analyze/` 下的脚本；后续本地源码仅迁入现有 `scripts/analysis/` 分类，脚本字节和计算逻辑不变。不修改正在运行的 SSD 代码、输出或历史收据；回放本轮实际运行时仍按 `45ffc26` 的原路径。

2026-10-13 的 HDD 实际分配探针报配额不足，用户明确授权本轮轻量评估使用同项目 SSD。该路径只供本次分析，不搬动历史资产：`/inspire/ssd/project/project-public/<inspire-account>/nancy/experiment1_analysis/`。其中 `code/` 为独立 Git 版本锁定的推理代码副本，原 HDD 的模型、数据和运行源码只读；输出保存到该 SSD 根下的 `outputs/analysis/stage05_feedback/<唯一名称>/`，收据保存到 `delivery/<唯一名称>/`。

这种运行使用显式的 `--project-root <HDD主项目>`、`--output-base <SSD专用分析父目录>` 和 `--output <SSD唯一输出目录>`。推理程序同时记录实际执行的 SSD Git HEAD 和 HDD 引擎/运行源码哈希。它不是已完成 HDD 主仓库同步的证明，HDD 配额恢复后再按正常流程同步；不能把 SSD 诊断目录当作完整项目迁移。

服务器项目根目录（已激活 grm-exp）运行：

```bash
python -B tests/test_feedback_paired_analysis.py
python -B tests/test_feedback_paired_evaluator.py
python -u -B scripts/analysis/evaluate_feedback_paired.py \
  --run-dir outputs/pilots/stage05_feedback/feedback_pilot_20261012_2150 \
  --output outputs/analysis/stage05_feedback/<新名称>
```

先在单 H100 上用 `--smoke` 检查前三组各两个 batch，smoke 不用于科研结论。正式 Job 使用另一新名称且不加 `--smoke`。运行状态、实际源码版本和是否复现，以原始收据为准，不将计划写成完成。

输出包含三份 `records.jsonl.gz`、`catalog_ids.json`、`paired_summary.json`、汇总 `README.md`、重放误差、输入/源码哈希和 `COMPLETE.json`。原快照留在服务器；这些轻量结果不是权重备份。
