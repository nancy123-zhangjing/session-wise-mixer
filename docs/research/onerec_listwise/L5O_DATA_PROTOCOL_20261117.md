# L5O数据与评测口径

执行依据：用户提供的 `L5O_EXPERIMENT_PLAN.md` v3 和 `L5O_ALIGNMENT_AND_RERUN_PLAN.md`。新版已定项优先：history=128、EV watch_ratio≥1、seed2024、不自动冻结。本轮只实现与短测，不校准SASRec、不启动正式六组。

## 数据

- 只读复用旧big_matrix去重事件表，排除无效/歧义时间戳记录；不切30分钟session，不按600秒跨度丢弃。
- 每用户按时间排序，最后**5条记录**作test、前5条作validation，再向前按stride=5构造训练块；所有split都保留重复商品。每个块仅看到此前最多128条历史，至少5条。
- 长度≥15的用户保留；15–19条只有训练前缀、没有完整监督训练块，此边界单独统计。此次真实数据不存在这类用户，不将计划中“15条保证有训练样本”的措辞当作事实。
- 训练已见商品集是验证/测试前的全部训练前缀，不包括本用户验证/测试目标。测试历史可包含其已经发生的验证记录；其他用户同期行为可能进入训练，是per-user L5O与旧全局时间切分的区别。
- 独立审计从原事件重建每条样本应取的目标、历史、watch_ratio和事件ID，检查记录守恒、缺失/重复块、冷启动和EV覆盖。双构建核对规范化样本哈希。
- 新数据目录 `outputs/kuairec_l5o_v1/records_20261117_run1`；旧数据和另一AI的distinct-item草稿run1/run2不覆盖、不删除。

## 评测

共同函数 `scripts/kuairec_data/l5o_metrics.py:slate_metrics` 输出曝光Hit/Recall/NDCG、EV Hit/Recall/NDCG及固定IDCG(5)旧NDCG。GT按集合去重，重复预测只算第一次命中；Recall分母为相关商品数，NDCG的IDCG为前min(5,相关商品数)位折扣和。

EV只在至少含一个EV商品的用户上平均，无EV用户该项为null；全体无EV则不允许据此选checkpoint。主阈值1，附带阈值2敏感性。主选点为验证EV-NDCG@5。短测沿用旧模型不屏蔽历史；未用test选择过滤策略。解码仍产出5个互异商品，训练目标允许重复，两者差异显式保留。

论文依据：[OGR实验设置](https://arxiv.org/html/2608.17613v1#S4.SS1)明确chronological、leave-five-out、history128，并区分正反馈与曝光指标；[HiGR](https://arxiv.org/html/2512.24787v1)区分相同评测类别。EV阈值、训练stride和最低长度等未明确公开的细节是本方案选择，不能声称逐位复现。SASRec达到计划区间仅是量级校准，不能证明所有协议完全一致或原低分唯一由划分造成。

现有草稿由另一轮工作产生，本次备份到 `.runtime/l5o_preexisting_20261117` 后修订；Git提交记录的是此次审查后的版本，不伪造草稿历史。
