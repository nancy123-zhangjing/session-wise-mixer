# 固定子集与结果复算

本文件记录评估工具的准备工作，不代表已经产生正式训练结果。

## 固定 1024 条验证样本

清单为 `validation_hash1024.json`。从完整 71,657 条验证记录中，计算
`SHA256(UTF8("stage09_validation_monitor_v1" + "\n" + sample_id))`，按哈希升序、sample_id 升序取前 1024 条。
sample_id 保留原始字符串，不做数字转换。清单保存源行索引、sample_id、选择哈希及验证文件/数据描述文件哈希。
选样只解析验证集的 sample_id 列，没有打开测试集，也没有读取标签列。

```text
python -m scripts.listwise_diffgrm.subset verify --dataset outputs/kuairec_listwise_v1/run1 --manifest experiments/09_kuairec_listwise/preparation_20261027/validation_hash1024.json
```

使用同一个已经按全量验证规则选出的 best，在同一清单上分别评估 beam32、beam256。
评估命令 `python -m scripts.listwise_diffgrm.run ... evaluate ...` 增加：

```text
--subset-manifest experiments/09_kuairec_listwise/preparation_20261027/validation_hash1024.json --beam 32
```

另一遍只把 beam 改为 256，并使用新输出目录。清单与 `--limit` 互斥；`--limit` 仍只是前 N 条的诊断入口。
子集用于搜索宽度敏感性检查，不能代替全量验证选模。CLI 仅接受完整 R 阶段结束、`update=0` 的正式 checkpoint，拒绝中间 T 状态及 preflight checkpoint。

## 独立指标复算

```text
python -m scripts.listwise_diffgrm.analysis recompute --predictions <评估目录>/per_sample.jsonl.gz --dataset outputs/kuairec_listwise_v1/run1 --output <新文件>/metric_audit.json
```

每个评估目录需保留同目录的 `summary.json`。复算只用真实商品 ID、列表分数和路径掩码，不导入正式的 `metrics.py` 或 `decoding.py`。
它逐样本复算单列表指标、候选聚合指标，并核对原记录、样本平均、用户平均及冷启动分组。
传入 `--dataset` 后，还会核对原始 split 文件哈希、sample_id、源行索引、user_id 和目标商品；全量结果必须完整覆盖原 split。
默认允许绝对误差 `2e-6`，用于兼容生产计算的 FP32 舍入；不一致时输出审计报告并返回退出码 2。

## 单 seed 的配对比较

```text
python -m scripts.listwise_diffgrm.analysis compare --baseline <J0评估目录>/per_sample.jsonl.gz --treatment <J1评估目录>/per_sample.jsonl.gz --repeats 10000 --seed 2024 --output <新文件>/J_pair_bootstrap.json
```

H0/H1 同理。比较工具要求同一训练 seed、架构、split、beam、商品目录、子集及指标口径，按 sample_id 配对并核对标签和用户。
所有指标先从预测独立复算，默认报告单列表 NDCG/Recall/HR@5，以及候选聚合 NDCG/Recall/HR@10、20、50。

- 主结果是样本平均。每次有放回抽取 U 个用户，每个被抽中的用户带入其全部样本；重复抽中则整组重复。每次统计量为抽中用户指标总和除以抽中样本数，分母会随重采样变化。
- 同时报告用户平均：先计算每个用户的样本平均，再给每个用户相同权重。
- 报告处理组减基线的绝对差、相对差和 95% percentile 区间，bootstrap 为 10,000 次，随机种子 2024，分块计算避免构造巨大重采样数组。
- 冷启动分组给出样本平均差值。用户 bootstrap 只刻画该次训练 seed 下的用户波动，不能作为跨训练 seed 稳定性的证据。

源码绑定统一使用 POSIX 相对路径，覆盖本地列表代码、实际 DiffGRM 模块和 genrec 基础模块，并拒绝误加载另一份 genrec。文档提交不会单独改变源码身份。
