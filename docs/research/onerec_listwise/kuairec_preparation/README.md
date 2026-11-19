# KuaiRec 五商品消费续接数据

本轮只准备数据。任务是用已观察历史预测同一人工会话内接下来的连续五件消费商品，不恢复真实请求，不把每件已播放商品解释成强偏好正例。两种后续列表生成架构共用这份数据。

## 固定协议

配置为 `configs/kuairec_listwise_v1.json`。使用完整 `big_matrix`，不拼入 `small_matrix`。按原始记录顺序生成从1开始的 `source_event_id`，物理CSV行号为该编号加1。微秒整数精确保存原始小数秒；原文件目前预期毫秒精度，但实际最大精度由全量统计决定。

1. 去除用户、商品、精确时间和反馈相同的重复事件，保留最早原始记录号，`duplicates.parquet`记录被删记录、代表记录及双方日期标签。不同时间的重复观看保留。反馈去重使用原始十进制字符串，避免float把不同反馈合并。确认`time`与`timestamp`一致后，冗余`date`差异不再让同一次事件变成两次；差异仍保存在去重记录中。非法记录的时间文本参与额外区分，避免合并冲突。
2. `time`按北京时间与精确Unix时间核对；二者冲突隔离。部分`date`标签可能不同于这两个一致的精确时钟：仅记录`date_mismatch`，不改变时间或删除记录。非法反馈及同用户同时间的剩余记录均隔离，不据商品ID推断先后；有无法定位时间的用户不产生目标，以免静默连接缺口。
3. 按所有可定位时间的观察事件分会话，间隔**严格大于1800秒**才切分。同会话保留原始事件位置，从头每五条组成一个不重叠块；不先滤零播放、重复商品再重新拼块。
4. 目标块须恰有五件不同商品、均播放>0、跨度≤600秒，不包含隔离事件。尾块和不合格块整块丢弃。历史为块开始前最近最多50条有效已播放记录，至少5条；历史重复保留，坏目标块中的有效观察仍可进入以后历史。
5. 从无歧义有效事件时间计算 nearest-rank（第ceil(qN)个值）80%/90%切点。训练`time<cut80`，验证`cut80<=time<cut90`，测试`time>=cut90`。整个人工会话满足`start<cut<=end`时，其所有块都不作目标；过去观察仍可供之后历史。实际样本比例不强求80/10/10。
6. 验证/测试采用滚动已观察历史。五件总播放≥60000毫秒仅在训练样本中保存可选标记，不作为主版本筛选条件。未根据验证/测试反馈调阈值。

人工会话内的等时记录只按原始行号稳定存放；这不是推断真实顺序。它们所在目标块不采用，并从有序历史中排除。完整会话的跨切点判定先于这些排除规则。

## 数据接口

主产物位于`outputs/kuairec_listwise_v1/run1/`，第二次独立全量复跑位于`run2/`，均不进入Git。

| 文件 | 用途 |
|---|---|
| train/validation/test.parquet | 每行一个五商品目标及可变长历史，保留时间、源事件号、反馈、mask、会话边界、原ID和映射ID |
| events/*.parquet | 去重后的事件及隔离/日期标签标志，保留精确原始字符串以回查 |
| duplicates.parquet、quarantine.jsonl | 去重映射和隔离原因 |
| user_map.json、item_map.json | 按原始整数ID排序映射到1起始的ID；0专用于PAD |
| item_text.jsonl、text_repairs.jsonl | 静态原字段、合成文本及逐条无损修复记录 |
| item_catalog.parquet、user_catalog.parquet | 映射ID、原ID、训练是否见过、文本缺失标记 |
| train_seen.json | 实际训练样本历史与目标覆盖的用户/商品集合；用于冷启动标记 |
| ingestion.json、construction.json | 全量清洗、切点、各原因剔除计数 |
| validation.json、split_statistics.json | 全量样本规则验收及分布统计 |
| trace_validation.json | 从原始ZIP独立重建32用户全部合格样本的回查结果 |
| loader_validation.json、reproducibility.json、manifest.json | 实际批次读取、双次复跑比较、版本和输入/输出哈希 |

`history_*`是长5–50的列表，`target_*`长5；样本内mask均有效，CPU loader再左侧补PAD到50。loader同时返回用户、历史、目标的冷启动标记；不删除训练未见用户或商品。商品文本缺失仍保留空串及标记，不编造内容。静态原字段和源事件表能够回查每条反馈；样本中的watch_ratio是float64，不用其舍入结果做去重。

完整静态商品目录的离线可用性是明确假设，文本未证实是行为时刻快照。不使用全时期热度或其他行为聚合特征。目录只为预处理和后续统一表示提供接口，本轮不生成embedding、不拟合PCA/OPQ/PSE。

## 复现命令（仓库根目录，PowerShell）

使用项目已有的CPU Python环境；独立环境只需`requirements.txt`中三个固定依赖。

```powershell
$py = '.runtime/stage08_restore_env/Scripts/python.exe'
& $py -m unittest discover -s tests -p 'test_kuairec*.py' -v
& $py -m scripts.kuairec_data.prepare --output outputs/kuairec_listwise_v1/run1
& $py -m scripts.kuairec_data.validate --dataset outputs/kuairec_listwise_v1/run1
& $py -m scripts.kuairec_data.trace_verify --dataset outputs/kuairec_listwise_v1/run1
& $py -m scripts.kuairec_data.finalize --dataset outputs/kuairec_listwise_v1/run1
& $py -m scripts.kuairec_data.loader --dataset outputs/kuairec_listwise_v1/run1 --split test --batch-size 64

& $py -m scripts.kuairec_data.prepare --output outputs/kuairec_listwise_v1/run2
& $py -m scripts.kuairec_data.validate --dataset outputs/kuairec_listwise_v1/run2
& $py -m scripts.kuairec_data.finalize --dataset outputs/kuairec_listwise_v1/run2
& $py -m scripts.kuairec_data.finalize --dataset outputs/kuairec_listwise_v1/run1 --compare outputs/kuairec_listwise_v1/run2
& $py -m scripts.kuairec_data.report --dataset outputs/kuairec_listwise_v1/run1
& $py -m scripts.kuairec_data.finalize --dataset outputs/kuairec_listwise_v1/run1
Copy-Item outputs/kuairec_listwise_v1/run1/manifest.json docs/research/onerec_listwise/kuairec_preparation/MANIFEST.JSON
```

输出目录必须不存在，脚本拒绝覆盖已有数据。如需再复现，请更换新目录名。正式执行也可拆为`prepare --stage ingest`和同目录`--stage build`，配置必须完全相同。映射、切分与样本内容复跑比较使用固定512行/固定schema的Arrow IPC规范化哈希，限定同一配置和依赖版本；不声称跨PyArrow版本、改变分桶参数仍是相同字节哈希。

后续模型需另做设计和授权。本轮没有候选列表聚合，也不把五件目标改成10/20/50件。
