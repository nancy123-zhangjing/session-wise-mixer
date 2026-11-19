# D/E调度实验与既有实验：飞书内容计划

用户于2026-11-02明确授权创建。待完整下载与E科学验收完成后发布；此文件不是最终报告。

## 主线

从“原H1后期下降，冻结后改善”出发，说明D/E分别把相同200次T更新分散到整个35R，试图保留继续学习且给R适应时间。先展示固定R35/T200的A/D/E，再单列各自验证选best，最后联系过程证据。不得把“少做T更新”“减小T学习率”“冻结T”“重置T”写成同一种干预。

## 简明设计表

| 组别 | 相比原H1改变 | 总终点 | 回答的问题 |
|---|---|---|---|
| 原H1 | 共享5R后10×(40T+3R) | R35/T400 | 持续联合训练的原始轨迹 |
| A | 沿原H1到R20/T200后冻结T，R继续 | R35/T200 | 保留已学编码，继续R是否改善 |
| D | 改为10×(20T+3R) | R35/T200 | 减少每块T更新，分散同一200T预算 |
| E | 改为5×(40T+6R) | R35/T200 | 保留40T块，延长两次T之间的R训练 |
| C | H0无Mixer，在R20/T200冻结T | R35/T200 | 同冻结点、同终点下的Mixer对照 |

Freeze R26与A属于同一冻结路线的不同节点，不计作新的独立模型机制；Decay R26是降低T学习率后继续更新，另列历史。H0 best、B晚冻结仅作补充背景，避免正文失焦。更早Beauty/其他结构实验不混进同口径排名。

## 必须保留的比较边界

- A/D/E共用H1 R5/T0起点、总R35/T200及数据/评价定义。A后15R固定SID；D/E最后一次T后分别只剩3R/6R，更新安排影响R收到的梯度，不能用总次数相同推出效果相同。
- E/D在R11/17/23/29/35的累计T相同，分别T40/80/120/160/200；适合作为过程对照。E相对D同时改变T块大小、SID刷新频率和连续R长度，不能唯一归因于适应时间。
- 原H1 R20/T200对D/E R35/T200：T数相同、R数和更新时点不同，是回答用户疑问的补充观察，不是等R对照。
- 固定R35主比较用测试NDCG@5、HR@5、Recall@5与E-D/E-A用户区间。各自best另表，明确R/T和验证选模网格。
- 过程图使用固定8192样本，完整验证为71657样本，测试53194样本/4153用户。三者不混用。
- SID变化采用实际相同40T端点。原H1仅保留T0/T200/T400，不能计算逐20T/40T曲线或从大区间除法估算。
- 原H1末期严重碰撞塌缩；D未复现塌缩但推荐仍下降。SID改变多或碰撞少本身不是质量指标。
- 时间切片属于静态验证集；降低T频率并不自动增加见到的新训练数据。调度结果不证明在线分布适应、收敛不一致或one-epoch overfitting。
- 单seed；用户bootstrap不覆盖训练随机性、选模和事后提出方案的不确定性。未实施MEDA重置。

## 已核对来源

- experiments/09_kuairec_listwise/REPORT.md
- experiments/09_kuairec_listwise/h1_recovery_test_20261029/REPORT.md
- experiments/09_kuairec_listwise/h_freeze35_20261029/REPORT.md
- experiments/09_kuairec_listwise/h1_spread200_20261031/final_report_r2/REPORT.md
- experiments/09_kuairec_listwise/h1_sid_change_20261101/REPORT.md
- docs/guides/NEW_CHAT_HANDOFF_20261030.md（仅作索引，旧待办不复活）

E新结果来自最终ANALYSIS/PROCESS/TEMPORAL及最终报告；必须在实际验收完成后填入。最终XML路径为仓库根draft_3cf6198c_folder/draft.xml，需parse、create、fetch核验正文/表格/图片且无未解决警告。
