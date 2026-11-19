# 检索边界与筛选记录

截止日期：2026-10-16（Asia/Shanghai）。检索目的：评估路线而非证明既定方向正确。先按问题分组检索，再用正文引用和最接近反证补查；在核心五类机制覆盖后停止扩张。

## 第一轮：已有推荐与跨领域先例

- 直接打开用户的四个链接，核对题名/版本，不能沿用对论文角色的旧猜测。
- 连续生成：DreamRec、DiffuRec、ContRec、flow matching recommendation、one-step continuous generative recommendation。
- 离散/混合/任务反馈：TIGER、COBRA、DIGER；工业采用定向搜索PinRec。
- 反演/摊销：pixel2style2pixel、PMLR v118 zhang20a及对应正式全文/作者代码。
- 连续检索参照：SASRec正文、谷歌NDR正式PDF；大库效率补查AVQ/ScaNN。

## 第二轮：针对新增价值和最强反证

- MIRAGE/FAVE由单步连续生成检索命中，继续阅读全文及可访问作者代码：检查“少步+学习商品空间”是否已被覆盖。
- frozen item encoder generative recommendation、decoder-aware tokenizer、amortized encoder recommendation等定向检索，找到EPIC后核对全文；不将冻结backbone的旁路推理误写为重学SID。
- 从各核心论文的related work、实验对照和后续方法中检查遗漏：SETRec的并行token、FAVE/MIRAGE的单步推理、EPIC的固定骨干低参数适配、AVQ的下游感知编码均进入创新性反证。
- 查作者README和固定代码版本：论文声称、公开实现、运行复现分别标记，不执行代码验证。
- 最后仅做两组“固定主干/商品向量低参数适配”反证搜索，补入FedPEFT的正式论文与方法正文（S06/E24）。REPREC只保留为用户侧旁路待核候选，不据其摘要定结论；到此停止扩张。

## 纳入原则

核心12篇：与A/B有直接机制重合，或是用户明确指定且需要澄清的基础先例。NDR、AVQ、DiffuRec、e4e、EPIC、FedPEFT作为6篇必要补证，不将条目数量作为完成依据。最终名单见RELATED_WORK.csv的tier字段。

主要使用arXiv全文、PMLR正式论文、作者项目页/固定代码、机构技术报告；博客/第三方聚合仅用来定位原始来源，不用其摘要证明性能或采用。

## 不继续展开的候选及原因

- 一般LLM4Rec/扩散推荐综述：可用于发现名称，不作为机制或落地的核心证据。
- CAPS-ID、FLASH、SST、DIRECTOR等检索候选：没有在本轮精读其完整方法，不据搜索摘要作结论；已有SETRec/EPIC/TIGER等直接证据覆盖相应必要反证。它们仍构成后续立项前的更新检索风险，而非被证明无关。
- 纯内容生成、生成广告素材、评分多热向量扩散：预测对象与“连续下一商品向量+检索”不同，不混入直接性能对照。
- 未来版本/未来会议模板：只采用截止前已公开版本。预印本页眉出现未来会议名，不据此认定已经接收。

## 证据限制

没有运行任何作者代码、重现训练或检索基准；公开工业A/B是作者/机构报告，不是本调研独立审计。未找到普及率统计，不能判断整个行业的采用比例；也没有据少见报道推断推广失败。没有主张完整穷尽全部2026年工作。
