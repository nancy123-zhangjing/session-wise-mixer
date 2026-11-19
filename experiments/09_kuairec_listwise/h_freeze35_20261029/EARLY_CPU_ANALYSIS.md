# H1提前测试复算记录

完整A模型包下载与CPU结果复算并行，以便测试核验后先向用户报告。它不替代后续本地分支分析、全部模型恢复或最终三组验收，不运行模型推理，也不使用GPU。

第一次CPU尝试于2026-10-29 19:22结束，未创建分析输出、未执行指标复算。原因是历史A验证R23/R26的原始摘要没有`groups`字段，当前严格读取器拒绝该格式；原摘要、预测、权重SHA及排名协议均正常。失败日志、启动参数和退出码保留在`early_a_cpu_attempt1/`。

修复仅针对两个SHA锁定的旧验证节点：核其原始来源、模型和选模分数，返回原metadata，并明确标记“仅核选模依据，未核原摘要未保存的组统计”。不改原始摘要/预测，也不放宽五份测试的真实分组、标签或指标核验。旧验证的正式独立审计可复用此前已验收结果，后续单独绑定证据。

第二次CPU尝试19:31启动，运行目录为远端个人Stage09下`h_freeze35_20261029_cpu_analysis_code_r2/`，输出为独立新目录`h_freeze35_20261029_cpu_analysis_r2/`。执行驱动SHA为`2ad69be29b3146d06f838e5c63c630e099140fb90cf51ad7551e0062bd87ed28`，部署tar SHA为`2f5d5e75b8ef88a81cbd262e76b29cb9b1f3ac29b4abcd612923bd074876f2df`。本地tar名为`tmp/h_freeze35_early_a_cpu_analysis_code_r3.tar`，上传后远端tar名为`h_freeze35_early_a_cpu_analysis_code_r2.tar`，字节SHA相同。

五份测试分别为本轮A R32/R35、旧H1 R20/Freeze26、旧H0 best R32；使用原始53,194条/4,153用户、beam32、FP32及原候选规则。全部逐样本复算通过后，才运行A R35及验证best R32分别对三个旧模型的配对用户bootstrap，2,000次、seed2024、95%区间。R32比较是用户要求的补充比较，不以测试重新选模。

19:34第二次CPU分析成功完成，5份逐样本审计和6份配对比较全部通过；19:35–19:37已先向用户报告H1测试，不等C。轻量结果包333,722字节，43个payload已独立解压逐SHA核验，见`early_a_cpu_r2/TRANSFER_VERIFICATION.json`。COMPLETE SHA为`301fd4519a5db9d1685acf456221c471fcc352584c041ccc2da8789a04b40256`。原科学执行副本、原`analyze_results.py`和`analysis.py`均未修改；B已完成报告的源文件绑定保持不变。完整模型下载及实际恢复不在本次签收范围。
