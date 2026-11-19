# 原 IncrementLinear R26/T280 冻结续训至 R35

同一父checkpoint与既有r26_hold、r26_low严格配对。只改变后续策略为冻结整个Tokenizer（基础码本、Mixer、SID、T优化器），跳过T；保持有效编码不突然解除历史门控。继续R27–R35，原35R/400T计划时钟、R学习率、Adam、样本顺序、batch1024、seed2024不变。终点R35/T280。

完整验证网格R26/29/32/35，父R26参与best选模，平局取早。测试best和R35（若相同只测一次）。保存节点checkpoint和8192诊断子集，沿用原独立验收。单张H100，独立目录，不覆盖前四组。

实现仅修改协议配置和运行辅助脚本；recovery_stability.py及recovery_stability_run.py训练/冻结/恢复逻辑逐字节复用。完整结果留远端，本地只保留源码和小收据。状态以STATUS.md为准。
