# L5O 阶段零：正确性和 H100 执行短测

只执行已授权的数据/指标适配验收与一张 H100 短测；不调用正式训练 runner，不启动 SASRec 或六卡实验。

- 数据协议及优化范围：`docs/research/onerec_listwise/L5O_IMPLEMENTATION_20261117.md`。
- `prepare_remote.py`：核对公开 KuaiRec 清洗事件的源 SHA，重建相同 L5O 数据，核对本地/远端规范化行哈希，复用静态 PSE 资产并另建数据绑定。
- `benchmark.py`：H0/H1 的原执行路径与候选路径各 200 更新，分为 80R/40T/80R；同一候选 checkpoint 上比较固定 256 条验证样本、beam32、batch4/128/256；精确核验恢复后下一次 GPU 更新。compile 单独限时 240 秒，未经额外一致性验收不采用。
- `control.py submit_gpu`：需有通过的远端准备回读收据，只提交一个两小时上限的 H100 作业；提交意图存在时拒绝重复申请。
- `http_transport.py`：标准 CLI 的浏览器连接失败后使用同一账号、同一临时 CPU 的 Jupyter HTTP/WebSocket；不修改安装的 CLI、不保存凭据。数据上传只包含公开 KuaiRec 事件和指定目录元数据，逐文件 SHA 校验，已匹配文件跳过，不覆盖不匹配文件。

远端根：`/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/l5o_stage0_20261117_v1`。
大数据、短测 checkpoint 和逐样本结果保留远端。本地只归档源码、数据核对摘要、源哈希和最终报告。`SOURCE_MANIFEST.json` 绑定实际发送字节；提交后补记源 Git commit。

本地已通过数据、指标、原训练/解码及新缓存对照检查；语法检查不代表 H100 已验收。GPU 结果以最终短测报告为准。`COMPLETE.json` 是主测试结果，compile 子测结束及平台作业终态须另行确认。

已完成的验收见 `REPORT.md` 和 `FINAL_ACCEPTANCE.json`。Git中的 `receipts/` 删除了原始JSON内重复的推荐列表，保留训练逐步loss、阶段耗时、验证一致率和指标；`RECEIPT_MANIFEST.json` 同时绑定远端原始文件及精简文件哈希。`summarize.py` 可从精简收据重建表格和曲线。临时平台响应不进入Git，远端原始产物不修改。
