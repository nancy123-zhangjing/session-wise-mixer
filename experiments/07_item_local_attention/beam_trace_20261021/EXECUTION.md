# 本轮执行与交付

已完成并验收：固定 Stage07 A/B 的 best25 权重与保存 SID，对同一验证集跑 beam32/256 逐轮搜索诊断。四组各 22,363 人，共 89,452 条结果；新增探针与原解码首批输出一致，全量逐用户记录也与此前评估完全一致。

- 作业：`nancy-stage07-beam-trace-20261021-r1`，最终 `SUCCEEDED`。
- 资源：单节点 4 张 H100，四个独立单卡进程；没有 DDP、重新训练、测试集评估或 SID 更新。
- 分配：2026-10-21 21:46:16 至 Pod 删除 21:53:41（北京时间），445 秒，合计 **0.494444 GPU 小时**。
- 容器运行至平台完成口径为 433 秒、0.481111 GPU 小时；包装器进程口径为 420.902 秒、0.467668 GPU 小时。三者边界不同，不当作已核实账单。
- 平台 `SuccessfulDeletePod` 事件确认释放；最终实例为历史成功状态。
- 启动前远程 CPU 10 项检查通过，模型、SID、13 项输入及源文件哈希通过。准备时发现的 Python3.10 CRLF 解析差异已通过统一源文本换行解决；发生在申请 GPU 之前，前次失败日志保留。
- 后置检查确认原训练目录、权重、SID、参考预测和执行脚本未改变。本地再次逐用户比较四组原始字段，全部相同。

远端产物根目录：
`/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage07/beam_trace_20261021_r1`

本地验收目录：
`C:/Users/nancy/Documents/experiment1/outputs/stage07_item_local_joint/beam_trace_20261021_r1/verified`

本地归档 16,397,745 字节，SHA256 `adca7572ba1618b63df12712fb2adba169907811852e1d2efb16c6dfc79c23c2`；69 项产物逐文件哈希全部通过，解压后数据约271MB。旧权重与复制的数据缓存不重复装入本轮轻量诊断包。

阅读顺序：`REPORT.md` 看结论，`SUMMARY.json` 看完整统计，`FINAL_ACCEPTANCE.json` 看本地独立复现与资源核验。`PROTOCOL.md` 记录实验约束，`deployment/DEPLOYMENT_MANIFEST.json` 绑定实际上传代码。
