# E 单卡最终统计入口

这些文件均从已验收 D 脚本作必要适配，尚未用 E 完整结果运行。不能将“脚本已准备”写为“验收已通过”。训练包/code/runtime/job 保持原冻结字节。

1. 完成训练、十个完整验证及固定35/best测试后，下载完整 E run、源码、subset、日志和收据。独立封包、SHA、重新解压沿用 D 交付流程，得到 LOCAL_EXTRACTION_VERIFIED.json。
2. analyze_e.py --run <完整本地run> --output <E/final_analysis>：核对全部正式样本顺序、阶段及双时钟，独立复算新预测，验证选模/测试次数；复用已验收 A、D，计算 E−D/E−A 用户聚类 bootstrap。原 A、D 只读。
3. cold_restore.py --extracted <独立解压根> --parent <原H1 initial.pt> --dataset <原数据> --assets <原assets> --output <E/COLD_RESTORE.json>：使用原 torch2.6 CPU 恢复环境 .runtime/stage08_restore_env/Scripts/python.exe，真实加载关键模型/优化器/checkpoint，检查36个边界状态与 R 间 T 不变。无训练、无推理；不声称 CPU 与 CUDA 轨迹等价。
4. process_e.py --run <完整本地run> --cold-restore <E/COLD_RESTORE.json> --output <E/final_process>：21个诊断状态、6个catalog；输出40T前后/1R/3R/6R曲线，并从 D 原快照计算相同累计40T端点变化，不加两个20T百分比。
5. temporal_e.py --analysis <E/final_analysis> --output <E/final_temporal>：仅处理 E 新 R35 验证预测；直接复用 A/D 已封存时间统计与数组；原切点、频次分类与bootstrap定义不变。

全部分析入口使用独占新输出目录，失败时保留失败证据，修正分析后选择新输出目录；禁止为修统计而重训模型、重跑解码。报告、曲线、完整包、独立恢复、远端同步和 GPU/CPU/heartbeat 关闭仍须实际完成。

备份入口 archive_delivery.py 已从 D 适配为 E 唯一根和 formal_r1 完成指针。只在作业真实终态且 FORMAL_COMPLETION/result/complete/test 证据齐备后执行 seal；其 extract 子命令校验完整包SHA、逐文件SHA并要求新目录。不要在训练期间执行 seal。后续故障若产生新attempt，必须按实际新完成指针再调整，不无条件接受任意目录。
