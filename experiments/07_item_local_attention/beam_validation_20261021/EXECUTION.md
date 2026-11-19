# 07 验证集搜索宽度诊断执行记录

已完成并验收：`nancy-stage07-beam-validation-20261021-r1`。四格指标、逐用户记录及GPU释放已核对，见 [REPORT.md](REPORT.md)。

- 新远程目录：`/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage07/beam_validation_20261021_r1`；产物在 `results/`。
- 单张 H100，平台 1 小时上限，进程 3400 秒，无自动重试。见 `JOB_SUBMISSION.json`、`JOB_DRY_RUN.json`。
- 新评估脚本 SHA256：`fde87ba653090fd73a7159dce29237da573167d487de80ab7d90d0e6a44ac1e1`。
- 新启动器 SHA256：`c001130ac3c39b740ba46862488e35acac86bf5df46225ff63b99e6b5c38a563`。
- 远端 CPU 预检实际通过：两 best 权重及 SID、13 件输入、冻结源码均绑定；未做推理。见 `GRID_PREFLIGHT.json`。
- 代码独立审查通过：保留原 forward 的 Accelerator BF16 包装及自定义 decoder 精度路径；只构建 val；沿用原 evaluator；四格均 confidence、Top10、batch8。beam32 的七项指标须与原 best25 记录差异不超过 1e-9。
- 评估进程及子进程继承独立进程组，失败/超时清理；原训练代码、权重、SID及结果目录保持只读。没有提交训练。

该Job为SUCCEEDED，2026-10-21 12:59:13容器删除，新增1067秒（0.296389 GPU小时）。四格均22,363用户，beam32七项指标精确复现，模型/SID未变；57件本地归档校验及逐用户重算通过。不要重复启动此已完成的验证。原07训练Goal已完成，本轮是单独授权的验证诊断，没有创建新Goal。
