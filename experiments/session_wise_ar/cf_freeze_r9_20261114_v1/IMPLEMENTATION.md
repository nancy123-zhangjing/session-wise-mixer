# 实现与启动记录

2026-11-14：用户在原五组 Goal 运行期间，明确追加授权两组 CF R9 冻结续训；不改变原 DIGER 两组。

模型代码提交：264880e。新增独立入口，原 CF 引擎与科学配置未改。目标、损失、batch、优化器、R学习率曲线、教师和评测噪声均沿用；只跳过R9后的T更新和目录刷新。

必要本地检查：`python -m unittest tests.test_cf_freeze -v` 通过（1项，内部覆盖none/intra两种模式）。实际合成训练至R9/T40再继续R11，验证Tokenzier/catalog/T优化器/T随机流保持相等，并验证恢复后下一步一致。不重跑旧检查。

远端来源、prepared、两个R9父checkpoint SHA均核对通过。原M1的训练根是`rqcd_targets_formal_20261113_v1/cf`，预训练根为`cf_rq`；原M0为`session_wise_ar_five_20261113_v1/runs/CF-M0/train`及`pretrain`。初次部署核对时发现M1目录层级不同，在提交作业前已更正。

远端独立根：`/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/cf_freeze_r9_20261114_v1`。`deploy.py`防重复提交；`status.py`只读跟踪；`audit_frozen.py`只读比较真实续训状态。不下载模型或预测大文件。

两张单独H100作业均已RUNNING：
- `nancy-cf-freeze-m0-1114-v1`
- `nancy-cf-freeze-m1-1114-v1`

GPU完整状态严格恢复回执见STARTUP_RECEIPT.json。开始后的真实GPU checkpoint对照见START_FREEZE_AUDIT.json：M0累计R更新6760、M1累计6560（父6237），两者T=40；推荐器参数已变化，Tokenizer、catalog（含SID）、T优化器、T数据流及三个T随机生成器保持一致；R optimizer步数一致。该核对是CPU读取GPU产生状态，不是新增GPU数值等价短测，也不是最终验收。

后续继续逐轮全验证、保存完整节点、best及R35全测试，核对所有冻结节点及资源释放。原R9候选沿用原验证结果，不重复验证；最终best可能仍为R9，届时明确标注无改善。


## 最终只读验收入口（2026-11-14补充）

audit_complete.py复用已有audit_cf_nodes.py与audit_frozen.py，检查R10至R35全部节点、R9父状态及继承验证、全节点冻结字段一致、最佳点选择包含R9、最终R35/T40/24255更新，以及best/final全测试53194条及预测SHA。只读CPU检查，不训练或推理。部署时保留两个实验目录的相对结构以导入已有审计模块。

目前仅完成Python编译检查；训练未结束，未执行最终真实产物验收，不将脚本存在视作通过。GPU恢复证据与资源释放仍须另行核对。
