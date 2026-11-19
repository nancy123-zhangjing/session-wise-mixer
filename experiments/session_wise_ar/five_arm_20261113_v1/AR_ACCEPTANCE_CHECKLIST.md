# AR 最终验收待办与证据边界

2026-11-14：训练期间静态核对，尚非最终验收。适用于 AR-T0、AR-D0、AR-D1；AR-T1 在推荐训练前失败，不能填入虚构的推荐测试结果。

## 每个验证节点

- 对照训练记录逐个核验 checkpoint SHA、完整验证样本数 71657、配置及父状态身份。
- 检查 tokenizer、recommender、optimizers、schedulers、stream、progress、rng、sid 均存在且与节点计数一致；核对来源、环境及优化器数量。
- 节点检查在远端执行，只保存小型路径和 SHA 清单，不下载权重。结构检查不等于 CUDA 数值恢复。

## 完成与测试

- 以 train/complete.json 的实际 best/final 为准；核对预算或原早停触发原因，不以当前最优节点代替最终点。
- 对照 test/complete.json 和逐点评测结果，确认测试样本数 53194、checkpoint SHA、配置及预测文件身份一致。
- 核对平台终态与实例资源释放，保留远端历史资产。

## 恢复证据的特别边界

已核对 scripts/session_wise_ar/runtime.py：_save 保存完整续训字段；_restore 恢复 Tokenizer、推荐器、优化器、调度器、数据流及 RNG，并要求配置、来源、父状态和环境一致。

但 AR 的 test() 只加载推荐器权重和已保存 SID，未调用 _restore，也没有恢复优化器或加载节点 Tokenizer。因此 best/final 测试成功只能证明推荐推理可加载，不能单独证明完整续训恢复通过。最终验收需要结合已通过的 GPU 恢复短测和全部节点的实际状态检查，并明确其覆盖范围；若补做真实节点恢复检查，仅加载和核验，不执行额外优化更新或推荐评测，不改动原作业。

以上为验收范围说明，不改变模型、训练预算、日程或实验组数。原 IMPLEMENTATION.md 的“尚未启动”描述属于实现阶段历史记录，不能作为当前运行状态。


## 35轮截断后的停止规则核验

用户后续授权DIGER两组35轮执行截断，保留原120轮学习率轨迹及patience=15。audit_ar_nodes.py现读取EXECUTION_CAP.json，区分执行预算与scheduler预算，检查最终步数不超过484995、停止来自预算或原早停、逐节点重新计算best及未提升计数，并拒绝达到停止条件后仍继续的记录。TIGER仍按原max_steps核验。

本地test_stop_policy.py四项检查通过：35轮截断、15次未提升早停、拒绝提前/超预算/错误best/达到早停后继续、拒绝将scheduler改为35轮。仅检查验收逻辑，未重跑训练或推理；DIGER尚未结束，实际最终停止审计未执行。
