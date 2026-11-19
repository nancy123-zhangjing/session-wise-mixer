# 最终 checkpoint 与 CPU 恢复独立复核

结论：**通过本项验收**。逐一核对最终 manifest、实际 A/B/C 恢复明细及原始边界收据，并独立重算归档、3,405 个解压文件和外部依赖 SHA；全部与已认证清单一致。原收据、归档和源码未修改。独立复核使用标准库读取 JSON/原始字节，没有反序列化 checkpoint、重建模型、执行推理或训练。

| 组 | 不可覆盖状态 | rolling last | 实际结构检查 | 关键角色 | 实际关键文件恢复 |
|---|---:|---:|---:|---:|---:|
| A | 14 | 1 | 15 | initial / R35 / best R32 | 3 |
| B | 6 | 1 | 7 | initial / R35 / best（固定 R35） | 2 |
| C | 43 | 1 | 44 | initial / R35 / R20 / best R23 | 4 |
| 合计 | 63 | 3 | 66 | 10 个角色 | 9 个唯一文件 |

应有节点由独立日程算式生成，再与 manifest、逐 checkpoint 收据及 worker 的 `checked_states` 比对，集合完全一致。35R 终点累计 24,255 次 R 更新，A/C 固定 200T，B 固定 360T；C 重放从 R5/T0 到 R20/T200。R35 未完成验证时保存的 training boundary（phase44/update693/prepared=true）与完整终点（phase45/update0）分别保留；三个 last 均与各自完整 R35 文件字节相同。C 的 R20 前冻结状态单独覆盖。

原真实恢复 worker 已对 66 个状态读取并校验字段、模型/alignment 形状/类型/有限性、优化器映射及 moment 形状/有限性、R/T Adam 时钟、原 35R LR 日程及 T LR、DIGER 时钟、冻结指纹、SID 范围/PAD、RNG 字段。独立复核核对这些实际明细并审查 SHA 绑定的校验实现；没有声称独立再次检验张量内容。逐节点 SHA、R/T 计数、阶段与关键恢复覆盖见 `CHECKPOINT_COVERAGE.csv`。

9 个关键文件确有模型、alignment、双优化器及 scheduler 的实际加载后指纹比较，CPU Python/NumPy/Torch RNG 恢复，真实 B2×L50 历史的有限前向（logits 2×4×256）。全部零训练更新、SID 不刷新、前向前后完整状态一致。每个关键来源文件在原恢复后已复哈希；本复核再次核全部 66 个状态，并在复核结束前再核一次，均匹配恢复前 manifest。B 的两个角色共享同一 R35 文件，去重并无遗漏。

路径重定位仅发生在内存中的已认证 dataset/PSE 绑定元数据；保留旧/新绑定指纹及 C 的 R20 gate 映射，不能据此改变科学参数或磁盘 checkpoint。C 的后冻结关键状态均正确绑定原 R20 replay gate。

范围：CPU 恢复没有实际恢复 CUDA RNG，只保留原 CUDA RNG 指纹；不能称 CUDA 精确续训验收。归档依赖外部 dataset、PSE 和 26 份历史结果，环境未打包，不能称单包自包含。这里只验收 checkpoint 覆盖和真实 CPU 恢复，不替代指标/bootstrap、冷启动补充、资源释放结算或整个 Goal 验收。原 `BACKUP_VERIFICATION` 的 model restore pending 是历史阶段边界，后续完成由本次 `FULL_MODEL_RESTORE` 及其明细补充证明，原收据保持不变。

最终归档 SHA256：`9006d8d835ed735bd79bfe382ae362ebc6266ba9c1522bf9a20e6b0dcb399f16`。
最终 manifest SHA256：`b4650e450f9d367e9ae5aa8780b393c2c7a79d8f9730c1aeafb2618a559cac11`。
认证字节验证收据 SHA256：`1d531247dd4cf57a0a422ff76091a579139045c29332da90e55065058d95550b`。

独立复核详情、原证据 SHA、实现审核位置与范围见 `INDEPENDENT_REVIEW.json`；`independent_review.py` 是新建离线复核工具，不属于冻结执行源码。
