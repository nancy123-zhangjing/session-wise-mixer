# C 补充统计准备记录

此记录保留输入就绪前的执行准备方法，不单独作为完成证明。2026-10-30 后续已按下述步骤运行；统计结果见 `c_supplement_r1/COMPLETE.json`，独立统计复核见 `VERIFY_C_SUPPLEMENT.json`，完整中文分析见 `FINAL_REPORT.md`。文字独立审查由主任务另行记录。

- 使用新增的 `prepare_c_supplement.py`，从已通过的 `../final_analysis/C/analysis_report.json` 提取 C 完整验证 R32 与原共同节点规则选出的 best R23；绑定原 selection、逐样本审计、预测、summary 和 checkpoint SHA256。
- 模型名为 `C_freeze_R32`（R32、22,176 个 R updates、T200）和 `C_best_R23`（R23、15,939 个 R updates、T200）。两份 checkpoint 不同，不去重。
- 只在完整包本地恢复且主 C 逐样本审计通过后运行。原 `analyze.py` 已支持 `--extra-models`，无需改变任何统计公式、初版结果或冻结代码。
- 调用原 `analyze.py --extra-models C_EXTRA_MODELS.json --outdir c_supplement_r1`，复算原三模型加新增 C 两模型；新增配对方向均为 `H1_freeze_R32 - C`，沿用原 2,000 次、seed 2024 用户配对 bootstrap 以及早晚段共同全体用户重采样规则。
- 调用原 `verify.py --result-dir c_supplement_r1 --output VERIFY_C_SUPPLEMENT.json`，独立复算全部中间数组、bootstrap 和分解；结果目录与核验文件必须全新，不覆盖。
- 报告写入 `REPORT_C_SUPPLEMENT.md`，重点比较相同 R32/T200、相同 R20 冻结时点的 A/C；另列各自验证 best 的 R32/R23 补充比较，明确 R updates 不相同。
- 保留原验证时间切点、五类训练覆盖/目标频次分层及原定义。分层命中率是逐目标出现次数的比例，与列表 HR@5 不混用；推荐 top10 集中度与训练热门 802 个物品占比单列。
- 本补充为选模验证集上的事后统计，不取代原固定 R35 测试主比较；单 seed、H1 观察后选 R20 冻结时点、选模不确定性及粗分层内用户/物品/时间组成变化仍限制因果解释。
- 原 `initial_r1`、`REPORT.md`、`analyze.py`、`verify.py` 保持不变；`STATUS.json` / `NOTIFICATIONS.json` 由主任务在全部验收后统一更新。
