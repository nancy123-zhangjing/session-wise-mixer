# 最终统计独立复核准备

本目录只保存独立 CPU 复核脚本和收据。准备完成不代表统计验收通过。

已读取冻结分析器、历史元数据兼容入口、独立指标/用户聚类 bootstrap 实现和 A/B 验收收据。2026-10-30 本轮现查 SHA-256 与原绑定一致：

| 文件 | SHA-256 |
|---|---|
| analyze_results.py | 47522df77e2eab06e598454ed331e6a939609bf0f63ddb9087d78f7f19aa9677 |
| analyze_results_compat.py | c5f6c819ea3889c8365bcee44336164b7617f9651745f5452c9061111ef8a7b4 |
| scripts/listwise_diffgrm/analysis.py | 4e056eecfded4ff2c1e25ecc1c32a348f01004f31d1b649946ea544f2aa79bdd |
| final_analysis/A/analysis_report.json | 66aec576ae2fe110ea593e4f83c6e24b8cca8bcac46a8595eb476f7593479937 |
| final_analysis/B/analysis_report.json | 7596546bb014f56b69d0a574d9cfd3a991e267bf667af6b80603cc6f48361a7e |

## 路径与入口

- 最终全包预期位于 outputs/stage09_restore/h_freeze35_20261029_final_r1_restored。主分析器通过分支目录标记将远端产物路径映射到本地，无须修改原始 JSON 或 checkpoint。
- 保留 early_A、early_B 和历史 reference 原位置；A/B 缓存报告绑定这些绝对路径及 SHA。
- 最终 --all-ready 应使用 analyze_results_compat.py。A 的历史 R23/R26 summary 缺少 groups，已有兼容入口严格绑定旧数据与已验收审计；裸入口重认证时会拒绝这两个旧节点。无需修改冻结分析器。
- B 的 selected_epoch=35 仅表示固定终点，不是 A/C 共同验证网格的 best。

## 待全包和主分析到齐后执行

1. 独立验证 C 的 13 份完整预测：10 个新验证节点、R35/R23 测试和历史 R5 验证；从原始 item ID 与 source split 重算逐样本指标、样本/用户/冷目标分组聚合，核对原 2e-6 容差。
2. 核对 A/C 只在共同验证节点 R20/23/26/29/32/35 中按 NDCG@5 最大、同分取最早节点选模；预期 A32、C23，不依赖测试值。
3. 对 C35−R5、C35−R20 验证，以及 A−C 固定 R35、共同节点 best 的 test/validation，独立复算原 2000 次、seed 2024、95% 用户聚类 bootstrap。每次抽 U 个用户，携带用户全部样本，以重抽样后的样本总数作为随机分母；用户等权估计单独核验。
4. A/B 分支既有审计不重跑。A 对照预测只为上述新 A−C 配对提取必要逐样本贡献，不重新生成 A 分支报告。C 起点只做已授权验证比较，不新增测试或推理。
5. 保存检查范围、输入 SHA、独立数值误差、问题和最终收据。不调用生产指标实现、Torch、GPU、网络；不修改任何冻结代码或源产物。
