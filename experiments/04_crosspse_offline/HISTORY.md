# 04 阶段版本演进

本页记录“为什么改、改了什么、留下什么证据”。主结论仅使用 [README](README.md) 的 Refined-PSE v2-catalog、Cross-PSE v3-catalog，并复用 03 阶段 Original PSE v6 基线。

| 顺序 | 版本 | 主要设计 / 调整 | 当前证据与观察 |
|---:|---|---|---|
| 1 | Cross-PSE v1 | 固定基础 PSE 码本；加入全局 Attention；可学习残差系数从 0 开始；用拟合集之外 40 商品选择 checkpoint | 有配置、脚本、设计文档；本地轻量包未含完整该版本 Tokenizer 产物，不写成已完成正式下游实验 |
| 2 | Cross-PSE v2 | 残差系数固定为 1，Attention 输出层置零；基础码本仍固定；从原拟合 catalog 划出 1206 商品验证 | best step 100；训练10855 / 验证1206，SID 改变较小；保留为 pilot |
| 3 | Refined-PSE v1 / Cross-PSE v3 | 基础码本可训练；加不加 Mixer 做成对对照；保留内部划分和 step 0 比较点 | 两组 best step 均为 0；未超过底座，不能称为有效学习到新 SID 的正式下游结果 |
| 4 | Refined-PSE v2-catalog / Cross-PSE v3-catalog | 与原 PSE 使用同一12061商品拟合catalog训练和选Tokenizer；40商品补集只诊断；基础码本LR1e-5、Mixer LR1e-4，上限3000步 | 正式离线组，best step 2900 / 2950，SID改变5.06% / 5.74%；再固定SID训练DiffGRM |
| 5 | 转入05阶段 | 检查仅靠离线目标是否不足，尝试将推荐梯度反馈到Tokenizer | 是新的联合训练研究问题，不把其结果倒填进04 |

## 内部 split 为什么需要另标

原始 PSE 底座已拟合全部 12,061 个 catalog 商品，后续再从其中划出 1,206 个作为 refinement 的验证集，并不构成“底座从未见过的独立验证集”。这会让初始底座与微调过程的训练 / 选择口径不对称，是协议问题的线索；**best step 0 本身不证明它就是唯一原因**。

正式 catalog 版改为同 catalog 的传导式拟合与选模，明确不宣称独立泛化验证。summary 中 legacy validation 字段须结合 selection_protocol 解释。

## 保留哪些 pilot 证据

已找到并保留来源的三组离线材料，每组包含 summary、training.csv、geometry_summary：

- outputs_tokenizers_diff_crosspse_Beauty_v2_seed2024
- outputs_tokenizers_diff_crosspse_Beauty_v3_seed2024
- outputs_tokenizers_diff_refinedpse_Beauty_v1_seed2024

它们原先位于 crosspse_report_20260829 轻量包的 tokenizers/pilots 目录。Cross v1 的设计材料与这三组“确有训练统计”的证据级别不同，不补造缺失结果。Smoke / profile 只用于检查管线，不计为正式方法或独立 seed。

## 与 03、05 的关系

- **03 → 04**：PSE v6 是复用基线，没有再训练一遍 Original PSE。
- **04 内部**：同一 PSE 底座，先离线训练 Tokenizer，再冻结 SID、从头训练 Decoder。
- **04 → 05**：本阶段两组最终 Tokenizer 和 Decoder 分别成为 05 的预训练起点；因此05是 continuation，不是把本阶段从头训练再重复一次。

代码 commit 缺口照实保留；当前源码锁定只能约束新的重跑。旧完整报告见 [previous_README.md](reproducibility/previous_README.md)。
