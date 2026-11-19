# 模型来源与项目适配

2026-10-15，本轮执行Goal尚在实现阶段，无GPU或正式推荐结果。

- eSASRec论文：[2508.06450v1](https://arxiv.org/html/2508.06450v1)。
- 作者代码：[blondered/transformer_benchmark](https://github.com/blondered/transformer_benchmark)，锁定 `2b039927ceb4c7654131d7ac7c43ea124b49d240`。
- 作者requirements固定RecTools 0.13.0；本轮对应源码：[MTSWebServices/RecTools v0.13.0](https://github.com/MTSWebServices/RecTools/tree/v0.13.0)，commit `6ffd25bf3ca8517c8bd1a7207d6b0e64c8e1912a`。

LiGR逐层按作者代码：pre-LayerNorm attention → sigmoid线性门控残差 → pre-LayerNorm SwiGLU → sigmoid线性门控残差；SwiGLU FFN无bias、隐层4d。普通SASRec采用该版本RecTools的PreLN块：GELU FFN、有bias，以及块末dropout。本轮称SASRec＋FullCE（RecTools PreLN变体），不声称逐字复现2018 TensorFlow SASRec。

两者共享输入/输出商品表，inverse learned position，无额外输出归一化，矩阵参数Xavier normal初始化。本轮明确不在输出商品表施加dropout，只在历史商品＋位置之和施加embedding dropout；与RecTools的复合ItemNet可配置dropout路径分别记录，不冒称完整RecTools逐字执行。

eSASRec损失采用作者uniform sampler：每个监督目标独立均匀有放回抽256/1024个真实商品；无PAD、无logQ、不排除重复负例或偶然抽到正例。正例与该负例multiset上的softmax CE，与全库FullCE的训练损失不混比；验证NLL统一使用全库。

作者Beauty配置为256维、1层、1头、dropout0.2、256负例。本Goal执行用户指定32组搜索、300轮上限和多seed预算，不能称为论文报告设置的严格复现。

原论文Shifted Sequence通常同时监督多个位置；本项目已展开前缀，每行只监督末位下一商品，禁止再对所有历史位置重复监督。此为共同项目协议适配。左PAD采用RecTools合并因果/padding mask并开放对角线的办法避免全遮罩NaN；真实查询仍不能看PAD或未来商品。PAD商品权重固定为0、分类与候选均完全排除PAD。

`reference_upstream/`保存未修改的作者网络源码供真实前向/梯度对照，`upstream_sources.json`保存URL、commit和字节哈希，`licenses/`保存Apache-2.0许可。训练不依赖RecTools/PyTorch Lightning，不改两个历史引擎。

## 2026-10-16 数值精度修复

原CUDA BF16默认efficient-attention路径在训练后的512维/2头checkpoint上产生异常反向梯度；即使loss有限且前向误差很小，固定样本/负例、关闭dropout后的梯度仍与FP32参考严重不符。不能只以推理cosine证明训练精度正确，详见[故障诊断](NUMERICAL_FAILURE.md)。

统一对两个模型采用CUDA FP32 Math注意力：关闭该区域autocast，Q/K/V投影、SDPA及输出投影均用FP32 Math，外层FFN和商品打分仍按原BF16策略。CPU保持原dtype，作者双精度方程/梯度对照仍通过。此为本项目数值策略适配，不改变网络方程、参数量、采样规则或搜索超参数，也不声称作者原程序存在同一故障。

固定checkpoint的3批梯度对照、514步恢复重放和原失败配置35轮从头压力检查已通过；这是有界数值证据，不等于证明所有配置长期稳定。新GPU gate进一步检查训练梯度，并在正式89项前对实际执行源码重复35轮回归。测速使用相同精度策略并明确记录；不能把旧默认后端测速与新结果混用。
