# 三组best完整验证/测试最近邻落地对照

用户授权扩大已有8192子集诊断到完整validation（71657）和test（53194），不训练。

- 使用原正式B0/R25、B1/R28、B2/R24验证选出的best，不按NN或测试成绩重新选点。
- 原模型、原噪声、BF16、20步DDIM、batch256不变；三组共六次完整推理。
- 捕获原RQ滚动推理中每个生成z。同一z分别用原RQ和全目录潜向量最近邻读出。
- 最近邻不改变Decoder前缀，各种读出分别排除自身已选商品，保持原协议允许历史商品。
- 核验全部原RQ预测与已有正式逐样本记录完全一致；保存模型未变和checkpoint/噪声/源码SHA证据。
- 报告NDCG/Recall/HR/命中session数、配对用户bootstrap及比较边界。
- 不重复真实向量oracle诊断。不测试其他checkpoint，不改距离数值算法，不启动新训练。
- 1张H100顺序执行，作业最长1小时；完成后释放GPU并关闭CPU文件通道。大向量与逐样本记录留远端。

远端根：`/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/rqcd_full_nn_20261112_v1`。
