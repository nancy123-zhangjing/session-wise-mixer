# 固定 R26 补充测试最终审计

日期：2026-10-29。结论：**PASS，未发现阻断交付的问题。**

本审计限于用户授权的两组固定 R26 完整测试及与原 H1 R20 测试的比较。复用已经完成的执行、独立复算及交付收据，只读核对现有文件和必要 SHA；未调用平台/GPU，未重跑训练、推理、bootstrap、解包或模型恢复。本轮没有新 Goal，也不改变此前已经 complete 的恢复 Goal。

| 核对项目 | 结论 | 实际证据及边界 |
|---|---|---|
| 三组身份与固定终点 | PASS | R20 复用原成功的 `H1_test_beam32_r2` 预测；Decay、Freeze 均为原认证恢复执行的固定 R26 `last.pt`。两份 runtime request 与 complete 的 parent、checkpoint、execution manifest、launcher SHA 一致。未按测试结果选轮次。 |
| 统一测试协议 | PASS | 三组完整 test 均为 53,194 样本、4,153 用户、seed2024、beam32、FP32、batch32；使用原五件列表及固定多列表聚合规则。新测试 R 更新数均为 18,018，T 更新数分别为 280/200。 |
| 无训练、状态未改变 | PASS | 两份 complete 均为 `training_performed=false`，控制器完整状态指纹 before=after，且与 request 对应。沿用已审 launcher 的诊断上下文和原评估入口；本轮不重新加载或备份模型。 |
| 逐样本独立复算 | PASS | 复用三个独立审计收据：合计 159,582 条；标签、用户、索引及冷启动分组绑定真实 test/train_seen；均 passed、mismatches=0，最大误差 4.80988114e−8。不是只核进程退出。 |
| 结果表与完整指标 | PASS | `METRICS.csv` 三行、每行 37 项指标共 111 值，与三个 audit 及 analysis_report 精确一致；`REPORT.md` 主要指标及相对变化的显示舍入正确。 |
| 三组配对区间 | PASS | 三份比较均为 2,000 次、seed2024、95% 用户聚类 bootstrap；主估计为样本平均，等权用户平均另列。REPORT 的差值、比较方向及区间与 JSON 一致。没有重跑 bootstrap。 |
| 输入及分析源码绑定 | PASS | 重新核对 `analysis_report.json` 中 17 个输入/实现 SHA，全部与当前文件相符；`inputs_unchanged=true`。原 R20 成功预测、summary、旧审计的三个 SHA 固定，未混入失败的旧部分输出。 |
| 本地交付 | PASS | 复用正式安全解包验收，另核整包及当前 11 个 payload 的大小/SHA；11 个正文文件加 MANIFEST 共 12 个落盘文件全部一致。未再次解包。归档不包含数据集或模型，原权重继续由上一轮已验收归档保存。 |
| 任务与资源释放 | PASS | 两份原 r1 CREATE 与最终 status 名称一致；均 SUCCEEDED、exit0，并有 SuccessfulDeletePod、JobTerminated。当前保存的提交和事件证据未见重提或重启；这一结论不扩展为账户全部作业库存审计。 |
| GPU 时长与费用 | PASS | Scheduled 至最后 JobTerminated：Decay 1,610 秒、Freeze 1,897 秒，各 1 GPU，合计 3,507 GPU秒=0.974166667 GPU小时；并行墙钟 1,923 秒。低于本轮 1.6 GPU小时上限，余量 0.625833333 小时。按 1 点/GPU小时估 0.97417 点，平台结算未核实，与旧恢复 Goal 单独记账。 |

## 结果与解释边界

| 模型 | 测试 NDCG@5 | 相对 R20 |
|---|---:|---:|
| H1 R20 | 0.00794367350449 | — |
| Decay R26 | 0.00456690936491 | −42.51% |
| Freeze R26 | 0.00915701606411 | +15.27% |

| 配对比较 | NDCG@5 绝对差 | 95% 用户聚类区间 |
|---|---:|---|
| Decay − Freeze | −0.00459010669920 | [−0.00506572774985, −0.00413981763001] |
| Decay − R20 | −0.00337676413958 | [−0.00381114282071, −0.00295922205815] |
| Freeze − R20 | +0.00121334255963 | [+0.00083370977738, +0.00159351041734] |

本次固定留出测试支持 Freeze 改善、当前 Decay 设置退化，与之前验证集方向一致。Freeze 保留 R20 已学到的基础码本、Full Mixer 和商品→SID 映射，并继续训练推荐器六轮；它不是 H0，也没有移除交互。不能据此推出交互优于无交互、查明退化机制或多 seed 稳定收益。

单份五件列表三组均完整、无重复/无效/缺失项；聚合候选覆盖没有全面改善。Freeze 平均不同候选 20.91，低于 R20 的 21.67；聚合 @50 是多份五件列表固定排序去重后的指标，不能称为完整生成 50 件商品。用户聚类区间仅描述这些固定模型的用户抽样不确定性，不覆盖训练 seed、历史 R20 选模及看到验证结果后提出策略的不确定性。主报告已保留这些限制。

## 关键绑定

| 资产 | SHA256 |
|---|---|
| R20 parent | `a672ea4d31be5281c2c24c1c7e774bc27eb7e679ec732181a197b703da890bf1` |
| Decay R26 checkpoint | `2ccf71e14ade6dbc58848a249f9396cd36d061f64abb65f56152e8afb84585c6` |
| Freeze R26 checkpoint | `1e39813b34709e4c5741fe5ac64f2e7f2778fbc321669985aad908726c58a3a9` |
| 原恢复 execution manifest | `5dec44adb958cc3f9c0d65e811986c2f40097f45ad9ef066daf24e4c654d797f` |
| 测试 launcher | `10cfbc7483aab5238a37beeb8afae8379f78a41de506d5561e02a92d8cf87593` |
| REPORT.md | `4bc567590685ba36decf51663a71e3f19930f902c54ed33976e9de90c16831e4` |
| METRICS.csv | `60cd1a2399e7f6c6b821883f67e681124b2609dea3f3d2ef9dacd5f067185218` |
| analysis_r1/analysis_report.json | `5da4b63d168412cba7075a431a26df47e66cb8bf5e14a2f0b3b60d9354aef747` |
| RESOURCE_ACCOUNTING.json | `4d4f92d001e555dc6c5549e49281391f59b2b07869c0b0a29fc1b3f5db2fdf96` |
| LOCAL_DELIVERY_VERIFICATION.json | `f6d0145aea45aa37a2f79484019ba59935be49b5c1b51bf9bc5f5b5303a04181` |
| ARCHIVE_RECEIPT.json | `96ddc8b80a7a41f01d0e9311db4f83fb9d4db5a0706bab48acea0e5244b9266c` |
| results.tar.gz（43,577,408 字节） | `3388e825658206ee7c3d8cfe501bb090c8c1978632d5c8dfac349164a1a80c76` |
| 包内 MANIFEST.json | `ac70b6d1659d125b2e5023a2ae451d7b173f2fb028c6e48eec01eccc4cd80fd8` |

测试入口来自提交 `81b655d`，运行绑定已保存于后续提交及两份 runtime request。当前结果/交接文件由主代理统一提交；本审计不自行提交 Git，也不覆盖旧恢复报告。
