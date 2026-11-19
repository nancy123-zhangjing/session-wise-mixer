> 2026-10-26退役说明：本文属于旧的单目标历史侧交互研究，不能作为OneRec请求级列表生成的当前实现或执行方案。相关代码、模型和恢复包已按用户要求删除；历史结果保留。当前方案见 docs/research/onerec_listwise/PLAN_20261026.md。

# 阶段08交接与恢复

2026-10-24。三组训练、完整验证、CPU控制审计和本轮GPU释放已完成。完整本地交付的最终验收，以交付目录中的 `FINAL_DELIVERY_BINDING.json` 和它绑定的恢复收据为准；本文件不单独代替恢复证明。

## 本轮结论

seed2024、同一起点、同样35R/60T，完整Beauty验证集22,363人、beam256，测试集未运行。固定R35时，axial的NDCG@10为0.059860，相对baseline/no_context约+1.10%/+1.11%；HR@5相对baseline约−1.30%。按预定weighted规则各自选best，baseline R15的NDCG@10为0.060606，仍高于axial R35的0.059860。

这是值得小范围复验的信号，尚不能称稳定收益。下一步最值得验证：固定R35的小幅优势能否跨seed复现，并在相同选模规则下仍然成立。本Goal没有追加seed或测试实验。完整结果、诊断和限制见[报告](REPORT.md)。

## 实际执行身份

- code：`152bea77d43284bb41f1887a8f43124be7e6db98`。
- engine：`a28faf6a24c667aa4f49f208fb93d294ad5aab4f`，完整源码树为`c724dfc0988686bcb3c5777fa485e5bf19a27a77`。原引擎浅历史保留，便携执行提交不声称恢复其缺失父历史。
- 原本地根仓库HEAD `037876cbb3de339ad011e5abcf0e66169967ce3b`、引擎HEAD `1c879252f6395d6e64462511823c3929b9c42c00`不是本轮执行版本。冻结时保全了原索引与其他任务改动；原文快照位于`outputs/stage08_session_codeword/control/preexisting_root_changes/`。
- pre-05 Aug28推荐器和canonical E0共同暖启动；没有使用05/07最终联合权重。正式input manifest SHA256：`6386be010907cf563753e448f24b3c465c830aa543e50a08f246a420c6f94abd`，14份实际输入。
- 正式共同initial文件SHA256：`1a1277e4e29940c1ac43f174be4937b24cdc2eba1b6fbc26e952434721caa96b`；完整tensor摘要：`d1a8d37ab940c3246b86b19a723ece8cc1322efca5426400f8d154735da8cc09`。

三组都开启原全局Full Mixer。baseline禁用新F；no_context保留同样参数和深度，但时间轴只看自身；axial使用因果历史。F处理64维物理码字，输出零初始化64→256残差，加入历史SID embedding后再走原item_mlp和推荐器。目标SID仍是全局编码；本轮没有改成连续商品向量生成，也没有减少解码步数。

R更新推荐器/W/F，冻结码本与全局Mixer；T更新码本/Mixer，冻结推荐器/W/F，但梯度仍经F输入回码本。5R后10次T+3R，合计35R、3640次R更新、60次T更新。主结果固定R35，含R0的weighted-best另列。现有训练早期前缀扩样和验证最近50件历史保持不变。

## 远端与本地

远端Goal根：

```text
/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_session_codeword/goal_20261024_session_codeword_v1
```

`code/`和`engine/`是冻结执行源码，`inputs/`为完整锁定输入，`formal/`保存共同initial、三组best/last和阶段状态，`reports/final_r1/`保存原始结果与CPU审计，`delivery/`保存平台原证据、工具及来源绑定。

完整包远端位置为该根下`delivery/final_package_r1/`。本地交付根：

```text
C:/Users/nancy/Documents/experiment1/outputs/stage08_session_codeword/delivery_20261024_final_r1/
```

- `package/`：PACKAGE.json和按SHA核验的全部分卷。
- `restored/goal_20261024_session_codeword_v1/`：从完整包全新恢复的Goal根及两个独立Git源码恢复。
- `restore_tools/`：与包内工具相同字节的外置启动工具。
- `verification/`：实际恢复收据。
- `final_delivery_supplement.zip`及最终binding：包生成后完成的本地恢复证明、最终验收矩阵和项目文档。

原远端数据、历史05/07产物、CPU桥接env-zzl和共享模型环境保留。本轮5次GPU分配均已核实为空并删除终态Job；没有停止其他任务资源。

## 恢复命令

以下在项目根目录执行；`package/`必须先下载齐并通过分卷SHA。verify输出必须使用全新目录，已有恢复目录不能原地覆盖。Windows下两条恢复命令都通过 `run_cpu_restore.py` 调用原冻结工具；wrapper只给当前子进程及其Git子进程追加 `core.longpaths=true`，不修改系统设置或global Git配置。

```powershell
$delivery = 'C:/Users/nancy/Documents/experiment1/outputs/stage08_session_codeword/delivery_20261024_final_r1'
$check = "$delivery/recheck_$(Get-Date -Format yyyyMMdd_HHmmss)"
New-Item -ItemType Directory -Path $check | Out-Null
$restore = "$check/goal_20261024_session_codeword_v1"
& ./.runtime/stage08_restore_env/Scripts/python.exe -X utf8 -B "$delivery/restore_tools/run_cpu_restore.py" "$delivery/restore_tools/pack_session_codeword.py" --mode verify --package-dir "$delivery/package" --output $restore
if ($LASTEXITCODE -ne 0) { throw '完整包验证或Git恢复失败；保留当前目录，停止后续恢复。' }
$receipt = Get-Content -LiteralPath "$delivery/package/PACKAGE.json" -Raw | ConvertFrom-Json
& ./.runtime/stage08_restore_env/Scripts/python.exe -X utf8 -B "$delivery/restore_tools/run_cpu_restore.py" "$restore/delivery/postprocess_v1/verify_full_model_restore.py" --restored-goal-root $restore --manifest-sha256 $receipt.manifest_sha256 --release "$restore/delivery/PLATFORM_RELEASE.json" --output "$check/FULL_MODEL_RESTORE.json" --cpu-threads 2
if ($LASTEXITCODE -ne 0) { throw '真实模型恢复未通过；保留日志和收据，不登记最终验收通过。' }
```

本机CPU恢复环境已准备；依赖与版本来源见包内`delivery/postprocess_v1/restore_requirements.txt`、环境收据和README。第一条命令用于核输入字节、独立Git恢复及六份状态；第二条进一步构建真实Beauty模型、strict加载六份权重、恢复参数对应的优化器/调度器并做有限CPU历史前向。保留原SID，不重新量化或刷新。CPU恢复不等于GPU训练步或推荐指标重演。本轮已实际取得包/双Git恢复及六个真实模型CPU恢复PASS；完整交付以随后实际生成并verify通过的 `FINAL_DELIVERY_BINDING.json` 所绑定收据为准。

### Windows已知问题与说明优先级

首次直接执行冻结 `pack --mode verify` 时，分卷、总归档、成员文件及输入字节校验已经通过，但在Windows Git checkout阶段遇到 `Filename too long`，因此该次不能登记为完整恢复通过。失败目录保留在交付根下 `restored/goal_20261024_session_codeword_v1_failed_windows_path_r1/`，不删除或覆盖。随后通过wrapper在全新目录重做的完整包/双Git恢复，以及六个真实模型CPU恢复均已PASS，分别见新恢复根的 `BACKUP_VERIFICATION.json` 与交付根的 `verification/FULL_MODEL_RESTORE.json`；最终交付仍以绑定收据为准。

wrapper在运行前核对冻结工具SHA256：`pack_session_codeword.py` 为 `b96288eccf10e2df2c9110a06fef2846b30b700fc1838830e95c77425d74bdb5`，`verify_full_model_restore.py` 为 `db0db45d94ebf69c5a9c0a066b721ccca25758165a6757f006e82d4083e46d28`；名称或字节不符就拒绝执行，不通过修改冻结源码绕过问题。

**最终supplement中的本版交接文档和Windows wrapper，覆盖基础包内较早的直接调用恢复说明。** 使用时先按 `FINAL_DELIVERY_BINDING.json` 核对supplement及wrapper绑定的SHA，再按上方两条命令恢复。基础包和包内冻结工具保留原字节；说明更新不代表旧失败目录通过，也不替代新的真实恢复收据。

## 资源与异常

平台Scheduled至finished的生命周期合计7.755278 GPU小时；Scheduled至首次确认占用为空的保守观测合计7.949226 GPU小时。后者用于本轮预算核对，均不等于已核实账单。按平台展示1点/GPU小时线性估算约7.95点，正式结算未核实。短验收保守口径308.158秒，小于1800秒；全部占用小于累计57600秒。

首次baseline在共同initial临时文件并发发布时失败，尚无已提交训练状态。已核对完整已发布common与H100初态一致，保持源码不变，仅baseline02重试一次；失败占用计入账本。后续新campaign应先由单个发布者完成共同初态核验，再并行启动，不能单凭同机文件锁检查保证跨节点串行。当前完成的common和产物不得删除或重新初始化。

推理统计为自然完整验证流的batch8同步Encoder+generate计时，含首batch，排除数据加载和CPU指标处理；不同节点与SID搜索差异限制了成本归因。SID级指标受碰撞影响，不能直接与06的item-level检索结果横比。下一轮复验需要独立授权与预算，本轮剩余额度不自动用于追加实验。

完整迁移须同时保留基础包11个分卷及PACKAGE、补充ZIP和外置binding，不能只移动ZIP。补充ZIP按finalizer的sources映射保留 `project/` 项目文件与 `delivery/` 下相对交付根的文件。本机已布置 `D/restore_tools` 可执行上述命令；异机应先按binding核验补充ZIP的SHA，再将wrapper恢复到相应的 `D/restore_tools`，冻结pack工具可从现有restore_tools或基础包来源取得，并按requirements准备依赖。历史收据中的Windows绝对路径仅作来源记录，不能用于读取旧机器目录；这里不保证任意新机器一键恢复。