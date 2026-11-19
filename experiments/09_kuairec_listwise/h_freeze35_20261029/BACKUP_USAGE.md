# 最终完整备份与CPU恢复：执行说明

本文件依据当前`backup_delivery.py`、`download_archive.py`的实际CLI编写，**没有执行以下命令，也没有生成完成收据**。只在A/B/C原Job全部结束、资源释放及输出稳定后走完整流程；不新建GPU，不另建B专用恢复入口。早期A/B结果可先交付，完整模型恢复统一在最终全包完成。

## 1. 固定来源、工具和容量

| 项目 | 本轮实际绑定 |
|---|---|
| 唯一运行根 | `/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/h_freeze35_20261029_r1` |
| 原科学恢复扩展提交 | `e4ca4b32920c95ef7190ad475d18e6f31fa3d276` |
| `code/FREEZE35_EXECUTION_MANIFEST.json` SHA256 | `9e50d158a68f20232119edee42fae6f5c42a518a27d889b835144aa33b51f28b` |
| `backup_delivery.py` SHA256 | `f99e2293ac7dc528b53151ecd0dddbb582095ef564846021fa9af12ef2df811d` |
| `download_archive.py` SHA256 | `1217d52db527bdb2d803047d74c56ccebf4f56530ed4cdc93b6440aa3ea8eec1` |
| 本地真实CPU环境 | `C:/Users/nancy/Documents/experiment1/.runtime/stage08_restore_env/Scripts/python.exe`，Torch `2.6.0+cpu` |

打包工具可以逐SHA上传到CPU Notebook的`/tmp/freeze35_backup_delivery.py`，不放入或修改冻结`code/`；远端和本地执行前均核对上述helper SHA，不能直接换成当前工作区另一版本。最终包会将实际打包helper收入`references/tools/backup_delivery.py`。恢复使用包内冻结源码和引擎，不能改用当前HEAD。

容量先按**6–7GB未压缩tar的预算估算**准备，最终以停止写入后的运行目录、额外父/祖先/历史选中状态及manifest实测为准。pack包含整个运行根，短测bundle也可能占空间；不能为缩小包擅自删除。远端需在运行根之外另留一整包空间；本地需同时保留tar和新鲜解压副本，按`tar大小＋payload大小`再留余量，暂按约20GB额外可用空间准备并在实际打包前复核。现有旧依赖和early包/解压目录不计作可删除空间。预算中的30–120分钟离线工作是操作估计，不是网络或恢复耗时保证。

## 2. 先生成真实释放收据，不能先填PASS

需要核对4次已知GPU分配：唯一短测、A、B、C；若实际新增过失败尝试，也一并入账。短测及B现有证据为`PREFLIGHT_EVENTS.json`/`PREFLIGHT_STATUS.json`、`B_RELEASE_EVENTS.json`/`STATUS_20261029T1605.json`。A/C须等各自真实终态及删除/终止事件；仅训练完成、测试生成完成或exit0不能代替资源释放。

根线程检查实际平台状态、Job名称、GPU数和事件关联，确认无本Goal仍占卡的Job/Pod，再将原始状态/事件JSON的**原字节副本**放到远端新目录，例如`运行根/delivery/final_lifecycle_r1/`，逐文件核SHA。不要仅上传本地Windows路径字符串：pack读取的是CPU Notebook能定位的远端绝对路径。

`backup_delivery.py pack --release-receipt`实际读取的schema只有以下必需项；**全部核验后才写值**：

| 字段 | 必需值及含义 |
|---|---|
| `status` | `verified`，由根线程实际核验产生 |
| `all_goal_jobs_released` | 布尔`true`，包含短测及全部正式/失败分配 |
| `evidence` | 非空数组，每条为`{"path": "CPU可读的真实证据绝对路径", "sha256": "该文件真实64位SHA256"}` |

建议额外记录`checked_utc`、逐Job名称/状态/分配秒数、事件证据对应关系和费用未结算状态。helper本身**不查询平台，也不解析这些证据证明语义**，只要求声明字段并核对每个证据文件的SHA；真实性仍由实际平台收据和最终审计支持。证据上传、报告副本等全部完成后保持根目录静止，再开始pack。pack期间不要继续往运行根写观察日志或报告。

pack将释放收据复制为`references/lifecycle/release.json`，其`evidence`按原数组顺序复制为`references/lifecycle/evidence000.json`等，并纳入成员SHA清单。原收据的远端`path`保留；本地审计应按数组序号及SHA定位这些已解压副本，不能因原远端路径仍写在JSON中就声称本地证据不可用，或跳过对原字节的核对。

最终本地分析/图表不会自动出现在远端全包：若需纳入本次tar，先将已完成分析的逐SHA副本放到运行根下全新`delivery/report_snapshot_r1/`再打包。pack后产生的恢复收据、最终审计/交接和Git记录另作有SHA绑定的交付件，不能声称已包含在较早tar中。

## 3. CPU Notebook远端pack

以下Bash路径是本轮可用的明确命名方案，使用前确认**tar及其`.receipt.json`均不存在**；若已存在，不覆盖，先查原执行结果，需要新包时换全新后缀。归档放在个人stage09根下、运行目录外，名称满足断点下载器的`h_freeze35_20261029*.tar`限制。

```bash
FREEZE35_BASE=/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09
FREEZE35_ROOT="$FREEZE35_BASE/h_freeze35_20261029_r1"
FREEZE35_ARCHIVE="$FREEZE35_BASE/h_freeze35_20261029_final_r1.tar"
FREEZE35_PY=/inspire/hdd/project/project-public/<inspire-account>/nancy/conda-envs/grm-exp/bin/python3.10
FREEZE35_TOOL=/tmp/freeze35_backup_delivery.py

sha256sum "$FREEZE35_TOOL"
"$FREEZE35_PY" -B "$FREEZE35_TOOL" pack \
  --root "$FREEZE35_ROOT" \
  --archive "$FREEZE35_ARCHIVE" \
  --execution-manifest-sha 9e50d158a68f20232119edee42fae6f5c42a518a27d889b835144aa33b51f28b \
  --release-receipt "$FREEZE35_ROOT/delivery/final_lifecycle_r1/RELEASE.json" \
  --references-mode remote
```

`RELEASE.json`是上一步根线程核验后新建的真实收据，不是本说明已提供的文件。pack会核代码/元数据清单、三臂完成/selection/测试、全部63检查点收据和last；同时打入3父状态、A祖先及实际需要的旧A best。运行根在打包前后须文件清单/SHA一致。失败留下的tar不能视为成功，也不自动删掉或重试覆盖。

成功后原路径旁生成`h_freeze35_20261029_final_r1.tar.receipt.json`，状态`packed_not_model_restored`，包含实际`bytes`、`sha256`、`manifest_sha256`。经CPU文件入口取回这份**原字节小收据**，保存为本任务目录的`FINAL_PACK_RECEIPT.json`并记录其来源/SHA，再作为下载强绑定参数；不能复用早期B包的大小或SHA。

## 4. Windows断点下载：必须先等SCP/其他writer彻底退出

下面PowerShell变量和命令均从仓库根执行；拟用目的地必须是普通文件/目录，不能经过symlink或junction。最终新鲜解压目录及CPU恢复收据目录此时必须不存在。

```powershell
$Repo = 'C:/Users/nancy/Documents/experiment1'
Set-Location -LiteralPath $Repo
$Task = Join-Path $Repo 'experiments/09_kuairec_listwise/h_freeze35_20261029'
$PythonCpu = Join-Path $Repo '.runtime/stage08_restore_env/Scripts/python.exe'
$BackupTool = Join-Path $Task 'backup_delivery.py'
$DownloadTool = Join-Path $Task 'download_archive.py'
$Archive = Join-Path $Repo 'outputs/stage09_restore/h_freeze35_20261029_final_r1.tar'
$Extracted = Join-Path $Repo 'outputs/stage09_restore/h_freeze35_20261029_final_r1_restored'
$CpuReceipts = Join-Path $Task 'final_cpu_restore_r1'
$RemoteArchive = '/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/h_freeze35_20261029_final_r1.tar'
$PackReceipt = Get-Content -LiteralPath (Join-Path $Task 'FINAL_PACK_RECEIPT.json') -Raw | ConvertFrom-Json
if ($PackReceipt.status -ne 'packed_not_model_restored') { throw 'Pack receipt is not accepted' }
if ((Get-FileHash -LiteralPath $BackupTool -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'f99e2293ac7dc528b53151ecd0dddbb582095ef564846021fa9af12ef2df811d') { throw 'Backup tool SHA mismatch' }
if ((Get-FileHash -LiteralPath $DownloadTool -Algorithm SHA256).Hash.ToLowerInvariant() -ne '1217d52db527bdb2d803047d74c56ccebf4f56530ed4cdc93b6440aa3ea8eec1') { throw 'Download tool SHA mismatch' }

& $PythonCpu -B $DownloadTool --remote $RemoteArchive --output $Archive `
  --bytes $PackReceipt.bytes --sha256 $PackReceipt.sha256 `
  --manifest-sha256 $PackReceipt.manifest_sha256 --other-writers-stopped
if ($LASTEXITCODE -ne 0) { throw 'Download failed; preserve prefix and inspect' }
```

默认仅通过CPU Notebook `nancy-stage09-prep-20261027-r1`、账号`sii-shared`执行只读SSH；确需换已认证CPU入口时，已有参数为`--notebook`及`--account`。8MiB块，先认证本地prefix及远端整包/嵌入manifest SHA，再逐块认证，当前块最多3次尝试；最终再核整包及嵌入manifest。不要并发SCP或启动第二writer；不要清空/截断/覆盖失败前缀。旧`.download.lock`须先核原进程确已退出，不自动删锁绕过。

成功文件为`<tar>.DOWNLOAD_BYTES_VERIFIED.json`，状态`BYTES_VERIFIED_ONLY`。它证明传输字节，**不证明成员安全解压、完整状态检查或模型恢复**。

## 5. 本地依赖保留、必要时显式重定位

默认直接用已经认证、仍存在的路径：

```powershell
$Dataset = Join-Path $Repo 'outputs/kuairec_listwise_v1/run1'
$Assets = Join-Path $Repo 'outputs/stage09_restore/goal_20261027_r1/assets/pse_v1'
$MapArgs = @()
```

数据manifest SHA应为`8fe1506fbbe3361048ad9b2862415a793ecefe538a8478cc3ab0d7e2a3b9b649`，PSE manifest为`7cc6dcdfcdeb4087e38fd2080f322a01375b189b284a59d5c9bfd279862525d4`；工具会核每个实际依赖文件。历史结果路径由`FREEZE35_REFERENCE_RESULTS.json`的`local`字段固定，分布于本地`goal_20261027_r1`、`h1_recovery_20261028_r2`、`h1_recovery_test_20261029_r1`。父/祖先及选中旧A状态另由pack加入archive的`references/`，不能把旧预测当作恢复权重。

**这些data/PSE/历史预测默认仍是外部依赖**；没有全部自包含时，只能称“归档＋已验外部依赖可恢复”，不能称“单包独立可恢复”。保留`outputs/stage09_restore/h_freeze35_20261029_early_B`、其原tar与传输/提取收据、`final_analysis/B`及后续early_A/分析引用的原始依赖。最终包并不自动重写先行分析报告的路径或SHA绑定，也不自动把它们当成同一存储位置；若迁移，须额外记录同SHA来源映射，不能直接删原路径。

只有实际迁移旧依赖时才写新的path-map JSON。例如将原`stage09_restore`迁到另一真实目录，可用以下**示意键值，必须替换成已经存在并验字节的实际目的地**：

```json
{"C:\\Users\\nancy\\Documents\\experiment1\\outputs\\stage09_restore": "D:\\verified\\stage09_restore"}
```

此时显式将`$Dataset`、`$Assets`设为新位置，设置`$MapArgs = @('--path-map', '该真实JSON绝对路径')`，下面verify和restore都传同一组参数。工具只对依赖路径作精确/最长前缀映射并重新核SHA，**不修改原manifest、原checkpoint或科学配置**；该参数不是忽略绑定错误的开关。

## 6. 新鲜安全解压、全量结构覆盖verify

```powershell
& $PythonCpu -B $BackupTool verify --archive $Archive --destination $Extracted `
  --archive-sha $PackReceipt.sha256 --manifest-sha $PackReceipt.manifest_sha256 `
  --dataset $Dataset --assets $Assets @MapArgs
if ($LASTEXITCODE -ne 0) { throw 'Verify failed; do not start restore' }
```

先核整包和manifest SHA，只允许manifest完整列出的非稀疏普通成员，拒绝路径穿越、链接、重复/大小写冲突及非新目标；解压后逐成员实际SHA再核一次，并核冻结代码、三臂收据及外部依赖。通过收据是`$Extracted/BACKUP_VERIFICATION.json`，状态`bytes_and_structural_coverage_passed`，明确`actual_model_restore=pending`。

应有**63份不可覆盖完整状态＋3份rolling last**：A14、B6、C43；已含三个`R35_training_boundary.pt`和C的`R20_pre_freeze.pt`。A/C的共同网格best由实际不可覆盖文件或认证旧A状态引用，不一定另有`best.pt`；不能自行要求不存在的文件名，也不能把最早59槽最低计划当最终覆盖总数。

## 7. 使用已审计入口统一进行真实CPU恢复

```powershell
$VerificationSha = (Get-FileHash -LiteralPath (Join-Path $Extracted 'BACKUP_VERIFICATION.json') -Algorithm SHA256).Hash.ToLowerInvariant()
& $PythonCpu -B $BackupTool restore --extracted $Extracted --output $CpuReceipts `
  --python $PythonCpu --verification-sha $VerificationSha `
  --dataset $Dataset --assets $Assets @MapArgs
if ($LASTEXITCODE -ne 0) { throw 'Actual CPU restore failed; preserve logs, no automatic retry' }
```

`--output`须全新且在解压树外；不要直接调用私有`_cpu-worker`，不要伪造`BACKUP_VERIFICATION.json`，没有`--branch`/`--reuse-B`恢复参数。现入口固定统一处理A/B/C；不另设早期B恢复程序。

该步骤从**解压后的冻结源码/引擎**构建真实模型及双优化器/scheduler。全部63＋3状态实际读取核字段、shape/dtype/有限性、R/T计数/阶段/原LR日程、冻结指纹、SID范围/PAD和RNG字段；关键roles为各臂initial与R35、C R20及每个选中best。重合路径去重实际加载，旧A best按真实选择走已认证旧envelope。对关键roles核恢复后model/alignment/双优化器/scheduler指纹及CPU RNG，并以真实B2×L50历史做无目标有限前向，要求SID未刷新、完整状态不变、零训练更新。

输出为`$CpuReceipts/{A,B,C}.json`及日志，最终`FULL_MODEL_RESTORE.json`状态`real_cpu_restores_passed`。其中`checked_states`须覆盖各臂所有命名状态＋last，`restored`须覆盖manifest的实际关键roles；不能只看总status。只允许经过核验的**内存路径元数据重定位**，并保留old/new绑定SHA，磁盘checkpoint原字节不改。CPU恢复把CUDA RNG保留为来源指纹而不实际恢复，**不是CUDA精确续训验收**；后者来自本轮独立H100短测证据。

## 8. 交付边界与最后签收

| 收据/文件 | 能证明 | 仍不能证明 |
|---|---|---|
| 远端`.tar.receipt.json` | 源稳定、manifest/整包字节、结构及依赖核对完成 | 下载、真实模型恢复、结算 |
| `.DOWNLOAD_BYTES_VERIFIED.json` | 正确prefix/逐块/整包及嵌入manifest字节 | 解压安全/覆盖、模型/指标验收 |
| `BACKUP_VERIFICATION.json` | 成员/落盘SHA、冻结源、63＋3结构、依赖 | 实际模型/优化器装载 |
| `FULL_MODEL_RESTORE.json`及A/B/C明细 | 全状态检查及关键真实CPU模型恢复 | CUDA轨迹、推荐指标正确、平台结算 |
| 独立分析/最终审计、释放台账、Git/交接 | 各自绑定范围内的科学和交付验收 | 不能被任一前述单收据代替 |

最后将真实收据、最终分析/图表/报告、源码执行版本、外部依赖清单及GPU终态/费用边界纳入交付索引，完成仅本任务Git提交和项目交接；全部完成后再停用heartbeat并保留回读。备份工具不自动完成Goal，也不自动核实实际账单。
