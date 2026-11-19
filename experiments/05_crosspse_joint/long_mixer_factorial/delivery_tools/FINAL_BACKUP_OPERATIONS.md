# 最终完整包后的执行清单（准备稿，尚未执行）

这里只准备后续命令。正式评估、最终artifact审计、7个本Goal Job的Pod释放核验、完整打包和下载尚需各自的实际证据。不得用本清单、语法检查或早先本地bundle探针冒充完成。

## 路径和资源边界

训练版本为code `69682698c15ba796b98e6329c950061a2f68a05b`、engine `1c879252f6395d6e64462511823c3929b9c42c00`；分析工具使用锁定的 `delivery/analysis_tools_v3r2`（分析提交 `4b40c90b56490868e5972e75bafbf8e50ca8a3de`）。不要更改这些目录中的源码。

远端CPU命令可以设置以下变量，再逐项执行。变量只缩写已确定的绝对路径，不改共享环境：

```bash
G=/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_training/goal_20261020_long_mixer_v1
P=/inspire/hdd/project/project-public/<inspire-account>/nancy/conda-envs/grm-exp/bin/python
T="$G/delivery/analysis_tools_v3r2"
C=goal_20261020_long_mixer_v1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
```

本轮评估前6个Job实际释放合计33609 GPU秒，评估cap3960秒，总预约37569秒，距43200秒上限余5631秒。最终记账仍要改用评估Job的真实Scheduled→SuccessfulDeletePod区间，不能把cap、进程退出或本清单当作资源释放证据。打包、下载、Git恢复和消费者恢复均在CPU执行，不留GPU等待。

## 门控、打包与归档目录

1. 回读四臂正式测试、选择封存及 `COMPUTE_COMPLETE.json`。使用独立完整artifact审计（不带 `--training-only`），核对9个完整状态、真实SID统计、封存的best选择及测试，保存新的 `AUDIT.json`。
2. 评估Job也必须查到平台终态及Pod释放事件；以7项完整平台inventory执行 `audit_mixer_long_resources.py --final`，核对所有本Goal Job及实际累计时长。保留raw status/events、最新完整Job listing和费用口径。
3. 保存实际曲线、原始CSV和表，准备恢复指南和源码锁；旧证据保留。选定的新目录必须未存在。
4. 将新辅助脚本单独放到本轮 `prep/` 或 `delivery/` 并记录SHA。只添加辅助文件，不修改冻结训练/分析目录。最终正式门控通过后，可先调用收集预览：

```bash
"$P" -B "$G/prep/inspect_final_pack_plan.py" --goal-root "$G" --artifact-audit "$G/delivery/final_audit_r1/AUDIT.json" --release "$G/delivery/RESOURCE_RELEASE_FINAL.json" --packer "$T/scripts/analysis/pack_mixer_long_delivery.py" --planned-output "$G/delivery/final_package_r1" --receipt "$G/delivery/FINAL_PACK_COLLECTION_FORECAST.json"
```

其中artifact审计与资源收据的文件名是建议名；调用时必须使用实际成功输出路径。预览调用冻结collector验证门控，只生成字节规模预测，不创建压缩包，也不宣称恢复成功。它要求9个正式状态和3个Git bundles，并拒绝将旧 `mixer_long_backup.tar.gz`、`PACKAGE.json` 或 `BYTE_VERIFICATION.json` 当作delivery/prep普通成员重收。

```bash
"$P" -B "$T/scripts/analysis/pack_mixer_long_delivery.py" --mode pack --goal-root "$G" --audit "$G/delivery/final_audit_r1/AUDIT.json" --release "$G/delivery/RESOURCE_RELEASE_FINAL.json" --output "$G/delivery/final_package_r1" --part-mib 512
```

包输出命名为 **`final_package*`**；解压命名为 **`package_extract*`**；CPU恢复scratch及新Git目录命名为 **`restore_check*`**，或放在本轮Goal以外。这些前缀受冻结collector明确排除。不要在 `delivery/some_backup` 下放旧包后再次打包；delivery/prep本身递归扫描，会重复带入没有排除前缀的旧包、解压输入或源码。

首次pack在收集完成后才创建输出，所以本次不会即时自吞正在写出的包；后续再次pack则必须继续遵守排除目录规则。排除是按路径成员检查，rglob仍会遍历目录，所以大型恢复工作放Goal外或使用上述小范围目录更省扫描时间。

## 下载与Windows本地验证

只下载 `PACKAGE.json` 和它列出的全部分卷。远端保留未切分tar.gz作为冗余副本，若PACKAGE列出的是part文件，本地无需再下载同内容完整tar.gz。下载前按最终 `archive_size_bytes`、`uncompressed_bytes` 核实磁盘容量，保留压缩包、解压数据、新Git快照及收据的空间。

建议本地使用较短的专有路径，例如：

```text
C:/Users/nancy/Documents/experiment1/backups/stage05_long/package_r1
C:/Users/nancy/Documents/experiment1/backups/stage05_long/extract_r1/goal_20261020_long_mixer_v1
C:/Users/nancy/Documents/experiment1/backups/stage05_long/git_restore_r1
```

以下命令在PowerShell执行；两个输出目录必须全新，所有输入必须来自最终下载包。

```powershell
python -B scripts/analysis/pack_mixer_long_delivery.py --mode verify --package-dir backups/stage05_long/package_r1 --output backups/stage05_long/extract_r1/goal_20261020_long_mixer_v1
python -B .runtime/stage05_long/restore_git_from_final_backup.py --package-dir backups/stage05_long/package_r1 --backup-root backups/stage05_long/extract_r1/goal_20261020_long_mixer_v1 --output-dir backups/stage05_long/git_restore_r1
```

先核对本地pack verifier哈希与锁定v3r2对应文件一致，再运行。Git辅助脚本会重核最终已下载的全部分卷和解压成员，将receipt绑定到相同PACKAGE/MANIFEST；从备份的三个bundles恢复两个新Git仓库，设置 `core.autocrlf=false`、`core.eol=lf`、`core.longpaths=true`，并在engine fetch前用ASCII无BOM写入shallow边界。最终逐一比较1493份跟踪源码与已验证备份的字节，核对两个HEAD/tree、`fsck`、干净状态和19份formal identity源码，成功后写 `FINAL_GIT_RESTORE_PASS.json`。

源快照/Git恢复不代表模型恢复。本地continuous_env可以运行Tensor及优化器内容独立审计，但缺少完整Pipeline依赖，不在本地虚报消费者恢复成功。

## 远端已有CPU环境执行完整消费者恢复

在已有CPU桥接的grm-exp环境，从**同一完整最终压缩包**解压出全新副本。不要直接把仍在原路径的live code/inputs/formal当作“从备份恢复”。

```bash
"$P" -B "$T/scripts/analysis/pack_mixer_long_delivery.py" --mode verify --package-dir "$G/delivery/final_package_r1" --output "$G/delivery/package_extract_r1/$C"
"$P" -B "$T/scripts/analysis/restore_mixer_long_backup.py" --backup-root "$G/delivery/package_extract_r1/$C" --scratch-root "$G/delivery/restore_check_cpu_r1/$C"
```

恢复器在新进程中强制CPU，重核所有manifest字节，并使用解压包自身code/engine和14件输入构造真实Pipeline；校验共同初态后，逐个恢复四臂best/last的strict模型、完整Adam、scheduler、Python/NumPy/CPU RNG、SID及非persistent E0 anchor。CUDA RNG只保留状态字节及哈希，CPU消费者不执行CUDA。CUDA同后端下一阶段重放另引真实H100验收收据。

把scratch中的 `RESTORE_PASS.json` 下载到本地备份旁的独立收据目录，与本地 `BYTE_VERIFICATION.json`、`FINAL_GIT_RESTORE_PASS.json`、PACKAGE和MANIFEST一起保留；比较PACKAGE/MANIFEST及byte-verification哈希，确认两地校验和CPU恢复实际绑定同一组最终压缩字节。侧边收据不改原压缩包和MANIFEST成员，避免循环重打包来“把恢复证明放进自身”。

## collector体积与状态覆盖的实际依据

只读核对冻结Git tree：code696有1456个tracked文件、31,372,597字节（29.92MiB）；engine1c有37文件、494,034字节（0.47MiB）。根tree没有tracked `.runtime/`、`backups/` 或 `formal/`；long_mixer_factorial中仅4份当时已跟踪的小文档/CPU日志。因此Git源码快照不会递归装入当前工作区的新训练、下载或恢复目录。

同一锁定CPU输入manifest有14件输入，共192,880,021字节（183.94MiB），其中最大的两个文件是Beauty原始压缩数据。正式共同初态及四臂best/last共9个完整状态由collector显式加入，不受delivery扩展名白名单影响；final与last为同一个R35完整状态，不需要虚构另一份final.pt。smoke状态另按其.pt白名单收集，包含GPU恢复证据。formal/tokenizer_boundary.pt不属于最终9个承诺状态；不能称全部中间checkpoint已归档。

delivery/prep的历史工具目录、分析tar.gz及source bundles可能有少量重复，但源码全tree约30.4MiB，不能据此声称会出现权重大规模丢失或无限递归。最终完整状态体积、smoke重复数量及额外delivery实际大小仍以门控后的collection预览和最终MANIFEST为准，当前本地静态树核对不能代替远端完整包清单。

只有正式四组、控制/指标、最终包本地逐字节验证、从最终下载包恢复Git、真实CPU消费者恢复和全部资源释放等证据齐全，才更新项目交接并申请Goal完成。
