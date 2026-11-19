# Joint r2正式交付运行说明

仅在B真实完成后执行。工具的CPU自测已经通过，无需重复；本说明不是正式产物通过收据。完整产物存在时不重训、不重复测试。所有输出选新路径，失败保留原收据并定位原因。

冻结训练根：`/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage07/goal_20261020_item_local_joint_r2`；root提交`842ff2aa3b0ba95d0ba2a66dea74c485abaa11a1`，engine提交`1c879252f6395d6e64462511823c3929b9c42c00`。远端CPU Python为`/inspire/hdd/project/project-public/<inspire-account>/nancy/conda-envs/grm-exp/bin/python`。本目录工具版本见[MANIFEST.json](MANIFEST.json)，与冻结训练源码分别绑定；已核对的远端副本位于`<goal>/delivery/`，直接复用，不新建另一套工具目录，也不更新运行用code/engine。

1. **先收齐完成及释放证据。** 两臂目录分别为`runs/stage07_item_local_joint_control_seed2024_20261020_r1`、`runs/stage07_item_local_joint_item_local_seed2024_20261020_r1`。核对实际`COMPLETE.json`的formal_completed及results SHA、`TEST_STARTED.json`/`TEST_RESULT.json`的一次测试与同一best/SID、`checkpoint_reload.json`、成功formal wrapper收据（本轮`delivery/formal_<variant>_attempt1/receipt.json`）。另保存B平台终态、含SuccessfulDeletePod的事件、Scheduled起点及最新完整作业清单。训练仍运行或只有marker时，不执行pack。

2. **补最终资源账。** 已释放7个作业共`19838=204+184+10885+364+102+207+7892`秒；再加B从Scheduled到Pod删除的实测秒数，累计上限28 GPU小时。以下从仓库根运行；B两个TERMINAL文件是完成后需新采集的文件名，不是当前已有证据。原两次pilot smoke由`--receipts-dir`自动计入，其他6个作业逐组三元组传入；同一类的status/events/start-events按相同顺序重复。

```powershell
$py = '.runtime/continuous_env/Scripts/python.exe'
$ea = 'experiments/07_item_local_attention/execution_acceptance'
$j = 'experiments/07_item_local_attention/joint_alternating'
& $py -X utf8 -B scripts/analysis/audit_item_local_resources.py `
  --receipts-dir $ea `
  --approved-budget-receipt "$ea/REPAIR_BUDGET_AUTHORIZATION_20261020.json" `
  --formal-status "$ea/FORMAL_CANCELLED_STATUS_20261020.json" `
  --formal-events "$ea/FORMAL_CANCELLED_EVENTS_20261020.json" `
  --formal-start-events "$ea/FORMAL_EVENTS_START_20261020.json" `
  --repair-status "$ea/ocn_repair_r1/PROFILE_TERMINAL_FOR_JOINT_SCOPE.json" `
  --repair-events "$ea/ocn_repair_r1/PROFILE_EVENTS_STOP_FOR_JOINT_SCOPE.json" `
  --repair-start-events "$ea/ocn_repair_r1/PROFILE_EVENTS_SCHEDULED.json" `
  --joint-scope-receipt "$j/SCOPE_OVERRIDE_20261020.json" `
  --joint-status "$j/SMOKE_STATUS_AFTER_WAIT.json" `
  --joint-events "$j/SMOKE_EVENTS_AFTER_WAIT.json" `
  --joint-start-events "$j/SMOKE_EVENTS_AFTER_WAIT.json" `
  --joint-status "$j/SMOKE_R2_STATUS_AFTER_EXIT.json" `
  --joint-events "$j/SMOKE_R2_EVENTS_AFTER_EXIT.json" `
  --joint-start-events "$j/SMOKE_R2_EVENTS_AFTER_EXIT.json" `
  --joint-status "$j/FORMAL_CONTROL_TERMINAL_STATUS_R2.json" `
  --joint-events "$j/FORMAL_CONTROL_TERMINAL_EVENTS_R2.json" `
  --joint-start-events "$j/FORMAL_CONTROL_EVENTS_SCHEDULED_R2.json" `
  --joint-status "$j/FORMAL_ITEM_LOCAL_TERMINAL_STATUS_R2.json" `
  --joint-events "$j/FORMAL_ITEM_LOCAL_TERMINAL_EVENTS_R2.json" `
  --joint-start-events "$j/FORMAL_ITEM_LOCAL_EVENTS_SCHEDULED_R2.json" `
  --output "$j/ALLOCATION_FINAL_R2_NEW.json"
```

先对最新完整inventory核对8个作业无漏项/重复、全部终态及Pod删除，再使用`--all-goal-jobs-accounted`并选择新输出文件，才可声明`all_goal_gpu_jobs_accounted=true`；若出现额外尝试，必须追加，不能仍称只有8个。此flag是调用者的完整清单声明，不是工具自动发现。将最终账本和平台收据副本放到远端goal的delivery下，供后续包收集；保留本地原件。

3. **远端CPU审计原目录。** 以下是CPU桥接进程中的bash命令；把`YYYYMMDD_HHMMSS`替换为本轮唯一标签。审计依赖原goal的code/engine/runs/inputs/delivery，不能直接对排除了输入的本地解压树运行。

```bash
G=/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage07/goal_20261020_item_local_joint_r2
PY=/inspire/hdd/project/project-public/<inspire-account>/nancy/conda-envs/grm-exp/bin/python
T="$G/delivery"
ROOT_SHA=842ff2aa3b0ba95d0ba2a66dea74c485abaa11a1
ENGINE_SHA=1c879252f6395d6e64462511823c3929b9c42c00
export CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=none
"$PY" -X utf8 -B "$T/audit_item_local_joint_pair.py" --goal-root "$G" --root-commit "$ROOT_SHA" --engine-commit "$ENGINE_SHA" --output "$G/delivery/JOINT_PAIR_AUDIT_YYYYMMDD_HHMMSS.json"
```

须退出码0且为`JOINT_PAIR_ARTIFACT_AUDIT_PASS`；检查实际两臂、源绑定、完整曲线、一次测试、选模、梯度/Adam/状态及比较字段。它不证明GPU释放，也不保证达到质量初筛标准。

4. **远端CPU打包。** 上一步真实通过后执行；新包目录在goal之外，保留原产物字节。packer固定engine为上述1c提交，`--root-commit`传训练提交而非后处理工具版本。

```bash
PKG=/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage07/joint_delivery_package_YYYYMMDD_HHMMSS
"$PY" -X utf8 -B "$T/pack_item_local_joint_delivery.py" --mode pack --goal-root "$G" --root-commit "$ROOT_SHA" --output "$PKG" --part-mib 1024
```

下载`$PKG/PACKAGE.json`和其中`parts[].member`列出的全部文件，保留逐件传输收据并核对SHA/size；小包可能是`joint_delivery.tar.gz`，大包是`.part0001`等，以PACKAGE清单为准。13输入在远端逐件验字节并留身份收据，原始数据/cache不在包内；不能声称本地已具备完整异地恢复输入。

5. **本地字节恢复。** 以下从`C:\Users\nancy\Documents\experiment1`执行；将标签替换为实际下载目录。PACKAGE及parts放在`$package`；`$backup`必须尚不存在。使用同目录已验收packer及其`verify_item_local_archive.py`。

```powershell
$py = '.runtime/continuous_env/Scripts/python.exe'
$j = 'experiments/07_item_local_attention/joint_alternating'
$tools = "$j/postprocess_tools_r2"
$package = 'outputs/stage07_item_local_joint/delivery_YYYYMMDD_HHMMSS/package'
$backup = 'outputs/stage07_item_local_joint/delivery_YYYYMMDD_HHMMSS/verified'
$env:CUDA_VISIBLE_DEVICES = ''
$env:NVIDIA_VISIBLE_DEVICES = 'none'
& $py -X utf8 -B "$tools/pack_item_local_joint_delivery.py" --mode verify --package-dir $package --output $backup
```

大文件放在仓库已忽略的`outputs/`下，轻量验收收据和报告仍放在`$j`。须退出码0，生成`$backup/BYTE_VERIFICATION.json`且为`BYTE_RESTORATION_VERIFIED`；它校验包/分卷/manifest以及每个还原文件的SHA/size，尚未加载权重。

6. **回读本地实际权重与状态，再画真实曲线。** 每条命令成功后才执行下一条，输出均须新建。

```powershell
& $py -X utf8 -B .runtime/stage07/verify_joint_backup_checkpoints.py --backup-root $backup --package-dir $package --output "$j/LOCAL_CHECKPOINT_READBACK_YYYYMMDD_HHMMSS.json"
& $py -X utf8 -B .runtime/stage07/render_joint_curves.py --backup-root $backup --output "$j/curves_YYYYMMDD_HHMMSS"
```

回读须为`LOCAL_CPU_CHECKPOINT_READBACK_PASS`，实际CPU加载两臂formal+smoke全部`.pt/.bin`，至少14份必要文件，检查有限Tensor、联合边界/Adam/scheduler/RNG/runtime及embedded-best/results/reload SHA。它不是完整异地resume实测。曲线生成PNG/PDF和`CURVE_INPUTS_AND_SELECTION.json`；epoch0保留为验证候选，Tokenizer loss和test点不参与曲线选模。

7. **报告与总验收。** 用真实B最佳验证/唯一测试、两臂最佳SID碰撞及成对审计差值补[REPORT.md](../REPORT.md)，按B相对A测试NDCG@10至少+3%且HR@10不降记录初筛；注明SID级、单seed、03仅背景。汇总远程成对审计、最终资源清单及释放、包/本地字节恢复、实际14份权重回读和曲线收据及SHA。工具自测、单个PASS或进程成功均不能替代这些真实产物验收；未齐全时不宣称Goal完成。
