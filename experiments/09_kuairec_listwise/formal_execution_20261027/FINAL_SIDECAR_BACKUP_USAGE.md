# 最终外围文件备份：已部署，尚未打包

新增 `backup_final_sidecars.py`，不修改训练源码或已有 `backup_completed_run.py`。运行时必须与原 helper 同目录，后者 SHA 固定为 `0800f6dd99b573b8780ebe8b1eea3ca64fe4713d527475e620a6b943fb65f403`。部署前须核对两文件 SHA。脚本不调用平台、不读取模型 state、不用 GPU。

固定远端根 G：`/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/goal_20261027_r1`。

## 两种固定范围

- `operations`：四组 complete/nolock、正式 supervisor 干净结束、原生平台正式 Job SUCCEEDED 且释放后才允许。收录固定运行配置、subset、warmup/branch binding、实际执行/备份 helper、输入来源清单、三个训练日志、`delivery/four_arm_supervisor_r1` 中明确允许的直接子文件，以及四 complete 与终态证据。不能递归整个 delivery，拒绝未知 supervisor 文件。
- `evaluations`：上述门槛之外，还要四个评估 Job 分别 SUCCEEDED/释放、四 best freeze SHA 与当前文件一致，12 个 summary/gzip 完整。检查4份完整 test 各53194行、8份固定 validation 各1024行、split/beam/architecture/seed/FP32、validation sample ID 集合及固定子集SHA。封包不重算推荐指标；独立 metric replay 仍需另外完成。

评估包包括12结果目录中的 summary/per_sample、4日志、配置/subset、正式 supervisor/四complete、final acceptance 与备份 helper，以及终态/冻结/额外 JSON 证据。两种包都不包含 checkpoint、原始数据、BGE/生成资产或已有 tar；这些由既有 run/asset/source 包交付。

## 先准备真实终态输入

由外部验收者只在实际终态后，把原生查询的 **未改写 JSON 副本** 放到 `G/delivery/backup_evidence/`；该目录由部署流程显式创建，helper不自动合成平台回执。

按 scope 创建 `operations_terminal.json` 或 `evaluations_terminal.json`。下面是形状示例，含占位符，**不是可用的完成凭据**：

```json
{
  "schema": "stage09_sidecar_terminal_inputs_v1",
  "root": "/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/goal_20261027_r1",
  "scope": "operations",
  "jobs": [{
    "name": "nancy-stage09-four-arm-20261027-r1",
    "status_receipt": {"path": "delivery/backup_evidence/TRAIN_STATUS.json", "sha256": "ACTUAL_64HEX"},
    "events_receipt": {"path": "delivery/backup_evidence/TRAIN_EVENTS.json", "sha256": "ACTUAL_64HEX"},
    "release_audit": {
      "job_name": "nancy-stage09-four-arm-20261027-r1",
      "all_allocated_pods_released": true,
      "native_events_query_was_scoped_to_this_job": true,
      "checked_at_utc": "ACTUAL_UTC",
      "auditor": "ACTUAL_VERIFIER"
    }
  }],
  "additional_json": []
}
```

原生 status 接受已观察到的 `success/data/name/status` 或 `success/data/items` 两种结构，必须精确命中 Job 名且 SUCCEEDED。events 按已有真实 `success/data/items` 结构检查 `SuccessfulDeletePod`。原生 events 本身通常没有 Job 名，因此必须由外部验收记录 scoped 查询与全部分配 Pod 释放；脚本不会把自写索引本身当作平台证明。脚本也不能保证下载很久以前的收据仍是最新状态，执行者必须先查实时状态、确认无重启。

`evaluations` 索引 jobs 必须恰为正式 Job 加以下4个：`nancy-stage09-final-eval-{j0,j1,h0,h1}-20261027-r1`。每项使用独立原生 status/events 文件与真实 SHA；若最终执行改了 Job 名/重试布局，应先审查后更新窄白名单，不可改写回执冒充这些名字。

评估索引还须有：

```json
"four_best_freeze": {
  "path": "delivery/backup_evidence/FOUR_BEST_FREEZE.json",
  "sha256": "ACTUAL_64HEX_OF_UNCHANGED_FINAL_ACCEPTANCE_OUTPUT"
}
```

它必须是实际 `formal_final_acceptance.py` 输出的 `stage09_four_arm_final_selection_v1` / passed，四best SHA/bytes均匹配当前文件。通过 `additional_json` 可补实际提交 argv/config、各次评估前后 SHA、资源/费用与源锁定审计等小型 JSON 的 `{path,sha256}`。路径只允许该 backup_evidence 目录下普通 `.json`；文件上限8MiB，拒绝任意路径/压缩包/二进制。

## 运行命令

远端 CPU（只有满足全部门槛后；本轮未执行）：

```bash
python -B "$G/delivery/backup_final_sidecars.py" pack --scope operations --terminal-index-sha256 ACTUAL_INDEX_SHA
python -B "$G/delivery/backup_final_sidecars.py" pack --scope evaluations --terminal-index-sha256 ACTUAL_INDEX_SHA
```

输出分别固定为 `G/delivery/stage09_operations_r1.tar` / `STAGE09_OPERATIONS_R1.json`、`stage09_evaluations_r1.tar` / `STAGE09_EVALUATIONS_R1.json`。manifest在tar外，源文件前/中/后SHA及metadata一致、归档成员独立复算后才发布PASS。不覆盖现存输出；失败留下的partial不是完成品，不会自动删除或覆盖它。

下载两包及各外置manifest后，使用远端PASS打印的 manifest SHA 独立恢复：

```powershell
.runtime/stage08_restore_env/Scripts/python.exe -B outputs/stage09_delivery/goal_20261027_r1/backup_final_sidecars.py restore --scope operations --archive ABS_TAR --manifest ABS_JSON --manifest-sha256 ACTUAL_SHA --restore-name goal_20261027_r1
.runtime/stage08_restore_env/Scripts/python.exe -B outputs/stage09_delivery/goal_20261027_r1/backup_final_sidecars.py restore --scope evaluations --archive ABS_TAR --manifest ABS_JSON --manifest-sha256 ACTUAL_SHA --restore-name goal_20261027_r1
```

目标分别为 `outputs/stage09_restore/goal_20261027_r1/sidecars/operations` 和 `.../evaluations`；必须是新目录。包内保留远端相对路径，因此12份评估结果最终位于 `outputs/stage09_restore/goal_20261027_r1/sidecars/evaluations/evaluations/<arm_variant>/`，与`FINAL_ANALYSIS_COMMANDS.md`一致。先验整个tar和每个payload，再解包，再逐文件独立SHA复验，最后写独立恢复收据。符号链接/reparse、越界、重复或非普通成员、现存目标均拒绝。

本地6项CPU检查及独立静态审查已通过，见`FINAL_SIDECAR_CPU_CHECKS.json`；这不是正式归档或真实Linux封包验收。脚本、测试和说明已保存到本目录；测试在outputs工作副本运行，以免把临时fixture写入证据目录。10月28日已部署helper并完成本地/远端SHA与依赖SHA回读，见`FINAL_SIDECAR_DEPLOYMENT_20261028.json`；没有生成两个最终包。

本地CPU小测试只覆盖合成文件、已知原生JSON形状的门槛、路径拒绝、真实小tar/新目录恢复及完整性拒绝；没有读取真实模型/数据、没有远端调用。Windows测试中的pack底层I/O是mock；不能把它声称为Linux远端no-follow或真实最终输入已验收。
