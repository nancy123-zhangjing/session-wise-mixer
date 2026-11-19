# Stage09 最终离线分析命令清单

本清单是执行前模板，所有正式复算现已完成。实际执行脚本为[完整验证](RUN_CACHED_VALIDATION_ANALYSIS_20261028.ps1)和[最终评估](RUN_FINAL_EVALUATION_ANALYSIS_20261028.ps1)，H1使用r2产物；已有输出为独占写入，勿直接重跑覆盖。


核对日期：2026-10-28。这里只准备命令，未执行最终评估、复算或 bootstrap；未修改冻结训练源码。已经验收的两份 R05 暖身复算不重复运行。

## 1. 输入范围与执行前条件

依据 `outputs/stage09_delivery/goal_20261027_r1/FINAL_EVAL_JOB_TEMPLATES.json`（SHA256 `6ee4e5ad2cfc617f8e11179096f868bf7ae4834aaa4e0cfe34f150bd929f2546`），最终结果恰好为以下 12 份。每个目录必须同时恢复 `per_sample.jsonl.gz` 和 `summary.json`。

| arm | test，beam32，53,194条 | 同一验证子集，beam32，1,024条 | 同一验证子集，beam256，1,024条 |
|---|---|---|---|
| J0 | `J0_test_beam32` | `J0_validation_subset_beam32` | `J0_validation_subset_beam256` |
| J1 | `J1_test_beam32` | `J1_validation_subset_beam32` | `J1_validation_subset_beam256` |
| H0 | `H0_test_beam32` | `H0_validation_subset_beam32` | `H0_validation_subset_beam256` |
| H1 | `H1_test_beam32` | `H1_validation_subset_beam32` | `H1_validation_subset_beam256` |

远端前缀为 `/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/goal_20261027_r1/evaluations/`。`backup_final_sidecars.py restore --scope evaluations --restore-name goal_20261027_r1` 的解包目标为 `C:/Users/nancy/Documents/experiment1/outputs/stage09_restore/goal_20261027_r1/sidecars/evaluations/`，包内继续保留 `evaluations/<arm_variant>/...`，所以12份结果的实际根目录是 **`C:/Users/nancy/Documents/experiment1/outputs/stage09_restore/goal_20261027_r1/sidecars/evaluations/evaluations/`**。恢复收据位于同一恢复根下 `sidecars/RESTORE_evaluations.json`。这里是待恢复路径，不表示现在已有12份结果。

模型、训练日志和训练期间缓存验证结果仍从独立 run 备份读取，路径为恢复根下 `runs/<arm>/...`；后面的 `$Evaluations` 专门指向最终评估 sidecar。

开始前需要：

1. 四臂完成固定 35R/400T，并取得 `formal_final_acceptance.py` 输出的 `stage09_four_arm_final_selection_v1`、`status=passed` 收据。
2. 填入这份四 best 冻结收据的**本地路径与 SHA256**，以及 J0/J1/H0/H1 四个 best 的 SHA256。best epoch 和被接受的 validation 路径直接从收据读取，不猜 `validation_Rxx`，不忽略可能的 `_retryN`。
3. 12 份结果及其命令/模型绑定收据已按备份清单校验 SHA。每个 arm 的三份结果必须来自同一冻结 best；`summary.json` 自身没有 checkpoint SHA，下面的指标复算不能独立证明使用了哪份模型。
4. 验证子集清单固定为 `experiments/09_kuairec_listwise/preparation_20261027/validation_hash1024.json`，SHA256 `c74343799024ea0ca0bdb806ab482f362ff227880a2676cf1541a3511a7942af`。不重选子集、不按 beam256 重选 best。

这里使用**当前工作树的已验收离线 analysis.py**，不要在独立恢复的冻结 `code/` 下运行。冻结远端维持 `1c423662d9a25a926328ee6466c30c0f2857b4f6`；当前离线 analysis 增加了从真实目标和 train_seen 独立核对冷标记的检查。

已核对 `formal_final_acceptance.py` 的真实收据结构：`arms.<arm>.best` 是包含 `path`、`bytes`、`sha256` 的对象；参数类型在 `arms.<arm>.best_state.stored_model_floating_dtype`，选中轮次在 `arms.<arm>.selection.best_epoch`。下面的字段访问与此一致。

## 2. 一次性设置与身份检查（PowerShell）

先替换下面六个占位值。后续各段在同一 PowerShell 会话运行。新分析目录是本清单指定的输出位置，不应覆盖已有目录。

```powershell
$Workspace = 'C:/Users/nancy/Documents/experiment1'
$Py = "$Workspace/.runtime/stage08_restore_env/Scripts/python.exe"
$RestoreRoot = "$Workspace/outputs/stage09_restore/goal_20261027_r1"
$Dataset = "$Workspace/outputs/kuairec_listwise_v1/run1"
$EvaluationSidecars = "$RestoreRoot/sidecars/evaluations"
$Evaluations = "$EvaluationSidecars/evaluations"
$AnalysisOut = "$Workspace/experiments/09_kuairec_listwise/final_analysis_20261027"
$FreezeReceipt = '__LOCAL_ALL_FOUR_BEST_FREEZE_RECEIPT_JSON__'
$FreezeSha = '__ALL_FOUR_BEST_FREEZE_RECEIPT_SHA256__'
$ExpectedBest = @{
    J0 = '__FINAL_BEST_J0_SHA256__'
    J1 = '__FINAL_BEST_J1_SHA256__'
    H0 = '__FINAL_BEST_H0_SHA256__'
    H1 = '__FINAL_BEST_H1_SHA256__'
}
Set-Location -LiteralPath $Workspace
if ((Get-FileHash -LiteralPath $FreezeReceipt -Algorithm SHA256).Hash.ToLowerInvariant() -ne $FreezeSha) {
    throw '四 best 冻结收据 SHA 不符'
}
$Freeze = Get-Content -LiteralPath $FreezeReceipt -Raw | ConvertFrom-Json
if ($Freeze.schema -ne 'stage09_four_arm_final_selection_v1' -or $Freeze.status -ne 'passed') {
    throw '需要正式四臂完成及选模验收收据'
}
foreach ($Arm in @('J0','J1','H0','H1')) {
    if ($Freeze.arms.$Arm.best.sha256 -ne $ExpectedBest[$Arm] -or
        $Freeze.arms.$Arm.best_state.stored_model_floating_dtype -ne 'torch.float32') {
        throw "$Arm 的冻结 best 身份或浮点参数类型不符"
    }
    # 最终模型本地回收后核对字节；不加载模型、不进行前向。
    if ((Get-FileHash -LiteralPath "$RestoreRoot/runs/$Arm/best.pt" -Algorithm SHA256).Hash.ToLowerInvariant() -ne $ExpectedBest[$Arm]) {
        throw "$Arm 的本地 best SHA 不符"
    }
}
$ExpectedSources = @{
    'scripts/listwise_diffgrm/analysis.py' = '4e056eecfded4ff2c1e25ecc1c32a348f01004f31d1b649946ea544f2aa79bdd'
    'scripts/listwise_diffgrm/subset.py' = '33f6fb79033ba31d2ec454c19e7b4c2cad38abaf3c926f263c6f6ff2e23aee2a'
}
foreach ($Name in $ExpectedSources.Keys) {
    if ((Get-FileHash -LiteralPath "$Workspace/$Name" -Algorithm SHA256).Hash.ToLowerInvariant() -ne $ExpectedSources[$Name]) {
        throw "离线源码改变，先复核当前接口：$Name"
    }
}
if (Test-Path -LiteralPath $AnalysisOut) { throw '分析输出目录已存在；检查已有收据，不覆盖重跑' }
New-Item -ItemType Directory -Path $AnalysisOut -ErrorAction Stop | Out-Null
```

## 3. 12 份结果分别独立复算

当前准确接口：`analysis recompute --predictions ... --dataset ... --tolerance 2e-6 --output ...`。必须保留 `--dataset`，这样才会核对源 split 的哈希、样本索引/ID、用户、真实目标及从 train_seen 重算的冷标记。

```powershell
$Stages = @('test_beam32','validation_subset_beam32','validation_subset_beam256')
foreach ($Arm in @('J0','J1','H0','H1')) {
    foreach ($Stage in $Stages) {
        $InputDir = "$Evaluations/${Arm}_${Stage}"
        $ExpectedN = if ($Stage -eq 'test_beam32') { 53194 } else { 1024 }
        $ExpectedSplit = if ($Stage -eq 'test_beam32') { 'test' } else { 'validation' }
        $ExpectedBeam = if ($Stage.EndsWith('beam256')) { 256 } else { 32 }
        $Meta = Get-Content -LiteralPath "$InputDir/summary.json" -Raw | ConvertFrom-Json
        if ($Meta.examples -ne $ExpectedN -or $Meta.split -ne $ExpectedSplit -or
            $Meta.beam -ne $ExpectedBeam -or $Meta.architecture -ne $Arm.Substring(0,1) -or
            $Meta.seed -ne 2024 -or $Meta.inference_precision -ne 'fp32') {
            throw "评估身份或样本数量不符：$InputDir"
        }
        if ($ExpectedSplit -eq 'test') {
            if ($Meta.full_split -ne $true -or $null -ne $Meta.subset) { throw 'test 必须是完整split' }
        } elseif ($Meta.full_split -ne $false -or
            $Meta.subset.manifest_sha256 -ne 'c74343799024ea0ca0bdb806ab482f362ff227880a2676cf1541a3511a7942af') {
            throw '验证子集必须是同一份固定1024清单'
        }
        & $Py -B -m scripts.listwise_diffgrm.analysis recompute `
            --predictions "$InputDir/per_sample.jsonl.gz" `
            --dataset $Dataset --tolerance 2e-6 `
            --output "$AnalysisOut/${Arm}_${Stage}_audit.json"
        if ($LASTEXITCODE -ne 0) { throw "独立复算未通过：$Arm $Stage" }
    }
}
```

每份应得到 `passed=true`、`mismatches=0`、`source_labels_verified=true`、`source_cold_groups_verified=true`，并保存输入预测/summary 的 SHA。单列表评价为真实 item 的多相关 NDCG/Recall/HR@5；@10/20/50 是最多32份候选列表按已锁定规则聚合后的指标，不能称为一份五件列表的质量。还核对样本平均、用户平均与冷/已见目标组均值。

冷目标定义保持不变：五个实际目标中**至少一个**不在绑定的 train_seen item 集合中，就属于 `cold_target`。

### 补充：已保存的 best/R35 全量验证结果复算

`formal_final_acceptance.py` 已明确遍历42份独立被接受验证结果，核对完整行数、样本ID和保存NDCG均值；这不是从源标签独立重算所有指标。最终报告只额外复算四臂各自 best 对应的缓存验证结果及 R35：最多8个请求，按冻结收据中的实际目录去重；best=R35只做一次，共享R5直接引用已经通过的暖身复算。**不重新生成、不盲目复算全部42份结果**。

```powershell
$ValidationRefs = @{}
foreach ($Arm in @('J0','J1','H0','H1')) {
    $Evidence = $Freeze.arms.$Arm
    foreach ($Role in @('selected_best','fixed_R35')) {
        $Epoch = if ($Role -eq 'selected_best') { [int]$Evidence.selection.best_epoch } else { 35 }
        $Matched = @($Evidence.accepted_validations | Where-Object { $_.epoch -eq $Epoch })
        if ($Matched.Count -ne 1) { throw "接受的验证记录不唯一：$Arm R$Epoch" }
        $Val = $Matched[0]
        if (-not $ValidationRefs.ContainsKey($Val.path)) {
            $ValidationRefs[$Val.path] = @{arm=$Arm; epoch=$Epoch; validation=$Val; roles=@()}
        }
        $ValidationRefs[$Val.path].roles += "${Arm}:$Role"
    }
}
$WarmupAuditSha = @{
    J = '94948289d6834a7d06042b06586b2c1a7e4d1d19e9384b4936231c0cc5e8f5fc'
    H = 'c525efad96e90e44bb2a3b089a049719b9bcc8d59d8dd0a660f6c7ca21595046'
}
$ValidationIndex = foreach ($RemotePath in ($ValidationRefs.Keys | Sort-Object)) {
    $Ref = $ValidationRefs[$RemotePath]
    if (-not $RemotePath.StartsWith($Freeze.root + '/')) { throw '验证路径不属于冻结Goal' }
    $LocalDir = Join-Path $RestoreRoot $RemotePath.Substring($Freeze.root.Length + 1)
    foreach ($Name in @('per_sample.jsonl.gz','summary.json')) {
        $ExpectedSha = if ($Name -eq 'summary.json') { $Ref.validation.summary_sha256 } else { $Ref.validation.predictions_sha256 }
        if ((Get-FileHash -LiteralPath "$LocalDir/$Name" -Algorithm SHA256).Hash.ToLowerInvariant() -ne $ExpectedSha) {
            throw "恢复的接受验证文件SHA不符：$LocalDir/$Name"
        }
    }
    $Reused = $Ref.epoch -eq 5
    if ($Reused) {
        $Architecture = $Ref.arm.Substring(0,1)
        $AuditPath = "$Workspace/experiments/09_kuairec_listwise/warmup_metric_recompute_20261027/${Architecture}_R05_RECOMPUTE.json"
        if ((Get-FileHash -LiteralPath $AuditPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $WarmupAuditSha[$Architecture]) {
            throw '已有暖身复算收据改变；检查来源，不直接重跑'
        }
    } else {
        $AuditPath = "$AnalysisOut/fullval_$($Ref.arm)_R$('{0:D2}' -f $Ref.epoch)_audit.json"
        & $Py -B -m scripts.listwise_diffgrm.analysis recompute `
            --predictions "$LocalDir/per_sample.jsonl.gz" --dataset $Dataset --tolerance 2e-6 `
            --output $AuditPath | Out-Host
        if ($LASTEXITCODE -ne 0) { throw "缓存全量验证复算失败：$RemotePath" }
    }
    $Audit = Get-Content -LiteralPath $AuditPath -Raw | ConvertFrom-Json
    if ($Audit.passed -ne $true -or $Audit.examples -ne 71657 -or
        $Audit.source_labels_verified -ne $true -or $Audit.source_cold_groups_verified -ne $true -or
        $Audit.predictions_sha256 -ne $Ref.validation.predictions_sha256 -or
        $Audit.summary_sha256 -ne $Ref.validation.summary_sha256) {
        throw "全量验证复算收据与冻结来源不符：$RemotePath"
    }
    [pscustomobject]@{
        remote_path=$RemotePath; epoch=$Ref.epoch; roles=$Ref.roles; audit_path=$AuditPath
        audit_sha256=(Get-FileHash -LiteralPath $AuditPath -Algorithm SHA256).Hash.ToLowerInvariant()
        reused_warmup=$Reused
    }
}
@{format='stage09_selected_full_validation_audit_index_v1'; rows=@($ValidationIndex)} |
    ConvertTo-Json -Depth 6 |
    Out-File -LiteralPath "$AnalysisOut/full_validation_selected_audits.json" -Encoding utf8 -NoClobber
```

## 4. 六组同架构、同 beam 的用户配对 bootstrap

只有上面12份复算全部通过后再运行。当前 CLI **没有** `compare --dataset`、`--confidence` 或 `--metric`；95% percentile 区间和默认12项主要指标来自函数默认设置。

```powershell
foreach ($Architecture in @('J','H')) {
    foreach ($Stage in $Stages) {
        & $Py -B -m scripts.listwise_diffgrm.analysis compare `
            --baseline "$Evaluations/${Architecture}0_${Stage}/per_sample.jsonl.gz" `
            --treatment "$Evaluations/${Architecture}1_${Stage}/per_sample.jsonl.gz" `
            --repeats 10000 --seed 2024 `
            --output "$AnalysisOut/${Architecture}_${Stage}_paired_users_10000.json"
        if ($LASTEXITCODE -ne 0) { throw "配对分析失败：$Architecture $Stage" }
    }
}
```

共6份：J/H × test_beam32、validation_subset_beam32、validation_subset_beam256。处理组是 Mixer 开启的 `1`，基线是 `0`，差值为 `1 - 0`。不能把 J0 与 H0、beam32 与 beam256 直接送入 compare；它会拒绝架构或 beam 不同的输入。

主统计量是样本平均：每次有放回抽取用户，被抽中的用户带入全部样本，指标总和除以抽到的样本总数；不是先把每个用户等权平均。另有 `user_average` 作为等权用户平均的辅助结果。`groups` 仅有冷/已见目标组均值与差值，**没有组内 bootstrap 区间**。10,000次用户重采样不等于多训练 seed，不能据此声称跨 seed 稳定。

## 5. 搜索宽度变化表：不伪造跨 beam 的 bootstrap

下面只整理已经复算的均值。两个 beam 必须对应同一冻结 best、同一1024条样本；不重新选模型或报告新的“最优 beam”。

```powershell
$BeamRows = foreach ($Arm in @('J0','J1','H0','H1')) {
    $Small = Get-Content -LiteralPath "$AnalysisOut/${Arm}_validation_subset_beam32_audit.json" -Raw | ConvertFrom-Json
    $Large = Get-Content -LiteralPath "$AnalysisOut/${Arm}_validation_subset_beam256_audit.json" -Raw | ConvertFrom-Json
    [pscustomobject]@{
        arm = $Arm
        list_ndcg5_beam32 = $Small.metrics.'list/ndcg@5'
        list_ndcg5_beam256 = $Large.metrics.'list/ndcg@5'
        list_ndcg5_delta256_minus32 = $Large.metrics.'list/ndcg@5' - $Small.metrics.'list/ndcg@5'
        aggregate_ndcg50_beam32 = $Small.metrics.'aggregate/ndcg@50'
        aggregate_ndcg50_beam256 = $Large.metrics.'aggregate/ndcg@50'
    }
}
$BeamRows | Export-Csv -LiteralPath "$AnalysisOut/beam_sensitivity.csv" -NoTypeInformation -NoClobber -Encoding utf8
```

交互增益在两种 beam 下是否变化，读取第4节对应的两份配对报告。当前工具没有“两个增益之差”的独立区间，不把两条区间肉眼比较当成正式显著性检验。

## 6. 训练/验证曲线与 best/R35 表

`analysis.py` **没有 curve/plot 子命令**。以下是单独的离线 Python 命令块，不是现有 CLI 的虚构参数。只读取已恢复日志和冻结收据指向的 summary，不加载 torch/checkpoint，不做新评估。当前 CPU 环境已确认有 matplotlib 3.10.7。

先恢复四臂 `training.jsonl` 和冻结收据列出的所有被接受验证目录。R05来自同架构共享暖身目录；每臂11个点（R5、8、…、35），画出44个逻辑点，但只有42份独立全量验证。曲线不用于重新选 best。R/T loss 分开画，不混淆不同目标函数。

```powershell
@'
import csv, hashlib, json, math, sys
from collections import defaultdict
from pathlib import Path, PurePosixPath
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

freeze_path, freeze_sha, restored, output = sys.argv[1:]
assert hashlib.sha256(Path(freeze_path).read_bytes()).hexdigest() == freeze_sha
freeze = json.loads(Path(freeze_path).read_text(encoding='utf-8'))
assert freeze['status'] == 'passed' and freeze['schema'] == 'stage09_four_arm_final_selection_v1'
restored, output = Path(restored), Path(output) / 'curves'
remote_root = PurePosixPath(freeze['root'])
arms, epochs = ('J0','J1','H0','H1'), list(range(5,36,3))
def local(remote):
    rel = PurePosixPath(remote).relative_to(remote_root)
    assert '..' not in rel.parts
    return restored.joinpath(*rel.parts)
def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4194304), b''): h.update(b)
    return h.hexdigest()
def save_csv(name, rows):
    with (output/name).open('x', encoding='utf-8', newline='') as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

metric_names = ['list/ndcg@5','list/recall@5','list/hr@5']
metric_names += [f'aggregate/{name}@{k}' for k in (10,20,50) for name in ('ndcg','recall','hr')]
points, best_r35, losses, sources, unique_validations = [], [], [], {}, set()
audit_index_path=output.parent/'full_validation_selected_audits.json'
audit_index=json.loads(audit_index_path.read_text(encoding='utf-8-sig'))
assert audit_index['format']=='stage09_selected_full_validation_audit_index_v1'
selected_audits={r['remote_path']:r for r in audit_index['rows']}
sources[str(audit_index_path)]=digest(audit_index_path)
for arm in arms:
    evidence=freeze['arms'][arm]
    assert [v['epoch'] for v in evidence['accepted_validations']] == epochs
    for val in evidence['accepted_validations']:
        path=local(val['path'])/'summary.json'; sources[str(path)]=digest(path)
        assert sources[str(path)] == val['summary_sha256']
        meta=json.loads(path.read_text(encoding='utf-8'))
        assert meta['examples']==71657 and meta['beam']==32 and meta['full_split'] and meta['subset'] is None
        assert meta['architecture']==arm[0] and meta['metrics']['list/ndcg@5']==val['ndcg_at_5']
        row=dict(arm=arm, epoch=val['epoch'], r_updates=meta['r_updates'], t_updates=meta['t_updates'],
                 **{name:meta['metrics'][name] for name in metric_names})
        points.append(row); unique_validations.add(val['path'])
    for label, epoch in (('selected_best', evidence['selection']['best_epoch']), ('fixed_R35',35)):
        row=next(r for r in points if r['arm']==arm and r['epoch']==epoch)
        val=next(v for v in evidence['accepted_validations'] if v['epoch']==epoch)
        ref=selected_audits[val['path']]; path=Path(ref['audit_path'])
        sources[str(path)]=digest(path); assert sources[str(path)]==ref['audit_sha256']
        audit=json.loads(path.read_text(encoding='utf-8'))
        assert audit['passed'] and audit['examples']==71657 and audit['predictions_sha256']==val['predictions_sha256']
        checked=dict(row); checked.update({name:audit['metrics'][name] for name in metric_names})
        best_r35.append(dict(role=label, metric_source='independent_recompute', **checked))
    # Logs are append-only across resumes: retain the last occurrence of each
    # (phase_index,update), then verify all final update counters are covered.
    latest={}
    for path in (restored/'runs'/('warmup_'+arm[0])/'training.jsonl', restored/'runs'/arm/'training.jsonl'):
        sources[str(path)]=digest(path)
        with path.open(encoding='utf-8') as f:
            for line in f:
                row=json.loads(line); latest[(row['phase_index'],row['update'])]=row
    for phase,total,counter in (('R',24255,'r_updates'),('T',400,'t_updates')):
        selected=[r for r in latest.values() if r['phase']==phase]
        assert len(selected)==total and {r[counter] for r in selected}==set(range(1,total+1))
    buckets=defaultdict(list)
    for row in latest.values(): buckets[row['phase_index']].append(row)
    assert len(buckets)==45
    for index, rows in sorted(buckets.items()):
        count=sum(r['lists'] for r in rows)
        mean=sum(r['loss']*r['lists'] for r in rows)/count
        assert math.isfinite(mean)
        losses.append(dict(arm=arm, phase_index=index, phase=rows[0]['phase'], epoch=rows[0]['epoch'],
            r_updates=max(r['r_updates'] for r in rows), t_updates=max(r['t_updates'] for r in rows),
            updates=len(rows), list_weighted_loss=mean))
assert len(unique_validations)==42 and len(points)==44
output.mkdir(exist_ok=False)
save_csv('validation_curve.csv',points)
save_csv('best_and_R35_validation.csv',best_r35)
save_csv('training_phase_loss.csv',losses)
fig,axes=plt.subplots(2,2,figsize=(12,8))
for arm in arms:
    values=[r for r in points if r['arm']==arm]
    for axis,key,title in ((axes[0,0],'list/ndcg@5','Single-list NDCG@5'),
                           (axes[0,1],'aggregate/ndcg@50','Aggregated candidates NDCG@50')):
        axis.plot([r['epoch'] for r in values],[r[key] for r in values],marker='o',label=arm)
        chosen=next(r for r in values if r['epoch']==freeze['arms'][arm]['selection']['best_epoch'])
        axis.scatter([chosen['epoch']],[chosen[key]],marker='*',s=120)
        axis.set(xlabel='Completed R epoch',ylabel=title)
    for axis,phase,key,title in ((axes[1,0],'R','epoch','R recommendation loss'),
                                (axes[1,1],'T','t_updates','T total loss')):
        rows=[r for r in losses if r['arm']==arm and r['phase']==phase]
        axis.plot([r[key] for r in rows],[r['list_weighted_loss'] for r in rows],label=arm)
        axis.set(xlabel='R epoch' if phase=='R' else 'Completed T updates',ylabel=title)
for axis in axes.flat: axis.grid(alpha=.25); axis.legend()
fig.tight_layout(); fig.savefig(output/'training_validation_curves.png',dpi=180); plt.close(fig)
with (output/'curve_sources.json').open('x',encoding='utf-8') as f:
    json.dump(dict(freeze_receipt_sha256=freeze_sha, source_sha256=sources,
        distinct_full_validations=42, plotted_validation_points=44,
        loss_log_dedup='last occurrence of (phase_index,update); not a GPU-work accounting rule',
        metric_scope='curves:42 hash-verified accepted summaries; best/R35 table: selected independent audits; not independent replay of all42 files'),f,indent=2)
print(output)
'@ | & $Py -B - $FreezeReceipt $FreezeSha $RestoreRoot $AnalysisOut
if ($LASTEXITCODE -ne 0) { throw '曲线输入身份、日志完整性或导出失败；保留现场，不覆盖重试' }
```

这段会输出3份小型CSV、一张四面板图及来源SHA收据。R05只是读取既有 summary/log 作为共享起点，不重跑暖身指标复算。图中的四条完整R loss轨迹共享前5轮，不能用图上重复点推导GPU用量。若恢复存在不完整日志、未回收的接受目录或新的schema，应停下补齐事实，不删断点、不补假数据。

## 7. 四组best/last的SID碰撞统计

共同资产的14.03%碰撞商品比例只属于初态，不能替代训练后模型。四臂final selection收据和本地模型恢复均完成后，用同一个冻结收据绑定8个角色的checkpoint字节与SID指纹，再统计保存的码字。仅CPU读取，不构建模型、不运行前向、不重新分配SID，不读取验证/测试标签。

```powershell
$SidDiagnostics = "$Workspace/experiments/09_kuairec_listwise/formal_execution_20261027/checkpoint_sid_diagnostics.py"
if ((Get-FileHash -LiteralPath $SidDiagnostics -Algorithm SHA256).Hash.ToLowerInvariant() -ne '22c9aec337efa241c595b08dded581c567ae854baa4a191e4a3f1002ee6b3e37') {
    throw 'SID统计脚本已改变，先复核来源'
}
& $Py -B $SidDiagnostics --selection-receipt $FreezeReceipt --selection-sha256 $FreezeSha --restore-root $RestoreRoot --output "$AnalysisOut/checkpoint_sid_diagnostics.json"
if ($LASTEXITCODE -ne 0) { throw '最终SID身份或统计失败，保留证据，不覆盖重跑' }
```

输出唯一SID数、碰撞商品数/比例、碰撞桶数、最大桶，以及各码位使用和前缀统计；排除内部商品0的PAD行，保留真实商品的全零码。分别报告best和固定R35的last，不能混用。当前仅[5项合成CPU检查](CHECKPOINT_SID_DIAGNOSTICS_PREPARATION.json)通过，尚未读取真实最终模型。

将该结果与预测复算得到的单列表完整率、聚合候选数、使用列表数和coverage一起解释。桶统计和覆盖不足并存不证明因果：不同候选列表的重复、有效完整beam数量及最多32份列表的聚合上限也会影响覆盖。同SID在同一列表中还可能依次映射到桶内不同商品。

## 8. 收尾与解释边界

```powershell
$AuditFiles = Get-ChildItem -LiteralPath $AnalysisOut -File -Recurse
$AuditFiles | ForEach-Object {
    [pscustomobject]@{path=$_.FullName; bytes=$_.Length; sha256=(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()}
} | ConvertTo-Json -Depth 4 | Out-File -LiteralPath "$AnalysisOut/OUTPUT_SHA256.json" -Encoding utf8 -NoClobber
```

最终同时保留四best冻结收据SHA、12份输入预测/summary SHA、去重后的best/R35缓存验证复算及暖身引用索引、6份配对报告、曲线来源SHA、8个best/last角色的SID统计及checkpoint身份，以及实际执行命令/耗时。当前 CLI 没有把冻结 checkpoint SHA 自动写入分析报告，所以还需与第1节的评估执行绑定收据一起交付。

最终主结论围绕每种列表生成结构下 Mixer± 的同条件差值；test结果不用于调参。验证曲线和 best/R35 表用于观察训练趋势、选模差异与最终固定轮次结果，不能替代 test 主比较。这里只准备命令，四best SHA、冻结收据路径/SHA和未来真实恢复文件仍待正式完成后填写/核实。
