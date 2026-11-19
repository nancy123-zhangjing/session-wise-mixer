$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8='1'
$env:PYTHONIOENCODING='utf-8'
$env:CUDA_VISIBLE_DEVICES=''
$env:OMP_NUM_THREADS='4'
$env:MKL_NUM_THREADS='4'
$Workspace = 'C:/Users/nancy/Documents/experiment1'
$Py = "$Workspace/.runtime/stage08_restore_env/Scripts/python.exe"
$RestoreRoot = "$Workspace/outputs/stage09_restore/goal_20261027_r1"
$Dataset = "$Workspace/outputs/kuairec_listwise_v1/run1"
$EvaluationSidecars = "$RestoreRoot/sidecars/evaluations"
$Evaluations = "$EvaluationSidecars/evaluations"
$AnalysisOut = "$Workspace/experiments/09_kuairec_listwise/final_analysis_20261027"
$FreezeReceipt = 'C:/Users/nancy/Documents/experiment1/experiments/09_kuairec_listwise/formal_execution_20261027/FOUR_BEST_FREEZE_20261028.json'
$FreezeSha = '4d0a0a17be2fd087db30c3dbbfe815a1d9243983619e7c8d47f1ca6fcd681ae2'
$ExpectedBest = @{
    J0 = '01fddc99e1d1a9a207ae2b21043474ed991a607a23c8cdb8aa8472deae20354a'
    J1 = '9baf4bbfa15551329c753f09f4c681a5a95a424f31249d7d2067d890c4fd456c'
    H0 = '8914c253a679e4ad18c213d4092733a8a6ce995a9b290d8bfedd6507a403b84e'
    H1 = 'a672ea4d31be5281c2c24c1c7e774bc27eb7e679ec732181a197b703da890bf1'
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
