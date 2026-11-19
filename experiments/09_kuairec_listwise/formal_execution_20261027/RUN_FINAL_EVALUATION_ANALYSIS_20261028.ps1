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
if (-not (Test-Path -LiteralPath "$AnalysisOut/full_validation_selected_audits.json")) { throw 'Run cached verification first' }

$Stages = @('test_beam32','validation_subset_beam32','validation_subset_beam256')
foreach ($Arm in @('J0','J1','H0','H1')) {
    foreach ($Stage in $Stages) {
        $Suffix = if ($Arm -eq 'H1') { '_r2' } else { '' }
        $InputDir = "$Evaluations/${Arm}_${Stage}${Suffix}"
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
foreach ($Architecture in @('J','H')) {
    foreach ($Stage in $Stages) {
        $Suffix = if ($Architecture -eq 'H') { '_r2' } else { '' }
        & $Py -B -m scripts.listwise_diffgrm.analysis compare `
            --baseline "$Evaluations/${Architecture}0_${Stage}/per_sample.jsonl.gz" `
            --treatment "$Evaluations/${Architecture}1_${Stage}${Suffix}/per_sample.jsonl.gz" `
            --repeats 10000 --seed 2024 `
            --output "$AnalysisOut/${Architecture}_${Stage}_paired_users_10000.json"
        if ($LASTEXITCODE -ne 0) { throw "配对分析失败：$Architecture $Stage" }
    }
}
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
