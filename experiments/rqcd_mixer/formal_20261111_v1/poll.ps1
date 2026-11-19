$ErrorActionPreference = 'Stop'
$env:NO_PROXY = '.sii.edu.cn'
$env:HTTP_PROXY = ''
$env:HTTPS_PROXY = ''
$env:ALL_PROXY = ''
$remote = '/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/rqcd_mixer_formal_20261111_v1'
$python = '/inspire/hdd/project/project-public/<inspire-account>/nancy/conda-envs/grm-exp/bin/python'
$raw = inspire --account sii-shared --json notebook exec nancy-rqcd-mixer-prep-1111-v1 --workspace CPU资源空间 --timeout 30 "$python $remote/progress.py # "
$response = $raw | ConvertFrom-Json
if (-not $response.success -or $response.data.returncode -ne 0) { throw $raw }
$line = ($response.data.output -split "`n")[0]
$data = $line | ConvertFrom-Json
$line | Set-Content -Encoding utf8 (Join-Path $PSScriptRoot 'PROGRESS.json')
$rows = foreach ($arm in @('B0', 'B1', 'B2')) {
    foreach ($v in $data.arms.$arm.joint.validation) {
        if ($v.count -ne 71657) { throw "Incomplete validation: $arm" }
        [pscustomobject]@{arm=$arm; R=$v.identity.R; T=$v.identity.T; count=$v.count; ndcg5=$v.metrics.'ndcg@5'; recall5=$v.metrics.'recall@5'; hr5=$v.metrics.'hr@5'; seconds=$v.seconds; checkpoint_sha256=$v.identity.checkpoint_sha256}
    }
}
$rows | Export-Csv -NoTypeInformation -Encoding utf8 (Join-Path $PSScriptRoot 'validation_progress.csv')
foreach ($arm in @('B0', 'B1', 'B2')) {
    $joint = $data.arms.$arm.joint
    $last = @($joint.validation) | Select-Object -Last 1
    [pscustomobject]@{arm=$arm; R=$joint.latest_by_kind.R_epoch.R; T=$joint.latest_by_kind.T_block.T; completed_validations=$joint.validation.Count; last_validation_R=$last.identity.R; last_ndcg5=$last.metrics.'ndcg@5'; progress=$joint.validation_progress; error=$joint.'ERROR.json'; training_complete=($null -ne $joint.'TRAINING_COMPLETE.json'); test_complete=($null -ne $joint.'TEST_COMPLETE.json')} | ConvertTo-Json -Depth 6 -Compress
}
