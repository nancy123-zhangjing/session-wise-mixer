param(
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DiffRoot = Join-Path $ProjectRoot "external/DiffGRM"
$ConfigPath = Join-Path $ProjectRoot "configs/ar_smoke_v2_unique.yaml"

if (-not (Test-Path $DiffRoot)) {
    throw "external/DiffGRM is missing. Run scripts/bootstrap_upstreams.ps1 first."
}

if (-not $env:HF_ENDPOINT) { $env:HF_ENDPOINT = "https://hf-mirror.com" }
if (-not $env:HF_HOME) { $env:HF_HOME = Join-Path $ProjectRoot "data/huggingface" }

& $PythonExe (Join-Path $ProjectRoot "scripts/test_collision_aware_eval.py")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Push-Location $DiffRoot
try {
    & $PythonExe main.py --model AR_GRM --dataset AmazonReviews2014 --config $ConfigPath
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
finally {
    Pop-Location
}
