param(
    [string]$PythonExe = 'python'
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DiffRoot = Join-Path $ProjectRoot 'external\DiffGRM'
$ConfigPath = Join-Path $ProjectRoot 'configs\ar_smoke_6gb.yaml'

if (-not (Test-Path $DiffRoot)) {
    throw 'external\DiffGRM is missing. Run scripts\bootstrap_upstreams.ps1 first.'
}

& $PythonExe -c "import torch, accelerate, datasets, faiss, sentence_transformers; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0) {
    throw 'The selected Python environment is incomplete. Create/activate environment.yml first.'
}

Push-Location $DiffRoot
try {
    & $PythonExe main.py --model AR_GRM --dataset AmazonReviews2014 --config $ConfigPath
    if ($LASTEXITCODE -ne 0) {
        throw "AR smoke run failed with exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}

