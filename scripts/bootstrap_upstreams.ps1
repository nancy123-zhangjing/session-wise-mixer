param(
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ExternalRoot = Join-Path $ProjectRoot 'external'
$PatchPath = Join-Path $ProjectRoot 'patches\diffgrm-position-loss.patch'
$GenRecPatchPath = Join-Path $ProjectRoot 'patches\genrec-rqvae-amp.patch'

New-Item -ItemType Directory -Force -Path $ExternalRoot | Out-Null

function Get-OrCloneRepository {
    param(
        [string]$Name,
        [string]$Url,
        [string]$Commit
    )

    $Target = Join-Path $ExternalRoot $Name
    if (-not (Test-Path $Target)) {
        git clone $Url $Target
    }
    elseif ($Force) {
        git -C $Target fetch origin
    }

    git -C $Target checkout --detach $Commit
}

Get-OrCloneRepository -Name 'genrec' -Url 'https://github.com/phonism/genrec.git' -Commit 'fb4bd1a1832df0312580a705d3440d1988856e31'
Get-OrCloneRepository -Name 'DiffGRM' -Url 'https://github.com/liuzhao09/DiffGRM.git' -Commit 'ad7b971c7e525e9fea6fb8e362a5c49dccb2473c'

$GenRecRoot = Join-Path $ExternalRoot 'genrec'
$PreviousErrorPreference = $ErrorActionPreference
$ErrorActionPreference = 'SilentlyContinue'
git -C $GenRecRoot apply --check $GenRecPatchPath 2>$null
$GenRecForwardCheck = $LASTEXITCODE
$ErrorActionPreference = $PreviousErrorPreference
if ($GenRecForwardCheck -eq 0) {
    git -C $GenRecRoot apply $GenRecPatchPath
    Write-Host 'Applied the CUDA-autocast Sinkhorn dtype fix to GenRec RQ-VAE.'
}
else {
    $ErrorActionPreference = 'SilentlyContinue'
    git -C $GenRecRoot apply --reverse --check $GenRecPatchPath 2>$null
    $GenRecReverseCheck = $LASTEXITCODE
    $ErrorActionPreference = $PreviousErrorPreference
    if ($GenRecReverseCheck -eq 0) {
        Write-Host 'GenRec RQ-VAE Sinkhorn dtype fix is already applied.'
    }
    else {
        throw 'GenRec RQ-VAE patch does not apply cleanly.'
    }
}

$DiffRoot = Join-Path $ExternalRoot 'DiffGRM'
$PreviousErrorPreference = $ErrorActionPreference
$ErrorActionPreference = 'SilentlyContinue'
git -C $DiffRoot apply --check $PatchPath 2>$null
$ForwardCheck = $LASTEXITCODE
$ErrorActionPreference = $PreviousErrorPreference

if ($ForwardCheck -eq 0) {
    git -C $DiffRoot apply $PatchPath
    Write-Host 'Applied AR position-loss, collision-suffix, and strict-evaluation instrumentation to DiffGRM.'
}
else {
    $ErrorActionPreference = 'SilentlyContinue'
    git -C $DiffRoot apply --reverse --check $PatchPath 2>$null
    $ReverseCheck = $LASTEXITCODE
    $ErrorActionPreference = $PreviousErrorPreference
    if ($ReverseCheck -eq 0) {
        Write-Host 'DiffGRM AR experiment instrumentation is already applied.'
    }
    else {
        throw 'DiffGRM patch does not apply cleanly. Check the pinned commit and local changes.'
    }
}
