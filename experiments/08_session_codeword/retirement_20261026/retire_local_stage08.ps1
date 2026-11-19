$ErrorActionPreference = 'Stop'
$workspaceRoot = (Resolve-Path -LiteralPath 'C:\Users\nancy\Documents\experiment1').Path.TrimEnd('\')
$receiptRoot = Join-Path $workspaceRoot 'experiments\08_session_codeword\retirement_20261026'
$relativeTargets = @(
  'outputs\stage08_session_codeword',
  'configs\08_session_codeword',
  'scripts\train\run_session_codeword_joint.py',
  'scripts\train\session_codeword_gpu_job.py',
  'scripts\checks\session_codeword_h100_gate.py',
  'scripts\checks\verify_session_codeword_cpu.py',
  'scripts\checks\session_inference_diagnostics.py',
  'scripts\analysis\report_session_codeword.py',
  'scripts\analysis\pack_session_codeword.py',
  'tests\test_session_codeword.py',
  'tests\test_session_codeword_backup.py',
  'tests\test_session_codeword_context.py',
  'tests\test_session_codeword_gpu_job.py',
  'tests\test_session_codeword_h100_gate.py',
  'tests\test_session_codeword_integration.py',
  'tests\test_session_codeword_report.py',
  'tests\test_session_codeword_runner.py',
  'tests\test_session_inference_diagnostics.py',
  'docs\research\session_wise_codebook\prototypes\session_router.py',
  'docs\research\session_wise_codebook\prototypes\verify_cpu.py',
  'docs\research\session_wise_codebook\prototypes\benchmark_cpu.py',
  'docs\research\session_wise_codebook\evidence\workspace_audit.py',
  'docs\research\session_wise_codebook\evidence\complete_audit.py'
)
$targets = [System.Collections.Generic.List[string]]::new()
foreach ($relative in $relativeTargets) {
  $candidate = Join-Path $workspaceRoot $relative
  if (!(Test-Path -LiteralPath $candidate)) { throw "Missing target before cleanup: $relative" }
  $resolved = (Resolve-Path -LiteralPath $candidate).Path
  if (!$resolved.StartsWith($workspaceRoot + '\', [StringComparison]::OrdinalIgnoreCase)) { throw "Path escaped workspace: $resolved" }
  $targets.Add($resolved)
  if ([IO.Path]::GetExtension($resolved) -eq '.py') {
    $cache = Join-Path ([IO.Path]::GetDirectoryName($resolved)) '__pycache__'
    if (Test-Path -LiteralPath $cache) {
      $stem = [IO.Path]::GetFileNameWithoutExtension($resolved)
      Get-ChildItem -LiteralPath $cache -File -Filter "$stem.*.pyc" | ForEach-Object { $targets.Add($_.FullName) }
    }
  }
}
$manifest = [System.Collections.Generic.List[object]]::new()
foreach ($target in $targets) {
  $entry = Get-Item -LiteralPath $target -Force
  $members = @($entry)
  if ($entry.PSIsContainer) { $members += @(Get-ChildItem -LiteralPath $target -Recurse -Force) }
  foreach ($member in $members) {
    if (($member.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) { throw "Refuse link: $($member.FullName)" }
    if (!$member.FullName.StartsWith($workspaceRoot + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Member escaped workspace' }
    if (!$member.PSIsContainer) {
      $row = @{ path = $member.FullName.Substring($workspaceRoot.Length + 1); bytes = $member.Length }
      if ($member.Extension -in '.py','.json','.pt','.bin') { $row.sha256 = (Get-FileHash -LiteralPath $member.FullName -Algorithm SHA256).Hash.ToLowerInvariant() }
      $manifest.Add($row)
    }
  }
}
$receipt = [ordered]@{
  scope = 'Only abandoned stage08 code/model/output copies; keep stage01-07, original data, reports and environments'
  workspace = $workspaceRoot
  targets = @($targets)
  files = @($manifest)
  file_count = $manifest.Count
  bytes = ($manifest | Measure-Object -Property bytes -Sum).Sum
  applied = $false
}
$receiptPath = Join-Path $receiptRoot 'LOCAL_DELETION.json'
if (Test-Path -LiteralPath $receiptPath) { throw 'Receipt already exists; refuse duplicate cleanup' }
$receipt | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $receiptPath -Encoding utf8
foreach ($target in $targets) {
  # All absolute targets and descendants were checked above; use native PowerShell throughout.
  Remove-Item -LiteralPath $target -Recurse -Force -ErrorAction Stop
}
$remaining = @($targets | Where-Object { Test-Path -LiteralPath $_ })
if ($remaining.Count) { throw "Cleanup incomplete: $remaining" }
$receipt.applied = $true
$receipt | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $receiptPath -Encoding utf8
[ordered]@{applied=$true;files=$receipt.file_count;bytes=$receipt.bytes;remaining=$remaining;receipt=$receiptPath} | ConvertTo-Json
