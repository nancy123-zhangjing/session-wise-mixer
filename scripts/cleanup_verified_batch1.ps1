$ErrorActionPreference='Stop'
$root=[IO.Path]::GetFullPath('C:/Users/nancy/Documents/experiment1')
$receipt=Get-Content -LiteralPath (Join-Path $root 'docs/research/CLEANUP_BATCH1_REMOTE_20261105.json') -Raw | ConvertFrom-Json
if ($receipt.status -ne 'remote_bytes_and_archive_read_passed') { throw 'Remote verification incomplete' }
$auditPath=Join-Path $root 'docs/research/CLEANUP_BATCH1_DELETED_20261105.json'
if (Test-Path -LiteralPath $auditPath) { throw 'Existing deletion receipt; inspect before continuing' }
$deleted=[Collections.Generic.List[object]]::new()
foreach($row in $receipt.files) {
 $path=[IO.Path]::GetFullPath((Join-Path $root $row.relative_path))
 if (-not $path.StartsWith($root+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)) {throw 'Outside workspace'}
 if (-not $row.remote_verified -or $row.sha256 -ne $row.remote_sha256 -or $row.bytes -ne $row.remote_bytes) {throw 'Remote mismatch'}
 if (-not $row.restore_verified) {
  $parent=$receipt.files | Where-Object { $_.relative_path -eq 'outputs/stage09_delivery/goal_20261027_r1/stage09_bge_model.tar' }
  if ($row.relative_path -notmatch 'stage09_bge_model\.tar\.part0[1-4]$' -or -not $parent.restore_verified) {throw 'No archive readability proof'}
 }
 $file=Get-Item -LiteralPath $path
 if ($file.Attributes -band [IO.FileAttributes]::ReparsePoint) {throw 'Reparse point'}
 if ($file.Length -ne $row.bytes -or (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $row.sha256) {throw 'Local changed'}
 Remove-Item -LiteralPath $path
 $deleted.Add([pscustomobject]@{local=$path;remote=$row.remote;bytes=$row.bytes;sha256=$row.sha256;deleted_utc=[datetime]::UtcNow.ToString('o')})
 [ordered]@{deleted=@($deleted.ToArray());bytes=($deleted|Measure-Object bytes -Sum).Sum;remote_receipt='CLEANUP_BATCH1_REMOTE_20261105.json';scope='Redundant archives only; extracted results retained'} | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $auditPath -Encoding utf8
 Write-Host ('Deleted verified redundant file: '+$row.relative_path)
}
