$ErrorActionPreference = 'Stop'
$root = 'C:\Users\nancy\Documents\experiment1'
$relative = @(
 'outputs/stage09_restore/h_freeze35_20261029_final_r1.tar',
 'outputs/stage09_restore/h_freeze35_20261029_A_export_r1.tar',
 'outputs/stage09_restore/h_freeze35_20261029_B_export_r1.tar',
 'experiments/09_kuairec_listwise/h1_spread200_20261031/delivery/stage09_h1_spread200_r1.tar.gz',
 'experiments/09_kuairec_listwise/h1_spread200_20261031/delivery/stage09_h1_spread200_r1.tar.gz.partial_r1',
 'experiments/09_kuairec_listwise/h1_block40_r6_single_20261101/delivery/stage09_h1_block40_single_r1.tar.gz',
 'outputs/stage09_delivery/goal_20261027_r1/stage09_bge_model.tar',
 'outputs/stage09_delivery/goal_20261027_r1/stage09_bge_model.tar.part01',
 'outputs/stage09_delivery/goal_20261027_r1/stage09_bge_model.tar.part02',
 'outputs/stage09_delivery/goal_20261027_r1/stage09_bge_model.tar.part03',
 'outputs/stage09_delivery/goal_20261027_r1/stage09_bge_model.tar.part04'
)
$rows = foreach ($name in $relative) {
 $p = Join-Path $root $name
 $file = Get-Item -LiteralPath $p
 $digest = (Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash.ToLowerInvariant()
 [pscustomobject]@{relative_path=$name;bytes=$file.Length;sha256=$digest;remote_verified=$false;restore_verified=$false;delete_allowed=$false}
 Write-Host ('Hashed: ' + $name)
}
$report = [ordered]@{timestamp_utc=[datetime]::UtcNow.ToString('o');status='local_inventory_only_remote_connection_reset';deleted_files=0;total_bytes=($rows | Measure-Object bytes -Sum).Sum;files=@($rows)}
$destination = Join-Path $root 'docs/research/CLEANUP_BATCH1_LOCAL_20261105.json'
$report | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $destination -Encoding utf8
Write-Host $destination
