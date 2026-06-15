$trackedDir = 'c:/asd_project/outputs/tracked'
$outDir = 'c:/asd_project/outputs/child_sequences'
if (-not (Test-Path $outDir)) { New-Item -ItemType Directory -Path $outDir | Out-Null }

Get-ChildItem $trackedDir -Filter '*_tracked.csv' | ForEach-Object {
    $trackedPath = $_.FullName
    Write-Host "Processing $($_.Name)..."
    python c:/asd_project/scripts/extract_child_track.py --input $trackedPath
    $videoId = $_.BaseName -replace '_tracked',''
    $reportPath = Join-Path $outDir "${videoId}_child_report.csv"
    python c:/asd_project/scripts/extract_child_sequence.py --tracked $trackedPath --report $reportPath --output-dir $outDir
}
