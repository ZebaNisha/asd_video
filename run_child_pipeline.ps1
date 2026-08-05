$projectRoot = $PSScriptRoot
$trackedDir = Join-Path $projectRoot "outputs/tracked"
$outDir = Join-Path $projectRoot "outputs/child_sequences"
if (-not (Test-Path $outDir)) { New-Item -ItemType Directory -Path $outDir | Out-Null }

Get-ChildItem $trackedDir -Filter '*_tracked.csv' | ForEach-Object {
    $trackedPath = $_.FullName
    Write-Host "Processing $($_.Name)..."
    python (Join-Path $projectRoot "scripts/extract_child_track.py") --input $trackedPath
    $videoId = $_.BaseName -replace '_tracked',''
    $reportPath = Join-Path $outDir "${videoId}_child_report.csv"
    python (Join-Path $projectRoot "scripts/extract_child_sequence.py") --tracked $trackedPath --report $reportPath --output-dir $outDir
}
