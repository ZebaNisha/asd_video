# Launch script for ASD Video Diagnostic System
$projectRoot = $PSScriptRoot

Write-Host "Starting Flask Backend Server..." -ForegroundColor Green
Start-Process powershell -ArgumentList '-NoExit',"-Command","cd '$projectRoot'; python run_backend.py"

Write-Host "Starting React Frontend Dev Server..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList '-NoExit',"-Command","cd '$projectRoot/frontend'; npm run dev"

Write-Host "ASD Diagnostic System launched successfully." -ForegroundColor Yellow
Write-Host "Backend running on http://127.0.0.1:5000"
Write-Host "Frontend running on http://127.0.0.1:5173"
