$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$PidFile = Join-Path $ProjectRoot "logs\api-server.pid"

if (-not (Test-Path $PidFile)) {
    Write-Host "No PID file found."
    exit 0
}

$ServerPid = Get-Content $PidFile -ErrorAction SilentlyContinue
if (-not $ServerPid) {
    Remove-Item $PidFile -Force
    Write-Host "Empty PID file removed."
    exit 0
}

$Process = Get-Process -Id ([int]$ServerPid) -ErrorAction SilentlyContinue
if ($Process) {
    Stop-Process -Id $Process.Id -Force
    Write-Host "Stopped api-server PID=$ServerPid"
} else {
    Write-Host "api-server PID=$ServerPid was not running."
}

Remove-Item $PidFile -Force
