param(
    [Parameter(Mandatory = $true)]
    [string]$JobPath,
    [int]$PollSeconds = 60
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$AbsoluteJobPath = Resolve-Path (Join-Path $ProjectRoot $JobPath)
$LogDir = Join-Path $ProjectRoot "logs"
$HeartbeatLog = Join-Path $LogDir "terminal-bench-watchdog.log"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Write-WatchdogLog {
    param([string]$Message)
    $Timestamp = (Get-Date).ToString("s")
    Add-Content -Path $HeartbeatLog -Value "[$Timestamp] $Message"
}

function Read-JobResult {
    $ResultPath = Join-Path $AbsoluteJobPath "result.json"
    if (-not (Test-Path $ResultPath)) {
        return $null
    }
    try {
        return Get-Content -Raw -Path $ResultPath | ConvertFrom-Json
    } catch {
        Write-WatchdogLog "result.json is not readable yet: $($_.Exception.Message)"
        return $null
    }
}

function Get-RunningHarborForJob {
    $JobPathText = [System.IO.Path]::GetFullPath($AbsoluteJobPath)
    Get-CimInstance Win32_Process |
        Where-Object {
            $_.Name -ieq "harbor.exe" -and
            $_.CommandLine -like "*job*resume*" -and
            ($_.CommandLine -like "*$JobPath*" -or $_.CommandLine -like "*$JobPathText*")
        }
}

Write-WatchdogLog "watchdog started for $AbsoluteJobPath with poll=${PollSeconds}s"

while ($true) {
    $Result = Read-JobResult
    if ($Result -and $Result.finished_at) {
        $Stats = $Result.stats
        Write-WatchdogLog "job finished: completed=$($Stats.n_completed_trials) errored=$($Stats.n_errored_trials) cancelled=$($Stats.n_cancelled_trials)"
        break
    }

    $StatsText = "result unavailable"
    if ($Result -and $Result.stats) {
        $Stats = $Result.stats
        $StatsText = "completed=$($Stats.n_completed_trials) running=$($Stats.n_running_trials) pending=$($Stats.n_pending_trials) errored=$($Stats.n_errored_trials)"
    }

    $Running = @(Get-RunningHarborForJob)
    if ($Running.Count -eq 0) {
        $Stamp = Get-Date -Format "yyyyMMdd-HHmmss"
        $OutLog = Join-Path $LogDir "terminal-bench-watchdog-resume-$Stamp.out.log"
        $ErrLog = Join-Path $LogDir "terminal-bench-watchdog-resume-$Stamp.err.log"
        Write-WatchdogLog "harbor not running; resuming job ($StatsText)"
        Start-Process `
            -FilePath "harbor" `
            -ArgumentList @("job", "resume", "-p", $AbsoluteJobPath) `
            -WorkingDirectory $ProjectRoot `
            -RedirectStandardOutput $OutLog `
            -RedirectStandardError $ErrLog `
            -PassThru | Out-Null
    } else {
        $Pids = ($Running | Select-Object -ExpandProperty ProcessId) -join ","
        Write-WatchdogLog "harbor running pid=$Pids ($StatsText)"
    }

    Start-Sleep -Seconds $PollSeconds
}
