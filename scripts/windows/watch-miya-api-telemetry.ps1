param(
    [string]$LogPath = "",
    [int]$Last = 20,
    [switch]$Follow,
    [switch]$Raw
)

$ErrorActionPreference = "Stop"

if (-not $LogPath) {
    $ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
    $LogPath = Join-Path $ProjectRoot "logs\api-server.out.log"
}

$LogDir = Split-Path -Parent $LogPath
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
if (-not (Test-Path $LogPath)) {
    New-Item -ItemType File -Force -Path $LogPath | Out-Null
}

function Show-TelemetryLine {
    param([string]$Line)

    if ([string]::IsNullOrWhiteSpace($Line)) {
        return
    }
    if ($Line -notlike '*"event":"api_usage"*') {
        if ($Raw) {
            Write-Host $Line
        }
        return
    }
    if ($Raw) {
        Write-Host $Line
        return
    }

    try {
        $event = $Line | ConvertFrom-Json
        $time = if ($event.timestamp_ms) {
            [DateTimeOffset]::FromUnixTimeMilliseconds([int64]$event.timestamp_ms).LocalDateTime.ToString("HH:mm:ss")
        } else {
            "--:--:--"
        }
        $requestId = if ($event.request_id) { $event.request_id } else { "-" }
        if ($requestId.Length -gt 8) {
            $requestId = $requestId.Substring(0, 8)
        }
        $lineText = "{0} req={1} model={2} effort={3} in={4} out={5} total={6} providers={7} tasks={8} children={9} tools={10} verified={11} route={12}" -f `
            $time,
            $requestId,
            $event.model,
            $event.reasoning_effort,
            $event.input_tokens,
            $event.output_tokens,
            $event.total_tokens,
            $event.provider_call_count,
            $event.task_count,
            $event.child_agent_count,
            $event.tool_call_count,
            $event.verification.passed,
            $event.route
        Write-Host $lineText
    } catch {
        Write-Host $Line
    }
}

Write-Host "Telemetry log: $LogPath"
Write-Host "Press Ctrl+C to stop watching."

if ($Follow) {
    Get-Content -Path $LogPath -Tail $Last -Wait | ForEach-Object {
        Show-TelemetryLine $_
    }
} else {
    Get-Content -Path $LogPath -Tail $Last | ForEach-Object {
        Show-TelemetryLine $_
    }
}
