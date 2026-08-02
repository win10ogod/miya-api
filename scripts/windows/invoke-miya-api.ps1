param(
    [string]$BaseUrl = "http://localhost:3100",
    [string]$Model = "local-model",
    [string]$MiyaApiKey = "miya-local-key",
    [ValidateSet("none", "low", "medium", "high", "xhigh")]
    [string]$Effort = "low",
    [int]$MaxParallelAgents = 0,
    [string]$Message = "OK",
    [string]$System = "Reply exactly OK.",
    [string]$LogPath = "",
    [string]$TrainingTracePath = "",
    [int]$TimeoutSec = 3700,
    [switch]$ShowTrainingTrace,
    [switch]$RawResponse,
    [switch]$RawTelemetry
)

$ErrorActionPreference = "Stop"

if (-not $LogPath) {
    $ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
    $LogPath = Join-Path $ProjectRoot "logs\api-server.out.log"
}
if (-not $TrainingTracePath) {
    if (-not $ProjectRoot) {
        $ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
    }
    $TrainingTracePath = Join-Path $ProjectRoot "logs\training-traces.jsonl"
}

$RequestId = [guid]::NewGuid().ToString()
$Headers = @{
    "Authorization" = "Bearer $MiyaApiKey"
    "x-request-id" = $RequestId
}
$Body = @{
    model = $Model
    reasoning = @{ effort = $Effort }
    messages = @(
        @{
            role = "system"
            content = $System
        },
        @{
            role = "user"
            content = $Message
        }
    )
}

if ($MaxParallelAgents -gt 0) {
    $Body["metadata"] = @{
        agent = @{
            max_parallel_agents = $MaxParallelAgents
        }
    }
}

$Body = $Body | ConvertTo-Json -Depth 10

Write-Host "API: $BaseUrl/v1/chat/completions"
Write-Host "request_id: $RequestId"
if ($MaxParallelAgents -gt 0) {
    Write-Host "request max_parallel_agents: $MaxParallelAgents"
}

$Response = Invoke-RestMethod `
    -Uri "$BaseUrl/v1/chat/completions" `
    -Method Post `
    -ContentType "application/json" `
    -Headers $Headers `
    -Body $Body `
    -TimeoutSec $TimeoutSec

if ($RawResponse) {
    $Response | ConvertTo-Json -Depth 30
} else {
    $Text = $Response.choices[0].message.content
    $Usage = $Response.usage
    Write-Host "response:"
    Write-Host $Text
    if ($Usage) {
        Write-Host ("response_usage prompt={0} completion={1} total={2}" -f `
            $Usage.prompt_tokens,
            $Usage.completion_tokens,
            $Usage.total_tokens)
    }
}

Start-Sleep -Milliseconds 300

if (-not (Test-Path $LogPath)) {
    Write-Host "telemetry log not found: $LogPath"
    exit 0
}

$TelemetryLine = Get-Content -Path $LogPath -Tail 500 |
    Where-Object { $_ -like "*$RequestId*" -and $_ -like '*"event":"api_usage"*' } |
    Select-Object -Last 1

if (-not $TelemetryLine) {
    Write-Host "no telemetry found for request_id=$RequestId"
    Write-Host "log: $LogPath"
    exit 0
}

Write-Host "telemetry:"
if ($RawTelemetry) {
    Write-Host $TelemetryLine
    exit 0
}

$Telemetry = $TelemetryLine | ConvertFrom-Json
Write-Host ("tokens input={0} output={1} total={2}" -f `
    $Telemetry.input_tokens,
    $Telemetry.output_tokens,
    $Telemetry.total_tokens)
Write-Host ("agents providers={0} tasks={1} children={2} tools={3}" -f `
    $Telemetry.provider_call_count,
    $Telemetry.task_count,
    $Telemetry.child_agent_count,
    $Telemetry.tool_call_count)
Write-Host ("verification passed={0} issues={1} unresolved_tools={2}" -f `
    $Telemetry.verification.passed,
    $Telemetry.verification.issue_count,
    $Telemetry.verification.unresolved_tool_call_count)
Write-Host ("route={0} model={1} effort={2} tenant={3}" -f `
    $Telemetry.route,
    $Telemetry.model,
    $Telemetry.reasoning_effort,
    $Telemetry.tenant_id)

if ($ShowTrainingTrace) {
    if (-not (Test-Path $TrainingTracePath)) {
        Write-Host "training trace not found: $TrainingTracePath"
        exit 0
    }
    $TraceLine = Get-Content -Path $TrainingTracePath -Tail 1
    if (-not $TraceLine) {
        Write-Host "training trace is empty: $TrainingTracePath"
        exit 0
    }
    Write-Host "latest_training_trace:"
    Write-Host $TraceLine
}
