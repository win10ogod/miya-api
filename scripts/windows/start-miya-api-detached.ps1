param(
    [string]$BindAddr = "127.0.0.1:3100",
    [string]$OpenAIBaseUrl = "http://localhost:8000/v1",
    [string]$OpenAIApiKey = "local-key",
    [string]$DefaultModel = "local-model",
    [string]$ExposedModels = "",
    [string]$GemmaModels = "",
    [string]$ContextStorePath = ".multi-agent-context\surrealkv",
    [int]$TenantMaxConcurrentRequests = 16,
    [int]$MaxParallelAgents = 4,
    [ValidateSet("request", "always", "never", "strip", "on", "off")]
    [string]$PublicReasoning = "always",
    [switch]$TrainingTrace,
    [string]$TrainingTracePath = "logs\training-traces.jsonl",
    [switch]$NoBuild
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$StartScript = Join-Path $PSScriptRoot "start-miya-api.ps1"
$LogDir = Join-Path $ProjectRoot "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$OutLog = Join-Path $LogDir "start-miya-api-detached-$Timestamp.out.log"
$ErrLog = Join-Path $LogDir "start-miya-api-detached-$Timestamp.err.log"

function Quote-Arg([string]$Value) {
    return '"' + ($Value -replace '"', '\"') + '"'
}

$ArgsList = @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", (Quote-Arg $StartScript),
    "-BindAddr", (Quote-Arg $BindAddr),
    "-OpenAIBaseUrl", (Quote-Arg $OpenAIBaseUrl),
    "-OpenAIApiKey", (Quote-Arg $OpenAIApiKey),
    "-DefaultModel", (Quote-Arg $DefaultModel),
    "-ContextStorePath", (Quote-Arg $ContextStorePath),
    "-TenantMaxConcurrentRequests", $TenantMaxConcurrentRequests,
    "-MaxParallelAgents", $MaxParallelAgents,
    "-PublicReasoning", (Quote-Arg $PublicReasoning)
)

if ($ExposedModels) {
    $ArgsList += @("-ExposedModels", (Quote-Arg $ExposedModels))
}
if ($GemmaModels) {
    $ArgsList += @("-GemmaModels", (Quote-Arg $GemmaModels))
}
if ($TrainingTrace) {
    $ArgsList += @("-TrainingTrace", "-TrainingTracePath", (Quote-Arg $TrainingTracePath))
}
if ($NoBuild) {
    $ArgsList += "-NoBuild"
}

$CommandLine = "cmd.exe /c cd /d `"$ProjectRoot`" && powershell.exe $($ArgsList -join ' ') 1> `"$OutLog`" 2> `"$ErrLog`""
$Result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine = $CommandLine
}

if ($Result.ReturnValue -ne 0) {
    throw "failed to start detached Miya API process, Win32 return value $($Result.ReturnValue)"
}

Write-Host "Started detached Miya API launcher. PID=$($Result.ProcessId)"
Write-Host "Stdout: $OutLog"
Write-Host "Stderr: $ErrLog"
