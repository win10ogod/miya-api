param(
    [string]$BindAddr = "127.0.0.1:3100",
    [string]$OpenAIBaseUrl = "http://localhost:8000/v1",
    [string]$OpenAIApiKey = "local-key",
    [string]$MiyaApiKey = "miya-local-key",
    [string]$DefaultModel = "local-model",
    [string]$ExposedModels = "",
    [string]$GemmaModels = "",
    [string]$ContextStorePath = ".multi-agent-context\surrealkv",
    [string]$DataDir = ".multi-agent-data",
    [int]$TenantMaxConcurrentRequests = 16,
    [int]$TenantQueueTimeoutMs = 30000,
    [int]$MaxParallelAgents = 4,
    [int]$ProviderMaxConcurrent = 64,
    [int]$ProviderQueueTimeoutMs = 30000,
    [int]$ProviderMaxRetries = 2,
    [int]$ProviderRetryBaseMs = 250,
    [int]$ProviderCircuitFailureThreshold = 5,
    [int]$ProviderCircuitCooldownMs = 30000,
    [int]$RequestTimeoutMs = 3600000,
    [int]$AgentTimeoutMs = 330000,
    [int]$StreamHeartbeatSec = 10,
    [int]$MaxConcurrentJobs = 4,
    [int]$BatchItemConcurrency = 8,
    [bool]$SemanticVerifier = $true,
    [int]$SemanticMaxRepairAttempts = 2,
    [string]$OtlpEndpoint = "",
    [ValidateSet("request", "always", "never", "strip", "on", "off")]
    [string]$PublicReasoning = "always",
    [switch]$TrainingTrace,
    [string]$TrainingTracePath = "logs\training-traces.jsonl",
    [switch]$NoBuild
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$LogDir = Join-Path $ProjectRoot "logs"
$PidFile = Join-Path $LogDir "api-server.pid"
$OutLog = Join-Path $LogDir "api-server.out.log"
$ErrLog = Join-Path $LogDir "api-server.err.log"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

if (-not $NoBuild) {
    Push-Location $ProjectRoot
    try {
        cargo.exe build -p api-server --release
    } finally {
        Pop-Location
    }
}

$Exe = Join-Path $ProjectRoot "target\release\api-server.exe"
if (-not (Test-Path $Exe)) {
    throw "api-server.exe not found at $Exe. Run without -NoBuild first."
}

if (Test-Path $PidFile) {
    $ExistingPid = Get-Content $PidFile -ErrorAction SilentlyContinue
    if ($ExistingPid) {
        $Existing = Get-Process -Id ([int]$ExistingPid) -ErrorAction SilentlyContinue
        if ($Existing) {
            Write-Host "Existing api-server process is running: PID $ExistingPid"
            Write-Host "Use scripts\windows\stop-miya-api.ps1 before starting another instance."
            exit 0
        }
    }
}

$env:BIND_ADDR = $BindAddr
$env:MULTI_AGENT_PROVIDER = "openai"
$env:OPENAI_BASE_URL = $OpenAIBaseUrl
$env:OPENAI_API_KEY = $OpenAIApiKey
$env:MIYA_API_KEY = $MiyaApiKey
if ($ExposedModels) {
    $env:MULTI_AGENT_MODELS = $ExposedModels
} else {
    $env:MULTI_AGENT_MODELS = "$DefaultModel,mock"
}
if ($GemmaModels) {
    $env:MIYA_GEMMA_MODELS = $GemmaModels
} else {
    Remove-Item Env:MIYA_GEMMA_MODELS -ErrorAction SilentlyContinue
}
$env:CONTEXT_STORE_PATH = $ContextStorePath
$env:MIYA_DATA_DIR = $DataDir
$env:TENANT_MAX_CONCURRENT_REQUESTS = [string]$TenantMaxConcurrentRequests
$env:MIYA_TENANT_QUEUE_TIMEOUT_MS = [string]$TenantQueueTimeoutMs
$env:MULTI_AGENT_MAX_PARALLEL_AGENTS = [string]$MaxParallelAgents
$env:MIYA_PROVIDER_MAX_CONCURRENT = [string]$ProviderMaxConcurrent
$env:MIYA_PROVIDER_QUEUE_TIMEOUT_MS = [string]$ProviderQueueTimeoutMs
$env:MIYA_PROVIDER_MAX_RETRIES = [string]$ProviderMaxRetries
$env:MIYA_PROVIDER_RETRY_BASE_MS = [string]$ProviderRetryBaseMs
$env:MIYA_PROVIDER_CIRCUIT_FAILURE_THRESHOLD = [string]$ProviderCircuitFailureThreshold
$env:MIYA_PROVIDER_CIRCUIT_COOLDOWN_MS = [string]$ProviderCircuitCooldownMs
$env:MIYA_REQUEST_TIMEOUT_MS = [string]$RequestTimeoutMs
$env:MIYA_AGENT_TIMEOUT_MS = [string]$AgentTimeoutMs
$env:MIYA_STREAM_HEARTBEAT_SECS = [string]$StreamHeartbeatSec
$env:MIYA_MAX_CONCURRENT_JOBS = [string]$MaxConcurrentJobs
$env:MIYA_BATCH_ITEM_CONCURRENCY = [string]$BatchItemConcurrency
$env:MIYA_SEMANTIC_VERIFIER = if ($SemanticVerifier) { "true" } else { "false" }
$env:MIYA_SEMANTIC_MAX_REPAIR_ATTEMPTS = [string]$SemanticMaxRepairAttempts
if ($OtlpEndpoint) {
    $env:OTEL_EXPORTER_OTLP_ENDPOINT = $OtlpEndpoint
} else {
    Remove-Item Env:OTEL_EXPORTER_OTLP_ENDPOINT -ErrorAction SilentlyContinue
}
$env:MIYA_PUBLIC_REASONING = $PublicReasoning
if ($TrainingTrace) {
    $env:TRAINING_TRACE = "enabled"
    $env:TRAINING_TRACE_PATH = $TrainingTracePath
} else {
    Remove-Item Env:TRAINING_TRACE -ErrorAction SilentlyContinue
    Remove-Item Env:TRAINING_TRACE_PATH -ErrorAction SilentlyContinue
}

$Process = Start-Process `
    -FilePath $Exe `
    -WorkingDirectory $ProjectRoot `
    -RedirectStandardOutput $OutLog `
    -RedirectStandardError $ErrLog `
    -PassThru

Set-Content -Path $PidFile -Value $Process.Id

$Port = ($BindAddr -split ":")[-1]
$HealthUrl = "http://127.0.0.1:$Port/health"
Start-Sleep -Seconds 2

try {
    $Health = Invoke-RestMethod -Uri $HealthUrl -Method Get -TimeoutSec 5
    Write-Host "api-server started. PID=$($Process.Id) health=$($Health.status)"
} catch {
    Write-Host "api-server process started. PID=$($Process.Id), but health check did not respond yet."
    Write-Host "Check logs: $OutLog and $ErrLog"
}

Write-Host "OpenAI-compatible API: http://127.0.0.1:$Port/v1"
Write-Host "Upstream backend: $OpenAIBaseUrl"
Write-Host "Shared Miya API key: $MiyaApiKey"
Write-Host "Default local model: $DefaultModel"
Write-Host "Exposed models: $env:MULTI_AGENT_MODELS"
Write-Host "Gemma-formatted models: $env:MIYA_GEMMA_MODELS"
Write-Host "Tenant max concurrent requests: $TenantMaxConcurrentRequests; queue timeout=$TenantQueueTimeoutMs ms"
Write-Host "Max parallel agents: $MaxParallelAgents"
Write-Host "Provider max concurrent calls: $ProviderMaxConcurrent"
Write-Host "Provider queue timeout: $ProviderQueueTimeoutMs ms"
Write-Host "Provider retries/circuit: $ProviderMaxRetries retries, threshold=$ProviderCircuitFailureThreshold cooldown=$ProviderCircuitCooldownMs ms"
Write-Host "Request/agent timeout: $RequestTimeoutMs/$AgentTimeoutMs ms"
Write-Host "Orchestration SSE heartbeat: $StreamHeartbeatSec sec"
Write-Host "Durable data: $DataDir; jobs=$MaxConcurrentJobs batch-item-concurrency=$BatchItemConcurrency"
Write-Host "Semantic verifier: $SemanticVerifier; max repairs=$SemanticMaxRepairAttempts"
Write-Host "OTLP endpoint: $OtlpEndpoint"
Write-Host "Public reasoning mode: $PublicReasoning"
if ($TrainingTrace) {
    Write-Host "Training trace: enabled -> $TrainingTracePath"
} else {
    Write-Host "Training trace: disabled"
}
