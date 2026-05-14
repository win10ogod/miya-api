param(
    [string]$TracePath = "",
    [int]$Last = 5,
    [switch]$Follow,
    [switch]$Raw
)

$ErrorActionPreference = "Stop"

if (-not $TracePath) {
    $ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
    $TracePath = Join-Path $ProjectRoot "logs\training-traces.jsonl"
}

$TraceDir = Split-Path -Parent $TracePath
New-Item -ItemType Directory -Force -Path $TraceDir | Out-Null
if (-not (Test-Path $TracePath)) {
    New-Item -ItemType File -Force -Path $TracePath | Out-Null
}

function Show-TrainingTraceLine {
    param([string]$Line)

    if ([string]::IsNullOrWhiteSpace($Line)) {
        return
    }
    if ($Raw) {
        Write-Host $Line
        return
    }

    try {
        $sample = $Line | ConvertFrom-Json
        $turnCount = if ($sample.conversations) { $sample.conversations.Count } else { 0 }
        $firstHuman = $sample.conversations |
            Where-Object { $_.from -eq "human" } |
            Select-Object -First 1
        $lastGpt = $sample.conversations |
            Where-Object { $_.from -eq "gpt" } |
            Select-Object -Last 1
        $toolCalls = @($sample.conversations | Where-Object { $_.from -eq "function_call" }).Count
        $observations = @($sample.conversations | Where-Object { $_.from -eq "observation" }).Count
        $humanText = if ($firstHuman) { $firstHuman.value } else { "" }
        $gptText = if ($lastGpt) { $lastGpt.value } else { "" }
        if ($humanText.Length -gt 80) { $humanText = $humanText.Substring(0, 80) + "..." }
        if ($gptText.Length -gt 80) { $gptText = $gptText.Substring(0, 80) + "..." }
        Write-Host ("turns={0} function_calls={1} observations={2}" -f $turnCount, $toolCalls, $observations)
        Write-Host ("  human: {0}" -f $humanText)
        Write-Host ("  gpt:   {0}" -f $gptText)
    } catch {
        Write-Host $Line
    }
}

Write-Host "Training trace: $TracePath"
Write-Host "Press Ctrl+C to stop watching."

if ($Follow) {
    Get-Content -Path $TracePath -Tail $Last -Wait | ForEach-Object {
        Show-TrainingTraceLine $_
    }
} else {
    Get-Content -Path $TracePath -Tail $Last | ForEach-Object {
        Show-TrainingTraceLine $_
    }
}
