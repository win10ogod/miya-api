param(
    [string]$TracePath = "",
    [string]$OutputPath = "",
    [switch]$Compact
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
if (-not $TracePath) {
    $TracePath = Join-Path $ProjectRoot "logs\training-traces.jsonl"
}
if (-not $OutputPath) {
    $OutputPath = Join-Path $ProjectRoot "logs\training-dataset.json"
}

if (-not (Test-Path $TracePath)) {
    throw "training trace file not found: $TracePath"
}

$Samples = New-Object System.Collections.Generic.List[object]
Get-Content -Path $TracePath | ForEach-Object {
    if (-not [string]::IsNullOrWhiteSpace($_)) {
        $Samples.Add(($_ | ConvertFrom-Json)) | Out-Null
    }
}

$Depth = 100
$Array = [object[]]$Samples.ToArray()
$Json = if ($Compact) {
    ConvertTo-Json -InputObject $Array -Depth $Depth -Compress
} else {
    ConvertTo-Json -InputObject $Array -Depth $Depth
}

$OutputDir = Split-Path -Parent $OutputPath
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
Set-Content -Path $OutputPath -Value $Json -Encoding UTF8

Write-Host "exported samples=$($Samples.Count)"
Write-Host "output=$OutputPath"
