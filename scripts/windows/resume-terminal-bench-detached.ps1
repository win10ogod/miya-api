param(
    [Parameter(Mandatory = $true)]
    [string]$JobPath,
    [string]$HarborExe = "C:\Users\jmes1\.local\bin\harbor.exe",
    [string]$LogPrefix = "terminal-bench-resume-detached"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$ResolvedJobPath = Resolve-Path $JobPath
$LogDir = Join-Path $ProjectRoot "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$OutLog = Join-Path $LogDir "$LogPrefix-$Timestamp.out.log"
$ErrLog = Join-Path $LogDir "$LogPrefix-$Timestamp.err.log"

if (-not (Test-Path $HarborExe)) {
    throw "harbor executable not found at $HarborExe"
}

$CommandLine = "cmd.exe /c cd /d `"$ProjectRoot`" && `"$HarborExe`" job resume -p `"$ResolvedJobPath`" 1> `"$OutLog`" 2> `"$ErrLog`""
$Result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine = $CommandLine
}

if ($Result.ReturnValue -ne 0) {
    throw "failed to start detached Harbor resume process, Win32 return value $($Result.ReturnValue)"
}

Write-Host "Started detached Harbor resume. PID=$($Result.ProcessId)"
Write-Host "Job: $ResolvedJobPath"
Write-Host "Stdout: $OutLog"
Write-Host "Stderr: $ErrLog"
