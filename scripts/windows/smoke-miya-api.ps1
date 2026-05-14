param(
    [string]$BaseUrl = "http://127.0.0.1:3100",
    [string]$Model = "local-model"
)

$ErrorActionPreference = "Stop"

$Health = Invoke-RestMethod -Uri "$BaseUrl/health" -Method Get -TimeoutSec 10
Write-Host "health=$($Health.status)"

$Models = Invoke-RestMethod -Uri "$BaseUrl/v1/models" -Method Get -TimeoutSec 10
if (-not ($Models.data.id -contains $Model)) {
    throw "$Model was not exposed by /v1/models"
}
Write-Host "models include $Model"

$Body = @{
    model = $Model
    reasoning = @{ effort = "none" }
    messages = @(
        @{
            role = "system"
            content = "Reply exactly OK."
        },
        @{
            role = "user"
            content = "OK"
        }
    )
} | ConvertTo-Json -Depth 8

$Response = Invoke-RestMethod `
    -Uri "$BaseUrl/v1/chat/completions" `
    -Method Post `
    -ContentType "application/json" `
    -Body $Body `
    -TimeoutSec 120

$Text = $Response.choices[0].message.content
if (-not $Text) {
    throw "chat completion returned an empty response"
}

Write-Host "chat completion ok:"
Write-Host $Text
