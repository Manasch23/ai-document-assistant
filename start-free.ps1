<#
.SYNOPSIS
Starts Folio using the free, on-device Ollama model.
.DESCRIPTION
Run this script from PowerShell after completing the README setup steps.
#>
$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    Write-Host "Ollama is not installed. Visit https://ollama.com/download, install it, then run this script again." -ForegroundColor Yellow
    exit 1
}

& ollama show llama3.2:3b 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Downloading the free local model once..." -ForegroundColor Cyan
    & ollama pull llama3.2:3b
}

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    python -m venv .venv
    .\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
}

if (-not (Test-Path ".env")) { Copy-Item .env.example .env }
Write-Host "Open http://127.0.0.1:8000 in your browser." -ForegroundColor Green
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
