param(
    [switch]$SkipWarmup
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
$pythonExe = if (Test-Path $venvPython) { $venvPython } else { "python" }

Push-Location $repoRoot
try {
    if (-not $SkipWarmup) {
        & (Join-Path $repoRoot "scripts\warmup_priori_x.ps1")
    }

    $apiCommand = "Set-Location '$repoRoot'; `$env:PYTHONPATH='src'; & '$pythonExe' -m uvicorn apps.api.main:app --reload"
    Start-Process powershell -ArgumentList "-NoExit", "-Command", $apiCommand | Out-Null

    $env:PYTHONPATH = "src"
    & $pythonExe -m streamlit run apps/research_console/app.py
}
finally {
    Pop-Location
}
