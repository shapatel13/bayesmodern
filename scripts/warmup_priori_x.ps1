$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
$pythonExe = if (Test-Path $venvPython) { $venvPython } else { "python" }

Push-Location $repoRoot
try {
    $env:PYTHONPATH = "src"

    Write-Host "PRIORI-X warmup: status"
    & $pythonExe -m eval.experiment_cli status

    foreach ($preset in @("clinical_reasoning_demo_lab", "ed_triage_demo_lab", "medication_safety_demo_lab")) {
        Write-Host "PRIORI-X warmup: running preset $preset"
        & $pythonExe -m eval.experiment_cli run-preset-rollout $preset | Out-Null
    }

    Write-Host "PRIORI-X warmup complete. Demo experiments are ready in artifacts/evals/experiments."
}
finally {
    Pop-Location
}
