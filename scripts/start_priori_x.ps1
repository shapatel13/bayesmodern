param(
    [switch]$Check,
    [switch]$NoApi,
    [switch]$Headless
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
$pythonExe = if (Test-Path $venvPython) { $venvPython } else { "python" }

Push-Location $repoRoot
try {
    $arguments = @((Join-Path $repoRoot "run_priori_x.py"))
    if ($Check) {
        $arguments += "--check"
    }
    if ($NoApi) {
        $arguments += "--no-api"
    }
    if ($Headless) {
        $arguments += "--headless"
    }
    & $pythonExe @arguments
}
finally {
    Pop-Location
}
