# bootstrap.ps1 - (re)create the repo venv from requirements.lock.txt, then smoke test.
# Usage (from E:\Basketball\basketball):  powershell -ExecutionPolicy Bypass -File .\bootstrap.ps1
# Does not request admin / UAC.

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

$Lock = Join-Path $PSScriptRoot 'requirements.lock.txt'
if (-not (Test-Path -LiteralPath $Lock)) {
    throw "Missing lockfile: $Lock"
}

$VenvPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $VenvPython)) {
    $Created = $false
    if (Get-Command py -ErrorAction SilentlyContinue) {
        Write-Host 'Creating .venv with py -3.14'
        py -3.14 -m venv .venv
        if ($LASTEXITCODE -eq 0 -and (Test-Path -LiteralPath $VenvPython)) {
            $Created = $true
        }
    }
    if (-not $Created) {
        $SystemPython = 'E:\Installs\Python314\python.exe'
        if (-not (Test-Path -LiteralPath $SystemPython)) {
            throw 'Could not create .venv: py -3.14 failed and E:\Installs\Python314\python.exe is missing'
        }
        Write-Host "Creating .venv with $SystemPython"
        & $SystemPython -m venv .venv
        if ($LASTEXITCODE -ne 0) {
            throw "venv creation failed (exit $LASTEXITCODE)"
        }
    }
    if (-not (Test-Path -LiteralPath $VenvPython)) {
        throw "venv created but $VenvPython is missing"
    }
}

Write-Host "Installing $Lock"
& $VenvPython -m pip install -r $Lock
if ($LASTEXITCODE -ne 0) {
    throw "pip install failed (exit $LASTEXITCODE)"
}

Write-Host 'Smoke test: python -m unittest discover -s tests'
& $VenvPython -m unittest discover -s tests
if ($LASTEXITCODE -ne 0) {
    throw "unit tests failed (exit $LASTEXITCODE)"
}

Write-Host 'bootstrap ok'
