$ErrorActionPreference = "Stop"
$project = $PSScriptRoot
$python = Join-Path $project ".venv\Scripts\python.exe"
$script = Join-Path $project "auto_potion.py"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Python virtual environment was not found: $python"
}

$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = [Security.Principal.WindowsPrincipal]::new($identity)
$isAdmin = $principal.IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator
)

if (-not $isAdmin) {
    Start-Process -FilePath $python `
        -ArgumentList @($script) `
        -WorkingDirectory $project `
        -Verb RunAs
    exit
}

Set-Location -LiteralPath $project
& $python $script
Read-Host "Press Enter to close"
