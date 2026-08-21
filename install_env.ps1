[CmdletBinding()]
param(
    [switch]$Recreate
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = [System.IO.Path]::GetFullPath($PSScriptRoot)
$VenvDirectory = Join-Path $ProjectRoot ".venv"
$VenvPython = Join-Path $VenvDirectory "Scripts\python.exe"
$RequirementsFile = Join-Path $ProjectRoot "requirements.txt"
$TorchBuild = "cu130"

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath,

        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $FilePath $($Arguments -join ' ')"
    }
}

function Get-CompatiblePython {
    $versionProbe = "import sys; v=sys.version_info[:3]; ok=(3, 10) <= v[:2] < (3, 15) and v != (3, 14, 1) and sys.maxsize > 2**32; print(sys.executable if ok else '')"
    $launcher = Get-Command "py.exe" -ErrorAction SilentlyContinue
    if ($null -ne $launcher) {
        foreach ($version in @("3.14", "3.13", "3.12", "3.11", "3.10")) {
            $resolved = & $launcher.Source "-$version" -c $versionProbe 2>$null
            if ($LASTEXITCODE -eq 0 -and $resolved) {
                return [string]($resolved | Select-Object -First 1)
            }
        }
    }

    $pythonCommand = Get-Command "python.exe" -ErrorAction SilentlyContinue
    if ($null -ne $pythonCommand) {
        $resolved = & $pythonCommand.Source -c $versionProbe 2>$null
        if ($LASTEXITCODE -eq 0 -and $resolved) {
            return [string]($resolved | Select-Object -First 1)
        }
    }

    $candidatePatterns = @(
        (Join-Path $env:LOCALAPPDATA "Programs\Python\Python3*\python.exe"),
        (Join-Path $env:ProgramFiles "Python3*\python.exe")
    )
    $candidates = Get-ChildItem -Path $candidatePatterns -File -ErrorAction SilentlyContinue |
        Sort-Object -Property FullName -Descending
    foreach ($candidate in $candidates) {
        $resolved = & $candidate.FullName -c $versionProbe 2>$null
        if ($LASTEXITCODE -eq 0 -and $resolved) {
            return [string]($resolved | Select-Object -First 1)
        }
    }

    return $null
}

function Assert-SafeVenvPath {
    $resolvedProject = [System.IO.Path]::GetFullPath($ProjectRoot).TrimEnd('\')
    $resolvedVenv = [System.IO.Path]::GetFullPath($VenvDirectory).TrimEnd('\')
    $expectedVenv = (Join-Path $resolvedProject ".venv").TrimEnd('\')
    if ($resolvedVenv -ne $expectedVenv -or $resolvedVenv -eq $resolvedProject) {
        throw "Refusing to modify unexpected virtualenv path: $resolvedVenv"
    }
}

Set-Location -LiteralPath $ProjectRoot
Write-Host "=== MapleStory environment installer ===" -ForegroundColor Cyan
Write-Host "Project: $ProjectRoot"
Write-Host "PyTorch build: $TorchBuild"

$PythonExecutable = Get-CompatiblePython
if (-not $PythonExecutable) {
    throw "No compatible 64-bit Python was found. This project supports Python 3.10 through 3.14, except Python 3.14.1."
}

$SourcePythonVersion = & $PythonExecutable -c "import platform; print(platform.python_version())"
Write-Host "Python: $SourcePythonVersion ($PythonExecutable)"

if ($Recreate -and (Test-Path -LiteralPath $VenvDirectory)) {
    Assert-SafeVenvPath
    Write-Host "Removing the existing .venv..." -ForegroundColor Yellow
    Remove-Item -LiteralPath $VenvDirectory -Recurse -Force
}

if (-not (Test-Path -LiteralPath $VenvPython -PathType Leaf)) {
    if (Test-Path -LiteralPath $VenvDirectory) {
        throw "The existing .venv is incomplete. Run install_env.bat -Recreate to rebuild it."
    }
    Write-Host "Creating .venv..."
    Invoke-Checked $PythonExecutable @("-m", "venv", $VenvDirectory)
}

$VenvVersion = & $VenvPython -c "import platform; print(platform.python_version())"
$VenvCompatible = & $VenvPython -c "import sys; v=sys.version_info[:3]; print(int((3, 10) <= v[:2] < (3, 15) and v != (3, 14, 1) and sys.maxsize > 2**32))"
if ($LASTEXITCODE -ne 0 -or $VenvCompatible -ne "1") {
    throw ".venv uses unsupported Python $VenvVersion. Run install_env.bat -Recreate."
}

if (-not (Test-Path -LiteralPath $RequirementsFile -PathType Leaf)) {
    throw "requirements.txt is missing from $ProjectRoot"
}

Write-Host "Updating pip build tools..."
Invoke-Checked $VenvPython @(
    "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"
)

$TorchIndex = "https://download.pytorch.org/whl/$TorchBuild"
Write-Host "Installing Python-compatible PyTorch / torchvision from $TorchIndex ..."
Invoke-Checked $VenvPython @(
    "-m", "pip", "install", "--upgrade",
    "torch", "torchvision",
    "--index-url", $TorchIndex
)

Write-Host "Installing project dependencies..."
Invoke-Checked $VenvPython @(
    "-m", "pip", "install", "--upgrade", "-r", $RequirementsFile
)

$RequiredFiles = @(
    "detect_game_yolo.py",
    "capture_game_window.py",
    "maplestory_02.pt"
)
foreach ($relativePath in $RequiredFiles) {
    $fullPath = Join-Path $ProjectRoot $relativePath
    if (-not (Test-Path -LiteralPath $fullPath -PathType Leaf)) {
        throw "Required project file is missing: $relativePath"
    }
}

Write-Host "Running environment verification..."
Invoke-Checked $VenvPython @("check_environment.py")

Write-Host ""
Write-Host "Installation completed successfully." -ForegroundColor Green
Write-Host "Run the detector with:"
Write-Host "  .venv\Scripts\python.exe detect_game_yolo.py" -ForegroundColor Cyan
