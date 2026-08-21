@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0install_env.ps1" %*
set "INSTALL_EXIT_CODE=%ERRORLEVEL%"

echo.
if not "%INSTALL_EXIT_CODE%"=="0" (
    echo Installation failed. Review the error message above.
) else (
    echo Environment is ready.
)
pause
exit /b %INSTALL_EXIT_CODE%
