@echo off
setlocal
cd /d "%~dp0"
powershell -ExecutionPolicy Bypass -File "%~dp0start.ps1" %*
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo MeshPool stopped or encountered an error.
    pause
)
