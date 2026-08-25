@echo off
rem MeshPool - double-click this to run MeshPool on your machine (Windows).
rem It installs what it needs on first run, then opens the dashboard in your browser.
setlocal enabledelayedexpansion
cd /d "%~dp0"

set "PY="
where py >nul 2>nul && set "PY=py"
if not defined PY ( where python >nul 2>nul && set "PY=python" )
if not defined PY (
  echo.
  echo Python 3.11+ is required but was not found.
  echo Install it from https://www.python.org/downloads/ and tick
  echo    "Add python.exe to PATH" during setup, then run this again.
  echo.
  pause
  exit /b 1
)

if not exist ".meshpool-deps-ok" (
  echo ==^> First run: installing dependencies, please wait a minute or two ...
  %PY% -m pip install --quiet --disable-pip-version-check -r requirements.txt
  if not errorlevel 1 (echo ok> ".meshpool-deps-ok")
)

%PY% -m meshpool.launcher solo %*
pause
