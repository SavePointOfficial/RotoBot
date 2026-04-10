@echo off
REM Rotobot Launcher
REM Uses the embedded Python installed by install_rotobot.bat

cd /d "%~dp0"

set "PYTHON_EXE=%~dp0venv\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
    echo [ERROR] Python venv not found.
    echo Please run install_rotobot.bat first.
    echo.
    pause
    exit /b 1
)

echo ============================================================
echo   ROTOBOT -- Automatic Rotoscoping Tool
echo ============================================================
echo.

"%PYTHON_EXE%" rotobot_gui.py

if errorlevel 1 (
    echo.
    echo Rotobot exited with an error. Press any key to close.
    pause >nul
)
