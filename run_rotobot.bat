@echo off
REM Rotobot Standalone Launcher
REM Activates the LabyrinthGameSandbox venv and runs the GUI.

cd /d "%~dp0"

REM 1. Priority: local venv created by install_rotobot.bat
set VENV=%~dp0venv\Scripts\activate.bat
if not exist "%VENV%" (
    REM 2. Fallback: LabyrinthGameSandbox
    set VENV=D:\PROGRAMS\LabyrinthGameSandbox\.venv\Scripts\activate.bat
)

if exist "%VENV%" (
    call "%VENV%"
) else (
    echo WARNING: Virtual environment not found.
    echo Trying system Python...
)

echo ============================================================
echo   ROTOBOT -- Automatic Rotoscoping Tool  (Standalone)
echo ============================================================
echo.

python rotobot_gui.py

if errorlevel 1 (
    echo.
    echo Rotobot exited with an error. Press any key to close.
    pause >nul
)
