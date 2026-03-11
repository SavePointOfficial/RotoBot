@echo off
TITLE Rotobot Portable Installer
echo ==================================================
echo   ROTOBOT VENV & DEPENDENCY INSTALLER
echo ==================================================
echo.

set "VENV_DIR=venv"
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo [INFO] Creating Python virtual environment...
    python -m venv %VENV_DIR%
) else (
    echo [INFO] Virtual environment already exists.
)

echo [INFO] Activating virtual environment...
call %VENV_DIR%\Scripts\activate.bat

echo [INFO] Upgrading pip...
python -m pip install --upgrade pip

echo [INFO] Installing required Python packages...
:: Install torch with CUDA support before requirements
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

echo [INFO] Setting up local models...
python setup_models.py

echo.
echo ==================================================
echo   INSTALLATION COMPLETE
echo ==================================================
echo To run Rotobot, use run_rotobot.bat
pause
