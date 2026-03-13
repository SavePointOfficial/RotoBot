@echo off
TITLE Rotobot Portable Installer
echo ==================================================
echo   ROTOBOT VENV ^& DEPENDENCY INSTALLER
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

echo [INFO] Upgrading pip and build tools...
python -m pip install --upgrade pip setuptools wheel

:: ========================================================
:: Auto-detect CUDA version from the NVIDIA driver
:: ========================================================
echo.
echo [INFO] Detecting NVIDIA GPU and CUDA support...

set "CUDA_INDEX=cpu"
set "CUDA_LABEL=CPU only"

:: Use PowerShell to reliably parse the CUDA version from nvidia-smi
for /f "usebackq delims=" %%v in (`powershell -NoProfile -Command "try { $m = nvidia-smi 2^>$null ^| Select-String 'CUDA Version: (\d+)\.(\d+)'; if ($m) { $m.Matches.Groups[1].Value + '.' + $m.Matches.Groups[2].Value } else { 'none' } } catch { 'none' }"`) do set "CUDA_VER=%%v"

if "%CUDA_VER%"=="none" (
    echo [WARNING] No NVIDIA GPU detected. Installing CPU-only PyTorch.
    goto :install_torch
)

:: Parse major.minor
for /f "tokens=1,2 delims=." %%a in ("%CUDA_VER%") do (
    set /a "CUDA_MAJOR=%%a"
    set /a "CUDA_MINOR=%%b"
)

echo [INFO] Detected driver CUDA version: %CUDA_VER%

:: Pick the best PyTorch CUDA build:
::   >= 12.8  ->  cu128  (sm_50 - sm_120: RTX 20xx through 50xx + Blackwell)
::   >= 12.4  ->  cu124  (sm_50 - sm_90:  RTX 20xx through 40xx)
::   >= 11.8  ->  cu118  (sm_37 - sm_90:  GTX 10xx through 40xx)

if %CUDA_MAJOR% GEQ 13 (
    set "CUDA_INDEX=cu128"
    set "CUDA_LABEL=CUDA 12.8+ (full GPU support including RTX 50xx)"
    goto :install_torch
)

if %CUDA_MAJOR% EQU 12 (
    if %CUDA_MINOR% GEQ 8 (
        set "CUDA_INDEX=cu128"
        set "CUDA_LABEL=CUDA 12.8 (full GPU support including RTX 50xx)"
        goto :install_torch
    )
    if %CUDA_MINOR% GEQ 4 (
        set "CUDA_INDEX=cu124"
        set "CUDA_LABEL=CUDA 12.4 (RTX 20xx - 40xx support)"
        goto :install_torch
    )
    set "CUDA_INDEX=cu118"
    set "CUDA_LABEL=CUDA 11.8 (broad compatibility)"
    goto :install_torch
)

if %CUDA_MAJOR% GEQ 11 (
    set "CUDA_INDEX=cu118"
    set "CUDA_LABEL=CUDA 11.8 (broad compatibility)"
    goto :install_torch
)

echo [WARNING] CUDA %CUDA_VER% may be too old. Trying cu118 as fallback.
set "CUDA_INDEX=cu118"
set "CUDA_LABEL=CUDA 11.8 (fallback)"

:install_torch
echo.
echo [INFO] Installing PyTorch with %CUDA_LABEL%...
if "%CUDA_INDEX%"=="cpu" (
    pip install torch torchvision torchaudio
) else (
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/%CUDA_INDEX%
)

echo [INFO] Installing remaining Python packages...
pip install -r requirements.txt

echo [INFO] Setting up local models...
python setup_models.py

echo.
echo ==================================================
echo   INSTALLATION COMPLETE
echo ==================================================
echo   PyTorch: %CUDA_LABEL%
echo   To run Rotobot, use run_rotobot.bat
echo ==================================================
pause
