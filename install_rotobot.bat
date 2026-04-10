@echo off
TITLE Rotobot Portable Installer
echo ==================================================
echo   ROTOBOT PORTABLE INSTALLER
echo   No prerequisites required (except NVIDIA drivers)
echo ==================================================
echo.

cd /d "%~dp0"
set "ROTOBOT_DIR=%~dp0"
set "PYTHON_DIR=%ROTOBOT_DIR%python"
set "PYTHON_EXE=%PYTHON_DIR%\python.exe"
set "PIP_EXE=%PYTHON_DIR%\Scripts\pip.exe"
set "PYTHON_VER=3.10.11"
set "PYTHON_ZIP=python-%PYTHON_VER%-embed-amd64.zip"
set "PYTHON_URL=https://www.python.org/ftp/python/%PYTHON_VER%/%PYTHON_ZIP%"

:: ========================================================
:: 1. Download embedded Python if not present
:: ========================================================
if exist "%PYTHON_EXE%" (
    echo [INFO] Embedded Python already installed.
    goto :have_python
)

echo [INFO] Downloading Python %PYTHON_VER% embedded...
powershell -NoProfile -Command ^
    "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; " ^
    "Invoke-WebRequest -Uri '%PYTHON_URL%' -OutFile '%ROTOBOT_DIR%%PYTHON_ZIP%'"

if not exist "%ROTOBOT_DIR%%PYTHON_ZIP%" (
    echo [ERROR] Failed to download Python. Check your internet connection.
    pause
    exit /b 1
)

echo [INFO] Extracting Python...
powershell -NoProfile -Command ^
    "Expand-Archive -Path '%ROTOBOT_DIR%%PYTHON_ZIP%' -DestinationPath '%PYTHON_DIR%' -Force"

del "%ROTOBOT_DIR%%PYTHON_ZIP%" 2>nul

if not exist "%PYTHON_EXE%" (
    echo [ERROR] Python extraction failed.
    pause
    exit /b 1
)

:: Enable site-packages by uncommenting "import site" in the ._pth file.
:: The embeddable Python ships with this commented out, which prevents pip.
echo [INFO] Configuring Python for package installation...
powershell -NoProfile -Command ^
    "$pth = Get-ChildItem '%PYTHON_DIR%\python*._pth' | Select-Object -First 1; " ^
    "if ($pth) { " ^
    "  $content = Get-Content $pth.FullName; " ^
    "  $content = $content -replace '^#import site', 'import site'; " ^
    "  $content = $content -replace '^#\s*import site', 'import site'; " ^
    "  Set-Content $pth.FullName $content; " ^
    "  Write-Host '  -> Enabled site-packages in' $pth.Name " ^
    "}"

:: Bootstrap pip
echo [INFO] Installing pip...
powershell -NoProfile -Command ^
    "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; " ^
    "Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile '%ROTOBOT_DIR%get-pip.py'"

"%PYTHON_EXE%" "%ROTOBOT_DIR%get-pip.py" --no-warn-script-location
del "%ROTOBOT_DIR%get-pip.py" 2>nul

if not exist "%PIP_EXE%" (
    echo [ERROR] pip installation failed.
    pause
    exit /b 1
)

echo [OK] Embedded Python %PYTHON_VER% ready.

:have_python

:: ========================================================
:: 2. Upgrade pip and build tools
:: ========================================================
echo.
echo [INFO] Upgrading pip and build tools...
"%PYTHON_EXE%" -m pip install --upgrade pip setuptools wheel --no-warn-script-location

:: ========================================================
:: 3. Auto-detect CUDA and install PyTorch
:: ========================================================
echo.
echo [INFO] Detecting NVIDIA GPU and CUDA support...

set "CUDA_INDEX=cpu"
set "CUDA_LABEL=CPU only"

:: Use PowerShell to reliably parse CUDA version from nvidia-smi
for /f "usebackq delims=" %%v in (`powershell -NoProfile -Command ^
    "try { $m = nvidia-smi 2^>$null ^| Select-String 'CUDA Version: (\d+)\.(\d+)'; " ^
    "if ($m) { $m.Matches.Groups[1].Value + '.' + $m.Matches.Groups[2].Value } " ^
    "else { 'none' } } catch { 'none' }"`) do set "CUDA_VER=%%v"

if "%CUDA_VER%"=="none" (
    echo [WARNING] No NVIDIA GPU detected. Installing CPU-only PyTorch.
    goto :install_torch
)

for /f "tokens=1,2 delims=." %%a in ("%CUDA_VER%") do (
    set /a "CUDA_MAJOR=%%a"
    set /a "CUDA_MINOR=%%b"
)

echo [INFO] Detected driver CUDA version: %CUDA_VER%

::   >= 12.8  ->  cu128  (sm_50 - sm_120: GTX 10xx through RTX 50xx)
::   >= 12.4  ->  cu124  (sm_50 - sm_90:  GTX 10xx through RTX 40xx)
::   >= 11.8  ->  cu118  (sm_37 - sm_90:  GTX 10xx through RTX 40xx)

if %CUDA_MAJOR% GEQ 13 (
    set "CUDA_INDEX=cu128"
    set "CUDA_LABEL=CUDA 12.8+ (all GPUs including RTX 50xx)"
    goto :install_torch
)
if %CUDA_MAJOR% EQU 12 (
    if %CUDA_MINOR% GEQ 8 (
        set "CUDA_INDEX=cu128"
        set "CUDA_LABEL=CUDA 12.8 (all GPUs including RTX 50xx)"
        goto :install_torch
    )
    if %CUDA_MINOR% GEQ 4 (
        set "CUDA_INDEX=cu124"
        set "CUDA_LABEL=CUDA 12.4 (RTX 20xx - 40xx)"
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
    "%PYTHON_EXE%" -m pip install torch torchvision torchaudio --no-warn-script-location
) else (
    "%PYTHON_EXE%" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/%CUDA_INDEX% --no-warn-script-location
)

:: ========================================================
:: 4. Install remaining packages
:: ========================================================
echo.
echo [INFO] Installing remaining Python packages...
"%PYTHON_EXE%" -m pip install -r requirements.txt --no-warn-script-location

:: ========================================================
:: 5. Set up models (GroundingDINO clone + weights)
:: ========================================================
echo.
echo [INFO] Setting up AI models...
"%PYTHON_EXE%" setup_models.py

:: ========================================================
:: Done
:: ========================================================
echo.
echo ==================================================
echo   INSTALLATION COMPLETE
echo ==================================================
echo   Python: %PYTHON_VER% (embedded)
echo   PyTorch: %CUDA_LABEL%
echo   To run Rotobot: double-click run_rotobot.bat
echo ==================================================
pause
