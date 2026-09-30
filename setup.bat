@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"
title UI Asset Extractor - Setup

rem ================================================================
rem  Cai dat toan bo moi truong vao thu muc .runtime (khong dung
rem  Python co san tren may). Chay lai file nay bat cu luc nao de
rem  sua loi / cap nhat. Tham so:  setup.bat cpu   -> ep dung CPU
rem ================================================================

set "ROOT=%~dp0"
set "RT=%ROOT%.runtime"
set "UV_DIR=%RT%\uv"
set "UV=%UV_DIR%\uv.exe"
set "UV_PYTHON_INSTALL_DIR=%RT%\python"
set "UV_CACHE_DIR=%RT%\cache"
set "UV_PYTHON_PREFERENCE=only-managed"
set "UV_LINK_MODE=copy"
set "VENV=%RT%\venv"
set "PY=%VENV%\Scripts\python.exe"
set "FORCE_CPU=0"
if /i "%~1"=="cpu" set "FORCE_CPU=1"

echo.
echo  ==== UI Asset Extractor - Cai dat ====
echo  Thu muc: %ROOT%
echo.

rem ---------- 1. uv (trinh quan ly Python) ----------
echo [1/5] Kiem tra uv...
if not exist "%UV%" (
    if not exist "%UV_DIR%" mkdir "%UV_DIR%"
    echo       Dang tai uv...
    powershell -NoProfile -ExecutionPolicy Bypass -Command ^
      "$env:UV_INSTALL_DIR='%UV_DIR%'; $env:UV_NO_MODIFY_PATH='1'; $env:INSTALLER_NO_MODIFY_PATH='1'; irm https://astral.sh/uv/install.ps1 | iex"
)
if not exist "%UV%" if exist "%UV_DIR%\bin\uv.exe" set "UV=%UV_DIR%\bin\uv.exe"
if not exist "%UV%" (
    echo.
    echo  [LOI] Khong tai duoc uv. Kiem tra ket noi mang roi chay lai setup.bat
    goto :fail
)
echo       OK

rem ---------- 2. Python 3.11 + moi truong ao ----------
echo [2/5] Chuan bi Python 3.11...
if not exist "%PY%" (
    "%UV%" venv "%VENV%" --python 3.11 --seed
    if errorlevel 1 goto :fail
)
echo       OK

rem ---------- 3. Thu vien ----------
echo [3/5] Cai thu vien (lan dau co the mat vai phut)...
"%UV%" pip install --python "%PY%" -r "%ROOT%requirements.txt"
if errorlevel 1 goto :fail

rem ---------- 4. ONNX Runtime: GPU (NVIDIA) hoac CPU ----------
echo [4/5] Chon ban ONNX Runtime...
set "ORT=onnxruntime"
set "DRV="
if "%FORCE_CPU%"=="0" (
    where nvidia-smi >nul 2>&1
    if not errorlevel 1 (
        for /f "tokens=1 delims=." %%a in ('nvidia-smi --query-gpu^=driver_version --format^=csv^,noheader 2^>nul') do (
            if not defined DRV set "DRV=%%a"
        )
    )
)
if defined DRV (
    echo       Phat hien GPU NVIDIA, driver %DRV%
    if %DRV% GEQ 580 (
        set "ORT=onnxruntime-gpu[cuda,cudnn]>=1.27"
    ) else if %DRV% GEQ 528 (
        set "ORT=onnxruntime-gpu[cuda,cudnn]==1.26.0"
    ) else (
        echo       Driver qua cu cho CUDA 12 - dung CPU. Cap nhat driver NVIDIA de dung GPU.
    )
) else (
    echo       Khong dung GPU - cai ban CPU.
)
"%UV%" pip uninstall --python "%PY%" onnxruntime onnxruntime-gpu >nul 2>&1
echo       Cai: !ORT!
"%UV%" pip install --python "%PY%" "!ORT!"
if errorlevel 1 (
    echo       Cai ban GPU loi - chuyen sang ban CPU.
    "%UV%" pip install --python "%PY%" onnxruntime
    if errorlevel 1 goto :fail
)

rem ---------- 5. Kiem tra + tai model OCR ----------
echo [5/5] Kiem tra va tai model...
"%PY%" -m app.warmup
if errorlevel 1 goto :fail

echo ok> "%RT%\installed.ok"
echo.
echo  ==== CAI DAT XONG ====
echo  Chay start.bat de mo tool.
echo.
if /i not "%~2"=="nopause" if /i not "%~1"=="nopause" pause
exit /b 0

:fail
echo.
echo  ==== CAI DAT THAT BAI ====
echo  Xem thong bao loi o tren. Co the chay lai setup.bat.
echo  Neu loi lien quan GPU, thu:  setup.bat cpu
echo.
pause
exit /b 1
