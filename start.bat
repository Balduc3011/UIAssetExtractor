@echo off
setlocal
cd /d "%~dp0"
title UI Asset Extractor

set "RT=%~dp0.runtime"
set "PY=%RT%\venv\Scripts\python.exe"

if not exist "%RT%\installed.ok" (
    echo Lan chay dau tien - dang cai dat moi truong...
    call "%~dp0setup.bat" nopause
    if errorlevel 1 exit /b 1
)
if not exist "%PY%" (
    call "%~dp0setup.bat" nopause
    if errorlevel 1 exit /b 1
)

echo Dang khoi dong UI Asset Extractor...
"%PY%" -m app.main %*
if errorlevel 1 (
    echo.
    echo Tool bi dung do loi. Xem thong bao o tren.
    echo Thu chay lai setup.bat neu loi lien quan thu vien.
    pause
)
