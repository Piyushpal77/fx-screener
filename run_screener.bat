@echo off
setlocal
title FX ^& Precious Metals Multi-Factor Terminal Screener
cd /d "%~dp0"

set "PY_EXE=C:\Users\PIYUSH\AppData\Local\Python\bin\python.exe"

if not exist "%PY_EXE%" (
    where python >nul 2>nul
    if %ERRORLEVEL% equ 0 (
        set "PY_EXE=python"
    ) else (
        echo [ERROR] Python executable not found.
        pause
        exit /b 1
    )
)

"%PY_EXE%" -m fx_screener %*
if %ERRORLEVEL% neq 0 (
    echo.
    echo [Screener exited with code %ERRORLEVEL%]
    pause
)
