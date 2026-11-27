@echo off
setlocal
title FX Screener 2.0 — Early Inflection ^& ASAP Exit Terminal
cd /d "%~dp0.."

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

"%PY_EXE%" -m fx_screener2 %*
if %ERRORLEVEL% neq 0 (
    echo.
    echo [FX Screener 2.0 exited with code %ERRORLEVEL%]
    pause
)
