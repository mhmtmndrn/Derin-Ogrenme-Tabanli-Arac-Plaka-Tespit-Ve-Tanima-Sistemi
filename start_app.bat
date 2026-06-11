@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" app.py %*
    goto :done
)

set "CODEX_PYTHON=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if exist "%CODEX_PYTHON%" (
    "%CODEX_PYTHON%" app.py %*
    goto :done
)

where python >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    python --version >nul 2>nul
    if %ERRORLEVEL% EQU 0 (
        python app.py %*
        goto :done
    )
)

where py >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    py -3 --version >nul 2>nul
    if %ERRORLEVEL% EQU 0 (
        py -3 app.py %*
        goto :done
    )
)

echo Python bulunamadi.
echo Once Python kurup su komutlari calistirin:
echo python -m venv .venv
echo .\.venv\Scripts\activate
echo pip install -r requirements.txt
pause

:done
endlocal
