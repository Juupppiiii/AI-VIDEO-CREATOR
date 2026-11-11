@echo off
title AI Video Creator
cd /d "%~dp0"
color 0F

echo.
echo    ╔══════════════════════════════════════╗
echo    ║       AI Video Creator  v2.0         ║
echo    ╚══════════════════════════════════════╝
echo.

:: ── Check Ollama ──
tasklist /FI "IMAGENAME eq ollama.exe" 2>NUL | find /I "ollama.exe" >NUL
if %ERRORLEVEL% NEQ 0 (
    echo    [*] Starting Ollama...
    start /MIN "" ollama serve
    timeout /t 3 /nobreak > NUL
    echo    [OK] Ollama started.
) else (
    echo    [OK] Ollama is already running.
)

echo.
echo    [*] Launching web UI...
echo    [*] Your browser will open automatically.
echo    [*] Press Ctrl+C here to stop the app.
echo.
echo    ──────────────────────────────────────
echo.

python server/app/main.py

echo.
echo    App closed. Press any key to exit.
pause > NUL
