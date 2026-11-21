@echo off
rem Vigil - tot sistemul CU camera reala (Vision -> ML1 -> ML2 -> dashboard).
rem Optiuni: --show (fereastra de debug a camerei), --camera 1 (alta camera), --no-llm, --voice-llm gemma3:4b
setlocal
chcp 65001 >nul
cd /d "%~dp0.."
".venv\Scripts\python.exe" START\run_vigil.py %*
echo.
pause
