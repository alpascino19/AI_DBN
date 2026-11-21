@echo off
rem Vigil - verifica tot ce trebuie pentru pornire (broker, Ollama, modele, pachete, dashboard).
setlocal
chcp 65001 >nul
cd /d "%~dp0.."
".venv\Scripts\python.exe" START\run_vigil.py --check %*
echo.
pause
