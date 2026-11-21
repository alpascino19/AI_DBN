@echo off
rem Vigil - tot sistemul cu DOUA camere pe ACEST PC: fata (index 0) + corp (index 1).
rem Vision = run_face + run_body + vigil_vision.merge -> vigil/vision -> ML1 -> ML2 -> dashboard.
rem Optiuni: --face-index 0 --body-index 2 (alte camere), --show (ferestrele camerelor), --no-llm, --voice-llm gemma3:4b
setlocal
chcp 65001 >nul
cd /d "%~dp0.."
".venv\Scripts\python.exe" START\run_vigil.py --cameras two %*
echo.
pause
