@echo off
rem Vigil - tot sistemul FARA camera: Vision trimite scenariul demo al echipei CV.
rem Optiuni (se pot adauga la final): --scenario chest ^| drowsy ^| phone ^| unconscious, --no-llm, --no-ml2
setlocal
chcp 65001 >nul
cd /d "%~dp0.."
".venv\Scripts\python.exe" START\run_vigil.py --fake %*
echo.
pause
