@echo off
rem Vigil - instalare completa (o singura data). Dublu-clic sau rulat din terminal.
rem Instaleaza: Mosquitto, Ollama, ffmpeg, Node.js (winget), mediul Python .venv cu Vision + ML1 + ML2,
rem modelul qwen2.5:1.5b, vocea offline Piper si pachetele dashboard-ului.
setlocal
chcp 65001 >nul
cd /d "%~dp0.."

echo.
echo === 1/6 Programe (winget): Mosquitto, Ollama, ffmpeg, Node.js
for %%P in (EclipseFoundation.Mosquitto Ollama.Ollama Gyan.FFmpeg OpenJS.NodeJS.LTS) do (
    winget list -e --id %%P >nul 2>&1 && (echo     %%P: deja instalat) || winget install -e --id %%P --accept-source-agreements --accept-package-agreements --silent
)

rem PATH-ul nou (ollama, npm, ffmpeg) fara repornirea terminalului
for /f "usebackq delims=" %%p in (`powershell -NoProfile -Command "[Environment]::GetEnvironmentVariable('Path','Machine') + ';' + [Environment]::GetEnvironmentVariable('Path','User')"`) do set "PATH=%%p;%PATH%"

echo.
echo === 2/6 Mediul Python (.venv) cu Vision + ML1 + ML2
if not exist ".venv\Scripts\python.exe" (
    python -m venv .venv || (echo EROARE: lipseste Python 3.11+ ^(winget install -e --id Python.Python.3.14^) & goto :fail)
)
".venv\Scripts\python.exe" -m pip install --upgrade pip -q
".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :fail

echo.
echo === 3/6 Modelul LLM local (ML1 + conversatia ML2): qwen2.5:1.5b, ~1 GB
ollama list >nul 2>&1 || (start "" "%LOCALAPPDATA%\Programs\Ollama\ollama app.exe" & timeout /t 8 >nul)
ollama pull qwen2.5:1.5b || echo ATENTIE: modelul nu s-a descarcat; ML1 merge si doar cu reguli

echo.
echo === 4/6 Vocea offline (Piper, rezerva cand nu e internet)
if not exist "ML2\voices\ro_RO-mihai-medium.onnx" (
    ".venv\Scripts\python.exe" -m piper.download_voices ro_RO-mihai-medium --download-dir ML2\voices
) else (echo     deja descarcata)

echo.
echo === 5/6 Dashboard-ul React (npm install)
pushd dashboard
call npm install --no-fund --no-audit || (popd & goto :fail)
popd

echo.
echo === 6/6 Verificare
".venv\Scripts\python.exe" START\run_vigil.py --check

echo.
echo Gata. Pentru dashboard (port 9001) si ESP32 (broker in retea) mai ruleaza o data,
echo ca administrator: START\0_broker_admin.bat
echo Optional, conversatie mai buna in romana:  ollama pull gemma3:4b
goto :end

:fail
echo.
echo Instalarea s-a oprit cu o eroare (vezi mai sus).
:end
echo.
pause
