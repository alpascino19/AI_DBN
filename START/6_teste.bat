@echo off
rem Vigil - toate testele automate: ML1, Vision, ML2, dashboard. Nu au nevoie de camera, Ollama sau broker.
setlocal
chcp 65001 >nul
cd /d "%~dp0.."
set "PY=%CD%\.venv\Scripts\python.exe"
set PYTHONUTF8=1
set FAILED=

echo === ML1
pushd ML1
"%PY%" -m unittest || set FAILED=%FAILED% ML1
popd

echo.
echo === Vision
pushd Vision
"%PY%" -m unittest discover -s tests || set FAILED=%FAILED% Vision
popd

echo.
echo === ML2
pushd ML2
"%PY%" test_executor.py || set FAILED=%FAILED% ML2-executor
"%PY%" test_voice.py || set FAILED=%FAILED% ML2-voce
popd

echo.
echo === Dashboard
pushd dashboard
call npm test || set FAILED=%FAILED% dashboard
popd

echo.
if defined FAILED (echo AU ESUAT:%FAILED%) else (echo TOATE TESTELE AU TRECUT)
echo.
pause
