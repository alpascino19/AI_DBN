@echo off
rem Vigil - ML1 primeste cadrele prin HTTP de la serverul Spring Boot (ML1\docs\API.md), port 8090.
rem Deciziile merg la ML2 prin broker, ca de obicei. Vision nu porneste (camera e la serverul Spring).
setlocal
chcp 65001 >nul
cd /d "%~dp0.."
".venv\Scripts\python.exe" START\run_vigil.py --input http %*
echo.
pause
