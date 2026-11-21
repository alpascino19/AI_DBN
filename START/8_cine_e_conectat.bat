@echo off
rem Vigil - pe laptopul 1 (cel cu brokerul): ce calculatoare sunt conectate ACUM la broker (laptopurile cu camere, ESP32).
rem Se poate rula oricand, si in timp ce sistemul merge. In fereastra lui 4c apar oricum liniile [NET] la fiecare conectare.
setlocal
chcp 65001 >nul
cd /d "%~dp0.."
echo IP-ul acestui laptop (il dati laptopurilor cu camere):
powershell -NoProfile -Command "Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -notlike '127.*' -and $_.IPAddress -notlike '169.254.*' } | ForEach-Object { '   ' + $_.IPAddress + '   (' + $_.InterfaceAlias + ')' }"
echo.
".venv\Scripts\python.exe" START\run_vigil.py --clients
echo.
pause
