@echo off
rem Vigil - PC-ul CENTRAL cand camerele sunt pe laptopuri diferite, legate la acelasi broker (docs\vision-doua-camere.md).
rem Aici ruleaza: vigil_vision.merge, ML1, ML2, dashboard-ul (si brokerul, daca e pe acest PC).
rem   4c_pornire_PC_central.bat                          camerele sunt pe alte 2 laptopuri
rem   4c_pornire_PC_central.bat --local-camera face      acest laptop are camera FETEI; corpul vine de pe celalalt
rem   4c_pornire_PC_central.bat --local-camera body      acest laptop are camera CORPULUI; fata vine de pe celalalt
rem   4c_pornire_PC_central.bat --broker 172.22.118.134  brokerul e pe alt calculator
rem Pe celelalte laptopuri: START\7_laptop_fata.bat IP / START\7_laptop_corp.bat IP (IP-ul brokerului).
rem Brokerul trebuie sa asculte in retea: o data START\0_broker_admin.bat pe calculatorul cu brokerul.
setlocal
chcp 65001 >nul
cd /d "%~dp0.."
echo IP-urile acestui PC (dati-l pe cel din reteaua hotspotului laptopurilor cu camere):
powershell -NoProfile -Command "Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -notlike '127.*' -and $_.IPAddress -notlike '169.254.*' } | ForEach-Object { '   ' + $_.IPAddress + '   (' + $_.InterfaceAlias + ')' }"
echo.
".venv\Scripts\python.exe" START\run_vigil.py --cameras remote %*
echo.
pause
