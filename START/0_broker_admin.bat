@echo off
rem Vigil - configurarea brokerului Mosquitto (o singura data, cere drepturi de administrator).
rem Copiaza ML2\mosquitto.conf peste configul serviciului si il reporneste:
rem   1883 pe toata reteaua (pentru ESP32)  +  9001 WebSocket (pentru dashboard-ul React).
rem Atentie: brokerul accepta conexiuni fara parola din reteaua locala. Folositi o retea de incredere
rem (hotspotul propriu la demo), nu Wi-Fi public.
setlocal
chcp 65001 >nul

net session >nul 2>&1
if errorlevel 1 (
    echo Cer drepturi de administrator...
    powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
    exit /b
)

set "SRC=%~dp0..\ML2\mosquitto.conf"
set "DST=%ProgramFiles%\mosquitto\mosquitto.conf"
if not exist "%DST%" (
    echo EROARE: nu gasesc %DST%. Instaleaza intai Mosquitto ^(START\1_instalare.bat^).
    goto :end
)
if not exist "%DST%.original" copy /y "%DST%" "%DST%.original" >nul
copy /y "%SRC%" "%DST%" >nul || (echo EROARE la copiere & goto :end)
powershell -NoProfile -Command "Restart-Service mosquitto"
timeout /t 2 >nul
echo.
echo Porturi deschise de broker (trebuie sa apara 1883 si 9001):
netstat -ano | findstr "LISTENING" | findstr ":1883 :9001"
echo.
echo Configul vechi e salvat in "%DST%.original".
:end
echo.
pause
