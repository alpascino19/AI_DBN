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

rem Firewall: fara aceasta regula, Windows blocheaza laptopurile cu camere (mai ales pe retele marcate "Public",
rem cum sunt hotspot-urile). Se aplica pe toate profilurile; regula veche cu acelasi nume e inlocuita.
netsh advfirewall firewall delete rule name="Vigil MQTT" >nul 2>&1
netsh advfirewall firewall add rule name="Vigil MQTT" dir=in action=allow protocol=TCP localport=1883,9001 profile=any >nul && (echo Firewall: portile 1883 si 9001 deschise pentru reteaua locala) || echo ATENTIE: nu am putut adauga regula de firewall
rem Ping: Windows nu raspunde implicit la ping pe retele "Public"; fara asta, "ping laptop1" da timeout chiar daca reteaua merge.
netsh advfirewall firewall delete rule name="Vigil ping" >nul 2>&1
netsh advfirewall firewall add rule name="Vigil ping" dir=in action=allow protocol=icmpv4:8,any profile=any >nul && echo Firewall: laptopul raspunde la ping (test de retea)
timeout /t 2 >nul
echo.
echo Porturi deschise de broker (trebuie sa apara 1883 si 9001):
netstat -ano | findstr "LISTENING" | findstr ":1883 :9001"
echo.
echo Configul vechi e salvat in "%DST%.original".
:end
echo.
pause
