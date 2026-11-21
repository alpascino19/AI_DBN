@echo off
rem Vigil - pe LAPTOPUL CU CAMERA FETEI. Trimite vigil/vision/face catre brokerul de pe PC-ul central.
rem   START\7_laptop_fata.bat 172.22.118.20            (IP-ul PC-ului central)
rem   START\7_laptop_fata.bat 172.22.118.20 --index 1  (alta camera)
rem Fara IP foloseste DEFAULT_IP de mai jos. Prima data instaleaza singur pachetele Vision in .venv.
setlocal
chcp 65001 >nul
cd /d "%~dp0.."
rem IP-ul laptopului 1 (brokerul). Se schimba aici daca hotspotul ii da alt IP; un IP dat ca argument are prioritate.
set "DEFAULT_IP=172.22.118.20"
set "IP=%~1"
if "%IP%"=="" set "IP=%DEFAULT_IP%"
echo Trimit catre brokerul de pe laptopul 1: %IP%:1883
call :venv || goto :end
cd Vision
"%PY%" -m vigil_vision.run_face --show --calib 5 --mute --publish --mqtt-host %IP% %2 %3 %4 %5 %6
goto :end

:venv
set "PY=%CD%\.venv\Scripts\python.exe"
if exist "%PY%" exit /b 0
echo Prima pornire: creez .venv si instalez pachetele Vision (cateva minute)...
python -m venv .venv || (echo EROARE: lipseste Python 3.11+ & exit /b 1)
"%PY%" -m pip install -r Vision\requirements.txt || exit /b 1
exit /b 0

:end
echo.
pause
