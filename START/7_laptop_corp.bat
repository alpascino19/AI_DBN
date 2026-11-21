@echo off
rem Vigil - pe LAPTOPUL CU CAMERA CORPULUI. Trimite vigil/vision/body catre brokerul de pe PC-ul central.
rem   START\7_laptop_corp.bat 172.22.118.134            (IP-ul PC-ului central)
rem   START\7_laptop_corp.bat 172.22.118.134 --index 1  (alta camera; implicit 0 = camera laptopului)
rem Fara IP, il cere. Prima data instaleaza singur pachetele Vision in .venv.
setlocal
chcp 65001 >nul
cd /d "%~dp0.."
set "IP=%~172.22.118.20"
if "%IP%"=="" set /p "IP=172.22.118.20"
if "%IP%"=="" (echo Lipseste IP-ul. & goto :end)
call :venv || goto :end
cd Vision
"%PY%" -m vigil_vision.run_body --show --mute --publish --mqtt-host %IP% --index 0 %2 %3 %4 %5 %6
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
