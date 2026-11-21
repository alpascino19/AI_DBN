@echo off
rem Vigil FATA (CV1): camera doar pentru fata. Argumente extra se adauga la comanda, ex: start_face.bat --index 0
cd /d "%~dp0"
set PYTHONUTF8=1
python -m vigil_vision.run_face --show --calib 5 %*
pause
