@echo off
rem Vigil CORP (CV2): camera doar pentru corp. Argumente extra se adauga la comanda, ex: start_body.bat --index 1
cd /d "%~dp0"
set PYTHONUTF8=1
python -m vigil_vision.run_body --show %*
pause
