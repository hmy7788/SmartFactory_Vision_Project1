@echo off
rem Checks that the model in checkpoints\model is wired correctly:
rem   accuracy on a labeled photo folder (<folder>\<class>\*.jpg) + same answers as src\models\convnext\inference.py.
rem Usage: double-click (finds a labeled folder next to this repo, e.g. ..\test photos), or drag a folder onto this file.
cd /d "%~dp0"
echo Looking for a python with torch + streamlit ...
set PY=
for /f "usebackq delims=" %%P in (`powershell -NoProfile -ExecutionPolicy Bypass -File tools\pyfind.ps1`) do set PY=%%P
if "%PY%"=="" (
  echo.
  echo [!] No python with torch + streamlit was found.
  echo     Open "Miniforge Prompt" ^(or Anaconda Prompt^), activate your env, then:
  echo         pip install -r requirements.txt
  echo     and run this file again ^(or run it from that prompt^).
  pause
  exit /b 1
)
for %%I in ("%PY%") do set PYDIR=%%~dpI
set PATH=%PYDIR%;%PYDIR%Scripts;%PYDIR%Library\bin;%PATH%
echo Using: %PY%
echo.
if "%~1"=="" (
  "%PY%" tools\check_model.py --compare
) else (
  "%PY%" tools\check_model.py --compare --folder "%~1"
)
echo.
pause
