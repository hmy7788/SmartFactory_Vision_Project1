@echo off
rem Installs streamlit-webrtc so the LIVE tab can use the BROWSER's camera
rem (so a teammate opening the page uses their own camera, not this computer's).
rem Without it the app still works - it falls back to this computer's camera.
cd /d "%~dp0"
echo Looking for a python with torch + streamlit ...
set PY=
for /f "usebackq delims=" %%P in (`powershell -NoProfile -ExecutionPolicy Bypass -File tools\pyfind.ps1`) do set PY=%%P
if "%PY%"=="" (
  echo.
  echo [!] No python with torch + streamlit was found. Run app\run_demo.bat first to see the same message.
  pause
  exit /b 1
)
echo Using: %PY%
echo.
"%PY%" -m pip install streamlit-webrtc
echo.
"%PY%" -c "import streamlit_webrtc, av; print('OK - streamlit-webrtc', streamlit_webrtc.__version__)" || echo [!] Import failed - the LIVE tab will use this computer's camera instead.
echo.
echo Restart the dashboard (app\run_demo.bat) for this to take effect.
pause
