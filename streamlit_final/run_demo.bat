@echo off
cd /d "%~dp0.."
set PY=.venv\Scripts\python.exe
if not exist "%PY%" set PY=python

echo Starting the demo (checkpoints\model).
echo If the browser does not open by itself, go to http://localhost:8501
echo Press Ctrl+C in this window to stop it.
echo.
"%PY%" -m streamlit run app\demo.py -- --checkpoint checkpoints\model
pause
