@echo off
rem Starts the dashboard AND a public https address in one go, so teammates can open it
rem on their own laptop and use their own camera.
rem
rem Do NOT run app\run_demo.bat first - this starts the dashboard itself. It has to,
rem because Streamlit must be told the public address at startup: otherwise the page tells
rem the browser to fetch its camera widget from localhost:8501, which on a teammate's
rem laptop is their own machine ("Failed to fetch dynamically imported module").
rem
rem First time only: run get_cloudflared.bat once.
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\share_https.ps1"
