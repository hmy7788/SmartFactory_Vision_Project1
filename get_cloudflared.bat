@echo off
rem Downloads cloudflared.exe (Cloudflare's tunnel client) into tools\, once.
rem share_https.bat needs it to give the dashboard a public https address.
rem
rem Pinned to one release and checked against its published SHA256, so what lands
rem on disk is exactly the file that was checked - not whatever is newest today.
setlocal
cd /d "%~dp0"

set VER=2026.9.1
set NAME=cloudflared-windows-amd64.exe
set SHA=2837888cc0f5d58f15b6dc478376de90b4d3ba5241c7947455d1e0a0df429712
set URL=https://github.com/cloudflare/cloudflared/releases/download/%VER%/%NAME%
set DEST=%~dp0tools\cloudflared.exe

if exist "%DEST%" (
  echo Already downloaded: %DEST%
  "%DEST%" --version
  echo.
  echo Nothing to do. Run share_https.bat next.
  pause
  exit /b 0
)

echo This will download one file:
echo.
echo   from : %URL%
echo   to   : %DEST%
echo   size : about 30 MB
echo.
echo It is Cloudflare's official tunnel client, from Cloudflare's GitHub releases.
echo The download is checked against the SHA256 published on that release page.
echo.
set /p GO="Download it now? (y/N): "
if /i not "%GO%"=="y" (
  echo Cancelled. Nothing was downloaded.
  pause
  exit /b 1
)

if not exist "%~dp0tools" mkdir "%~dp0tools"
echo.
echo Downloading ...
powershell -NoProfile -Command "$ProgressPreference='SilentlyContinue'; try{Invoke-WebRequest -Uri '%URL%' -OutFile '%DEST%' -UseBasicParsing}catch{Write-Host $_.Exception.Message; exit 1}"
if errorlevel 1 goto failed
if not exist "%DEST%" goto failed

echo Checking the file ...
powershell -NoProfile -Command "$h=(Get-FileHash -Algorithm SHA256 '%DEST%').Hash.ToLower(); if($h -ne '%SHA%'){Write-Host ('  got      ' + $h); Write-Host ('  expected %SHA%'); exit 1}"
if errorlevel 1 (
  echo.
  echo [!] The downloaded file does not match the published checksum. Deleting it.
  echo     Do not use it. Try again later, or download by hand from:
  echo     https://github.com/cloudflare/cloudflared/releases
  del "%DEST%" >nul 2>nul
  pause
  exit /b 1
)

echo   checksum OK
"%DEST%" --version
echo.
echo Done. Next: run app\run_demo.bat, then share_https.bat.
pause
exit /b 0

:failed
echo.
echo [!] Download failed ^(no internet, proxy, or the release moved^).
echo     You can download it by hand instead:
echo       1. open  https://github.com/cloudflare/cloudflared/releases
echo       2. get   %NAME%
echo       3. rename it to cloudflared.exe and put it in  %~dp0tools\
del "%DEST%" >nul 2>nul
pause
exit /b 1
