# Gives the dashboard a public https address so teammates can open it on THEIR laptop
# and use THEIR OWN camera.
#
# Why this starts the dashboard itself instead of wrapping a running one:
#   Streamlit bakes an absolute address into the page so the browser knows where to fetch
#   its component files (the webrtc camera widget is one). By default that address is
#   localhost:8501 - which on a teammate's laptop points at their own machine, where nothing
#   is running. Result: "Failed to fetch dynamically imported module".
#   The fix is --browser.serverAddress=<public host>, and the public host does not exist
#   until the tunnel is up. So: tunnel first, read its address, then start Streamlit with it.

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

function Test-Port([int]$p) {
    $c = New-Object Net.Sockets.TcpClient
    try { $c.Connect('127.0.0.1', $p); $true } catch { $false } finally { $c.Dispose() }
}

# ---- 1. port must be free: this script starts Streamlit itself -------------------------
if (Test-Port 8501) {
    Write-Host ""
    Write-Host "[!] Port 8501 is already in use - the dashboard is probably already running."
    Write-Host "    Close that window first (Ctrl+C), then run this again."
    Write-Host "    This script has to start the dashboard itself, so that it can tell it"
    Write-Host "    the public address."
    Write-Host ""
    Read-Host "Press Enter to close"
    exit 1
}

# ---- 2. cloudflared --------------------------------------------------------------------
$cf = Join-Path $PSScriptRoot 'cloudflared.exe'
if (-not (Test-Path $cf)) {
    $onPath = Get-Command cloudflared -ErrorAction SilentlyContinue
    if ($onPath) { $cf = $onPath.Source } else {
        Write-Host ""
        Write-Host "[!] cloudflared.exe not found."
        Write-Host "    Run get_cloudflared.bat once - it downloads it for you."
        Write-Host ""
        Read-Host "Press Enter to close"
        exit 1
    }
}

# ---- 3. python with torch + streamlit ---------------------------------------------------
Write-Host "Looking for a python with torch + streamlit ..."
$py = & (Join-Path $PSScriptRoot 'pyfind.ps1')
if (-not $py) {
    Write-Host ""
    Write-Host "[!] No python with torch + streamlit was found."
    Write-Host "    Open Miniforge/Anaconda Prompt, activate your env, then:"
    Write-Host "        pip install -r requirements.txt"
    Write-Host ""
    Read-Host "Press Enter to close"
    exit 1
}
$pyDir = Split-Path $py -Parent
$env:PATH = "$pyDir;$pyDir\Scripts;$pyDir\Library\bin;$env:PATH"
Write-Host "Using: $py"

# ---- 4. tunnel first, then read the address it hands out --------------------------------
$outLog = Join-Path $env:TEMP 'cf_tunnel.out.log'
$errLog = Join-Path $env:TEMP 'cf_tunnel.err.log'
Remove-Item $outLog, $errLog -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "Opening a public https address ..."
$proc = Start-Process -FilePath $cf `
    -ArgumentList 'tunnel', '--url', 'http://localhost:8501' `
    -RedirectStandardOutput $outLog -RedirectStandardError $errLog `
    -NoNewWindow -PassThru

$publicUrl = $null
try {
    foreach ($i in 1..60) {
        Start-Sleep -Seconds 1
        if ($proc.HasExited) { break }
        $text = (Get-Content $outLog, $errLog -Raw -ErrorAction SilentlyContinue) -join "`n"
        $m = [regex]::Match($text, 'https://[a-z0-9-]+\.trycloudflare\.com')
        if ($m.Success) { $publicUrl = $m.Value; break }
    }

    if (-not $publicUrl) {
        Write-Host ""
        Write-Host "[!] The tunnel did not give an address within 60 seconds."
        Write-Host "    This network may be blocking it. Log:"
        Get-Content $outLog, $errLog -ErrorAction SilentlyContinue | Select-Object -Last 15
        Write-Host ""
        Write-Host "    Fall back to running the dashboard locally: app\run_demo.bat"
        Read-Host "Press Enter to close"
        exit 1
    }

    $publicHost = ([Uri]$publicUrl).Host      # NOTE: not $host - that name is taken by PowerShell

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "  Send your teammates this link:"
    Write-Host ""
    Write-Host "      $publicUrl"
    Write-Host ""
    Write-Host "  They open it, allow the camera, and their own camera is used."
    Write-Host "  Nothing to install on their side."
    Write-Host ""
    Write-Host "  Keep THIS window open. Closing it kills the link."
    Write-Host "  The link is new every time - share it after starting, not before."
    Write-Host "============================================================"
    Write-Host ""

    # ---- 5. start Streamlit, telling it the public address --------------------------
    & $py -m streamlit run app\demo.py `
        --server.headless=true `
        --browser.serverAddress=$publicHost `
        --browser.serverPort=443 `
        -- --checkpoint checkpoints\model
}
finally {
    if ($proc -and -not $proc.HasExited) {
        Write-Host ""
        Write-Host "Closing the tunnel ..."
        $proc.Kill()
    }
}

Write-Host ""
Write-Host "Stopped. The link is dead now; run this again to get a new one."
Read-Host "Press Enter to close"
