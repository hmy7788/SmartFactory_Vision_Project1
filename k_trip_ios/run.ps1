param([int]$Port = 8765)
$ErrorActionPreference = 'Stop'
$env:PYTHONPATH = "$PSScriptRoot\.packages;$PSScriptRoot"
Set-Location $PSScriptRoot
python -m uvicorn server:app --host 127.0.0.1 --port $Port --ws-max-size 6291456
