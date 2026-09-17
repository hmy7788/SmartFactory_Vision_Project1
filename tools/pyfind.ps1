# Finds a python.exe that has torch + torchvision + streamlit + opencv + pyyaml, and prints its path.
# Order: .venv in the repo -> python on PATH (an activated conda env wins) -> miniforge3 / anaconda3 / miniconda3 and their envs.
# Used by app\run_demo.bat and check_model.bat so the demo can be double-clicked without opening a conda prompt.
$ErrorActionPreference = "SilentlyContinue"
$cands = New-Object System.Collections.Generic.List[string]
if (Test-Path ".venv\Scripts\python.exe") { $cands.Add((Resolve-Path ".venv\Scripts\python.exe").Path) }
$onPath = Get-Command python -ErrorAction SilentlyContinue
if ($onPath -and $onPath.Source -notlike "*WindowsApps*") { $cands.Add($onPath.Source) }   # skip the Microsoft Store stub
$roots = @("$env:USERPROFILE\miniforge3", "$env:USERPROFILE\anaconda3", "$env:USERPROFILE\miniconda3",
           "$env:LOCALAPPDATA\miniforge3", "$env:LOCALAPPDATA\anaconda3", "$env:LOCALAPPDATA\miniconda3",
           "$env:ProgramData\miniforge3", "$env:ProgramData\anaconda3", "$env:ProgramData\miniconda3",
           "$env:USERPROFILE\.conda")
foreach ($root in $roots) {
    if (Test-Path "$root\envs") {
        Get-ChildItem "$root\envs" -Directory | ForEach-Object {
            if (Test-Path "$($_.FullName)\python.exe") { $cands.Add("$($_.FullName)\python.exe") }
        }
    }
    if (Test-Path "$root\python.exe") { $cands.Add("$root\python.exe") }   # base env last: usually has no torch
}
$origPath = $env:PATH
foreach ($p in ($cands | Select-Object -Unique)) {
    $dir = Split-Path $p -Parent
    $env:PATH = "$dir;$dir\Scripts;$dir\Library\bin;$origPath"   # conda DLLs (MKL etc.) live in Library\bin
    [Console]::Error.WriteLine("  checking $p")
    & $p -c "import torch, torchvision, streamlit, cv2, yaml" 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) { Write-Output $p; exit 0 }
}
exit 1
