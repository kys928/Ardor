$ErrorActionPreference = "Stop"
python -m pip install --upgrade pip pyinstaller
pyinstaller ArdorGUI.spec --clean
Write-Host "Built -> dist\ArdorGUI\ArdorGUI.exe"
