# EXE_CREATOR.ps1 — Build Ardor app from ProjectArdor1.2 to Desktop
$ErrorActionPreference = 'Stop'

# Paths
$ROOT  = 'C:\Users\adm\PycharmProjects\ProjectArdor1.2'
$DIST  = 'C:\Users\adm\Desktop\ArdorApp'      # where the app folder will appear
$BUILD = "$ROOT\build"
$SPEC  = "$ROOT\build\spec"
$ICON  = "$ROOT\Assets\ardor.ico"              # produced by make_icon.py
$ENTRY = "$ROOT\GUI_Cortex.py"                  # entry point GUI

# Clean old build/dist
if (Test-Path $BUILD) { Remove-Item -Recurse -Force $BUILD }
if (Test-Path $DIST)  { Remove-Item -Recurse -Force $DIST  }

# Ensure icon exists
if (-not (Test-Path $ICON)) {
    Write-Host "[!] Icon not found: $ICON — run 'python make_icon.py' first." -ForegroundColor Yellow
}

# Build
python -m PyInstaller --noconfirm --windowed `
  --name ArdorGUI `
  --distpath "$DIST" `
  --workpath "$BUILD" `
  --specpath "$SPEC" `
  --icon "$ICON" `
  --collect-all tokenizers `
  --collect-all tqdm `
  --collect-submodules torch `
  --add-data "$ROOT\Cerebrum;Cerebrum" `
  --add-data "$ROOT\Cortex;Cortex" `
  --add-data "$ROOT\ProjectTokenizer\ardor_tokenizer;ProjectTokenizer\ardor_tokenizer" `
  --add-data "$ROOT\Models;Models" `
  --add-data "$ROOT\REM.py;." `
  --add-data "$ROOT\neural_plasticity_training.py;." `
  "$ENTRY"

# Write ardor_home.txt so the GUI can locate its root if needed
New-Item -ItemType File -Path (Join-Path $DIST 'ArdorGUI' 'ardor_home.txt') -Force -Value $PSScriptRoot | Out-Null
Write-Host "\nBuild complete: $DIST\ArdorGUI" -ForegroundColor Green