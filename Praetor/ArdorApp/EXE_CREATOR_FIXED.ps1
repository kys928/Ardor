param(
    [string]$RootDir
)
$ErrorActionPreference = 'Stop'

# --- Resolve project root ---
if ($RootDir) {
    $ROOT = (Resolve-Path $RootDir).Path
} else {
    # Try to auto-detect by walking up to find GUI_Cortex.py
    $ROOT = $null
    $cur = $PSScriptRoot
    for ($i=0; $i -lt 5 -and $null -eq $ROOT; $i++) {
        if (Test-Path (Join-Path $cur 'GUI_Cortex.py')) { $ROOT = $cur; break }
        $cur = Split-Path $cur -Parent
        if (-not $cur) { break }
    }
    if (-not $ROOT) {
        throw "Could not find GUI_Cortex.py. Re-run with: -RootDir 'C:\Path\To\Your\Project'"
    }
}

# --- Paths ---
$DESKTOP = [Environment]::GetFolderPath('Desktop')
$DIST    = Join-Path $DESKTOP 'ArdorApp'          # output root (folder mode)
$BUILD   = Join-Path $ROOT 'build'
$SPEC    = Join-Path $ROOT 'build\spec'
$ICON    = Join-Path $ROOT 'Assets\ardor.ico'     # run make_icon.py first to create this
$ENTRY   = Join-Path $ROOT 'GUI_Cortex.py'

Write-Host "ROOT     = $ROOT"
Write-Host "ENTRY    = $ENTRY"
Write-Host "ICON     = $ICON"
Write-Host "DIST     = $DIST"

# --- Clean old build/dist ---
if (Test-Path $BUILD) { Remove-Item -Recurse -Force $BUILD }
if (Test-Path $DIST)  { Remove-Item -Recurse -Force $DIST  }

# --- Warn if icon missing (not fatal) ---
if (-not (Test-Path $ICON)) {
    Write-Host "[!] Icon not found: $ICON — run 'python make_icon.py' first." -ForegroundColor Yellow
}

# --- Build args (no fragile backticks) ---
$pyArgs = @(
    '-m','PyInstaller',
    '--noconfirm',
    '--windowed',
    '--name','ArdorGUI',
    '--distpath', $DIST,
    '--workpath', $BUILD,
    '--specpath', $SPEC,
    '--icon', $ICON,
    '--collect-all','tokenizers',
    '--collect-all','tqdm',
    '--collect-submodules','torch',
    '--add-data', (Join-Path $ROOT 'Cerebrum') + ';Cerebrum',
    '--add-data', (Join-Path $ROOT 'Cortex') + ';Cortex',
    '--add-data', (Join-Path $ROOT 'ProjectTokenizer\ardor_tokenizer') + ';ProjectTokenizer\ardor_tokenizer',
    '--add-data', (Join-Path $ROOT 'Models') + ';Models',
    '--add-data', (Join-Path $ROOT 'REM.py') + ';.',
    '--add-data', (Join-Path $ROOT 'neural_plasticity_training.py') + ';.',
    $ENTRY
)

# --- Build ---
Write-Host "`nRunning: python $($pyArgs -join ' ')" -ForegroundColor Cyan
python @pyArgs

# --- Marker to help the app locate its root, if needed ---
$AppDir = Join-Path $DIST 'ArdorGUI'
New-Item -ItemType Directory -Path $AppDir -Force | Out-Null
Set-Content -Path (Join-Path $AppDir 'ardor_home.txt') -Value $ROOT -Encoding UTF8

Write-Host ""
Write-Host ("Build complete: " + (Join-Path $DIST 'ArdorGUI')) -ForegroundColor Green
