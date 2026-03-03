# EXE_CREATOR_FIXED_PATHS_v3.ps1 — Build Ardor GUI (Praetor layout, fixed icon)
# - No 'param' block (so it runs anywhere)
# - Robust Python selection (avoids WindowsApps stub)
# - ENTRY -> run_ardor.py in Praetor\ArdorApp

$ErrorActionPreference = 'Stop'

# --- Fixed paths ---
$APP_ROOT     = 'C:\Users\adm\PycharmProjects\ProjectArdor\Praetor'
$PROJECT_ROOT = 'C:\Users\adm\PycharmProjects\ProjectArdor'
$ICON         = 'C:\Users\adm\PycharmProjects\ProjectArdor\Praetor\ArdorApp\assets\ardor.ico'
$ENTRY        = 'C:\Users\adm\PycharmProjects\ProjectArdor\Praetor\ArdorApp\run_ardor.py'

# --- Derived paths ---
$DESKTOP = [Environment]::GetFolderPath('Desktop')
$DIST    = Join-Path $DESKTOP 'ArdorApp'
$BUILD   = Join-Path $APP_ROOT 'ArdorApp\build'
$SPEC    = Join-Path $APP_ROOT 'ArdorApp\build\spec'

Write-Host ("APP_ROOT     = {0}" -f $APP_ROOT)
Write-Host ("PROJECT_ROOT = {0}" -f $PROJECT_ROOT)
Write-Host ("ENTRY        = {0}" -f $ENTRY)
Write-Host ("ICON         = {0}" -f $ICON)
Write-Host ("DIST         = {0}" -f $DIST)

# Basic checks
foreach ($p in @($APP_ROOT,$PROJECT_ROOT,$ENTRY,$ICON)) {
    if (-not (Test-Path $p)) { throw ("Path not found: {0}" -f $p) }
}

# --- Resolve Python (avoid WindowsApps stub) ---
function Get-NonStubPython {
    param([string[]]$Candidates)
    foreach ($c in $Candidates) {
        if (-not $c) { continue }
        if (-not (Test-Path $c)) { continue }
        try {
            $len = (Get-Item $c).Length
            if ($len -gt 0 -and ($c -notlike '*WindowsApps*')) { return $c }
        } catch { }
    }
    return $null
}

# Strip WindowsApps from PATH for this session
$env:Path = ($env:Path -split ';' | Where-Object { $_ -notlike '*WindowsApps*' }) -join ';'

# Candidate interpreters (edit or add if needed)
$cands = @(
    "$env:LocalAppData\Programs\Python\Python312\python.exe",
    'C:\Users\adm\ardor312\Scripts\python.exe',
    'C:\Users\adm\venv\Scripts\python.exe'
)
try { $cmd = (Get-Command python -ErrorAction SilentlyContinue).Source } catch { $cmd = $null }
if ($cmd) { $cands += $cmd }

$PY = Get-NonStubPython -Candidates $cands
if (-not $PY) { throw "Could not locate a valid Python (non-WindowsApps). Please install Python 3.12 from python.org or update the candidates." }

Write-Host ("Using Python: {0}" -f $PY)

# --- Clean old build/dist ---
if (Test-Path $BUILD) { Remove-Item -Recurse -Force $BUILD }
if (Test-Path $DIST)  { Remove-Item -Recurse -Force $DIST  }

# --- PyInstaller args ---
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
    '--collect-submodules','torch'
)

function Add-DataIfExists {
    param([string]$src, [string]$destRel)
    if (Test-Path $src) {
        $pyArgs += @('--add-data', ($src + ';' + $destRel))
    } else {
        Write-Host ("[!] Skipping missing path: {0}" -f $src) -ForegroundColor Yellow
    }
}

# Add from PROJECT_ROOT
Add-DataIfExists (Join-Path $PROJECT_ROOT 'Cerebrum') 'Cerebrum'
Add-DataIfExists (Join-Path $PROJECT_ROOT 'Cortex') 'Cortex'
Add-DataIfExists (Join-Path $PROJECT_ROOT 'ProjectTokenizer\ardor_tokenizer') 'ProjectTokenizer\ardor_tokenizer'
Add-DataIfExists (Join-Path $PROJECT_ROOT 'Models') 'Models'
Add-DataIfExists (Join-Path $PROJECT_ROOT 'REM.py') '.'
Add-DataIfExists (Join-Path $PROJECT_ROOT 'neural_plasticity_training.py') '.'

# Fallback: also include under APP_ROOT if present
Add-DataIfExists (Join-Path $APP_ROOT 'Cerebrum') 'Cerebrum'
Add-DataIfExists (Join-Path $APP_ROOT 'Cortex') 'Cortex'
Add-DataIfExists (Join-Path $APP_ROOT 'ProjectTokenizer\ardor_tokenizer') 'ProjectTokenizer\ardor_tokenizer'
Add-DataIfExists (Join-Path $APP_ROOT 'Models') 'Models'
Add-DataIfExists (Join-Path $APP_ROOT 'REM.py') '.'
Add-DataIfExists (Join-Path $APP_ROOT 'neural_plasticity_training.py') '.'

# Entry
$pyArgs += $ENTRY

# Ensure PyInstaller exists in chosen Python
& $PY -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('PyInstaller') else 1)"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing PyInstaller in selected Python..." -ForegroundColor Yellow
    & $PY -m pip install -U pip pyinstaller pyinstaller-hooks-contrib
    if ($LASTEXITCODE -ne 0) { throw "Failed to set up PyInstaller." }
}

# Build
Write-Host ("`nRunning: {0} {1}" -f $PY, ($pyArgs -join ' ')) -ForegroundColor Cyan
& $PY @pyArgs
if ($LASTEXITCODE -ne 0) { throw ("PyInstaller failed with exit code {0}" -f $LASTEXITCODE) }

# Marker
$AppDir = Join-Path $DIST 'ArdorGUI'
New-Item -ItemType Directory -Path $AppDir -Force | Out-Null
Set-Content -Path (Join-Path $AppDir 'ardor_home.txt') -Value $PROJECT_ROOT -Encoding UTF8

Write-Host ''
$msg = 'Build complete: ' + (Join-Path $DIST 'ArdorGUI')
Write-Host $msg -ForegroundColor Green