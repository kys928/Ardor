# EXE_CREATOR_FIXEDICON_PARAM.ps1 - Fixed icon path, param for App root (Praetor), robust Python resolution.
# Usage: -AppRoot "C:\Users\adm\PycharmProjects\ProjectArdor\Praetor" [-PythonExe "C:\path\to\venv\Scripts\python.exe"]

param(
    [string]$AppRoot,
    [string]$PythonExe
)
$ErrorActionPreference = 'Stop'

if (-not $AppRoot) { throw "Usage: -AppRoot 'C:\Users\adm\PycharmProjects\ProjectArdor\Praetor'" }
$APP_ROOT = (Resolve-Path $AppRoot).Path
$PROJECT_ROOT = Split-Path $APP_ROOT -Parent

# Fixed icon path under Praetor
$ICON = Join-Path $APP_ROOT 'ArdorApp\assets\ardor.ico'
$DESKTOP = [Environment]::GetFolderPath('Desktop')
$DIST    = Join-Path $DESKTOP 'ArdorApp'
$BUILD   = Join-Path $APP_ROOT 'build'
$SPEC    = Join-Path $APP_ROOT 'build\spec'
$ENTRY   = Join-Path $APP_ROOT 'GUI_Cortex.py'

Write-Host ("APP_ROOT     = {0}" -f $APP_ROOT)
Write-Host ("PROJECT_ROOT = {0}" -f $PROJECT_ROOT)
Write-Host ("ENTRY        = {0}" -f $ENTRY)
Write-Host ("ICON         = {0}" -f $ICON)
Write-Host ("DIST         = {0}" -f $DIST)

if (-not (Test-Path $ENTRY)) { throw ("Entry point not found: {0}" -f $ENTRY) }
if (-not (Test-Path $ICON))  { throw ("Icon not found: {0}" -f $ICON) }

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

$cands = @()
if ($PythonExe) { $cands += $PythonExe }
$cands += (Join-Path $PROJECT_ROOT 'venv\Scripts\python.exe')
$cands += (Join-Path $APP_ROOT     'venv\Scripts\python.exe')
$cands += 'C:\Users\adm\venv\Scripts\python.exe'
try { $cmd = (Get-Command python -ErrorAction SilentlyContinue).Source } catch { $cmd = $null }
if ($cmd) { $cands += $cmd }

$PY = Get-NonStubPython -Candidates $cands
if (-not $PY) { throw "Could not locate a valid Python interpreter. Pass -PythonExe 'C:\Path\to\venv\Scripts\python.exe'." }

Write-Host ("Using Python: {0}" -f $PY)

# Clean
if (Test-Path $BUILD) { Remove-Item -Recurse -Force $BUILD }
if (Test-Path $DIST)  { Remove-Item -Recurse -Force $DIST  }

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

Add-DataIfExists (Join-Path $PROJECT_ROOT 'Cerebrum') 'Cerebrum'
Add-DataIfExists (Join-Path $PROJECT_ROOT 'Cortex') 'Cortex'
Add-DataIfExists (Join-Path $PROJECT_ROOT 'ProjectTokenizer\ardor_tokenizer') 'ProjectTokenizer\ardor_tokenizer'
Add-DataIfExists (Join-Path $PROJECT_ROOT 'Models') 'Models'
Add-DataIfExists (Join-Path $PROJECT_ROOT 'REM.py') '.'
Add-DataIfExists (Join-Path $PROJECT_ROOT 'neural_plasticity_training.py') '.'

# Entry
$pyArgs += $ENTRY

# Ensure PyInstaller
& $PY -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('PyInstaller') else 1)"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing PyInstaller in selected Python..." -ForegroundColor Yellow
    & $PY -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw "Failed to upgrade pip." }
    & $PY -m pip install pyinstaller
    if ($LASTEXITCODE -ne 0) { throw "Failed to install PyInstaller." }
}

Write-Host ("`nRunning: {0} {1}" -f $PY, ($pyArgs -join ' ')) -ForegroundColor Cyan
& $PY @pyArgs
if ($LASTEXITCODE -ne 0) { throw ("PyInstaller failed with exit code {0}" -f $LASTEXITCODE) }

$AppDir = Join-Path $DIST 'ArdorGUI'
New-Item -ItemType Directory -Path $AppDir -Force | Out-Null
Set-Content -Path (Join-Path $AppDir 'ardor_home.txt') -Value $PROJECT_ROOT -Encoding UTF8

Write-Host ''
$msg = 'Build complete: ' + (Join-Path $DIST 'ArdorGUI')
Write-Host $msg -ForegroundColor Green
