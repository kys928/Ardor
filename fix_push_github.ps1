param(
  [string]$RemoteUrl = "https://github.com/kys928/Ardor.git",
  [string]$Branch    = "main",
  [switch]$RebuildMainIfPushFails = $true,
  [int]$PushRetries = 5
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function RunGit([string[]]$GitArgs) {
  $pretty = "git " + ($GitArgs -join " ")
  Write-Host ">> $pretty"
  & git @GitArgs
  if ($LASTEXITCODE -ne 0) { throw "Command failed: $pretty" }
}

function TryGit([string[]]$GitArgs) {
  $pretty = "git " + ($GitArgs -join " ")
  Write-Host ">> $pretty"
  & git @GitArgs
  return $LASTEXITCODE
}

function Ensure-FileHasBlock([string]$path, [string]$block) {
  if (!(Test-Path $path)) { New-Item -ItemType File -Path $path | Out-Null }
  $content = Get-Content -Raw -ErrorAction SilentlyContinue $path
  if ($null -eq $content) { $content = "" }
  if ($content -notlike "*$block*") { Add-Content -Path $path -Value "`r`n$block`r`n" }
}

function Ensure-GitIgnore {
  $block = @"
# ======================
# Ardor repo hygiene
# ======================
__pycache__/
*.pyc
*.pyo
*.pyd
*.log

build/
dist/
**/build/
**/dist/
*.spec
*.toc
*.pkg
*.pyz
base_library.zip

.vscode/
.idea/
.DS_Store
Thumbs.db
"@
  Ensure-FileHasBlock ".gitignore" $block
}

function Ensure-GitAttributesLFS {
  $patterns = @(
    "*.pt","*.safetensors","*.bin","*.onnx","*.npy","*.npz","*.pth","*.ckpt",
    "*.zip","*.7z","*.tar","*.gz","*.tgz","*.parquet","*.jsonl"
  )
  if (!(Test-Path ".gitattributes")) { New-Item -ItemType File -Path ".gitattributes" | Out-Null }
  $existing = Get-Content ".gitattributes" -ErrorAction SilentlyContinue
  $added = 0
  foreach ($p in $patterns) {
    $line = "$p filter=lfs diff=lfs merge=lfs -text"
    if ($existing -notcontains $line) { Add-Content -Path ".gitattributes" -Value $line; $added++ }
  }
  if ($added -gt 0) { Write-Host "Added $added LFS patterns to .gitattributes" }
  else { Write-Host ".gitattributes already contains all LFS patterns." }
}

function Untrack-GeneratedJunkFromIndex {
  $paths = @("**/__pycache__","**/*.pyc","build","dist","**/build","**/dist")
  foreach ($p in $paths) {
    TryGit @("rm","-r","--cached","--ignore-unmatch",$p) | Out-Null
  }
}

function Configure-PushStability {
  RunGit @("config","http.postBuffer","524288000")
  RunGit @("config","http.version","HTTP/1.1")
  RunGit @("config","http.lowSpeedLimit","0")
  RunGit @("config","http.lowSpeedTime","999999")
}

function Try-PushWithRetries([int]$retries) {
  for ($i=1; $i -le $retries; $i++) {
    Write-Host ""
    Write-Host "[PUSH] Attempt $i / $retries"

    $c1 = TryGit @("push","-u","origin",$Branch)
    if ($c1 -eq 0) { return $true }

    Write-Host "Push failed (code $c1). Retrying with --no-thin..."
    $c2 = TryGit @("push","-u","origin",$Branch,"--no-thin")
    if ($c2 -eq 0) { return $true }

    Start-Sleep -Seconds ([Math]::Min(10 * $i, 60))
  }
  return $false
}

function Rebuild-Main-AsSingleCommit {
  $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
  $tmpBranch = "rebuild_main_$stamp"

  Write-Host ""
  Write-Host "=== Rebuilding '$Branch' as a clean single commit (history rewrite) ==="

  RunGit @("checkout","--orphan",$tmpBranch)
  TryGit @("rm","-r","--cached",".") | Out-Null

  Ensure-GitIgnore
  Ensure-GitAttributesLFS

  RunGit @("add","-A")
  RunGit @("commit","-m","Rebuild main clean snapshot")

  RunGit @("push","-f","origin",("$tmpBranch`:$Branch"))
  RunGit @("checkout",$Branch)
}

# ======================
# MAIN
# ======================

Write-Host ""
Write-Host "[1] Probing git + lfs..."
RunGit @("version")
RunGit @("lfs","version")
RunGit @("lfs","install")

Write-Host ""
Write-Host "[2] Ensuring remote origin is correct..."
$origin = (& git remote get-url origin 2>$null)
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($origin)) {
  Write-Host "No origin remote found. Adding origin => $RemoteUrl"
  RunGit @("remote","add","origin",$RemoteUrl)
  $origin = $RemoteUrl
} else {
  Write-Host "Current origin: $origin"
  if ($origin -ne $RemoteUrl) {
    Write-Host "Updating origin to: $RemoteUrl"
    RunGit @("remote","set-url","origin",$RemoteUrl)
    $origin = $RemoteUrl
  } else {
    Write-Host "Origin already correct."
  }
}

Write-Host ""
Write-Host "[3] Hygiene (LFS + ignore + untrack caches)..."
Ensure-GitAttributesLFS
Ensure-GitIgnore
RunGit @("add",".gitattributes",".gitignore")
Untrack-GeneratedJunkFromIndex

Write-Host ""
Write-Host "[4] Commit if needed..."
RunGit @("add","-A")
$status = & git status --porcelain
if ($status -and $status.Length -gt 0) {
  $msg = "Repo hygiene - ignore caches/build, prep LFS, shrink push pack"
  RunGit @("commit","-m",$msg)
} else {
  Write-Host "No changes to commit."
}

Write-Host ""
Write-Host "[5] Push..."
Configure-PushStability

$ok = Try-PushWithRetries $PushRetries
if ($ok) {
  Write-Host ""
  Write-Host "✅ Push succeeded to $origin ($Branch)."
  exit 0
}

if ($RebuildMainIfPushFails) {
  Write-Host ""
  Write-Host "⚠️ Push keeps failing (GitHub 500 likely). Rebuilding main..."
  Rebuild-Main-AsSingleCommit
  Write-Host ""
  Write-Host "✅ Rebuild + force-push completed."
  exit 0
}

throw "Push failed after retries, and rebuild mode disabled."