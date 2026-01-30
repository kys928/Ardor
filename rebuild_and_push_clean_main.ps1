param(
  [string]$RemoteUrl = "https://github.com/kys928/Ardor.git",
  [string]$Branch    = "main"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function RunGit([string[]]$a) {
  $pretty = "git " + ($a -join " ")
  Write-Host ">> $pretty"
  & git @a
  if ($LASTEXITCODE -ne 0) { throw "Command failed: $pretty" }
}

function TryGit([string[]]$a) {
  $pretty = "git " + ($a -join " ")
  Write-Host ">> $pretty"
  & git @a
  return $LASTEXITCODE
}

Write-Host "`n[0] Sanity..."
RunGit @("version")
RunGit @("lfs","install")

Write-Host "`n[1] Ensure origin..."
$origin = (& git remote get-url origin 2>$null)
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($origin)) {
  RunGit @("remote","add","origin",$RemoteUrl)
} elseif ($origin -ne $RemoteUrl) {
  RunGit @("remote","set-url","origin",$RemoteUrl)
}

Write-Host "`n[2] Ensure .gitattributes (LFS patterns present)..."
if (!(Test-Path ".gitattributes")) { New-Item -ItemType File -Path ".gitattributes" | Out-Null }
$lfs = @(
  "*.pt","*.safetensors","*.bin","*.onnx","*.npy","*.npz","*.pth","*.ckpt",
  "*.zip","*.7z","*.tar","*.gz","*.tgz","*.parquet","*.jsonl"
)
$existingAttr = Get-Content ".gitattributes" -ErrorAction SilentlyContinue
foreach ($p in $lfs) {
  $line = "$p filter=lfs diff=lfs merge=lfs -text"
  if ($existingAttr -notcontains $line) { Add-Content ".gitattributes" $line }
}

Write-Host "`n[3] Ensure .gitignore (hard excludes)..."
if (!(Test-Path ".gitignore")) { New-Item -ItemType File -Path ".gitignore" | Out-Null }
$ignoreBlock = @"
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

# Big generated artifacts
Praetor/ArdorApp/build/
Praetor/ArdorApp/dist/
Praetor/ArdorApp/*.spec
_internal/

.vscode/
.idea/
.DS_Store
Thumbs.db
"@
$ig = Get-Content -Raw ".gitignore"
if ($ig -notlike "*Big generated artifacts*") {
  Add-Content ".gitignore" "`r`n$ignoreBlock`r`n"
}

Write-Host "`n[4] Rebuild '$Branch' as ONE clean commit (robust)..."
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$tmpBranch = "clean_main_$stamp"

# Create orphan branch
RunGit @("checkout","--orphan",$tmpBranch)

# IMPORTANT: clear any staged/index weirdness
# This makes the index match the working tree (even on orphan), then we can safely remove.
TryGit @("reset","--mixed") | Out-Null

# Remove EVERYTHING from index WITHOUT touching disk
# Using 'git ls-files -z' avoids pathspec + staged-conflict issues.
Write-Host ">> clearing index (git ls-files | git rm --cached)..."
$files = & git ls-files -z
if ($files) {
  $tmp = [System.IO.Path]::GetTempFileName()
  [System.IO.File]::WriteAllBytes($tmp, $files)
  # feed the NUL-separated list into git rm
  $bytes = [System.IO.File]::ReadAllBytes($tmp)
  Remove-Item $tmp -Force | Out-Null
  # PowerShell can't easily pipe raw bytes as-is, so call via cmd to preserve NULs:
  # We instead use a safer approach: remove from index by pattern '.' after a reset,
  # but with '--ignore-unmatch' and allowing errors (some git versions are picky).
  TryGit @("rm","-r","--cached","--ignore-unmatch",".") | Out-Null
}

# Now add only what we want (honors .gitignore)
RunGit @("add",".gitattributes",".gitignore")
RunGit @("add","-A")

# If there is nothing to commit, bail early.
$status = & git status --porcelain
if ([string]::IsNullOrWhiteSpace($status)) {
  throw "Nothing to commit after rebuild. That usually means your .gitignore excludes everything unexpectedly."
}

RunGit @("commit","-m","Clean snapshot: Ardor monorepo (pruned generated artifacts)")

Write-Host "`n[5] Push clean snapshot to main (force)..."
RunGit @("config","http.version","HTTP/1.1")
RunGit @("config","http.postBuffer","524288000")

RunGit @("push","-f","origin",("$tmpBranch`:$Branch"))

Write-Host "`n✅ Done. main now points to the clean snapshot commit."