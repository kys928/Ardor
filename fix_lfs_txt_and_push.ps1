# fix_lfs_txt_and_push.ps1
# Run from anywhere: powershell -ExecutionPolicy Bypass -File .\fix_lfs_txt_and_push.ps1

param(
  [string]$RepoPath = "C:\Users\adm\PycharmProjects\ProjectArdor",
  [string]$Branch   = "clean_main_20260130_115441",
  [string]$Remote   = "origin",
  [string]$Target   = "main"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function RunGit([string[]]$a) {
  $pretty = "git " + ($a -join " ")
  Write-Host ">> $pretty"
  & git @a
  if ($LASTEXITCODE -ne 0) { throw "Command failed: $pretty" }
}

Write-Host "=== Ardor: fix .vs + LFS-track big txt + force-push ===`n"

if (!(Test-Path $RepoPath)) { throw "RepoPath not found: $RepoPath" }
Set-Location $RepoPath

# 0) Sanity
RunGit @("version")
RunGit @("lfs","install")

# 1) Checkout the orphan clean branch
RunGit @("checkout",$Branch)

# 2) Ignore VS junk + remove it from index if present
$vsIgnoreBlock = @"
# Visual Studio junk
.vs/
*.sqlite
*.db
*.db-shm
*.db-wal
*.wsuo
"@

if (!(Test-Path ".gitignore")) { New-Item -ItemType File -Path ".gitignore" | Out-Null }
$ig = Get-Content -Raw ".gitignore" -ErrorAction SilentlyContinue
if ($null -eq $ig) { $ig = "" }

if ($ig -notlike "*# Visual Studio junk*") {
  Add-Content ".gitignore" "`r`n$vsIgnoreBlock`r`n"
  Write-Host "Appended VS ignore block to .gitignore"
} else {
  Write-Host ".gitignore already has VS ignore block"
}

# Remove .vs from the git index (does not delete files on disk)
& git rm -r --cached --ignore-unmatch .vs | Out-Null

# 3) LFS-track the offending big file
$bigFile = "Cerebrum/Training Data/GravisCorpora.txt"
RunGit @("lfs","track",$bigFile)

# 4) Re-add big file as LFS pointer (remove from index first)
& git rm --cached $bigFile 2>$null | Out-Null

RunGit @("add",".gitattributes",".gitignore")
RunGit @("add",$bigFile)

# 5) Add everything else (honors .gitignore)
RunGit @("add","-A")

# 6) Amend the clean snapshot commit (keeps single commit)
RunGit @("commit","--amend","-m","Clean snapshot: Ardor monorepo (pruned artifacts; LFS for large assets)")

# 7) Force-push branch -> main
RunGit @("push","-f",$Remote,("$Branch`:$Target"))

Write-Host "`n✅ Done. Pushed $Branch -> $Remote/$Target (with LFS for $bigFile and .vs ignored)."