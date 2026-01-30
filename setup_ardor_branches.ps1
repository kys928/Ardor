# setup_ardor_branches.ps1  (FIXED: safe git invocation + LFS attributes)
$ErrorActionPreference = "Stop"

$RepoRoot = "C:\Users\adm\PycharmProjects\ProjectArdor"
$RemoteName = "origin"
$RemoteUrl  = "https://github.com/kys928/Ardor.git"

$Modules = @(
  "Cerebrum","Aeternum","Sentinel","Chronos","Akasha","Hermes","Thanatos",
  "Praetor","Erratum","Hephaestus","Nemesis","Assets"
)

# LFS patterns: tune as needed
$LfsPatterns = @(
  "*.pt","*.pth","*.ckpt","*.bin","*.safetensors","*.onnx",
  "*.npy","*.npz","*.pkl","*.joblib",
  "*.zip","*.7z","*.tar","*.tar.gz","*.tgz"
)

function ExecGit {
  param([Parameter(Mandatory=$true)][string[]]$Args)

  $pretty = "git " + ($Args -join " ")
  Write-Host ">> $pretty" -ForegroundColor Cyan

  & git @Args
  if ($LASTEXITCODE -ne 0) { throw "Command failed: $pretty" }
}

function Ensure-RepoRoot($path) {
  if (!(Test-Path $path)) { throw "Repo root not found: $path" }
  Set-Location $path
  if (!(Test-Path ".git")) { throw "No .git directory found in $path." }
}

Ensure-RepoRoot $RepoRoot

Write-Host "`n[1/6] Checking git + lfs..." -ForegroundColor Green
ExecGit @("--version")
ExecGit @("lfs","version")
ExecGit @("lfs","install")

Write-Host "`n[2/6] Setting remote '$RemoteName' => $RemoteUrl" -ForegroundColor Green
$remotes = (git remote) 2>$null
if ($remotes -match "^$RemoteName$") { ExecGit @("remote","set-url",$RemoteName,$RemoteUrl) }
else { ExecGit @("remote","add",$RemoteName,$RemoteUrl) }

Write-Host "`n[3/6] Writing .gitattributes for LFS patterns (safe mode)..." -ForegroundColor Green
$attrPath = Join-Path (Get-Location) ".gitattributes"
if (!(Test-Path $attrPath)) { New-Item -ItemType File -Path $attrPath | Out-Null }

$existing = Get-Content $attrPath -ErrorAction SilentlyContinue
$linesToAdd = @()

foreach ($p in $LfsPatterns) {
  $line = "$p filter=lfs diff=lfs merge=lfs -text"
  if ($existing -notcontains $line) { $linesToAdd += $line }
}

if ($linesToAdd.Count -gt 0) {
  Add-Content -Path $attrPath -Value $linesToAdd
  Write-Host "Added $($linesToAdd.Count) LFS patterns to .gitattributes" -ForegroundColor Yellow
} else {
  Write-Host ".gitattributes already contains all LFS patterns." -ForegroundColor Yellow
}

ExecGit @("add",".gitattributes")

Write-Host "`n[4/6] Ensuring main branch + committing current state..." -ForegroundColor Green
$branch = (git rev-parse --abbrev-ref HEAD).Trim()
if ($branch -ne "main") {
  $hasMain = (git branch --list main).Count -gt 0
  if ($hasMain) { ExecGit @("checkout","main") } else { ExecGit @("checkout","-b","main") }
}

ExecGit @("add","-A")
$status = (git status --porcelain)
if ($status) {
  ExecGit @("commit","-m","Ardor monorepo snapshot (main) + LFS attributes")
} else {
  Write-Host "No changes to commit on main." -ForegroundColor Yellow
}

Write-Host "`n[5/6] Pushing main..." -ForegroundColor Green
ExecGit @("push","-u",$RemoteName,"main")

Write-Host "`n[6/6] Creating + pushing subtree branches..." -ForegroundColor Green
foreach ($m in $Modules) {
  if (Test-Path $m) {
    # create/refresh split branch
    $exists = (git branch --list $m).Count -gt 0
    if ($exists) { ExecGit @("branch","-D",$m) }

    ExecGit @("subtree","split","--prefix=$m","-b",$m)
    ExecGit @("push","-u",$RemoteName,$m,"--force")
  } else {
    Write-Host "Skipping '$m' (folder not found)" -ForegroundColor Yellow
  }
}

Write-Host "`n✅ Done." -ForegroundColor Green