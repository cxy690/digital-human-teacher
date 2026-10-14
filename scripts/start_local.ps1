<#
  start_local.ps1 - one-click LOCAL start (Windows)

    -Engine musetalk : MuseTalk v1.5 engine   http://127.0.0.1:8012   (highest quality, slow on 8GB GPU)
    -Engine wav2lip  : Wav2Lip engine         http://127.0.0.1:8011   (fast, keeps the ORIGINAL video best)
    and the student app on http://127.0.0.1:8001

  Usage:
    powershell -ExecutionPolicy Bypass -File scripts\start_local.ps1 -Engine wav2lip
    powershell -ExecutionPolicy Bypass -File scripts\start_local.ps1 -Engine musetalk
    powershell -ExecutionPolicy Bypass -File scripts\start_local.ps1 -Stop

  The script also syncs .env (AVATAR_BACKEND / AVATAR_API_BASE) to the chosen engine,
  so the app always talks to the engine that was just started.

  Overrides (env vars):
    EDU_PY   engine/app python   default D:\An\envs\tea\python.exe
#>
param(
    [ValidateSet('musetalk', 'wav2lip')] [string]$Engine = 'musetalk',
    [switch]$Stop,
    [switch]$NoApp
)
$ErrorActionPreference = 'SilentlyContinue'
$Root = Split-Path -Parent $PSScriptRoot
$Py = if ($env:EDU_PY) { $env:EDU_PY } else { 'D:\An\envs\tea\python.exe' }
Set-Location $Root

$engineScript = if ($Engine -eq 'wav2lip') { 'scripts\avatar_engine_wav2lip.py' } else { 'scripts\avatar_engine_musetalk_v15.py' }
$enginePort   = if ($Engine -eq 'wav2lip') { 8011 } else { 8012 }
$engineName   = if ($Engine -eq 'wav2lip') { 'Wav2Lip' } else { 'MuseTalk v1.5' }
$backend      = if ($Engine -eq 'wav2lip') { 'http' } else { 'musetalk' }
$engineUrl    = "http://127.0.0.1:$enginePort"

function Stop-Svc([string]$pat) {
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -like "*$pat*" } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
}
function Stop-Port([int]$port) {
    # uvicorn reload 的 worker 子进程命令行不含脚本名，只杀 reloader 会留孤儿占端口；这里按端口清
    $owners = (netstat -ano | Select-String ":$port\s" | Select-String 'LISTENING') |
        ForEach-Object { ($_ -split '\s+')[-1] } | Sort-Object -Unique
    foreach ($p in $owners) { if ($p -and $p -ne '0') { Stop-Process -Id $p -Force -ErrorAction SilentlyContinue } }
}
function Wait-Health([string]$url, [string]$name, [int]$sec) {
    $t0 = Get-Date
    while (((Get-Date) - $t0).TotalSeconds -lt $sec) {
        try {
            if ((Invoke-WebRequest -UseBasicParsing -Uri $url -TimeoutSec 3).StatusCode -eq 200) {
                Write-Host ("  [OK]   {0} ready -> {1}" -f $name, $url) -ForegroundColor Green
                return $true
            }
        } catch {}
        Start-Sleep -Seconds 3
    }
    Write-Host ("  [WAIT] {0} not ready in {1}s -> {2}" -f $name, $sec, $url) -ForegroundColor Yellow
    return $false
}
function Sync-Env([string]$file, [string]$backend, [string]$apiBase) {
    if (-not (Test-Path $file)) { return }
    $out = New-Object System.Collections.Generic.List[string]
    $bFound = $false; $aFound = $false
    foreach ($l in (Get-Content $file)) {
        if ($l -match '^\s*AVATAR_BACKEND\s*=')    { $out.Add("AVATAR_BACKEND=$backend");  $bFound = $true }
        elseif ($l -match '^\s*AVATAR_API_BASE\s*=') { $out.Add("AVATAR_API_BASE=$apiBase"); $aFound = $true }
        else { $out.Add($l) }
    }
    if (-not $bFound) { $out.Add("AVATAR_BACKEND=$backend") }
    if (-not $aFound) { $out.Add("AVATAR_API_BASE=$apiBase") }
    # 无 BOM 的 UTF-8，避免 python-dotenv 读到 BOM
    [System.IO.File]::WriteAllLines($file, $out, (New-Object System.Text.UTF8Encoding($false)))
}

if (-not (Test-Path (Join-Path $Root 'server\app.py'))) {
    Write-Host "[x] project root (education/) not found. Keep this script under scripts\ and run it." -ForegroundColor Red
    exit 1
}

Write-Host "== stopping old processes ==" -ForegroundColor Cyan
Stop-Svc 'avatar_engine_musetalk_v15'
Stop-Svc 'avatar_engine_wav2lip'
Stop-Svc 'server.app'
Stop-Port 8001; Stop-Port 8011; Stop-Port 8012
Start-Sleep -Seconds 2
if ($Stop) { Write-Host "stopped." -ForegroundColor Green; exit 0 }

if (-not (Test-Path $Py)) { Write-Host "[x] python not found: $Py  (set env EDU_PY)" -ForegroundColor Red; exit 1 }

Write-Host "== syncing .env -> AVATAR_BACKEND=$backend , AVATAR_API_BASE=$engineUrl ==" -ForegroundColor Cyan
Sync-Env (Join-Path $Root '.env') $backend $engineUrl

Write-Host "== starting $engineName engine (:$enginePort) ==" -ForegroundColor Cyan
Start-Process -FilePath $Py -ArgumentList $engineScript -WorkingDirectory $Root -WindowStyle Minimized

Write-Host "   waiting for engine to load (MuseTalk first run ~2-3 min)..." -ForegroundColor DarkGray
Wait-Health "$engineUrl/health" "$engineName engine" 300 | Out-Null

if (-not $NoApp) {
    Write-Host "== starting student app (:8001) ==" -ForegroundColor Cyan
    Start-Process -FilePath $Py -ArgumentList '-m', 'server.app' -WorkingDirectory $Root -WindowStyle Minimized
    Wait-Health 'http://127.0.0.1:8001/health' 'student app' 60 | Out-Null
}

Write-Host ""
Write-Host "================ READY ================" -ForegroundColor Green
Write-Host ("engine ({0}):  {1}/health" -f $Engine, $engineUrl)
Write-Host "student app:       http://127.0.0.1:8001"
Write-Host "switch engine:     scripts\start_local.bat -Engine wav2lip   (or musetalk)"
Write-Host "stop all:          scripts\start_local.bat -Stop"
