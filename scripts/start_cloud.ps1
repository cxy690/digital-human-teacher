<#
  start_cloud.ps1 - 一键全部启动（云端 MuseTalk 引擎 + 云端 CosyVoice TTS + 本机 SSH 隧道 + 本机主应用）

  适用架构：数字人引擎（:8012）与音色克隆 TTS（:8090）跑在算力云；
            本机 _tunnel.py 通过 SSH 把它们映射到本地同名端口；
            本机主应用（:8001）再经隧道调用云端引擎。

  启动顺序（脚本自动完成；已在跑的组件自动复用，不再重启）：
    1) 云端 MuseTalk 引擎  -> scripts/cloud_ctl.py restart（仅当进程不在时）
    2) 云端 CosyVoice TTS  -> scripts/cloud_ctl.py restart_tts（仅当进程不在时；加载需 1-3 分钟）
    3) 本机 SSH 隧道       -> _tunnel.py（本地 8012/8090 -> 云端）
    4) 本机主应用          -> -m server.app（本地 8001）
    5) 轮询各健康接口，全部就绪后打印 READY

  用法：
    powershell -ExecutionPolicy Bypass -File scripts\start_cloud.ps1
    powershell -ExecutionPolicy Bypass -File scripts\start_cloud.ps1 -Stop          # 停止全部（本机隧道+应用，云端引擎+TTS）
    powershell -ExecutionPolicy Bypass -File scripts\start_cloud.ps1 -ForceRestart  # 云端引擎/TTS 即使健康也重启
    powershell -ExecutionPolicy Bypass -File scripts\start_cloud.ps1 -NoEngine      # 不碰云端引擎
    powershell -ExecutionPolicy Bypass -File scripts\start_cloud.ps1 -NoTts         # 不碰云端 TTS（不用克隆音色时）
    powershell -ExecutionPolicy Bypass -File scripts\start_cloud.ps1 -NoApp         # 只起引擎+隧道，不起主应用

  可用环境变量覆盖：
    EDU_PY        主应用 python（默认自动探测 D:\anaconda3\python.exe 等，需带 fastapi/uvicorn）
    EDU_PY_BASE   带 paramiko 的 python，用于隧道与云端控制（默认自动探测）
#>
param(
    [switch]$Stop,
    [switch]$ForceRestart,
    [switch]$NoEngine,
    [switch]$NoTts,
    [switch]$NoApp
)

$ErrorActionPreference = 'SilentlyContinue'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$EngineHealth = 'http://127.0.0.1:8012/health'
$TtsHealth    = 'http://127.0.0.1:8090/health'
$AppHealth    = 'http://127.0.0.1:8001/health'

# ---------------------------------------------------------------- helpers

function Get-EnvVar([string]$name) {
    foreach ($level in 'Process', 'User', 'Machine') {
        $v = [Environment]::GetEnvironmentVariable($name, $level)
        if ($v) { return $v }
    }
    return $null
}

function Resolve-Py([string]$envName, [string]$probe, [string[]]$candidates) {
    $list = @()
    $v = Get-EnvVar $envName
    if ($v) { $list += $v }
    $list += $candidates
    foreach ($c in $list) {
        $path = $c
        if ($c -eq 'python') {
            $cmd = Get-Command python -ErrorAction SilentlyContinue
            if (-not $cmd) { continue }
            $path = $cmd.Source
        }
        if (-not (Test-Path $path)) { continue }
        & $path -c $probe 2>$null | Out-Null
        if ($LASTEXITCODE -eq 0) { return $path }
    }
    return $null
}

function Stop-ByCmd([string]$pattern) {
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -like "*$pattern*" } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
}

function Stop-ByPort([int]$port) {
    (netstat -ano | Select-String ":$port\s" | Select-String 'LISTENING') |
        ForEach-Object { ($_ -split '\s+')[-1] } | Sort-Object -Unique |
        ForEach-Object { if ($_ -and $_ -ne '0') { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue } }
}

function Get-Procs([string]$pattern) {
    @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
      Where-Object { $_.CommandLine -like "*$pattern*" })
}

function Test-PortOpen([int]$port) {
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $async = $client.BeginConnect('127.0.0.1', $port, $null, $null)
        if (-not $async.AsyncWaitHandle.WaitOne(800)) { return $false }
        $client.EndConnect($async)
        return $true
    } catch { return $false } finally { $client.Close() }
}

function Test-Health([string]$url) {
    try { return ((Invoke-WebRequest -UseBasicParsing -Uri $url -TimeoutSec 3).StatusCode -eq 200) }
    catch { return $false }
}

function Wait-Health([string]$url, [string]$name, [int]$sec) {
    $t0 = Get-Date
    while (((Get-Date) - $t0).TotalSeconds -lt $sec) {
        if (Test-Health $url) {
            $elapsed = [int]((Get-Date) - $t0).TotalSeconds
            Write-Host ("  [OK]   {0} -> {1}  ({2}s)" -f $name, $url, $elapsed) -ForegroundColor Green
            return $true
        }
        Start-Sleep -Seconds 4
    }
    Write-Host ("  [WAIT] {0} not ready in {1}s: {2}" -f $name, $sec, $url) -ForegroundColor Yellow
    return $false
}

function Invoke-Cloud([string]$action) {
    (& $PyBase (Join-Path $Root 'scripts\cloud_ctl.py') $action 2>$null) -join "`n"
}

# ---------------------------------------------------------------- python 解释器探测

$PyBase = Resolve-Py 'EDU_PY_BASE' 'import paramiko' @(
    'D:\anaconda3\python.exe', 'D:\An\python.exe', 'D:\An\envs\tea\python.exe', 'python')
$PyApp = Resolve-Py 'EDU_PY' 'import fastapi, uvicorn' @(
    'D:\anaconda3\python.exe', 'D:\An\envs\tea\python.exe', 'python')

if (-not $PyBase) {
    Write-Host '[x] python with paramiko not found (set EDU_PY_BASE)' -ForegroundColor Red
    exit 1
}
if (-not $PyApp) {
    Write-Host '[x] python with fastapi/uvicorn not found (set EDU_PY)' -ForegroundColor Red
    exit 1
}
Write-Host ("python: app={0}  base={1}" -f $PyApp, $PyBase) -ForegroundColor DarkGray

# 隧道/云控凭据：把用户级变量补进当前进程，确保子进程能继承
foreach ($name in 'TUNNEL_SSH_HOST', 'TUNNEL_SSH_PORT', 'TUNNEL_SSH_USER', 'TUNNEL_SSH_PASSWORD',
                  'CLOUD_HOST', 'CLOUD_PORT', 'CLOUD_USER', 'CLOUD_PWD', 'CLOUD_ROOT') {
    if (-not (Get-Item "Env:$name" -ErrorAction SilentlyContinue)) {
        $v = Get-EnvVar $name
        if ($v) { Set-Item "Env:$name" $v }
    }
}

# ---------------------------------------------------------------- -Stop：停止全部

if ($Stop) {
    Write-Host '== stop all ==' -ForegroundColor Cyan
    Stop-ByCmd 'server.app'
    Stop-ByCmd '_tunnel.py'
    Stop-ByPort 8001
    Stop-ByPort 8012
    Stop-ByPort 8090
    Write-Host '  local tunnel + app stopped' -ForegroundColor Green
    Write-Host '  stopping cloud MuseTalk engine + CosyVoice TTS ...' -ForegroundColor DarkGray
    Invoke-Cloud 'stop'     | Out-Null
    Invoke-Cloud 'stop_tts' | Out-Null
    Write-Host '  cloud engines stopped' -ForegroundColor Green
    Write-Host 'done.' -ForegroundColor Green
    exit 0
}

if (-not (Test-Path (Join-Path $Root 'server\app.py'))) {
    Write-Host '[x] project root (education/) not found. Run this script under scripts\.' -ForegroundColor Red
    exit 1
}

# ---------------------------------------------------------------- 1/4 云端 MuseTalk 引擎

$cloudOk = $true
$engineState = ''
if (-not $NoEngine) {
    Write-Host '== 1/4 cloud MuseTalk engine (:8012) ==' -ForegroundColor Cyan
    $engineState = Invoke-Cloud 'engine_state'
    if (-not $engineState) {
        $cloudOk = $false
        Write-Host '  [WARN] cloud unreachable (SSH) - skipping cloud engine step' -ForegroundColor Yellow
    } elseif ($engineState -match 'RUNNING') {
        if ($ForceRestart) {
            Write-Host '  running -> force restart (loading may take 1-3 min)'
            Invoke-Cloud 'restart' | Out-Null
        } else {
            Write-Host '  already running -> reuse' -ForegroundColor Green
        }
    } else {
        Write-Host '  not running -> restart (loading may take 1-3 min)'
        Invoke-Cloud 'restart' | Out-Null
    }
} else {
    Write-Host '== 1/4 cloud MuseTalk engine: skipped (-NoEngine) ==' -ForegroundColor DarkGray
}

# ---------------------------------------------------------------- 2/4 云端 CosyVoice TTS

if (-not $NoTts -and $cloudOk) {
    Write-Host '== 2/4 cloud CosyVoice TTS (:8090) ==' -ForegroundColor Cyan
    $ttsState = Invoke-Cloud 'tts_state'
    if ($ttsState -match 'RUNNING') {
        if ($ForceRestart) {
            Write-Host '  running -> force restart (loading may take 1-3 min)'
            (Invoke-Cloud 'restart_tts') | ForEach-Object { if ($_) { Write-Host "  $_" -ForegroundColor DarkGray } }
        } else {
            Write-Host '  already running -> reuse' -ForegroundColor Green
        }
    } else {
        Write-Host '  not running -> restart (loading may take 1-3 min)'
        (Invoke-Cloud 'restart_tts') | ForEach-Object { if ($_) { Write-Host "  $_" -ForegroundColor DarkGray } }
    }
} else {
    Write-Host '== 2/4 cloud CosyVoice TTS: skipped ==' -ForegroundColor DarkGray
}

# ---------------------------------------------------------------- 3/4 本机 SSH 隧道

Write-Host '== 3/4 local SSH tunnel (:8012 + :8090 -> cloud) ==' -ForegroundColor Cyan
$tprocs = Get-Procs '_tunnel.py'
if ($tprocs.Count -eq 1 -and (Test-PortOpen 8012)) {
    Write-Host '  already running -> reuse' -ForegroundColor Green
} else {
    if ($tprocs.Count -gt 0) {
        Write-Host ("  cleaning {0} stale tunnel process(es)" -f $tprocs.Count)
        Stop-ByCmd '_tunnel.py'
    }
    Stop-ByPort 8012
    Stop-ByPort 8090
    Start-Process -FilePath $PyBase -ArgumentList '_tunnel.py' -WorkingDirectory $Root -WindowStyle Hidden `
        -RedirectStandardOutput 'logs\_tunnel_out.log' -RedirectStandardError 'logs\_tunnel_err.log'
    Write-Host '  tunnel started (logs\_tunnel_out.log)'
}

# ---------------------------------------------------------------- 4/4 本机主应用

if (-not $NoApp) {
    Write-Host '== 4/4 local app (:8001) ==' -ForegroundColor Cyan
    $aprocs = Get-Procs 'server.app'
    if ((Test-Health $AppHealth) -and $aprocs.Count -ge 1) {
        Write-Host '  already running -> reuse' -ForegroundColor Green
    } else {
        if ($aprocs.Count -gt 0 -or (Test-PortOpen 8001)) {
            Write-Host '  cleaning stale app process(es)'
            Stop-ByCmd 'server.app'
            Stop-ByPort 8001
            Start-Sleep -Seconds 2
        }
        Start-Process -FilePath $PyApp -ArgumentList '-u', '-m', 'server.app' -WorkingDirectory $Root -WindowStyle Hidden `
            -RedirectStandardOutput 'logs\_server_out.log' -RedirectStandardError 'logs\_server_err.log'
        Write-Host '  app started (logs\_server_out.log)'
    }
} else {
    Write-Host '== 4/4 local app: skipped (-NoApp) ==' -ForegroundColor DarkGray
}

# ---------------------------------------------------------------- 等待就绪

Write-Host ''
Write-Host '== waiting for readiness ==' -ForegroundColor Cyan
if (-not $NoEngine)  { Wait-Health $EngineHealth 'cloud engine (via tunnel)' 360 | Out-Null }
if (-not $NoTts)     { Wait-Health $TtsHealth    'cloud TTS    (via tunnel)' 240 | Out-Null }
if (-not $NoApp)     { Wait-Health $AppHealth    'student app' 120 | Out-Null }

Write-Host ''
Write-Host '================ READY ================' -ForegroundColor Green
Write-Host 'student app:   http://127.0.0.1:8001/'
Write-Host 'engine 8012:   http://127.0.0.1:8012/health'
Write-Host 'tts    8090:   http://127.0.0.1:8090/health'
Write-Host 'stop all:      powershell -ExecutionPolicy Bypass -File scripts\start_cloud.ps1 -Stop'
Write-Host 'logs:          logs\_server_out.log  logs\_tunnel_out.log'
