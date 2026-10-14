<#
  start_all.ps1 - one-click start on the GPU machine:
    (1) digital-human render engine (Wav2Lip) listening on 0.0.0.0 (LAN reachable)
    (2) student app (FastAPI, port from .env PORT)
  and prints the LAN IP + the config teammates should put in their .env.

  Usage (from the project root education/):
    powershell -ExecutionPolicy Bypass -File scripts\start_all.ps1

  Optional overrides (env vars):
    WAV2LIP_PY   default D:\An\envs\wav2lip\python.exe   (engine python)
    TEA_PY       default D:\An\envs\tea\python.exe        (student app python)
    WAV2LIP_DIR  default D:\wav2lip                       (Wav2Lip code+weights)
    ENGINE_PORT  default 8011                             (render engine port)
#>
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path (Join-Path $Root "server\app.py"))) {
    Write-Host "[x] project root (education/) not found. Keep this script under scripts\ and run it." -ForegroundColor Red
    exit 1
}

$Wav2lipPy  = if ($env:WAV2LIP_PY)  { $env:WAV2LIP_PY }  else { "D:\An\envs\wav2lip\python.exe" }
$TeaPy      = if ($env:TEA_PY)      { $env:TEA_PY }      else { "D:\An\envs\tea\python.exe" }
$Wav2lipDir = if ($env:WAV2LIP_DIR) { $env:WAV2LIP_DIR } else { "D:\wav2lip" }
$EnginePort = if ($env:ENGINE_PORT) { $env:ENGINE_PORT } else { "8011" }

if (-not (Test-Path $Wav2lipPy)) { Write-Host "[x] engine python not found: $Wav2lipPy  (set env WAV2LIP_PY)" -ForegroundColor Red; exit 1 }
if (-not (Test-Path $TeaPy))     { Write-Host "[x] app python not found: $TeaPy  (set env TEA_PY)" -ForegroundColor Red; exit 1 }

# Local LAN IPv4 candidates (the address teammates will use)
$ips = @(Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
        Where-Object { $_.IPAddress -notlike "127.*" -and $_.IPAddress -notlike "169.254.*" } |
        Select-Object -ExpandProperty IPAddress)
if (-not $ips.Count) {
    $ips = @([System.Net.Dns]::GetHostAddresses([System.Net.Dns]::GetHostName()) |
          Where-Object { $_.AddressFamily -eq 'InterNetwork' -and $_.IPAddressToString -notlike '127.*' } |
          Select-Object -ExpandProperty IPAddressToString)
}
$ip = if ($ips.Count) { $ips[0] } else { "<YOUR-LAN-IP>" }

# --- 1) render engine, exposed on 0.0.0.0 ---
$env:WAV2LIP_DIR = $Wav2lipDir
$env:ENGINE_HOST = "0.0.0.0"
$env:ENGINE_PORT = "$EnginePort"
Write-Host "[1/2] starting render engine on 0.0.0.0:$EnginePort ..." -ForegroundColor Cyan
Start-Process -FilePath $Wav2lipPy -ArgumentList "scripts\avatar_engine_wav2lip.py" -WorkingDirectory $Root

# --- 2) student app ---
Remove-Item Env:ENGINE_HOST -ErrorAction SilentlyContinue
Write-Host "[2/2] starting student app ..." -ForegroundColor Cyan
Start-Process -FilePath $TeaPy -ArgumentList "-m", "server.app" -WorkingDirectory $Root

Start-Sleep -Seconds 3
Write-Host ""
Write-Host "================ started ================" -ForegroundColor Green
Write-Host ("Engine  (this machine):   http://127.0.0.1:{0}" -f $EnginePort)
Write-Host  "Student (this machine):  http://127.0.0.1:8001   (port = .env PORT)"
Write-Host ""
Write-Host "==== teammates: put these in their .env, then restart their app ====" -ForegroundColor Yellow
Write-Host  "AVATAR_BACKEND=http"
Write-Host ("AVATAR_API_BASE=http://{0}:{1}" -f $ip, $EnginePort)
if ($ips.Count -gt 1) { Write-Host ("  (detected IPs: " + ($ips -join ", ") + ") - pick the one teammates can ping") -ForegroundColor DarkGray }
Write-Host ""
Write-Host ("Teammate check: open  http://{0}:{1}/health  (expect: {{""status"":""ok""...}})" -f $ip, $EnginePort)
Write-Host "Allow through firewall (run once in an ADMIN PowerShell):" -ForegroundColor Yellow
Write-Host ("  New-NetFirewallRule -DisplayName 'AI Teacher Engine {0}' -Direction Inbound -LocalPort {0} -Protocol TCP -Action Allow" -f $EnginePort)
