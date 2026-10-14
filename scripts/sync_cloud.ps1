<#
  sync_cloud.ps1 - 一键把本地项目同步到算力云（增量，只增不删）

  用法：
    powershell -ExecutionPolicy Bypass -File scripts\sync_cloud.ps1          # 增量同步
    powershell -ExecutionPolicy Bypass -File scripts\sync_cloud.ps1 -Dry     # 只预览
    powershell -ExecutionPolicy Bypass -File scripts\sync_cloud.ps1 -Force   # 全量重传

  说明：调用 scripts/sync_cloud.py（SFTP 增量上传）。
        跳过 __pycache__ / .venvs / 日志 / data 缓存 / .env 等，不会删除云端文件。
        连接信息可用环境变量覆盖：CLOUD_HOST / CLOUD_PORT / CLOUD_USER / CLOUD_PWD / CLOUD_ROOT
        python 路径可用 EDU_PY_BASE 覆盖（需含 paramiko）。
#>
param(
    [switch]$Dry,
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$PyBase = if ($env:EDU_PY_BASE) { $env:EDU_PY_BASE } else { 'D:\An\python.exe' }
if (-not (Test-Path $PyBase)) {
    Write-Host "[x] 基础 python 不存在：$PyBase （可用 EDU_PY_BASE 覆盖）" -ForegroundColor Red
    exit 1
}

$argv = @((Join-Path $Root 'scripts\sync_cloud.py'))
if ($Dry)   { $argv += '--dry' }
if ($Force) { $argv += '--force' }

& $PyBase @argv
