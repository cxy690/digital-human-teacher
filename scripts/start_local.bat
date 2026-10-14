@echo off
rem 本地一键启动（双击本文件即可）。等价于：
rem   powershell -ExecutionPolicy Bypass -File scripts\start_local.ps1
powershell -ExecutionPolicy Bypass -File "%~dp0start_local.ps1" %*
pause
