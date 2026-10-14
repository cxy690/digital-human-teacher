<#
.SYNOPSIS
    用本机 PowerPoint 把 .pptx 逐页导出成 p1.png / p2.png …（课件页背景图）。

.DESCRIPTION
    课件（PPT）讲解是「页图 + 上层标注」模型：
      · 页图（Slide.image）决定学生看到什么；
      · 元素框（SlideElement）决定 AI 圈哪里（由 python-pptx 解析）。
    python-pptx 只能读元素框、不能渲染画面，所以页图必须预先导出。
    core/courseware/pptx_source.py 只认 p1.png / p-1.png / slide1.png / 1.png
    这几种命名，本脚本统一输出 p{i}.png。

    为什么用 PowerPoint COM 而不是 LibreOffice：本机已装 Office，无需额外下载
    数 GB 的 LibreOffice；PowerPoint 渲染结果也最贴近「所见即所得」。

.PARAMETER Pptx
    源 .pptx 的绝对路径。

.PARAMETER OutDir
    输出目录。约定为 data/courseware/<deck_id>/，且 <deck_id> 必须等于 pptx 文件名
    （loader.py 用 deck_id 拼页图 URL，否则页图会 404）。

.PARAMETER Width / Height
    导出分辨率，默认 1600x900（16:9）。与 PPT 页宽高比不一致时，PowerPoint 会等比
    留白，不影响标注（元素框是 0~1 归一化的）。

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\export_pptx_images.ps1 `
      -Pptx "c:\Users\me\Desktop\有的人.pptx" `
      -OutDir "c:\Users\me\Desktop\education\data\courseware\有的人"
#>
param(
    [Parameter(Mandatory = $true)][string]$Pptx,
    [Parameter(Mandatory = $true)][string]$OutDir,
    [int]$Width = 1600,
    [int]$Height = 900
)

$ErrorActionPreference = 'Stop'

if (-not (Test-Path -LiteralPath $Pptx)) { throw "找不到 PPT：$Pptx" }
$Pptx = (Resolve-Path -LiteralPath $Pptx).Path
if (-not (Test-Path -LiteralPath $OutDir)) { New-Item -ItemType Directory -Force -Path $OutDir | Out-Null }
$OutDir = (Resolve-Path -LiteralPath $OutDir).Path

Write-Host "[1/3] 启动 PowerPoint ..."
$app = New-Object -ComObject PowerPoint.Application

$pres = $null
try {
    # Open(FileName, ReadOnly, Untitled, WithWindow)
    #   ReadOnly = msoTrue(-1)  : 只读打开，绝不回写源文件
    #   Untitled = msoFalse(0)  : 保持原文件名
    #   WithWindow = msoFalse(0): 无窗口后台打开，避免抢焦点
    Write-Host "[2/3] 打开：$Pptx"
    $pres = $app.Presentations.Open($Pptx, -1, 0, 0)

    $count = $pres.Slides.Count
    Write-Host "      共 $count 页，开始导出 $Width x $Height PNG ..."

    for ($i = 1; $i -le $count; $i++) {
        $out = Join-Path $OutDir ("p{0}.png" -f $i)
        # Slide.Export(FileName, FilterName, ScaleWidth, ScaleHeight)
        $pres.Slides.Item($i).Export($out, "PNG", $Width, $Height)
        Write-Host ("      -> p{0}.png" -f $i)
    }

    Write-Host "[3/3] 完成，共导出 $count 张到：$OutDir"
}
finally {
    if ($pres) { try { $pres.Close() } catch {} }
    try { $app.Quit() } catch {}
    # 释放 COM 引用，否则 POWERPNT.EXE 会残留在后台
    foreach ($o in @($pres, $app)) {
        if ($o) { try { [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($o) } catch {} }
    }
    [GC]::Collect(); [GC]::WaitForPendingFinalizers()
}
