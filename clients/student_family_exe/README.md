# student_family_exe（家庭版学生端 · 桌面客户端）

面向家庭 PC 的学生端桌面应用（Windows/macOS），基于 `clients/student_tablet`
的同一套前端逻辑封装为可执行程序。

- 打包建议：Electron / Tauri / PyWebView + PyInstaller；
- 启动后自动打开本地学生端页面并连接 `server` 的 WebSocket；
- 负责摄像头/麦克风权限申请与音视频设备枚举。

> 纯 AI 教师模式下仅保留学生端，前端页面见 `clients/student_tablet/index.html`。
