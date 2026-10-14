# local_guard（本地守护 / 防逆向客户端）

与主服务配合的本地守护进程，职责：

- 校验客户端签名（HMAC）与会话时效，拦截被二次分发的白嫖脚本；
- 上报设备指纹、录屏/多开等异常信号，供 `core/anti_rev.py` 决策；
- 为离线场景提供本地缓存与断线重连。

> 说明：本目录为打包/运行占位目录。实现建议用 Python（PyInstaller）或
> 原生壳（Electron/Tauri）封装，通过 HTTPS + 签名与 `server` 通信。
