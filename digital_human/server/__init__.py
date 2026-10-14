# 数字人服务层（项目自有代码）
# 结构：
#   config.py  - 配置加载（config.yaml + 命令行覆盖）
#   tts.py     - 语音合成（edge-tts）
#   engine.py  - 渲染引擎（MuseTalk 推理 + 待机循环 + 播报任务）
#   webrtc.py  - WebRTC 视频/音频轨道（aiortc）
#   app.py     - HTTP/信令服务（aiohttp）
