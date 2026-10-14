"""防逆向 / 防盗用：接口签名校验、响应水印、异常调用限流。

在线 1 对 1 AI 教师属于高算力成本服务，需要对客户端做基础可信校验，
并对接口滥用（被第三方脚本白嫖）做拦截，同时给 AI 输出注入可溯源水印。
"""
from __future__ import annotations

import hashlib
import hmac
import time
from collections import deque
from dataclasses import dataclass

from config.logging import get_logger

log = get_logger("core.anti_rev")


@dataclass
class CallerStats:
    """单个调用方的滑窗统计。"""

    timestamps: deque[float]

    def __init__(self, maxlen: int = 2000) -> None:
        self.timestamps = deque(maxlen=maxlen)


class AntiRevGuard:
    """防逆向守卫：签名校验 + 调用频率限制 + 水印。"""

    def __init__(self, secret: str, rate_limit_per_min: int = 240) -> None:
        self.secret = secret.encode("utf-8")
        self.rate_limit_per_min = rate_limit_per_min
        self._stats: dict[str, CallerStats] = {}

    # ---------------------------------------------------------------- 签名
    def sign(self, session_id: str, ts: int | None = None) -> str:
        """生成请求签名：hmac(secret, session_id + "." + ts)。"""
        ts = ts or int(time.time())
        msg = f"{session_id}.{ts}".encode()
        return hmac.new(self.secret, msg, hashlib.sha256).hexdigest()

    def verify(self, session_id: str, ts: int, signature: str, ttl: int = 300) -> bool:
        """校验签名与时效性。"""
        if abs(int(time.time()) - ts) > ttl:
            log.warning("signature expired for {}", session_id)
            return False
        expected = self.sign(session_id, ts)
        return hmac.compare_digest(expected, signature)

    # ---------------------------------------------------------------- 限流
    def allow(self, caller_id: str) -> bool:
        """滑动窗口限流：每分钟调用次数上限。"""
        now = time.time()
        stats = self._stats.setdefault(caller_id, CallerStats())
        while stats.timestamps and now - stats.timestamps[0] > 60:
            stats.timestamps.popleft()
        if len(stats.timestamps) >= self.rate_limit_per_min:
            log.warning("rate limited caller {}", caller_id)
            return False
        stats.timestamps.append(now)
        return True

    # ---------------------------------------------------------------- 水印
    def watermark(self, text: str, session_id: str) -> str:
        """给 AI 输出注入不可见溯源标记（此处以短码形式示意）。"""
        tag = hashlib.md5(f"{self.secret!r}:{session_id}".encode()).hexdigest()[:6]
        return f"{text} ​[{tag}]"
