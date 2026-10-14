"""弹性能力：重试、超时、熔断、降级，保证实时交互不因单点抖动卡死。

LLM / ASR / TTS / OCR 都是外部依赖，任何一次慢调用都直接影响答疑体验，
因此统一在这里收敛重试与降级策略。
"""
from __future__ import annotations

import asyncio
import functools
import time
from typing import Any, Awaitable, Callable, TypeVar

from config.logging import get_logger

log = get_logger("core.resilience")

T = TypeVar("T")


class CircuitBreaker:
    """简易熔断器：连续失败达到阈值后开路，冷却后半开探活。"""

    def __init__(self, fail_threshold: int = 5, reset_timeout: float = 20.0) -> None:
        self.fail_threshold = fail_threshold
        self.reset_timeout = reset_timeout
        self._fail_count = 0
        self._opened_at: float = 0.0

    @property
    def is_open(self) -> bool:
        if self._opened_at == 0.0:
            return False
        if time.time() - self._opened_at >= self.reset_timeout:
            # 冷却结束，进入半开
            self._opened_at = 0.0
            self._fail_count = 0
            return False
        return True

    def on_success(self) -> None:
        self._fail_count = 0
        self._opened_at = 0.0

    def on_failure(self) -> None:
        self._fail_count += 1
        if self._fail_count >= self.fail_threshold:
            self._opened_at = time.time()
            log.warning("circuit breaker opened after {} failures", self._fail_count)


async def with_timeout(coro: Awaitable[T], seconds: float, default: T) -> T:
    """超时保护：超时返回默认值，而不是阻塞实时链路。"""
    try:
        return await asyncio.wait_for(coro, timeout=seconds)
    except (asyncio.TimeoutError, TimeoutError):
        log.warning("call timed out after {}s, fallback to default", seconds)
        return default


def async_retry(
    attempts: int = 3,
    base_delay: float = 0.2,
    breaker: CircuitBreaker | None = None,
):
    """异步重试装饰器，指数退避；可选熔断。"""

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            if breaker is not None and breaker.is_open:
                raise RuntimeError("circuit_open")
            last_err: Exception | None = None
            for i in range(attempts):
                try:
                    result = await func(*args, **kwargs)
                    if breaker is not None:
                        breaker.on_success()
                    return result
                except Exception as err:  # noqa: BLE001 - 统一收敛后重试
                    last_err = err
                    if breaker is not None:
                        breaker.on_failure()
                    if i < attempts - 1:
                        await asyncio.sleep(base_delay * (2**i))
            assert last_err is not None
            raise last_err

        return wrapper

    return decorator


async def first_success(*coros: Awaitable[T]) -> T:
    """降级链：按顺序尝试，返回第一个成功结果。"""
    last_err: Exception | None = None
    for coro in coros:
        try:
            return await coro
        except Exception as err:  # noqa: BLE001
            last_err = err
    if last_err is not None:
        raise last_err
    raise RuntimeError("no candidates provided")
