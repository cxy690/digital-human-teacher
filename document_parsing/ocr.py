"""OCR 客户端：识别学生上传的试卷/习题/作业图片，提取题干文字与定位框。

默认实现基于 PaddleOCR（可选依赖）；缺失时降级为 Mock，保证链路可跑。
"""
from __future__ import annotations

import asyncio
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config.logging import get_logger
from config.settings import settings

log = get_logger("document_parsing.ocr")

# PaddleX 的 C++ 推理层无法打开含非 ASCII 字符的模型路径（如中文用户名下的 ~/.paddlex），
# 会把模型文件读成空内容并报 JSON 解析错误；这里统一把模型缓存重定向到项目内的 ASCII 路径。
# 必须在导入 paddleocr/paddlex 之前设置（导入是延迟的，放在模块加载时即可生效）。
os.environ.setdefault(
    "PADDLE_PDX_CACHE_HOME",
    str(Path(__file__).resolve().parents[1] / "data" / "models" / "paddlex"),
)


@dataclass
class OCRBox:
    """一行/一块文字及其归一化外接框。"""

    text: str
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2（0~1 归一化）
    confidence: float = 1.0


@dataclass
class OCRResult:
    text: str
    boxes: list[OCRBox] = field(default_factory=list)


class OCRClient:
    """OCR 客户端协议（duck typing）。"""

    async def recognize(self, image: bytes) -> OCRResult:  # pragma: no cover - 接口
        raise NotImplementedError


class PaddleOCRClient(OCRClient):
    """基于 PaddleOCR 的本地识别（兼容 3.x 与 2.x 两代 API）。"""

    def __init__(self, lang: str = "ch") -> None:
        from paddleocr import PaddleOCR  # noqa: PLC0415 - 可选依赖

        # enable_mkldnn=False：规避 paddlepaddle 3.3 的 PIR+oneDNN 指令转换 bug
        # （文本检测推理时报 ConvertPirAttribute2RuntimeAttribute not support）
        try:
            # PaddleOCR 3.x：use_angle_cls 更名为 use_textline_orientation，且不再有 show_log 参数
            self._ocr = PaddleOCR(lang=lang, use_textline_orientation=True, enable_mkldnn=False)
        except Exception:  # noqa: BLE001 - 兼容旧版参数名
            try:
                self._ocr = PaddleOCR(  # 2.x
                    use_angle_cls=True, lang=lang, show_log=False, enable_mkldnn=False
                )
            except Exception:  # noqa: BLE001
                self._ocr = PaddleOCR(lang=lang)  # 兜底：仅传两代共有参数
        self._legacy = not hasattr(self._ocr, "predict")

    async def recognize(self, image: bytes) -> OCRResult:
        return await asyncio.to_thread(self._recognize_sync, image)

    def _recognize_sync(self, image: bytes) -> OCRResult:
        import io  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415

        # PaddleX 管道按 BGR 约定接收 ndarray
        arr = np.asarray(Image.open(io.BytesIO(image)).convert("RGB"))[:, :, ::-1]
        h, w = arr.shape[:2]
        raw = self._ocr.ocr(arr, cls=True) if self._legacy else self._ocr.predict(arr)

        boxes: list[OCRBox] = []
        lines: list[str] = []
        for page in raw or []:
            entries: list[tuple[str, float, Any]] = []
            if hasattr(page, "get"):  # 3.x：OCRResult（dict 风格，含 rec_texts / rec_scores / rec_polys）
                texts = page.get("rec_texts") or []
                scores = page.get("rec_scores") or []
                polys = page.get("rec_polys")
                if polys is None:
                    polys = page.get("dt_polys") or []
                entries = [(str(t), float(s), poly) for t, s, poly in zip(texts, scores, polys)]
            else:  # 2.x：[[polygon, (text, conf)], ...]
                for item in page or []:
                    polygon, (text, conf) = item[0], item[1]
                    entries.append((str(text), float(conf), polygon))
            for text, conf, polygon in entries:
                xs = [float(p[0]) for p in polygon]
                ys = [float(p[1]) for p in polygon]
                boxes.append(
                    OCRBox(
                        text=text,
                        bbox=(min(xs) / w, min(ys) / h, max(xs) / w, max(ys) / h),
                        confidence=conf,
                    )
                )
                lines.append(text)
        return OCRResult(text="\n".join(lines), boxes=boxes)


class MockOCRClient(OCRClient):
    """离线 Mock：返回一段样例题干。"""

    async def recognize(self, image: bytes) -> OCRResult:
        text = "已知一个长方形的长为 8 厘米，宽为 5 厘米，求它的面积是多少平方厘米？"
        return OCRResult(
            text=text,
            boxes=[OCRBox(text=text, bbox=(0.08, 0.10, 0.92, 0.32), confidence=0.99)],
        )


_ocr_client: OCRClient | None = None
_ocr_lock = threading.Lock()


def get_ocr() -> OCRClient:
    """获取 OCR 客户端（进程级单例，线程安全）。

    PaddleOCR 不能并发初始化：预热线程与首个会话可能同时触发，
    原生层并发初始化会死锁，这里用锁串行化，保证只初始化一次。
    """
    global _ocr_client
    if _ocr_client is not None:
        return _ocr_client
    with _ocr_lock:
        if _ocr_client is None:
            client: OCRClient | None = None
            if settings.ocr_provider == "paddle":
                try:
                    client = PaddleOCRClient()
                except Exception as err:  # noqa: BLE001 - 可选依赖缺失
                    log.warning("paddleocr unavailable ({}), fallback to MockOCR", err)
            _ocr_client = client or MockOCRClient()
    return _ocr_client
