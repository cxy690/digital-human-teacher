"""题目解析流水线：OCR -> 结构化题干（题型 / 条件 / 学科）。

对应 PRD「AI 大模型核心辅助能力」第 1、2 条：OCR 提取题干与条件，
并按学科生成标准化解析所需的结构化输入。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config.logging import get_logger
from document_parsing.ocr import OCRBox, OCRClient, get_ocr
from document_parsing.region import DEFAULT_MAX_AREA, BBox, question_region
from models.base import BaseLLMClient
from models.llm import get_llm

log = get_logger("document_parsing.pipeline")

_STRUCT_INSTRUCTION = """你是题目结构化助手。请把下面 OCR 文本解析为 JSON：
{{"stem": "题干", "conditions": ["已知条件"], "question_type": "题型", "subject": "学科"}}
只输出 JSON。"""


@dataclass
class ParsedQuestion:
    """结构化后的题目。"""

    raw_text: str
    stem: str = ""
    conditions: list[str] = field(default_factory=list)
    question_type: str = "简答题"
    subject: str = "综合"
    boxes: list[OCRBox] = field(default_factory=list)

    @property
    def content(self) -> str:
        parts = [self.stem] + [f"已知：{c}" for c in self.conditions]
        return "\n".join(p for p in parts if p)

    def question_region(self, *, max_area: float = DEFAULT_MAX_AREA) -> BBox | None:
        """题目区域外接框（归一化 0~1）；定位不出来时返回 ``None``。

        拍照讲题时用它圈出「这一道题」（详见 :mod:`document_parsing.region`）——
        整页多题的照片不会再被罩上一个大框。优先用结构化后的题干挑块，
        未结构化时退化为 OCR 原文。
        """
        return question_region(self.boxes, self.stem or self.raw_text, max_area=max_area)


class QuestionParser:
    """把图片转成结构化题目。"""

    def __init__(self, ocr: OCRClient | None = None, llm: BaseLLMClient | None = None) -> None:
        self.ocr = ocr or get_ocr()
        self.llm = llm or get_llm()

    async def parse(self, image: bytes) -> ParsedQuestion:
        try:
            ocr_result = await self.ocr.recognize(image)
        except Exception as err:  # noqa: BLE001 - 图片损坏/OCR 失败不应中断会话
            log.warning("ocr recognize failed: {}", err)
            return ParsedQuestion(raw_text="")
        parsed = ParsedQuestion(raw_text=ocr_result.text, stem=ocr_result.text, boxes=ocr_result.boxes)
        if not ocr_result.text.strip():
            return parsed

        data = await self._structure(ocr_result.text)
        if data:
            parsed.stem = str(data.get("stem") or ocr_result.text)
            parsed.conditions = [str(c) for c in (data.get("conditions") or [])]
            parsed.question_type = str(data.get("question_type") or parsed.question_type)
            parsed.subject = str(data.get("subject") or parsed.subject)
        return parsed

    async def _structure(self, text: str) -> dict[str, Any]:
        try:
            return await self.llm.chat_json(
                [
                    {"role": "system", "content": _STRUCT_INSTRUCTION},
                    {"role": "user", "content": text},
                ]
            )
        except Exception as err:  # noqa: BLE001 - 结构化失败不影响主流程
            log.warning("structure parse failed: {}", err)
            return {}
