"""document_parsing 包：OCR 识别与题目结构化。"""
from document_parsing.ocr import (
    MockOCRClient,
    OCRBox,
    OCRClient,
    OCRResult,
    PaddleOCRClient,
    get_ocr,
)
from document_parsing.pipeline import ParsedQuestion, QuestionParser

__all__ = [
    "MockOCRClient",
    "OCRBox",
    "OCRClient",
    "OCRResult",
    "PaddleOCRClient",
    "get_ocr",
    "ParsedQuestion",
    "QuestionParser",
]
