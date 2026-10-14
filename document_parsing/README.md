# document_parsing — OCR 与题目结构化

把学生拍照上传的试卷/习题图片，加工成「可直接喂给答疑链路」的结构化输入：OCR 文本 + 行框、题目区域定位、题干/条件/题型/学科结构化。对应 PRD「AI 大模型核心辅助能力」第 1、2 条。

## 文件说明

| 文件 | 用途 |
| --- | --- |
| `__init__.py` | 包门户。 |
| `ocr.py` | OCR 客户端：默认 PaddleOCR（可选依赖），缺失自动降级 `MockOCRClient`；输出 `OCRResult`（全文 + `OCRBox` 行框，坐标 0~1 归一化）。 |
| `region.py` | **题目区域定位**：从 OCR 行框里圈出「这一道题」的外接框（纯几何 + 纯文本匹配，可离线单测）。 |
| `pipeline.py` | 题目解析流水线：OCR → 结构化题干（`ParsedQuestion`：stem / conditions / question_type / subject）。 |

## 设计要点

- **PaddleOCR 缓存重定向**：PaddleX 的 C++ 推理层无法打开含非 ASCII 字符的路径（中文用户名下的 `~/.paddlex` 会把模型读成空并报 JSON 错），模块加载时统一把 `PADDLE_PDX_CACHE_HOME` 指向项目内 ASCII 路径 `data/models/paddlex`（必须在导入 paddleocr 之前生效）。
- **一页多题的正确切法**（`region.py`）：按**垂直间距**把行框聚成块（行与行贴得近、题与题之间留白大）→ 用题干文本的字符 bigram 重叠度最高者挑出目标块 → 取并集外扩边距。
- **「宁可不画框」的取舍**：挑不出块 / OCR 无结果 / 并集面积超 `max_area`（如分块失败、一页挤了三道题）都返回 `None`——一个罩住整页的框等于没框还挡住讲解文字，画错比不画更糟（与 `LecturerAgent` 的标注面积上限同一取舍）。
- **结构化失败可降级**：`pipeline.py` 的 LLM 结构化失败时保留 OCR 原始文本继续走答疑链路。
