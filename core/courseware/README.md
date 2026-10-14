# core/courseware — 课件（PPT 讲解）

课件的数据模型、仓库加载与真实 `.pptx` 解析。对外只需 `get_store()`——讲解 / 标注逻辑不感知课件来自 JSON 还是真实 .pptx。

## 文件说明

| 文件 | 用途 |
| --- | --- |
| `__init__.py` | 包门户：数据模型 + 仓库加载 + .pptx 解析。 |
| `models.py` | 数据模型 `SlideDeck → Slide → SlideElement`；`match_element` 按「完全相等→包含→双向包含」取首个命中。 |
| `loader.py` | 扫描 `data/courseware/` 提供课件清单与取页服务（「一个子目录 = 一份课件」，`deck.json` 即序列化格式）；`_safe_name()` 去路径穿越字符。 |
| `pptx_source.py` | 真实 `.pptx` 解析成 `SlideDeck`：页面图路径 + 元素框 + 文本提取。 |

## 设计要点

- **坐标全部 0~1 归一化**：python-pptx 的 EMU 除以页宽高即可落位，业务层零换算；`_clamp01()` 与白板 validator 判据一致。
- **「页面图决定学生看到什么，元素框决定 AI 圈哪里」**：用导出的页图做白板背景（所见即 PPT 原样），元素框只作为标注落点；LLM 引用元素 id/文字，不让模型凭空猜坐标。
- **元素框按文字宽度收窄**：按全角 1em / 半角 0.55em 估算，避免 PPT 文本框远宽于文字导致「框比字大一圈」。
- **页图来源**：`data/courseware/<deck_id>/p{i}.png`，由 `scripts/export_pptx_images.ps1` 用本机 PowerPoint COM 导出（`pptx_source.py` 只认这几种命名）。
