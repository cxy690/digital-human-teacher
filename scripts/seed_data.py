"""数据种子脚本：确保 data/ 目录结构齐全（示例内容已随仓库提供）。

运行： python -m scripts.seed_data
"""
from __future__ import annotations

from pathlib import Path

_DIRS = {
    "data/textbook": ["math_primary.txt"],
    "data/curriculum": ["math_grade5.txt"],
    "data/question_bank": ["math_samples.txt"],
    "data/teaching_materials": ["notes.md"],
    "data/error_patterns": ["common_errors.jsonl"],
    "data/wrong_book": [],
    "data/raw": [],
}


def main() -> None:
    for d, names in _DIRS.items():
        path = Path(d)
        path.mkdir(parents=True, exist_ok=True)
        if not names and not any(path.iterdir()):
            (path / ".gitkeep").write_text("# 运行时数据目录\n", encoding="utf-8")
        print(f"[ok] {d} (示例文件 {len(names)} 个)")
    print("数据目录已就绪。")


if __name__ == "__main__":
    main()
