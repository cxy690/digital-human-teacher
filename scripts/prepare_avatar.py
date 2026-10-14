"""一次性工具：裁掉生成图的水印、输出 3:4 标准形象图，并绘制口型/眨眼锚点标尺图用于校准。

用法：
    python scripts/prepare_avatar.py <源图> [--save clients/student_tablet/ai_teacher2.png]
输出：形象图 + 同目录 *_anchors.png 校准预览图（画出口型/眼睛锚点椭圆）。
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

# 当前 CSS 中的锚点（与 index.html 的 mask-image / transform-origin 对应）
MOUTH = (50.0, 48.2, 8.0, 5.0)      # cx%, cy%, rx%, ry%
EYE_L = (41.6, 33.2, 4.2, 2.8)
EYE_R = (58.4, 33.2, 4.2, 2.8)


def crop_watermark(img: Image.Image, bottom_px: int = 72) -> Image.Image:
    """裁掉底部水印条，并同步裁掉两侧以保持 3:4 比例。"""
    w, h = img.size
    new_h = h - bottom_px
    new_w = int(round(new_h * 3 / 4))
    left = max(0, (w - new_w) // 2)
    return img.crop((left, 0, left + new_w, new_h))


def draw_anchors(path: Path, targets: list[tuple[float, float, float, float]]) -> Path:
    img = Image.open(path).convert("RGB")
    d = ImageDraw.Draw(img)
    w, h = img.size
    for cx_pct, cy_pct, rx_pct, ry_pct in targets:
        cx, cy = cx_pct / 100 * w, cy_pct / 100 * h
        rx, ry = rx_pct / 100 * w, ry_pct / 100 * h
        d.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], outline=(255, 0, 0), width=3)
        d.line([cx - 12, cy, cx + 12, cy], fill=(255, 0, 0), width=2)
        d.line([cx, cy - 12, cx, cy + 12], fill=(255, 0, 0), width=2)
    out = path.with_name(path.stem + "_anchors.png")
    img.save(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--save", default="clients/student_tablet/ai_teacher2.png")
    ap.add_argument("--bottom", type=int, default=72)
    args = ap.parse_args()

    src = Image.open(args.source).convert("RGB")
    cropped = crop_watermark(src, args.bottom)
    out = Path(args.save)
    out.parent.mkdir(parents=True, exist_ok=True)
    cropped.save(out)
    print(f"saved {out} size={cropped.size} aspect={cropped.size[0] / cropped.size[1]:.3f}")
    preview = draw_anchors(out, [MOUTH, EYE_L, EYE_R])
    print(f"anchor preview -> {preview}")


if __name__ == "__main__":
    main()
