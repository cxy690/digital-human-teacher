"""校验一位老师的形象是否配置正确（换完底片/改完教师档案后跑这个）。

检查项（按数据流顺序，从档案一路查到引擎渲染）：

1. **档案**：``teacher.json`` 能否被 ``config.teachers`` 正确解析；
2. **底片规格**：分辨率 / 帧率 / 时长是否符合数字人要求（1280x720 @ 24fps）；
3. **标定**：眼位 / 待机帧是否落在归一化人脸几何上，``blink_src`` 是否指向已不存在的旧素材；
4. **同科老师列表**：该学科下是否出现了不该出现的档案（例如把备份目录放进了 data/teachers/）；
5. **形象缩略图**：能否从底片抽出待机帧；
6. **待机素材**：能否生成 ``video/still/blink`` —— 这一步需要数字人引擎在跑
   （``scripts/start_local.ps1 -Engine musetalk``），没起引擎时会全为 None，
   前端表现为「教师画面空白」。

用法::

    python -m scripts.verify_teacher_avatar english
    python -m scripts.verify_teacher_avatar 英语
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EXPECTED_SIZE = (1280, 720)
EXPECTED_FPS = 24.0
# 归一化后人脸落在同一几何上，因此眼位 y 应该在同一高度（见 add_teacher_variant.py）
EXPECTED_FACE_CX = 0.50
EXPECTED_FACE_CY = 0.374


def check_profile(subject: str) -> bool:
    from config.teachers import default_teacher_key, teacher_for_subject, teacher_variants

    prof = teacher_for_subject(subject)
    print("=== 1. 档案 ===")
    if prof is None:
        print("  [FAIL] 找不到该学科的教师档案（data/teachers/<key>/teacher.json）")
        return False
    print(f"  [OK]   key={prof.key}  name={prof.name}  subject={prof.subject}")
    print(f"         voice={prof.voice_name()}  engine={prof.voice_engine}"
          f"  gender={prof.gender or '(未设置)'}  is_variant={prof.is_variant}")

    print("=== 2. 底片规格 ===")
    talk = prof.talk_path
    if not talk.is_file():
        print(f"  [FAIL] 底片不存在：{talk}")
        return False
    print(f"         路径 {talk}")
    print(f"         大小 {talk.stat().st_size / 1024:.0f}KB")
    try:
        import cv2

        cap = cv2.VideoCapture(str(talk))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        print(f"         {w}x{h} @ {fps:.2f}fps  {n} 帧  {n / max(1.0, fps):.2f}s")
        if (w, h) != EXPECTED_SIZE:
            print(f"  [WARN] 分辨率应为 {EXPECTED_SIZE[0]}x{EXPECTED_SIZE[1]}，实际 {w}x{h}"
                  f"（跑 add_teacher_variant.py 归一化）")
        elif abs(fps - EXPECTED_FPS) > 0.5:
            print(f"  [WARN] 帧率应为 {EXPECTED_FPS:g}，实际 {fps:.2f}（口型会对不上）")
        else:
            print("  [OK]   规格符合数字人要求")
    except Exception as err:  # noqa: BLE001
        print(f"  [WARN] 规格检查失败：{type(err).__name__}: {err}")

    print("=== 3. 标定 ===")
    (e1, e2) = prof.eye_points
    print(f"         眼位 {prof.eye_points_env}  eye_r={prof.eye_r}"
          f"  frame_at={prof.frame_at}s  blink_at={prof.blink_at}")
    if prof.blink_src and prof.blink_src_path is None:
        print(f"  [FAIL] blink_src 指向不存在的文件：{prof.blink_src}"
              f"（换了底片后必须删掉这个字段，否则会贴旧形象的闭眼帧）")
    elif prof.blink_src_path is not None:
        print(f"  [INFO] 使用独立闭眼帧：{prof.blink_src_path.name}")
    else:
        print("  [OK]   无独立闭眼帧 -> 前端走「眼区压合」合成眨眼（几何对齐，不会错位）")
    if abs(e1[1] - e2[1]) > 12:
        print(f"  [WARN] 左右眼高度差 {abs(e1[1] - e2[1])}px 偏大，眨眼/口型可能不协调")
    else:
        print("  [OK]   左右眼高度一致（共享归一化标定生效）")

    print("=== 4. 同科老师列表 ===")
    variants = teacher_variants(subject)
    for v in variants:
        print(f"         {v['key']:<22} {v['display_name']:<16} default={v['is_default']}"
              f"  voice={v['voice']}")
    if len(variants) == 1:
        print("  [INFO] 该学科只有一位老师：前端会跳过「选老师」直接进课堂")
    suspicious = [v["key"] for v in variants if ".bak" in v["key"] or ".orig" in v["key"]]
    if suspicious:
        print(f"  [FAIL] 列表里混入了备份目录：{suspicious}"
              f"（data/teachers/ 下的任何 */teacher.json 都会被当成一位老师，请把备份移出去）")
    print(f"         默认老师 = {default_teacher_key(subject)}")
    return True


async def check_assets(teacher_key: str) -> None:
    from core.avatar_idle import ensure_idle_assets, ensure_teacher_portrait

    print("=== 5. 形象缩略图 ===")
    try:
        url = await ensure_teacher_portrait(teacher_key)
        print(f"  {url or '[FAIL] 生成失败（底片不可读？）'}")
    except Exception as err:  # noqa: BLE001
        print(f"  [FAIL] {type(err).__name__}: {err}")

    print("=== 6. 待机素材（需要引擎在跑）===")
    try:
        assets = await ensure_idle_assets(teacher_key)
    except Exception as err:  # noqa: BLE001
        print(f"  [FAIL] {type(err).__name__}: {err}")
        return
    for k, v in assets.items():
        print(f"  {k:<6} = {v or '(无)'}")
    if not any(assets.values()):
        print("  [WARN] 三个素材全为空：数字人引擎没在跑（8012/8011），"
              "前端会拿不到教师画面 -> 表现为「教师区空白」")
        print("         启动引擎：scripts\\start_local.ps1 -Engine musetalk")

    print("=== 产物 ===")
    stream = ROOT / "data" / "avatar_stream"
    if stream.is_dir():
        for f in sorted(stream.glob(f"*{teacher_key}*")):
            dims = ""
            try:
                from PIL import Image

                if f.suffix.lower() == ".png":
                    with Image.open(f) as im:
                        dims = f"  {im.size[0]}x{im.size[1]}"
            except Exception:  # noqa: BLE001
                pass
            print(f"  {f.name:<36}{f.stat().st_size / 1024:>9.0f}KB{dims}")


def main() -> int:
    arg = (sys.argv[1] if len(sys.argv) > 1 else "english").strip()
    from config.teachers import teacher_for_subject

    prof = teacher_for_subject(arg)
    if prof is None:
        print(f"找不到教师档案：{arg}")
        return 2
    print(f"######## 校验老师：{prof.key}（{prof.subject}）########\n")
    if not check_profile(arg):
        return 2
    asyncio.run(check_assets(prof.key))
    print("\n[完成] 第 6 项若为空，先启动数字人引擎再跑一次。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
