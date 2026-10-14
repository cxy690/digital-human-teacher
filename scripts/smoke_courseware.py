"""课件（PPT）端到端冒烟测试。

覆盖「上传 PPT -> 白板呈现 -> AI 讲解圈画」这条链路的关键环节，
不依赖浏览器：

1. ``GET /api/courseware``        课件清单里能看到目标课件；
2. ``GET /api/courseware/<id>``   每页都有页图 URL 与元素框；
3. ``GET <页图 URL>``             页图能真的取到（中文路径也要能取）；
4. ``WS /ws``                     ``courseware`` 指令后，服务端确实下发
                                  ``slide``（换页）+ ``slide_state``（页码栏）
                                  + 文字/标注指令。

用法::

    python -m server.app                      # 先起服务（默认 8001）
    python -m scripts.smoke_courseware        # 测全部课件
    python -m scripts.smoke_courseware 有的人   # 只测指定 deck_id
"""
from __future__ import annotations

import asyncio
import json
import sys
import urllib.parse
import urllib.request
from typing import Any

BASE = "http://127.0.0.1:8001"
WS_URL = "ws://127.0.0.1:8001/ws"

_ok_count = 0
_fail_count = 0


def _check(name: str, passed: bool, detail: str = "") -> None:
    global _ok_count, _fail_count
    mark = "PASS" if passed else "FAIL"
    if passed:
        _ok_count += 1
    else:
        _fail_count += 1
    print(f"[{mark}] {name}" + (f"  -> {detail}" if detail else ""))


def _get(path: str) -> tuple[int, str, bytes]:
    with urllib.request.urlopen(BASE + path, timeout=15) as resp:
        return resp.status, resp.headers.get("Content-Type", ""), resp.read()


def check_http(deck_id: str) -> dict[str, Any] | None:
    """校验清单 / 详情 / 页图三件事，返回课件详情。"""
    try:
        _, _, body = _get("/api/courseware?reload=true")
        decks = json.loads(body).get("decks") or []
    except Exception as err:  # noqa: BLE001 - 冒烟脚本要给出明确原因而不是堆栈
        _check("GET /api/courseware", False, f"{type(err).__name__}: {err}")
        return None

    ids = [d.get("deck_id") for d in decks]
    _check("GET /api/courseware 含目标课件", deck_id in ids, f"现有课件 {ids}")
    if deck_id not in ids:
        return None

    try:
        _, _, body = _get("/api/courseware/" + urllib.parse.quote(deck_id))
        deck = json.loads(body)["deck"]
    except Exception as err:  # noqa: BLE001
        _check("GET /api/courseware/<id>", False, f"{type(err).__name__}: {err}")
        return None

    slides = deck.get("slides") or []
    with_img = [s for s in slides if s.get("image_url")]
    _check(
        "课件详情：每页都有页图 URL",
        len(with_img) == len(slides) and bool(slides),
        f"{len(with_img)}/{len(slides)} 页有图",
    )
    _check(
        "课件详情：页图坐标已归一化（非全 0）",
        any(
            any((el.get("width") or 0) > 0 for el in (s.get("elements") or []))
            for s in slides
        ),
        "若全为 0，多半是 EMU/px 量纲搞混（见 pptx_source.py 注释）",
    )

    # 只抽查首/中/末页，避免 39 页全下造成无谓等待
    probes = sorted({0, len(slides) // 2, len(slides) - 1})
    for i in probes:
        url = slides[i].get("image_url")
        try:
            status, ctype, blob = _get(url)
            _check(
                f"页图可访问 第{i + 1}页",
                status == 200 and ctype.startswith("image/"),
                f"{status} {ctype} {len(blob) // 1024}KB {url}",
            )
        except Exception as err:  # noqa: BLE001
            _check(f"页图可访问 第{i + 1}页", False, f"{type(err).__name__}: {err}")
    return deck


async def check_ws(deck_id: str, timeout: float = 30.0) -> None:
    """通过 WebSocket 真实打开课件，确认服务端下发了课件指令。"""
    try:
        import websockets  # type: ignore[import-not-found]
    except ImportError:
        _check("WebSocket 测试", False, "未安装 websockets（pip install websockets），已跳过")
        return

    got: dict[str, Any] = {"ops": [], "slide_state": None, "speech": ""}
    try:
        async with websockets.connect(WS_URL, max_size=None) as ws:
            await ws.send(json.dumps({"type": "init", "student_id": "smoke-courseware"}))
            await ws.send(json.dumps({"type": "courseware", "deck_id": deck_id, "index": 0}))
            while True:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
                except asyncio.TimeoutError:
                    break
                msg = json.loads(raw)
                kind = msg.get("type")
                payload = msg.get("payload")
                if kind == "board_ops":
                    got["ops"].extend(payload or [])
                elif kind == "slide_state":
                    got["slide_state"] = payload
                elif kind in ("speech", "subtitle", "ai_speech") and not got["speech"]:
                    got["speech"] = payload if isinstance(payload, str) else json.dumps(
                        payload, ensure_ascii=False
                    )
                ops = got["ops"]
                if (
                    any(o.get("op") == "slide" for o in ops)
                    and got["slide_state"]
                    and any(o.get("op") == "text" for o in ops)
                ):
                    break  # 换页 + 页码栏 + 讲解文字都到齐，链路已证明可用
    except Exception as err:  # noqa: BLE001
        _check("WebSocket 打开课件", False, f"{type(err).__name__}: {err}")
        return

    ops = got["ops"]
    slide_ops = [o for o in ops if o.get("op") == "slide"]
    text_ops = [o for o in ops if o.get("op") == "text"]
    mark_ops = [o for o in ops if o.get("op") in ("annotate", "highlight")]

    _check("WS 下发 slide 换页指令", bool(slide_ops),
           (slide_ops[0].get("meta", {}) or {}).get("image_url", "") if slide_ops else "未收到")
    _check("WS 下发 slide_state（页码栏）", bool(got["slide_state"]),
           json.dumps(got["slide_state"], ensure_ascii=False) if got["slide_state"] else "未收到")
    _check("WS 下发讲解文字", bool(text_ops), f"{len(text_ops)} 行")
    _check("WS 下发页内标注", bool(mark_ops),
           f"{len(mark_ops)} 处（LLM 未给 annotations 时为 0 也属预期）")
    # 前端靠 op.deck_id 判断「标注坐标是相对 PPT 页还是整块白板」，
    # 一旦缺失，课件页上的红框会整体偏移（见 student_tablet 的 opRect）。
    _check("页内标注带 deck_id（前端对齐参照系）",
           all(o.get("deck_id") for o in mark_ops) if mark_ops else True,
           "缺失会导致课件页标注偏移")
    if got["speech"]:
        print(f"\n讲解话术：{got['speech'][:200]}")


def main() -> int:
    deck_arg = sys.argv[1] if len(sys.argv) > 1 else ""
    deck_id = deck_arg.strip()
    if not deck_id:
        try:
            _, _, body = _get("/api/courseware?reload=true")
            decks = json.loads(body).get("decks") or []
            pptx_decks = [d for d in decks if d.get("source") == "pptx"] or decks
            if not pptx_decks:
                print("课件仓库为空：先把 PPT 放进 data/courseware/<deck_id>/ 再跑")
                return 1
            deck_id = pptx_decks[0]["deck_id"]
        except Exception as err:  # noqa: BLE001
            print(f"服务不可达（{BASE}）：{type(err).__name__}: {err}")
            print("请先启动：python -m server.app")
            return 1

    print(f"=== 课件冒烟测试：{deck_id} ===\n")
    if check_http(deck_id) is None:
        print(f"\n结果：{_ok_count} 通过 / {_fail_count} 失败")
        return 1
    print()
    asyncio.run(check_ws(deck_id))
    print(f"\n结果：{_ok_count} 通过 / {_fail_count} 失败")
    return 0 if _fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
