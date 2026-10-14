"""学科护栏回归测试。

覆盖两个历史坑：
1. 合并新增「科学」学科时，把「实验/观察/探究/猜想」等跨学科通用词写进了科学的特征词，
   导致物理/化学/生物学生一提「实验」就被误判为跨学科（触发教学护栏误拦）。
2. 「化学方程式」同时命中数学的「方程」和化学的「化学方程式」，按学科顺序会误判成数学。
"""
from __future__ import annotations

import pytest

from config.subjects import SUBJECTS, _GENERIC_TERMS, detect_other_subject


# 本学科内的正常提问，不应该被判成跨学科
@pytest.mark.parametrize(
    ("subject", "question"),
    [
        ("物理", "这个实验怎么做？"),
        ("化学", "这个实验的现象是什么？"),
        ("生物", "做实验时要注意什么？"),
        ("物理", "请观察一下这个现象"),
        ("化学", "我们提出一个猜想"),
        ("物理", "电压越大电流越大吗？"),
        ("物理", "电路图怎么画？"),
        ("数学", "这道题的思路是什么？"),
        ("生物", "光合作用需要什么条件？"),
        ("历史", "辛亥革命的意义是什么？"),
        ("地理", "季风气候是怎么形成的？"),
    ],
)
def test_same_subject_not_flagged(subject: str, question: str) -> None:
    assert detect_other_subject(question, subject) is None


# 真正的跨学科提问，且要判对学科（取最长特征词，而非学科列表顺序）
@pytest.mark.parametrize(
    ("subject", "question", "expected"),
    [
        ("物理", "这道题的化学方程式怎么写？", "化学"),
        ("数学", "帮我翻译这个英语单词", "英语"),
        ("数学", "这个英语语法怎么理解？", "英语"),
        ("物理", "光合作用是什么？", "生物"),
    ],
)
def test_cross_subject_detected(subject: str, question: str, expected: str) -> None:
    assert detect_other_subject(question, subject) == expected


def test_generic_terms_never_trigger_cross_subject() -> None:
    """通用词单独出现时，不能把任何学科判成跨学科。"""
    for term in sorted(_GENERIC_TERMS):
        for subject in ("物理", "化学", "生物", "数学", "语文"):
            assert detect_other_subject(f"这个{term}是什么意思？", subject) is None, term


def test_all_subjects_have_teacher_and_film() -> None:
    """11 个学科都必须能解析到老师档案，且底片文件真实存在。"""
    from pathlib import Path

    from config.teachers import teacher_for_subject

    assert len(SUBJECTS) == 11
    root = Path(__file__).resolve().parents[1]
    for item in SUBJECTS:
        profile = teacher_for_subject(item["name"])
        assert profile is not None, f"{item['name']} 没有老师档案"
        assert profile.key == item["key"] or profile.key.startswith(item["key"] + "_")
        assert (root / profile.talk).is_file(), f"{profile.key} 底片缺失: {profile.talk}"


def test_every_teacher_profile_has_film() -> None:
    """**含同科变体**在内，每个教师档案都必须有真实底片。

    回归自真实故障：数学的「严谨型」变体只落了 ``teacher.json`` 与干声、没有
    ``talk.mp4``，学生一选中它整节课的数字人都是空白（待机素材三项为 null，
    逐句渲染 404，聊天区反复出现「数字人引擎正在加载」）。
    """
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    missing: list[str] = []
    for cfg in sorted((root / "data" / "teachers").glob("*/teacher.json")):
        data = json.loads(cfg.read_text(encoding="utf-8"))
        talk = str(data.get("talk") or f"data/teachers/{cfg.parent.name}/talk.mp4")
        film = Path(talk)
        film = film if film.is_absolute() else (root / film)
        if not (film.is_file() and film.stat().st_size > 0):
            missing.append(f"{cfg.parent.name}（{talk}）")
    assert not missing, "以下教师档案缺少底片，会导致数字人整节课空白：" + "、".join(missing)


def test_filmless_profile_is_hidden_and_falls_back(tmp_path, monkeypatch) -> None:
    """档案齐备但底片缺失时：不进可选列表，且被选中要回退到同科可用老师。"""
    import json

    from config import teachers as T

    for key, variant_order, with_film in (("math", 0, True), ("math_x", 10, False)):
        d = tmp_path / key
        d.mkdir()
        (d / "teacher.json").write_text(
            json.dumps(
                {
                    "key": key,
                    "subject": "数学",
                    "name": "数学老师",
                    "talk": str(d / "talk.mp4"),        # 绝对路径，指向隔离目录
                    "variant_order": variant_order,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        if with_film:
            (d / "talk.mp4").write_bytes(b"fake-film")

    monkeypatch.setattr(T, "_teachers_dir", lambda: tmp_path)
    T.load_teacher.cache_clear()
    T._load_all_profiles.cache_clear()
    try:
        assert [p.key for p in T.profiles_for_subject("数学")] == ["math"]
        assert [v["key"] for v in T.teacher_variants("数学")] == ["math"]
        # 老会话残留的坏 key：静默回退到同科可用老师，而不是返回一个渲染必 404 的档案
        prof = T.teacher_for_subject("数学", "math_x")
        assert prof is not None and prof.key == "math"
        assert T.has_film(prof)
    finally:
        T.load_teacher.cache_clear()
        T._load_all_profiles.cache_clear()
        T._filmless_warned.clear()
