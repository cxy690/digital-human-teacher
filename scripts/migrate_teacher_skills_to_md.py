# -*- coding: utf-8 -*-
"""一次性迁移：把两层教师技能（.py 硬编码）合并生成 ``.agents/skills/<key>/SKILL.md``。

背景与目标见 ``.agents/notes/teacher-skills-plan.md``：
* 合并前有两份「按学科」的技能资产，内容重叠、改一处漏一处：
  - ``config/teacher_skills/<key>.py``        —— 说话风格 / 讲课方式 / 深度 / 语速；
  - ``core/skills/teacher_personality/<key>_teacher.py`` —— 人格与九项教学行为。
* 合并后 **一份 md 承载两者**，运行期由 ``core/skills/md_skill.py`` 解析，Python 只留适配器。

本脚本是**一次性迁移工具**（已执行完毕），保留在仓库里作为「这些 md 从哪来」的可复现依据：

    python scripts/migrate_teacher_skills_to_md.py            # 生成 / 覆盖 .agents/skills/**
    python scripts/migrate_teacher_skills_to_md.py --check     # 只做往返自校验，不写文件

自校验：生成后立即用 PyYAML 读回，逐字段比对源对象，任一字段不一致即报错退出。

⚠️ **前置条件**：脚本读的是**迁移前的旧 `.py` 与旧设计稿目录**，这些在迁移后已删除 ——
若要重跑，需先从 git 历史取回：

    git show <迁移前提交>:config/teacher_skills/math.py             > config/teacher_skills/math.py
    git show <迁移前提交>:core/skills/teacher_personality/math_teacher.py > core/skills/teacher_personality/math_teacher.py
    git checkout <迁移前提交> -- docs/teacher_skills_src

日常维护**不要**用它：直接改 ``.agents/skills/<学科>/SKILL.md`` 即可（见
``.agents/notes/README.md``），本脚本会把 md 覆盖回迁移时的快照。
"""
from __future__ import annotations

import importlib
import inspect
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:            # 直跑脚本时把项目根加进 import 路径
    sys.path.insert(0, str(ROOT))

SKILLS_DIR = ROOT / ".agents" / "skills"
FLOW_SRC = ROOT / "docs" / "teacher_skills_src"

# 学科 key（= data/teachers/<key>/ = config/subjects.py 的 key），顺序与 SUBJECTS 对齐
SUBJECT_KEYS: tuple[str, ...] = (
    "chinese", "math", "english", "science", "physics", "chemistry",
    "biology", "politics", "morality", "geography", "history",
)

# 学科 key -> 流派设计稿目录名（`docs/teacher_skills_src/<flow>`）；科学无设计稿
FLOW_BY_KEY: dict[str, str] = {
    "chinese": "chinese-text-close-reading",
    "math": "math-modeling-studio",
    "english": "english-task-immersion",
    "physics": "physics-phenomenon-inquiry",
    "chemistry": "chemistry-lab-inquiry",
    "biology": "biology-life-concepts",
    "politics": "politics-issue-based",
    "morality": "morals-case-dialogue",
    "geography": "geography-spatial-field",
    "history": "history-evidence-story",
}

#: 正文小节标题 -> TeacherPersonalitySkill 的行为方法名（顺序即 md 中的出现顺序）
SECTION_METHODS: tuple[tuple[str, str], ...] = (
    ("开场方式", "opening_style"),
    ("追问方式", "questioning_style"),
    ("鼓励方式", "encouragement_style"),
    ("表扬方式", "praise_style"),
    ("纠错方式", "correction_style"),
    ("学生说「我不会」时", "confused_student_strategy"),
    ("学生连续答错时", "repeated_error_strategy"),
    ("白板组织", "whiteboard_style"),
    ("语气与节奏", "tone_rhythm"),
)

#: 内容超出「一位老师的教学行为」、但仍随 md 保存的额外小节（不作为行为注入）。
EXTRA_SECTIONS: tuple[str, ...] = ("讲课方式", "人格与教学", "说话风格")


def _q(value: Any) -> str:
    """把标量写成 YAML 双引号字符串（中文/冒号/破折号一律安全）。"""
    text = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"'


def _block(text: str) -> str:
    """多行文本写成 YAML 块标量（`|-`：不做换行折叠、结尾不带空行）。"""
    body = str(text).replace("\r\n", "\n").strip("\n")
    return "|-\n" + "\n".join("  " + line for line in body.split("\n"))


def _list_block(name: str, items: tuple[str, ...]) -> list[str]:
    if not items:
        return []
    lines = [f"{name}:"]
    lines += [f"  - {_q(x)}" for x in items]
    return lines


def _build_md(key: str, style: Any, persona_skill: Any, system_prompt: str) -> str:
    """拼装一份 SKILL.md（frontmatter + 正文小节）。"""
    flow = FLOW_BY_KEY.get(key, "")
    greeting_default = f"今天的{persona_skill.subject_name}课"
    greeting = persona_skill.greeting("{where}")

    front: list[str] = ["---"]
    front.append(f"name: {_q(f'{persona_skill.subject_name} · {persona_skill.display_name}')}")
    front.append(f"subject_key: {_q(key)}")
    front.append(f"subject_name: {_q(persona_skill.subject_name)}")
    front.append(f"skill_id: {_q(persona_skill.skill_id)}")
    front.append(f"class_name: {_q(type(persona_skill).__name__)}")
    front.append(f"display_name: {_q(persona_skill.display_name)}")
    front.append(f"persona: {_q(persona_skill.persona)}")
    front.append(f"tone: {_q(persona_skill.tone)}")
    front.append(f"depth: {_q(style.depth)}")
    front.append(f"speech_rate: {style.speech_rate}")
    front.append(f"teaching_style: {_q(style.teaching_style)}")
    if flow:
        front.append(f"source_flow: {_q(flow)}")
    front.append(f"greeting_default: {_q(greeting_default)}")
    front.append(f"style_persona: {_block(style.persona)}")
    front += _list_block("catchphrases", tuple(style.catchphrases))
    front += _list_block("do", tuple(style.do))
    front += _list_block("dont", tuple(style.dont))
    front.append("---")

    flow_hint = (
        f"流派设计稿（人读，不注入提示词）见 `references/{flow}.md`。"
        if flow else "本学科暂无独立流派设计稿。"
    )

    body: list[str] = [
        f"# {persona_skill.subject_name}老师 · {persona_skill.display_name}",
        "",
        "> 本文件是该科数字人老师的**运行期事实源**：`core/skills/md_skill.py` 解析上面的",
        "> frontmatter 与下面的小节，构建「说话风格层」（`TeacherSkill`）与「人格行为层」",
        "> （`TeacherPersonalitySkill`）。**改这里即生效，不需要改任何 Python。**",
        f"> {flow_hint}",
        "",
        "## 说话风格",
        "",
        str(style.persona).strip(),
        "",
        "## 讲课方式",
        "",
        str(style.teaching_way).strip(),
        "",
        "## 人格与教学",
        "",
        str(system_prompt).strip(),
        "",
    ]
    for title, method in SECTION_METHODS:
        body += [f"## {title}", "", str(getattr(persona_skill, method)()).strip(), ""]
    body += [
        "## 进场问候",
        "",
        str(greeting).strip(),
        "",
        f"> `{{where}}` 替换为当前课题；留空时用「{greeting_default}」。",
        "",
    ]

    # 深度档位速查：写成提示而不是隐藏规则，改 md 时看得见
    depth_note = {
        "elementary": "浅显（elementary）：多打比方，概念一出现就用生活例子解释，不超课标。",
        "standard": "标准（standard）：概念讲到位、步骤给依据，不炫技不超纲。",
        "advanced": "深入（advanced）：讲来龙去脉与因果脉络，以学生能消化为限。",
    }.get(str(style.depth), "")
    if depth_note:
        body += [f"<!-- 讲解深度：{depth_note} -->", ""]

    return "\n".join(front) + "\n\n" + "\n".join(body).rstrip("\n") + "\n"


def _load_persona(key: str) -> tuple[Any, str, type]:
    mod = importlib.import_module(f"core.skills.teacher_personality.{key}_teacher")
    cls = next(
        obj for _, obj in inspect.getmembers(mod, inspect.isclass)
        if issubclass(obj, object)
        and obj.__module__ == mod.__name__
        and getattr(obj, "subject_id", "") == key
    )
    system_prompt = getattr(mod, "_SYSTEM_PROMPT", "")
    if not system_prompt:
        raise RuntimeError(f"{key}: 模块内未找到 _SYSTEM_PROMPT")
    return cls(), str(system_prompt), cls


def _load_style(key: str) -> Any:
    mod = importlib.import_module(f"config.teacher_skills.{key}")
    return getattr(mod, "SKILL")


def _verify(key: str, md_text: str, style: Any, persona_skill: Any) -> None:
    """往返自校验：md 读回的字段必须与源对象逐项一致。"""
    assert md_text.startswith("---\n"), f"{key}: frontmatter 缺失"
    _, raw_front, body = md_text.split("---", 2)
    front = yaml.safe_load(raw_front) or {}

    expect = {
        "subject_key": key,
        "subject_name": persona_skill.subject_name,
        "skill_id": persona_skill.skill_id,
        "class_name": type(persona_skill).__name__,
        "display_name": persona_skill.display_name,
        "persona": persona_skill.persona,
        "tone": persona_skill.tone,
        "depth": style.depth,
        "speech_rate": style.speech_rate,
        "teaching_style": style.teaching_style,
        "style_persona": str(style.persona).strip(),
        "catchphrases": list(style.catchphrases),
        "do": list(style.do),
        "dont": list(style.dont),
    }
    for field, want in expect.items():
        got = front.get(field)
        if isinstance(want, list):
            assert list(got or []) == want, f"{key}.{field} 往返不一致：{got!r} != {want!r}"
        else:
            assert str(got) == str(want), f"{key}.{field} 往返不一致：{got!r} != {want!r}"

    assert "## 讲课方式" in body and str(style.teaching_way).strip() in body, f"{key}: 讲课方式丢失"
    for title, method in SECTION_METHODS:
        want = str(getattr(persona_skill, method)()).strip()
        assert f"## {title}" in body, f"{key}: 缺少小节「{title}」"
        assert want in body, f"{key}: 小节「{title}」内容丢失"


def main() -> None:
    check_only = "--check" in sys.argv
    for key in SUBJECT_KEYS:
        style = _load_style(key)
        persona_skill, system_prompt, _ = _load_persona(key)
        md_text = _build_md(key, style, persona_skill, system_prompt)
        _verify(key, md_text, style, persona_skill)

        out_dir = SKILLS_DIR / key
        if check_only:
            existing = (out_dir / "SKILL.md")
            assert existing.is_file(), f"{key}: 缺少 {existing}"
            assert existing.read_text(encoding="utf-8") == md_text, f"{key}: 与源对象不一致"
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "SKILL.md").write_text(md_text, encoding="utf-8")
        print(f"  [md]  .agents/skills/{key}/SKILL.md")

        flow = FLOW_BY_KEY.get(key)
        if not flow:
            continue
        src = FLOW_SRC / flow
        if not src.is_dir():
            print(f"  [skip] {src} 不存在")
            continue
        ref_dir = out_dir / "references"
        ref_dir.mkdir(parents=True, exist_ok=True)
        design = src / "SKILL.md"
        if design.is_file():
            shutil.copyfile(design, ref_dir / f"{flow}.md")
            print(f"  [ref] .agents/skills/{key}/references/{flow}.md")
        child = src / "references"
        if child.is_dir():
            for fp in sorted(child.glob("*.md")):
                shutil.copyfile(fp, ref_dir / fp.name)
                print(f"  [ref] .agents/skills/{key}/references/{fp.name}")

    print("[done] 全部通过往返自校验" if not check_only else "[ok] 自校验通过")


if __name__ == "__main__":
    main()
