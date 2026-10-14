"""教师档案：每位老师（= 一个学科）的底片、标定、音色与说话方式。

档案来源：``data/teachers/<学科key>/teacher.json``（由标定流程生成）。
学科 key 与 :mod:`config.subjects` 的 key 完全一致（chinese/math/...），
因此「按学科自动选老师」= 直接用学科 key 取档案，无需额外的映射表。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from config.logging import get_logger
from config.settings import settings
from config.subjects import SUBJECT_BY_KEY, SUBJECT_BY_NAME

log = get_logger("config.teachers")

ROOT = Path(__file__).resolve().parent.parent

# 每位老师的「说话方式」（可被 teacher.json 的 persona 覆盖）：
# 让同一套讲解流程在不同学科呈现不同人格。
DEFAULT_PERSONAS: dict[str, str] = {
    "chinese": "你是一位温婉知性的语文老师，说话有画面感，喜欢用诗词、典故和生活场景作比，"
               "讲字词句篇时特别注重语感与表达。",
    "math": "你是一位沉稳理性的数学老师，说话条理清晰、层层递进，习惯先把条件和目标写清楚再动笔，"
            "强调「不跳步、每一步都有依据」。",
    "english": "你是一位明快干练的英语老师，说话轻快、富有节奏。**全程用普通话讲解**，"
               "只在举例时给必要的英文单词/短语并立刻用中文解释；鼓励学生大胆开口、多造句。",
    "physics": "你是一位严谨又亲切的物理老师，喜欢从生活现象追问「为什么」，"
               "特别强调物理量的单位、适用条件和因果关系。",
    "chemistry": "你是一位细致耐心的化学老师，喜欢用实验现象和日常类比把抽象概念讲具体，"
                 "提醒学生注意反应条件与守恒关系。",
    "biology": "你是一位亲切自然的生物老师，善于用身体、饮食和自然界的例子讲解，"
               "喜欢帮学生把零散知识串成规律。",
    "politics": "你是一位端庄稳重的政治老师，说话条理分明、紧扣材料与观点，"
                "引导学生从多个角度分析问题。",
    "morality": "你是一位温和可信赖的道德与法治老师，喜欢用校园和家庭里的真实情境讲道理，"
                "重视规则意识与同理心。",
    "geography": "你是一位阳光健谈的地理老师，喜欢用地图、旅行见闻和时事讲地理，"
                 "善用「位置—气候—人文」的因果链。",
    "history": "你是一位儒雅沉静的历史老师，喜欢讲故事讲脉络，注重史料依据与因果分析，"
               "常提醒学生「把事件放回它发生的年代」。",
    "science": "你是一位爱探索、爱提问的科学老师，喜欢把生活中的小事变成科学现场，"
               "总先问「你看到了什么、你猜为什么」，再带着学生一起找证据。",
}


# 开课问候池（每科多条，开课时随机取一条 —— 每次进课堂第一句话不重样）。
# 写问候语的约定：口语化、10~40 字、贴合该科 persona（见 DEFAULT_PERSONAS）；
# 句式尽量错开（提问式 / 期待式 / 生活化 / 悬念式），不要每条都是「我是你的XX老师」。
# 同科变体老师（如 math_strict）档案里的专属 greeting 会并入池中一起随机，不再一辈子只念同一句。
DEFAULT_GREETINGS: dict[str, tuple[str, ...]] = {
    "chinese": (
        "同学你好呀！我是你的语文老师，今天想读点什么、写点什么？",
        "欢迎回来～字词句篇，咱们一段一段慢慢品。",
        "你好呀！语文的美藏在每个字里，准备好了吗？",
        "今天的课堂上，我们和文字交个朋友吧。",
        "同学好～有读不懂的句子随时问我，我们一起琢磨。",
    ),
    "math": (
        "同学你好！我是你的数学老师，今天想先攻克哪一类题？",
        "准备好了吗？数学讲究步步有据，我们一步步来。",
        "你好呀！有题目就直接发给我，我们边做边讲。",
        "欢迎回来～先把上次的疑问解决掉，再往下走。",
        "同学好！遇到卡住的地方，恰恰是进步的开始。",
    ),
    "english": (
        "同学你好！我是你的英语老师，今天我们边聊边练，大胆开口说！",
        "准备好了吗？英语就是用出来的——多说一句，就多会一点。",
        "你好呀！不会的单词先别慌，我们放到句子里猜一猜。",
        "欢迎回来～今天先热热身，再挑一个你喜欢的表达练熟它。",
        "同学好！哪怕只说对一个句子，也值得表扬自己。",
    ),
    "physics": (
        "同学你好！我是你的物理老师，今天想弄明白哪个「为什么」？",
        "准备好了吗？我们先看清现象，再找背后的规律。",
        "你好呀！物理不怕错，就怕不问——有疑问尽管来。",
        "欢迎回来～记得带上单位，我们一起把每个量算清楚。",
        "同学好！生活里处处是物理，今天看看能发现什么。",
    ),
    "chemistry": (
        "同学你好！我是你的化学老师，今天想研究哪种「变化」？",
        "准备好了吗？我们像做实验一样，一步一步看清楚。",
        "你好呀！记不住的反应，我们试着用生活里的东西打个比方。",
        "欢迎回来～方程式别死背，先想清楚谁变成了谁。",
        "同学好！化学的乐趣在于变化，我们一起来观察。",
    ),
    "biology": (
        "同学你好！我是你的生物老师，今天想认识哪个生命的小秘密？",
        "准备好了吗？从我们自己身上讲起，生物一点都不远。",
        "你好呀！记不住的知识点，我们把它串成一张网。",
        "欢迎回来～今天用身边的例子，把知识学活。",
        "同学好！有问题随时问，探索生命不着急。",
    ),
    "politics": (
        "同学你好！我是你的政治老师，今天我们从材料出发，理清思路。",
        "准备好了吗？答题先审清问题，再分层作答。",
        "你好呀！有疑问就提出来，我们一个角度一个角度分析。",
        "欢迎回来～把观点讲清楚，比背得多更重要。",
        "同学好！我们一起来读材料、找关键。",
    ),
    "morality": (
        "同学你好！我是你的道德与法治老师，今天从一件小事聊起。",
        "准备好了吗？懂规矩、有同理心，从我们的校园生活说起。",
        "你好呀！遇到想不通的事，我们可以一起分析对与错。",
        "欢迎回来～今天的故事，说不定就发生在你身边。",
        "同学好！有事随时和我说，我们慢慢聊。",
    ),
    "geography": (
        "同学你好！我是你的地理老师，今天想去哪儿看看？",
        "准备好了吗？打开地图，我们从位置开始讲起。",
        "你好呀！天气、旅行、新闻里，都藏着地理。",
        "欢迎回来～今天用「位置—气候—人文」这条线串一串。",
        "同学好！有问题随时问，我们边走边看。",
    ),
    "history": (
        "同学你好！我是你的历史老师，今天想听哪一段故事？",
        "准备好了吗？把事件放回它发生的年代，一切就通顺了。",
        "你好呀！历史不是死记硬背，是理清来龙去脉。",
        "欢迎回来～今天讲脉络、找因果，慢慢来。",
        "同学好！有疑问尽管问，我们把时间线捋直。",
    ),
    "science": (
        "同学你好！我是你的科学老师，今天先从「你看到了什么」说起。",
        "准备好了吗？大胆猜一猜「为什么」，再用证据验证。",
        "你好呀！生活里的小事，都是科学现场。",
        "欢迎回来～今天带着好奇心，一起找答案。",
        "同学好！没有笨问题，每个问题都值得研究。",
    ),
}

# 通用问候池：老师档案缺失 / 无学科问候池时的兜底（``{subject}`` 占位学科名）。
# 同样多条随机，避免「每次开课都是同一句模板」。
GENERIC_GREETINGS: tuple[str, ...] = (
    "同学你好呀！我是你的{subject}老师，今天我们来一起学习{subject}，准备好了吗？",
    "{subject}课开始啦！有不懂的直接问我，我们一起把它弄明白。",
    "同学好！今天想先学点{subject}的什么？告诉我就行。",
    "欢迎来到{subject}课堂～放松点，有疑问随时提。",
    "你好呀！我是教{subject}的老师，今天我们来解决你真正想问的问题。",
)


# 每位老师的 edge-tts 兜底音色（无克隆干声时用；可被 teacher.json 的 voice 覆盖）
DEFAULT_VOICES: dict[str, str] = {
    "chinese": "zh-CN-XiaoxiaoNeural",
    "math": "zh-CN-YunxiNeural",
    "english": "zh-CN-XiaoyiNeural",
    "physics": "zh-CN-YunyangNeural",
    "chemistry": "zh-CN-XiaoyiNeural",
    "biology": "zh-CN-XiaoxiaoNeural",
    "politics": "zh-CN-YunxiaNeural",
    "morality": "zh-CN-YunjianNeural",
    "geography": "zh-CN-YunxiNeural",
    # 历史：素材里几乎无有效语音，克隆不可靠 -> 用女声兜底（该老师为女性）
    "history": "zh-CN-XiaoxiaoNeural",
    # 科学（合并新增第 11 科）：暂无专属干声，先用 edge-tts 自带音色
    "science": "zh-CN-YunxiNeural",
}


# edge-tts 中文音色 -> 性别。用途：
#   1) TTS 依「老师形象性别」校验音色是否配错（男老师配女声 → 告警，而不是静默播错）；
#   2) 兜底音色白名单（TTS_VOICE 填成 OpenAI 的 alloy/英语音色时收敛回中文音色）。
EDGE_VOICE_GENDER: dict[str, str] = {
    "zh-CN-XiaoxiaoNeural": "female",
    "zh-CN-XiaoyiNeural": "female",
    "zh-CN-YunxiNeural": "male",
    "zh-CN-YunyangNeural": "male",
    "zh-CN-YunjianNeural": "male",
    "zh-CN-YunxiaNeural": "male",
    # 方言音色（东北 / 陕西）：**已不再指派给任何老师**（讲课一律用普通话标准音色，
    # 见 scripts/apply_edge_voices.py）。这里保留映射只为「性别校验」兜底：
    # 万一有档案残留方言音色，仍能正确判定性别而不是误报「音色配错」。
    "zh-CN-liaoning-XiaobeiNeural": "female",
    "zh-CN-shaanxi-XiaoniNeural": "female",
}


def voice_gender(voice: str) -> str:
    """音色名 -> 性别（``male`` / ``female`` / 空串=未知）。"""
    return EDGE_VOICE_GENDER.get((voice or "").strip(), "")


# 音高偏移（Hz）：让「形象性别」在声音上也能听出来（女声略高、男声略低）。
# 数值刻意保守，只做辨识度微调，不改变老师本人的声线特征。
_PITCH_BY_GENDER: dict[str, int] = {"female": 4, "male": -4}

# teacher.json 里的 gender 历史上有多种写法（female/女/女性…），统一归一化再比较，
# 否则「女」会被当成未知值，导致对每位老师误报「音色性别与形象不符」。
_GENDER_ALIASES: dict[str, str] = {
    "female": "female", "f": "female", "女": "female", "女性": "female", "女士": "female",
    "male": "male", "m": "male", "男": "male", "男性": "male", "男士": "male",
}


def _norm_gender(value: str | None) -> str:
    """把各种性别写法归一为 ``male`` / ``female``（未知返回空串）。"""
    return _GENDER_ALIASES.get((value or "").strip().lower(), "")


@dataclass(frozen=True)
class TeacherProfile:
    """一位老师的完整档案。"""

    key: str                       # 学科 key（= 目录名 = 引擎用的 teacher_id）
    subject: str                   # 学科中文名
    name: str                      # 显示名，如「语文老师」
    talk: str                      # 底片路径（相对项目根或绝对）
    voice_ref_audio: str = ""      # 干声路径
    voice_ref_text: str = ""       # 干声对应文本
    eye_points: tuple[tuple[int, int], tuple[int, int]] = ((605, 235), (730, 230))
    eye_r: tuple[int, int] = (46, 22)
    frame_at: float = 0.13         # 抽帧点（睁眼闭嘴）
    # 底片中真实闭眼时刻；None = 该底片里没有可用眨眼帧，前端退化为「眼区压合」合成
    blink_at: float | None = 6.25
    # 闭眼帧两只眼睛中心相对待机帧的像素偏移 [[右眼dx,dy],[左眼dx,dy]]：
    # 贴回眼区时按此对齐，避免「眨眼不协调」。
    blink_shift: tuple[tuple[int, int], tuple[int, int]] = ((0, 0), (0, 0))
    # 独立闭眼帧素材（整帧图片路径，可选）：底片本身是「静态图转的视频」、
    # 里面根本没有眨眼时，blink_at 无从标定，就用这张图提供真实闭眼。
    blink_src: str = ""
    teaching_style: str = "plain"
    persona: str = ""
    # 老师性别（male / female）：前端浏览器回退朗读时据此挑音色，
    # 否则会用系统默认女声 —— 男老师开口变成女声就是这么来的。
    gender: str = ""
    voice: str = ""                # edge-tts 音色名（留空则用学科默认）
    # 发声方式：clone = 用 voice_ref.wav 做零样本音色克隆；
    # edge = 该素材有效语音太少（克隆不可靠），退回 edge-tts 自带音色。
    voice_engine: str = "clone"
    # ------------------------------------------------ 同科多老师（风格变体）扩展
    # 同一学科可以挂多位老师：key 不再等于学科 key（如数学有 math / math_b），
    # 由 subject 字段把它们归到同一学科，由以下字段描述"风格"，供选择界面展示。
    variant_name: str = ""         # 风格名，如「严谨型」「亲和型」；空 = 学科默认老师
    variant_desc: str = ""         # 一句话介绍（选择界面副标题）
    variant_order: int = 0         # 同科排序：小的在前（0 通常是默认老师）
    portrait: str = ""             # 选择界面头像（相对项目根的图片路径或 URL；空则前端用 emoji）
    # 技能继承：变体没有自己的 config/teacher_skills/<key>.py 时，
    # 复用该 key 的技能模块（教学法/深度/语速），只把「说话风格」换成自己的 persona。
    # 留空则按 subject 自动推导（数学变体 -> math）。
    skill_key: str = ""
    # 专属开场白（同科多老师时体现风格差异）；留空用学科通用模板。
    greeting: str = ""
    # 专属开场白**池**（teacher.json 可写多条，开课时随机取一条；单条 greeting 兼容保留）。
    # 与学科内置池（DEFAULT_GREETINGS）合并后随机，见 :meth:`greeting_pool`。
    greetings: tuple[str, ...] = ()

    # ---------------------------------------------------------------- 便捷属性
    @property
    def talk_path(self) -> Path:
        """底片绝对路径。"""
        p = Path(self.talk)
        return p if p.is_absolute() else (ROOT / p)

    @property
    def voice_ref_path(self) -> Path | None:
        """干声绝对路径；不存在返回 None（调用方回退默认音色）。"""
        if not self.voice_ref_audio:
            return None
        p = Path(self.voice_ref_audio)
        p = p if p.is_absolute() else (ROOT / p)
        return p if p.is_file() else None

    @property
    def blink_src_path(self) -> Path | None:
        """独立闭眼帧素材的绝对路径；未配置或文件不存在返回 None。"""
        if not self.blink_src:
            return None
        p = Path(self.blink_src)
        p = p if p.is_absolute() else (ROOT / p)
        return p if p.is_file() else None

    @property
    def eye_points_env(self) -> str:
        """``x1,y1;x2,y2`` 形式（与 AVATAR_EYE_POINTS 一致）。"""
        (x1, y1), (x2, y2) = self.eye_points
        return f"{x1},{y1};{x2},{y2}"

    @property
    def avatar_id(self) -> str:
        """渲染引擎里用于选择底片的 ID。"""
        return self.key

    def _base_skill(self):
        """本档案复用的技能模块：自己的 key 优先，其次 skill_key / 学科技能。

        返回 ``TeacherSkill`` 或 None。技能模块管「怎么讲课」，
        teacher.json 管「形象/音色」，二者解耦。
        """
        from config.teacher_skills import load_skill  # 局部导入避免循环依赖

        own = load_skill(self.key)
        if own is not None:
            return own
        base = self.skill_key.strip() or teacher_key_for_subject(self.subject)
        return load_skill(base) if base else None

    def persona_text(self) -> str:
        """说话方式提示词。

        解析顺序：
        1. 有自己的技能模块（``teacher_skills/<key>.py``）→ 直接用（老老师走的路径）；
        2. 变体自带 ``persona`` + 可继承学科技能 → 用学科的**教学法**，只换**说话风格**；
        3. 学科技能模块；
        4. 档案 ``persona`` / 学科默认人格（``DEFAULT_PERSONAS``）。
        """
        from config.teacher_skills import load_skill  # 局部导入避免循环依赖

        own_skill = load_skill(self.key)
        if own_skill is not None:
            return own_skill.prompt_text()
        base = self._base_skill()
        if base is None:
            return self.persona or DEFAULT_PERSONAS.get(self.key, "")
        if self.persona:
            # 变体：保留学科的教学法/深度/语速/口头禅，只替换「说话风格」这一句
            from dataclasses import replace as _dc_replace

            return _dc_replace(base, persona=self.persona).prompt_text()
        return base.prompt_text()

    @property
    def speech_rate(self) -> float:
        """语速倍率（来自技能模块；供 TTS 引擎使用，未配置技能则 1.0）。"""
        skill = self._base_skill()
        return float(skill.speech_rate) if skill is not None else 1.0

    def voice_name(self) -> str:
        """该老师的 edge-tts 音色名（档案未写则用学科默认；仍为空则由调用方兜底）。"""
        return self.voice or DEFAULT_VOICES.get(self.key, "")

    def voice_pitch_hz(self) -> int:
        """该老师音色的音高偏移（Hz）；音色/形象性别均未知时返回 0（不加偏移）。"""
        gender = voice_gender(self.voice_name()) or _norm_gender(self.gender)
        return _PITCH_BY_GENDER.get(gender, 0)

    def voice_matches_gender(self) -> bool:
        """音色性别是否与老师形象一致；任一侧未知一律视为一致（不告警）。"""
        want = _norm_gender(self.gender)
        got = voice_gender(self.voice_name())
        if not want or not got:
            return True
        return want == got

    # ------------------------------------------------------- 同科多老师辅助属性
    @property
    def is_variant(self) -> bool:
        """是否是同科的「风格变体」（有风格名即视为变体）。"""
        return bool(self.variant_name.strip())

    def display_name(self) -> str:
        """选择界面用显示名：``数学老师 · 严谨型``；无风格名时就是 ``name``。"""
        label = self.variant_name.strip()
        return f"{self.name} · {label}" if label else self.name

    @property
    def style_label(self) -> str:
        """风格标签（无风格名时回退到学科默认人格的简称）。"""
        return self.variant_name.strip()

    def portrait_path(self) -> Path | None:
        """选择界面头像绝对路径；未配置或文件不存在返回 None。"""
        if not self.portrait:
            return None
        # 允许直接写 URL（前端原样使用）
        if "://" in self.portrait or self.portrait.startswith("/"):
            return None
        p = Path(self.portrait)
        p = p if p.is_absolute() else (ROOT / p)
        return p if p.is_file() else None

    def greeting_pool(self) -> tuple[str, ...]:
        """开课问候池：变体专属问候 + 学科内置池（合并去重；开课时随机取一条）。

        每次开课都随机 —— 解决「每位老师每次开场都是同一句」的单调感；
        变体档案里的专属 greeting（如「我讲课比较严格」）仍然参与随机，风格不被抹掉。
        """
        parts: list[str] = []
        if self.greeting.strip():
            parts.append(self.greeting.strip())
        parts.extend(g for g in self.greetings if g.strip())
        key = teacher_key_for_subject(self.subject) or self.key.split("_", 1)[0]
        parts.extend(DEFAULT_GREETINGS.get(key, ()))
        return tuple(dict.fromkeys(parts))


def _teachers_dir() -> Path:
    p = Path(settings.teachers_dir or "data/teachers")
    return p if p.is_absolute() else (ROOT / p)


def _shift_from(raw: object) -> tuple[tuple[int, int], tuple[int, int]]:
    """解析 ``blink_shift``（退化/缺省时返回 ((0,0),(0,0))）。"""
    try:
        (r0, r1), (l0, l1) = raw                       # type: ignore[misc]
        return ((int(r0), int(r1)), (int(l0), int(l1)))
    except Exception:  # noqa: BLE001 - 档案缺字段/格式异常一律视为无偏移
        return ((0, 0), (0, 0))


def _str_list(raw: object) -> tuple[str, ...]:
    """解析字符串列表字段（缺省 / 格式异常一律视为空）。"""
    if isinstance(raw, (list, tuple)):
        return tuple(str(x).strip() for x in raw if str(x).strip())
    return ()


def _from_dict(key: str, data: dict) -> TeacherProfile:
    eyes = data.get("eye_points") or [[605, 235], [730, 230]]
    eye_r = data.get("eye_r") or [46, 22]
    blink_raw = data.get("blink_at")
    return TeacherProfile(
        key=key,
        subject=str(data.get("subject") or key),
        name=str(data.get("name") or f"{data.get('subject') or key}老师"),
        talk=str(data.get("talk") or f"data/teachers/{key}/talk.mp4"),
        voice_ref_audio=str(data.get("voice_ref_audio") or ""),
        voice_ref_text=str(data.get("voice_ref_text") or ""),
        eye_points=((int(eyes[0][0]), int(eyes[0][1])), (int(eyes[1][0]), int(eyes[1][1]))),
        eye_r=(int(eye_r[0]), int(eye_r[1])),
        frame_at=float(data.get("frame_at") if data.get("frame_at") is not None else 0.13),
        blink_at=(float(blink_raw) if blink_raw is not None else None),
        blink_shift=_shift_from(data.get("blink_shift")),
        blink_src=str(data.get("blink_src") or ""),
        teaching_style=str(data.get("teaching_style") or "plain"),
        persona=str(data.get("persona") or ""),
        gender=str(data.get("gender") or "").strip().lower(),
        voice=str(data.get("voice") or ""),
        voice_engine=str(data.get("voice_engine") or "clone"),
        variant_name=str(data.get("variant_name") or ""),
        variant_desc=str(data.get("variant_desc") or ""),
        variant_order=int(data.get("variant_order") or 0),
        portrait=str(data.get("portrait") or ""),
        skill_key=str(data.get("skill_key") or ""),
        greeting=str(data.get("greeting") or ""),
        greetings=_str_list(data.get("greetings")),
    )


@lru_cache(maxsize=64)
def load_teacher(key: str) -> TeacherProfile | None:
    """按学科 key 载入档案；不存在或解析失败返回 None。"""
    key = (key or "").strip()
    if not key:
        return None
    cfg = _teachers_dir() / key / "teacher.json"
    if not cfg.is_file():
        return None
    try:
        data = json.loads(cfg.read_text(encoding="utf-8"))
    except Exception as err:  # noqa: BLE001 - 档案坏掉不应阻断答疑
        log.warning("教师档案解析失败 {}：{}", cfg, err)
        return None
    if not isinstance(data, dict):
        return None
    return _from_dict(key, data)


def teacher_key_for_subject(subject: str | None) -> str:
    """学科（中文名或英文 key）-> 教师档案 key；无法识别时原样返回。"""
    raw = (subject or "").strip()
    if not raw:
        return ""
    meta = SUBJECT_BY_NAME.get(raw) or SUBJECT_BY_KEY.get(raw)
    return str((meta or {}).get("key") or raw)


def _subject_name(subject: str | None) -> str:
    """学科（key 或中文名）-> 中文名；无法识别时原样返回。"""
    raw = (subject or "").strip()
    if not raw:
        return ""
    meta = SUBJECT_BY_NAME.get(raw) or SUBJECT_BY_KEY.get(raw)
    return str((meta or {}).get("name") or raw)


def profiles_for_subject(subject: str | None) -> list[TeacherProfile]:
    """该学科下**全部**老师档案（同科多老师），默认老师排第一。

    归属判定用档案里的 ``subject`` 字段（而非目录名），因此同科的
    ``math`` / ``math_b`` 都会被归到「数学」。
    """
    name = _subject_name(subject)
    if not name:
        return []
    out: list[TeacherProfile] = []
    for prof in _load_all_profiles():
        if _subject_name(prof.subject) != name:
            continue
        if not has_film(prof):
            # 无底片的档案不放进选择列表：否则学生一旦选中，数字人整节课都是空白
            _warn_filmless(prof, "已从可选老师列表忽略")
            continue
        out.append(prof)
    out.sort(key=lambda p: (p.variant_order, 0 if p.key == teacher_key_for_subject(subject) else 1, p.key))
    return out


def default_teacher_key(subject: str | None) -> str:
    """该学科默认老师的 key（同科多老师时取排序第一；无档案时回退学科 key）。"""
    profiles = profiles_for_subject(subject)
    if profiles:
        return profiles[0].key
    return teacher_key_for_subject(subject)


def has_film(profile: TeacherProfile | None) -> bool:
    """该老师档案的底片（``talk.mp4``）是否真的存在且非空。

    为什么必须校验：``teacher.json`` 完全可能齐备（音色 / 人格 / 开场白都在）
    而底片缺失 —— 典型是「同科风格变体」档案先落盘、底片转码失败或未同步。
    这时数字人链路会**整节课失效**：``/api/teacher`` 的待机素材三项全为 null
    （前端教师区一片空白），逐句渲染一律 404（学生只看到
    「（数字人引擎正在加载，先用语音讲解～）」）。因此无底片的档案
    既不能进入可选老师列表，也不能被选中。
    """
    if profile is None:
        return False
    try:
        film = profile.talk_path
        return film.is_file() and film.stat().st_size > 0
    except OSError:  # 路径非法 / 无权限：一律当作不可用
        return False


# 已就「缺底片」告警过的老师 key：只报一次，避免每次取列表都刷同一条日志
_filmless_warned: set[str] = set()


def _warn_filmless(profile: TeacherProfile, why: str) -> None:
    """档案存在但底片缺失：首次使用时告警一次，便于定位坏档案。"""
    if profile.key in _filmless_warned:
        return
    _filmless_warned.add(profile.key)
    log.warning(
        "{}：老师 {} 的底片不存在（{}），该档案不参与数字人形象选择",
        why, profile.key, profile.talk_path,
    )


def _belongs_to_subject(prof: TeacherProfile, subject: str | None) -> bool:
    """该老师档案是否属于指定学科（**防串台**）。

    目前的历史问题：``teacher_for_subject("物理", "math_strict")`` 会把数学老师整个
    带进来（形象 / 音色 / 人格全线串台），因为 ``load_teacher`` 只按 key 取档案、
    不校验学科。这里用「同科多老师 = ``<学科key>_<变体>`` 前缀」这一既有约定做校验。

    学科无法识别时（如「综合」、空串）一律放行，保持旧行为不变。
    """
    raw = (subject or "").strip()
    meta = SUBJECT_BY_NAME.get(raw) or SUBJECT_BY_KEY.get(raw)
    expected = str((meta or {}).get("key") or "")
    if not expected:
        return True
    return prof.key == expected or prof.key.startswith(expected + "_")


def teacher_for_subject(
    subject: str | None, teacher_key: str | None = None
) -> TeacherProfile | None:
    """按学科取老师档案；该学科没有独立档案时返回 None（调用方回退全局配置）。

    ``teacher_key`` 非空时优先用它（同科多老师：学生选中的那位）；
    **但它必须属于该学科** —— 否则视为串台（老会话残留的 teacher、前端传错 key），
    丢弃并回退到学科默认老师。该 key 不存在时同样静默回退，保证老前端也能正常开课。
    """
    tid = (teacher_key or "").strip()
    if tid:
        prof = load_teacher(tid)
        if prof is not None:
            if not has_film(prof):
                # 档案在、底片没了（如同科变体只落了 teacher.json）：回退学科默认老师。
                # 否则引擎逐句 404、待机素材三项全为 null，整节课看不到数字人。
                _warn_filmless(prof, "所选老师底片缺失，已回退学科默认老师")
            elif _belongs_to_subject(prof, subject):
                return prof
            else:
                log.warning(
                    "老师档案与学科不符，已丢弃以免串台：teacher={}（属 {}）!= subject={}",
                    prof.key, prof.subject, subject,
                )
    key = teacher_key_for_subject(subject)
    prof = load_teacher(key)
    if has_film(prof):
        return prof
    if prof is not None:
        _warn_filmless(prof, "学科默认老师底片缺失")
    profiles = profiles_for_subject(subject)
    return profiles[0] if profiles else None


def teacher_variants(subject: str | None) -> list[dict]:
    """选择界面用的老师列表（同科多老师）；单老师学科返回长度为 1 的列表。"""
    default_key = default_teacher_key(subject)
    return [
        {
            "key": prof.key,
            "name": prof.name,
            "display_name": prof.display_name(),
            "subject": prof.subject,
            "variant_name": prof.variant_name,
            "variant_desc": prof.variant_desc,
            "portrait": prof.portrait,
            "voice": prof.voice_name(),
            "voice_engine": prof.voice_engine,
            # 形象性别：选择页据此标出「男声 / 女声」，一眼看出音色有没有配错
            "gender": prof.gender,
            "teaching_style": prof.teaching_style,
            "is_default": prof.key == default_key,
        }
        for prof in profiles_for_subject(subject)
    ]


@lru_cache(maxsize=4)
def _load_all_profiles() -> tuple[TeacherProfile, ...]:
    """扫描 ``teachers_dir`` 下全部教师档案（同科多老师归组的数据来源）。

    按 mtime 监听目录变化会让缓存失效逻辑复杂化；这里用进程级缓存，
    新增/修改档案后重启服务即可生效（与 ``load_teacher`` 的 lru_cache 行为一致）。
    """
    d = _teachers_dir()
    if not d.is_dir():
        return ()
    profiles: list[TeacherProfile] = []
    for cfg in sorted(d.glob("*/teacher.json")):
        prof = load_teacher(cfg.parent.name)
        if prof is not None:
            profiles.append(prof)
    return tuple(profiles)


def available_keys() -> list[str]:
    """已配置档案的学科 key 列表。"""
    d = _teachers_dir()
    if not d.is_dir():
        return []
    return sorted(p.parent.name for p in d.glob("*/teacher.json"))
