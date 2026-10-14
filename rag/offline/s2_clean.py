#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 2 · 清洗（rag/offline/rag/offline/s2_clean.py）
=====================================
把 s1 解析出的原始文本洗成「可信、可入库、可合规审计」的教学文本。处理顺序是**有意设计**的：

1. **去时间戳 / 页码 / 页眉页脚** —— 先做，否则页码会被当成数字内容进入后续统计；
2. **去 ASR 口头禅与重复** —— 再削掉口语噪声，让句子更像成稿；
3. **隐私脱敏** —— 在"文本已经稳定"之后做，避免口头禅删除把手机号切成两半导致漏检；
4. **全角半角、标点归一** —— 最后统一形态，产出下游可直接使用的文本。

> 隐私脱敏是面向未成年人场景的**硬性要求**：手机号、身份证号、学生真实姓名、
> 学校名、QQ/微信 一律替换为 ``{student_name}`` 之类的占位符，并在 ``stats.pii_hits``
> 里逐类计数。任何新增正则都必须同步在 ``rag/offline/tests/test_pipeline.py`` 里加断言，
> 不允许出现"改了正则但测试没跟上"的情况。

输出 ``data/clean/<name>.json``::

    {
      "source_name": "…", "source_type": "…", "source_hash": "…",
      "text": "清洗后的全文（块之间空行分隔）",
      "blocks": [{..., "text": "清洗后", "raw_text": "清洗前", "pii_hits": {...}}, ...],
      "stats": {
        "raw_chars": 1000, "clean_chars": 820,
        "char_delete_ratio": 0.18,          # 删除率（字符级）
        "removed": {"filler": 12, "repeat": 3, "noise_line": 2},
        "pii_hits": {"phone": 1, "id_card": 0, "student_name": 2, "school": 1, "im_account": 0},
        "pii_hit_total": 4,
        "segments_before": 30, "segments_after": 28
      },
      "warnings": []
    }

CLI::

    python rag/offline/rag/offline/s2_clean.py --in data/parsed --out data/clean
    python rag/offline/rag/offline/s2_clean.py --text "大家好 我是三年二班的张小明" --show   # 单条排查用
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:  # pragma: no cover
    from . import common
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import common  # type: ignore

log = common.get_logger("s2_clean")


# =========================================================================== #
# 一、正则表（集中定义，便于教研 review 与单测引用）
# =========================================================================== #
# 口头禅：只删"独立成词"的语气词，避免误伤"把握/啊呀"之类实义词。
# 设计取舍：不删孤立的"啊/呀/嗯"——"嗯，你算对了"里的"嗯"是教学语气的肯定应答，
# 删掉会改变课堂语感。只删「连续重复」与「后接停顿标点」的纯填充词。
FILLER_PATTERNS: list[tuple[str, str]] = [
    (r"(?:嗯|呃|唔|欸|诶)\s*[，,、]\s*", ""),          # 嗯，… / 呃，…
    (r"(?:那个|这个)\s*(?:那个|这个)\s*[，,]?\s*", ""),  # 那个那个 / 这个那个
    (r"就是说\s*[，,]?\s*", ""),                        # 就是说…
    (r"然后呢\s*[，,]?\s*", ""),                        # 然后呢…
    (r"你知道吧\s*[，,]?\s*", ""),
    (r"对吧对吧\s*[，,]?\s*", ""),
    (r"是不是啊\s*[，,]?\s*", ""),
    (r"(?:呢|啊|吧)\s*[，,]\s*(?=(?:我们|那么|所以|大家))", ""),  # "…呢，我们来看" → "…我们来看"
]

# 连续重复词 / 短语（"同学们同学们" / "看看看" / "哈哈哈哈"）
REPEAT_PATTERNS: list[tuple[str, str]] = [
    (r"([\u4e00-\u9fff]{2,4})\1+", r"\1"),   # 2~4 字词整段重复，只保留一份
    (r"([\u4e00-\u9fff])\1{3,}", r"\1"),      # 单字连续≥4 次才折叠（保留"哈哈哈"这类正常语气）
]

# 行级噪声：时间戳、页码、页眉页脚、"第 N 页 共 M 页"
NOISE_LINE_PATTERNS: list[tuple[str, str]] = [
    (r"^\s*\[?\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d{1,3})?\s*(?:[-~—>]{1,2}>?\s*\[?\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d{1,3})?)?\]?\s*$", ""),
    (r"^\s*\[?\d{1,2}:\d{2}(?::\d{2})?\]?\s*", ""),        # 行首时间戳（正文前的噪声）
    (r"^\s*第\s*\d+\s*页\s*(?:/\s*共?\s*\d+\s*页|共\s*\d+\s*页)?\s*$", ""),
    (r"^\s*[-—]?\s*\d{1,3}\s*[-—]?\s*$", ""),               # 纯页码行
    (r"^\s*共\s*\d+\s*页\s*$", ""),
    (r"^\s*(?:课件|讲义|教案)\s*[·:：]?\s*第?\s*\d*\s*页?\s*$", ""),
    (r"^\s*(?:人教版|北师大版|苏教版|外研版)?\s*(?:数学|语文|英语|物理|化学|生物|历史|地理|政治)\s*[·\-—]?\s*(?:三年级|四年级|五年级|六年级|七年级|八年级|九年级|高一|高二|高三)?[上下]册\s*$", ""),
]
NOISE_LINE_RE = re.compile("|".join(f"(?:{p})" for p, _ in NOISE_LINE_PATTERNS))

#: 页眉页脚重复行识别阈值：在多块文本中重复出现、且不超过 30 字的行
HEADER_FOOTER_MIN_REPEAT = 3
HEADER_FOOTER_MAX_CHARS = 30

# --------------------------------------------------------------------------- #
# 隐私正则（未成年人场景硬要求）
# --------------------------------------------------------------------------- #
PII_PATTERNS: dict[str, list[str]] = {
    # 手机号：11 位，1 开头，第二位 3-9；允许 138 1234 5678 / 138-1234-5678 写法
    "phone": [
        r"(?<!\d)1[3-9]\d[\s\-]?\d{4}[\s\-]?\d{4}(?!\d)",
        r"(?<!\d)0\d{2,3}[\s\-]?\d{7,8}(?!\d)",     # 座机
    ],
    # 身份证：18 位（末位可为 X）或 15 位老号
    "id_card": [
        r"(?<!\d)[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx](?!\d)",
        r"(?<!\d)[1-9]\d{5}\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}(?!\d)",
    ],
    # QQ / 微信：允许「是/为/号/号码/加/：」等连接词（真实课堂里"我QQ是123456789"比"QQ：123"更常见）。
    # 数组顺序 = 最长/最具体优先（redact_pii 用交替正则单次扫描，先匹配到的优先）。
    "im_account": [
        r"(?:微信号|微信号码|wechat\s*id|WeChat\s*ID)\s*(?:是|为|:|：)?\s*[A-Za-z][\w\-]{4,19}",
        r"(?:微信|weixin|wechat|WX|wx|VX|vx)\s*(?:号|号码)?\s*(?:是|为|:|：)?\s*[A-Za-z][\w\-]{4,19}",
        r"(?:QQ|qq|Qq)\s*(?:号|号码)?\s*(?:是|为|＝|=|:|：)?\s*\d{5,12}",
    ],
    # 学校名：只接受「行政区划 + 校名」或带明确命名词（实验/中心/附属/第X）的形式。
    # 不能再用「任意汉字 + 小学」兜底，否则「那小学的时候」也会被误删。
    "school": [
        r"(?:[\u4e00-\u9fff]{1,8}(?:省|市|区|县|镇|乡|街道|路|街)){1,3}"
        r"[\u4e00-\u9fff]{1,10}(?:大学|学院|中学|初中|高中|小学|学校|实验学校|附属小学|附属中学|附小|附中|幼儿园|教育培训|培训学校|教育集团|辅导班)",
        r"[\u4e00-\u9fff]{1,8}(?:一小|二小|三小|一中|二中|三中|实验小学|实验中学|实验学校|中心小学|附属小学|附属中学|附小|附中)",
    ],
}

#: 学生姓名识别的"上下文锚点"。中文姓名没有区分词边界，纯靠 2~4 个汉字猜命中率太低，
#: 因此只在高精度上下文里替换：① 面称（"张小明同学"）；② 动作/言语框架（"张小明说"）。
STUDENT_NAME_ANCHORS = r"(?:同学|小朋友|小同学|学生|宝宝)"
#: 自我介绍锚点：真实课堂上最高频的姓名暴露场景（"我是三年二班的张小明""我叫李红"）。
#: 用「贪婪 + 回退」锁定最后一个"班的/组的"之后的名字，避免把"三年二班"整段吃掉。
SELF_INTRO_ANCHORS = (
    r"我是[\u4e00-\u9fff]{0,12}班的",
    r"我是[\u4e00-\u9fff]{0,12}组的",
    r"我是[\u4e00-\u9fff]{0,8}学校的",
    r"我叫",
    r"我的名字叫",
    r"我的名字是",
)
STUDENT_ACTION_ANCHORS = (
    r"(?:说|回答|问|讲|读|写|做|画|举手|来|试试|分享|上台)"
)

#: 出现在姓名位置的常见非姓名词（否则"各位同学""每个同学"会被误脱敏）
NAME_STOPWORDS = {
    "各位", "每个", "这位", "那位", "这些", "那些", "我们", "你们", "他们", "她们", "咱们",
    "全班", "大家", "所有", "其他", "别的", "有的", "很多", "几个", "一个", "两个", "三个",
    "哪些", "什么", "哪个", "谁的", "名字", "姓名", "一位", "两位", "几位", "两名", "三名",
    "小组", "男生", "女生", "孩子", "小朋友", "新来的", "穿红", "戴眼镜", "最后一", "第一",
    "下面", "上面", "前面", "后面", "左边", "右边", "中间", "今天", "刚才", "现在", "下次",
    "请你", "请问", "谢谢", "老师", "咱们班", "咱们组",
    # 单字虚词：贪婪匹配可能把它们裹进"姓名"，命中即放弃本次替换（宁可少脱敏，不可误伤）
    "的", "了", "是", "在", "和", "与", "请", "让", "找", "叫", "给", "对", "跟", "向",
}

#: 姓名占位符（与 taxonomy.utterance_function[UF_ROLL_CALL].slot_required 的槽位名一致）
PII_PLACEHOLDERS = {
    "phone": "{phone}",
    "id_card": "{id_card}",
    "im_account": "{im_account}",
    "school": "{school}",
    "student_name": "{student_name}",
}

# --------------------------------------------------------------------------- #
# 标点归一
# --------------------------------------------------------------------------- #
# 策略：先把常见英文标点收成中文形态（课堂文本以中文为主），
# 再把"中文紧邻英文标点"的情形补成中文标点，最后清多余空白。
# 数字/公式里的 `.` `-` `%` 等不受影响（它们两侧不是汉字）。
ASCII_PUNCT_MAP = [
    ("...", "……"), ("…", "……"), ("~", "～"),
    ("?", "？"), ("!", "！"), (";", "；"), (",", "，"), (":", "："),
]
CJK_PUNCT_MAP = [
    (r"(?<=[\u4e00-\u9fff])\.(?=[\u4e00-\u9fff]|$)", "。"),
    (r"(?<=[。！？；，]{1})\.+", ""),
]
MATH_MAP = [("＝", "="), ("＋", "+"), ("－", "−"), ("％", "%")]

#: 各类敏感信息的初始计数（student_name 走上下文匹配，不在 PII_PATTERNS 里）
_EMPTY_HITS = {k: 0 for k in (*PII_PATTERNS, "student_name")}


# =========================================================================== #
# 二、清洗原语（单测直接引用这些函数）
# =========================================================================== #
def remove_fillers(text: str) -> tuple[str, int]:
    """去 ASR 口头禅与重复，返回 ``(新文本, 命中次数)``。"""
    hits = 0
    out = text
    for pat, repl in FILLER_PATTERNS + REPEAT_PATTERNS:
        out, n = re.subn(pat, repl, out)
        hits += n
    # 清理被删空后的残留标点（如 "，，" 或行首 "，…"）
    out = re.sub(r"[，,]{2,}", "，", out)
    out = re.sub(r"^\s*[，,、]\s*", "", out)
    out = re.sub(r"[，,]\s*(?=[。！？；])", "", out)
    return out, hits


def remove_noise_lines(text: str, repeated_lines: set[str] | None = None) -> tuple[str, int]:
    """去时间戳行、页码行、页眉页脚行，返回 ``(新文本, 删除行数)``。

    行级噪声（整行就是页码/时间码）直接删行；行首时间码只剥掉前缀，
    因为后面往往跟着真实讲授内容（"00:01:20 我们来看第三题"）。
    """
    repeated = repeated_lines or set()
    kept: list[str] = []
    removed = 0
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            kept.append("")
            continue
        if NOISE_LINE_RE.match(stripped) or stripped in repeated:
            removed += 1
            continue
        out = re.sub(r"^\s*\[?\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d{1,3})?\]?[\s:：\-—]*", "", stripped)
        out = out.strip()
        if not out:
            removed += 1
            continue
        kept.append(out)
    return "\n".join(kept), removed


def find_repeated_lines(blocks: list[dict]) -> set[str]:
    """跨块统计重复短行 → 页眉页脚候选（同一行出现 ≥3 次且 ≤30 字）。"""
    counter: Counter[str] = Counter()
    for b in blocks:
        seen_in_block = set()
        for line in str(b.get("text") or "").split("\n"):
            s = line.strip()
            if 0 < len(s) <= HEADER_FOOTER_MAX_CHARS and s not in seen_in_block:
                counter[s] += 1
                seen_in_block.add(s)
    return {line for line, n in counter.items() if n >= HEADER_FOOTER_MIN_REPEAT}


#: 常见中文姓氏（单姓）——把"任意 2~4 汉字 + 动词"这种极松的启发式收紧成
#: "以真实姓氏开头"。中文没有词边界，实测这条守卫是必需的：
#:     「欢迎来到今天的数学课」 → 旧规则把「欢迎」当姓名 → "{student_name}来到今天的数学课"
#:     「谁来说说你的答案？」   → 旧规则把「谁来」当姓名 → "{student_name}说你的答案？"
#: 两处都是**把正文改坏**（不只是漏检/误检），TTS 会真的念出占位符。
#: 用姓氏表后：欢/谁/这 都不是姓氏 → 不再误伤；张小明/李红 → 正常命中。
#: 取舍：召回率不是 100%（生僻姓氏会漏），因此另有 detect_pii_leaks 兜底告警；
#: 生产环境仍应叠加花名册词典（见 docs/02 §4.1）。
COMMON_SURNAMES = set(
    "赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜"
    "戚谢邹喻柏水窦章云苏潘葛奚范彭郎鲁韦昌马苗凤花方俞任袁柳酆鲍史唐"
    "费廉岑薛雷贺倪汤滕殷罗毕郝邬安常乐于时傅皮卞齐康伍余元卜顾孟平黄"
    "和穆萧尹姚邵湛汪祁毛禹狄米贝明臧计伏成戴谈宋茅庞熊纪舒屈项祝董梁"
    "杜阮蓝闵席季麻强贾路娄危江童颜郭梅盛林刁钟徐邱骆高夏蔡田樊胡凌霍"
    "虞万支柯昝管卢莫经房裘缪干解应宗丁宣贲邓郁单杭洪包诸左石崔吉钮龚"
    "程嵇邢滑裴陆荣翁荀羊惠甄曲家封芮羿储靳汲邴糜松井段富巫乌焦巴弓"
    "牧隗山谷车侯宓蓬全郗班仰秋仲伊宫宁仇栾暴甘厉戎祖武符刘景詹束龙"
    "叶幸司韶郜黎蓟薄印宿白怀蒲邰从鄂索咸籍赖卓蔺屠蒙池乔阴胥能苍双"
    "闻莘党翟谭贡劳逄姬申扶堵冉宰郦雍桑桂濮牛寿通边扈燕冀郏浦尚农"
    "温别庄晏柴瞿阎充慕连茹习宦艾鱼容向古易慎戈廖庾终暨居衡步都耿满弘"
    "匡国文寇广禄阙东欧殳沃利蔚越夔隆师巩聂晁勾敖融冷訾辛阚那简饶空"
    "曾毋沙乜养鞠须丰巢关蒯相查后荆红游竺权逯盖益桓公"
)

#: 常见复姓（两字），命中则直接认可
COMPOUND_SURNAMES = (
    "欧阳", "太史", "端木", "上官", "司马", "东方", "独孤", "南宫", "万俟", "闻人",
    "夏侯", "诸葛", "尉迟", "公羊", "赫连", "澹台", "皇甫", "宗政", "濮阳", "公冶",
    "太叔", "申屠", "公孙", "慕容", "仲孙", "钟离", "长孙", "宇文", "司徒", "鲜于",
    "司空", "闾丘", "子车", "亓官", "司寇", "巫马", "公西", "颛孙", "壤驷", "公良",
    "漆雕", "乐正", "宰父", "谷梁", "拓跋", "夹谷", "轩辕", "令狐", "段干", "百里",
    "呼延", "东郭", "南门", "羊舌", "微生", "梁丘", "左丘", "东门", "西门", "南荣",
)


def _is_name_candidate(cand: str) -> bool:
    """姓名候选是否可信：长度 2~4、不含虚词/集体称呼、且**以真实姓氏开头**。

    三处守卫缺一不可：
      1. 长度 2~4；
      2. 停用词保护——**多字词按"包含"判断**。旧实现只做整词相等
         （``cand in NAME_STOPWORDS``），于是「这位同学」逃过保护被当成姓名；
      3. 姓氏守卫——见 ``COMMON_SURNAMES`` 的说明，这条专门挡「欢迎来到」「谁来说说」。
    """
    if len(cand) < 2 or len(cand) > 4:
        return False
    if cand in NAME_STOPWORDS:
        return False
    # 多字停用词：包含即否决（"这位同学"含"这位"）
    if any(w in cand for w in NAME_STOPWORDS if len(w) >= 2):
        return False
    # 单字停用词：逐字否决
    if any(ch in NAME_STOPWORDS for ch in cand):
        return False
    # 姓氏守卫
    if cand[:2] in COMPOUND_SURNAMES:
        return True
    return cand[0] in COMMON_SURNAMES


def redact_student_names(text: str) -> tuple[str, int]:
    """脱敏学生真实姓名，返回 ``(新文本, 命中次数)``。

    四类高精度上下文（都不依赖姓名库，也不做"任意 2~4 汉字一律脱敏"）：

    1. **自我介绍**（真实课堂最高频）：``我是三年二班的张小明`` / ``我叫李红`` /
       ``我的名字叫王芳`` —— 锚点用贪婪匹配 + 回退，锁定最后一个"班的"之后的名字；
    2. **面称**：``张小明同学``；
    3. **动作框架**：``李红说…`` / ``王强举手``；
    4. 以上都受 ``NAME_STOPWORDS`` 保护（见 ``_is_name_candidate``）。

    设计取舍：宁可少脱敏（另有 ``detect_pii_leaks`` 兜底告警 + 教研人工复核），
    也不把"长度单位""各位同学""我们一起"打成占位符——脏数据比漏检更难挽回。
    """
    hits = 0
    # 三种姓名窗口，各自带不同的边界守卫。
    #
    # name_post 的两处关键设计（都来自实测踩坑）：
    #   · **必须以真实姓氏开头**（复姓优先），并配**非贪婪**长度。
    #     旧实现是"任意 2~4 汉字 + 非贪婪退化"，靠 `(?<![\u4e00-\u9fff])`
    #     要求匹配起点前不能是汉字——那条守卫挡住了大量真实场景：
    #     「请欧阳娜娜来回答」「那么张小明来说说」前面都是汉字，名字反而漏掉了。
    #     现在改由姓氏表保证精度，"必须紧跟在标点/句首"的限制就可以去掉。
    #   · **非贪婪**：`[\u4e00-\u9fff]{1,3}?` 先从 2 字试起，
    #     否则「张小明来说说」会贪婪吃掉"来"变成"张小明来"，
    #     把姓名和动词黏在一起（旧实现靠左边界守卫间接规避，代价是召回率）。
    surname_single = "".join(sorted(COMMON_SURNAMES))
    surname_alt = "|".join(sorted(COMPOUND_SURNAMES, key=len, reverse=True))
    name_intro = r"(?<!\{{)(?!student_name\}})[\u4e00-\u9fff]{2,4}"
    name_post = (rf"(?:(?:{surname_alt})[\u4e00-\u9fff]{{1,2}}?"
                 rf"|[{surname_single}][\u4e00-\u9fff]{{1,3}}?)")
    # 动作锚点前还必须存在点名语境或句子边界。没有这层左上下文，
    # 「能够写」「分数来说」会分别被误判成姓名「能够」「分数」。
    name_prefix = r"(?:请|让|叫|由|那么|现在|下面|接下来|^|[\s，。！？；：、])"
    out = str(text or "")

    def _repl_anchored(m: re.Match) -> str:
        nonlocal hits
        prefix, cand = m.group(1), m.group(2)
        if not _is_name_candidate(cand):
            return m.group(0)
        hits += 1
        return prefix + PII_PLACEHOLDERS["student_name"]

    def _repl_name(cand: str) -> str:
        """自我介绍锚点专用：候选可信才替换，否则原样返回。"""
        nonlocal hits
        if not _is_name_candidate(cand):
            return cand
        hits += 1
        return PII_PLACEHOLDERS["student_name"]

    # 1) 自我介绍：锚点在前、姓名在后（"我叫李红" → "我叫{student_name}"）
    for anchor in SELF_INTRO_ANCHORS:
        out, _ = re.subn(rf"({anchor})({name_intro})",
                         lambda m: m.group(1) + _repl_name(m.group(2)), out)
    # 2)+3) 面称/动作框架：姓名在前、锚点在后的 lookahead
    for anchor in (STUDENT_NAME_ANCHORS, STUDENT_ACTION_ANCHORS):
        out, _ = re.subn(
            rf"({name_prefix})({name_post})(?={anchor})", _repl_anchored, out
        )
    return out, hits


def redact_pii(text: str) -> tuple[str, dict[str, int]]:
    """隐私脱敏主函数，返回 ``(新文本, {类别: 命中数})``。

    两处"必须这样写"的细节：

    1. **顺序固定**：先长模式（身份证）后短模式（手机号），再 IM 号/学校名，最后姓名。
       否则身份证里的 11 位数字片段会先被手机号规则吃掉，造成"半截身份证"漏网。
    2. **IM 号单次匹配**：QQ/微信的几条正则彼此有重叠（放宽连接词之后更明显），
       若逐条 ``subn`` 会把同一处命中两次、``pii_hits`` 计数虚高。因此把 IM 相关模式
       合成一条交替正则，**一次扫描 + 最长匹配优先**（Python ``re`` 交替天然左优先，
       把"更长/更具体"的模式排在前面），保证同一处只替换、只计数一次。
    """
    hits = dict(_EMPTY_HITS)
    out = text
    for kind in ("id_card", "phone", "school"):
        for pat in PII_PATTERNS[kind]:
            out, n = re.subn(pat, PII_PLACEHOLDERS[kind], out)
            hits[kind] += n
    # IM 号：单次扫描器（见 docstring 第 2 点）
    out, n_im = _redact_im_accounts(out)
    hits["im_account"] += n_im
    out, n_name = redact_student_names(out)
    hits["student_name"] += n_name
    return out, hits


def _im_placeholder(m: re.Match) -> str:
    """把 ``QQ是123456789`` 整体换成 ``{im_account}``，但保留可读的连接词。"""
    frag = m.group(0)
    label = re.match(r"\s*(?:QQ|qq|Qq|微信|weixin|wechat|WeChat|WX|wx|VX|vx|微信号|微信号码)", frag)
    return (label.group(0).strip() + PII_PLACEHOLDERS["im_account"]) if label else PII_PLACEHOLDERS["im_account"]


def _redact_im_accounts(text: str) -> tuple[str, int]:
    """QQ/微信单次扫描替换：返回 ``(新文本, 命中数)``，同一处只计一次。"""
    combined = "|".join(f"(?:{p})" for p in PII_PATTERNS["im_account"])
    return re.subn(combined, _im_placeholder, str(text or ""))


#: "漏检自检"用到的模式：清洗后仍能命中，说明脱敏可能没盖住，必须显式告警而非静默放过
LEAK_HINTS: list[tuple[str, str]] = [
    (r"我是.{0,6}班的[\u4e00-\u9fff]{2,4}", "自我介绍里的疑似姓名"),
    (r"我叫[\u4e00-\u9fff]{2,4}", "自我介绍里的疑似姓名"),
    (r"我的名字(?:叫|是)[\u4e00-\u9fff]{2,4}", "自我介绍里的疑似姓名"),
    (r"[\u4e00-\u9fff]{2,4}同学", "疑似学生姓名（面称）"),
    (r"(?:QQ|qq|微信|wechat|WX|wx|VX|vx)\s*(?:号|号码)?\s*(?:是|为|:|：)?\s*[A-Za-z0-9][\w\-]{4,}", "疑似 QQ/微信号"),
    (r"(?<!\d)1[3-9]\d{9}(?!\d)", "疑似手机号"),
    (r"(?<!\d)\d{17}[\dXx](?!\d)", "疑似身份证号"),
]


def detect_pii_leaks(text: str) -> list[str]:
    """漏检自检：在**已脱敏**的文本上再扫一遍，返回疑似残留的提示。

    为什么需要它：正则脱敏必然有召回上限（姓名尤甚），而"静默漏检"是合规上最危险的形态——
    文档写着"入库前必须脱敏"，实现却悄悄漏掉，出事时无法追溯。因此这里宁可多提示，
    把可疑片段交给人工复核（教研在 ``data/review`` 阶段处理）。

    为避免噪声：命中片段里若只含占位符或已在 ``NAME_STOPWORDS`` 中的集体称呼，则不算漏检。
    """
    if not text:
        return []
    leaks: list[str] = []
    for pattern, label in LEAK_HINTS:
        for m in re.finditer(pattern, text):
            frag = m.group(0)
            if "{student_name}" in frag or "{phone}" in frag or "{im_account}" in frag:
                continue
            # 抽掉锚点/后缀，只留"疑似姓名"部分再判定
            core = re.sub(
                r"(?:同学|小朋友|学生|老师|我的名字(?:叫|是)|我是|我叫|班的|QQ|qq|微信|号|号码)",
                "",
                frag,
            )
            if not core or all(ch in NAME_STOPWORDS for ch in core):
                continue
            if "姓名" in label and not any(
                _is_name_candidate(core[-size:])
                for size in (2, 3, 4)
                if len(core) >= size
            ):
                continue
            # 集体称呼（"各位同学"）不是姓名，属于已知的可接受漏检，不报噪声
            if any(w in frag for w in ("各位", "每个", "这位", "那位", "全班", "大家",
                                      "所有", "其他", "有的", "一个", "两个", "三个")):
                continue
            leaks.append(f"{label}『{frag}』可能未被脱敏")
    # 同类提示去重，避免同一片段刷屏
    return list(dict.fromkeys(leaks))


def normalize_punct(text: str) -> str:
    """全角/半角与中英文标点归一（保留中文标点形态，TTS 依赖它们断句）。"""
    out = common.fullwidth_to_halfwidth(str(text or ""))
    for src, dst in ASCII_PUNCT_MAP:
        out = out.replace(src, dst)
    for pat, repl in CJK_PUNCT_MAP:
        out = re.sub(pat, repl, out)
    for src, dst in MATH_MAP:
        out = out.replace(src, dst)
    out = re.sub(r"[ \t]{2,}", " ", out)
    out = re.sub(r"[ \t]+(?=[，。！？；：、）】」])", "", out)   # 中文标点前的空格
    out = re.sub(r"(?<=[（【「])[ \t]+", "", out)
    out = re.sub(r"[，,]{2,}", "，", out)
    return out.strip()


def clean_text(text: str, repeated_lines: set[str] | None = None) -> tuple[str, dict[str, Any]]:
    """对单块文本跑完整清洗链，返回 ``(清洗后文本, 本块统计)``。

    ``stats["warnings"]`` 会带出"漏检自检"的结果——**不阻塞流程，但绝不静默**。
    """
    raw = str(text or "")
    stats: dict[str, Any] = {"raw_chars": len(raw)}
    out, removed_noise = remove_noise_lines(raw, repeated_lines)
    out, removed_filler = remove_fillers(out)
    out, pii_hits = redact_pii(out)
    out = normalize_punct(out)
    leaks = detect_pii_leaks(out)
    stats.update({
        "clean_chars": len(out),
        "removed": {"noise_line": removed_noise, "filler": removed_filler},
        "pii_hits": pii_hits,
        "pii_hit_total": sum(pii_hits.values()),
        "pii_leak_warnings": leaks,
    })
    return out, stats


def clean_document(parsed: dict) -> dict:
    """清洗一份 s1 解析结果，返回 ``data/clean/<name>.json`` 的结构。"""
    blocks = parsed.get("blocks") or []
    repeated = find_repeated_lines(blocks)
    warnings: list[str] = list(parsed.get("warnings") or [])
    if repeated:
        warnings.append(f"识别到疑似页眉页脚重复行 {len(repeated)} 条，已删除")

    clean_blocks: list[dict] = []
    agg = {"noise_line": 0, "filler": 0}
    pii_total = dict(_EMPTY_HITS)
    raw_chars = clean_chars = 0
    for b in blocks:
        text, st = clean_text(b.get("text") or "", repeated)
        raw_chars += st["raw_chars"]
        clean_chars += st["clean_chars"]
        for k, v in st["removed"].items():
            agg[k] = agg.get(k, 0) + v
        for k, v in st["pii_hits"].items():
            pii_total[k] += v
        nb = {k: v for k, v in b.items() if k != "text"}
        nb["raw_text"] = b.get("text") or ""
        nb["text"] = text
        nb["pii_hits"] = st["pii_hits"]
        nb["pii_leak_warnings"] = st.get("pii_leak_warnings") or []
        clean_blocks.append(nb)

    non_empty = [b for b in clean_blocks if b["text"]]
    leak_warnings = list(dict.fromkeys(w for b in non_empty for w in (b.get("pii_leak_warnings") or [])))
    stats = {
        "raw_chars": raw_chars,
        "clean_chars": clean_chars,
        "char_delete_ratio": round(1 - (clean_chars / raw_chars), 4) if raw_chars else 0.0,
        "removed": agg,
        "pii_hits": pii_total,
        "pii_hit_total": sum(pii_total.values()),
        "pii_leak_warnings": leak_warnings,
        "segments_before": len(blocks),
        "segments_after": len(non_empty),
        "empty_blocks_dropped": len(blocks) - len(non_empty),
    }
    if stats["empty_blocks_dropped"]:
        warnings.append(f"清洗后 {stats['empty_blocks_dropped']} 个块为空，已丢弃")
    # 漏检自检：不阻塞，但必须显式提示人工复核（合规硬要求，绝不静默）
    for w in leak_warnings:
        warnings.append(f"【脱敏漏检自检】{w}；请人工确认是否需要补充脱敏")
    return {
        "source_name": parsed.get("source_name"),
        "source_path": parsed.get("source_path"),
        "source_type": parsed.get("source_type"),
        # 素材角色必须一路带下去：s3 靠它决定"该不该切分"（教案不切分、扫描件先 OCR）。
        # 早期这里漏了透传，导致 s3 拿不到 source_role，路由形同虚设。
        "source_role": parsed.get("source_role"),
        "source_hash": parsed.get("source_hash"),
        "parser": parsed.get("parser"),
        "cleaned_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "text": "\n\n".join(b["text"] for b in non_empty),
        "blocks": non_empty,
        "meta": parsed.get("meta") or {},
        "stats": stats,
        "warnings": warnings,
    }


def clean_dir(in_dir: str | Path, out_dir: str | Path) -> list[Path]:
    """批量清洗目录下所有 ``*.json``，输出同名文件到 ``out_dir``。"""
    files = common.iter_input_files(in_dir, [".json"])
    if not files:
        log.warning("输入目录 %s 下没有 s1 解析产物（*.json），请先跑 s1_parse.py", in_dir)
        return []
    out = common.ensure_dir(out_dir)
    written: list[Path] = []
    for f in files:
        try:
            doc = clean_document(common.read_json(f))
        except Exception as exc:  # noqa: BLE001
            log.error("清洗失败 %s：%s: %s", f.name, type(exc).__name__, exc)
            continue
        target = out / f.name
        common.write_json(target, doc)
        written.append(target)
        s = doc["stats"]
        log.info("清洗完成 %s → 删除率 %.1f%%，脱敏 %d 处（%s）",
                 f.name, s["char_delete_ratio"] * 100, s["pii_hit_total"],
                 {k: v for k, v in s["pii_hits"].items() if v})
        if s["pii_hit_total"]:
            log.warning("  ↑ 该素材含敏感信息，已按硬性要求脱敏，请人工复核占位符是否合理")
    log.info("本批共清洗 %d/%d 个文件，输出目录 %s", len(written), len(files), out)
    return written


# =========================================================================== #
# 三、CLI
# =========================================================================== #
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="s2_clean",
        description="Stage2 清洗：去口头禅/噪声 + 隐私脱敏 + 标点归一，输出 data/clean/<name>.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  python rag/offline/rag/offline/s2_clean.py --in data/parsed --out data/clean\n"
            "  python rag/offline/rag/offline/s2_clean.py --text \"大家好，我是三年二班的张小明\" --show\n"
        ),
    )
    common.add_in_out_args(ap, "data/parsed", "data/clean")
    ap.add_argument("--text", help="只清洗这一条文本（排查正则用，不读目录）")
    ap.add_argument("--show", action="store_true", help="配合 --text：打印清洗前后对照与统计")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.text is not None:
        cleaned, stats = clean_text(args.text)
        if args.show:
            print(f"原文：{args.text}")
            print(f"清洗：{cleaned}")
            print(f"统计：{stats}")
        else:
            print(cleaned)
        return 0
    in_dir = Path(args.in_dir)
    if not in_dir.exists():
        log.error("输入目录不存在：%s（请先跑 python rag/offline/rag/offline/s1_parse.py）", in_dir)
        return 1
    written = clean_dir(in_dir, args.out_dir)
    return 0 if written or not common.iter_input_files(in_dir, [".json"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
