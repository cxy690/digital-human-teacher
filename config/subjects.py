"""学科元数据：学科选择页与后端提示词的单一事实来源。

新增学科只需在此追加一条；前端学科卡与 ``GET /api/subjects`` 会自动生效，
无需改动其它代码（避免学科清单散落在前后端两处）。
"""
from __future__ import annotations

from typing import Any

# key 用于 URL 传参（英文，稳定）；name 用于显示与提示词（中文）
SUBJECTS: list[dict[str, Any]] = [
    {
        "key": "chinese",
        "name": "语文",
        "emoji": "📖",
        "desc": "字词句篇 · 阅读理解 · 古诗文 · 写作表达",
        "grades": "小学 · 初中",
        "color": "#f59e0b",
    },
    {
        "key": "math",
        "name": "数学",
        "emoji": "➗",
        "desc": "数与代数 · 图形几何 · 应用题与方程",
        "grades": "小学 · 初中",
        "color": "#1677ff",
    },
    {
        "key": "english",
        "name": "英语",
        "emoji": "🔤",
        "desc": "词汇 · 语法 · 句型 · 听说读写",
        "grades": "小学 · 初中",
        "color": "#22c55e",
    },
    {
        "key": "science",
        "name": "科学",
        "emoji": "🔬",
        "desc": "观察与实验 · 生命与环境 · 物质与能量",
        "grades": "小学",
        "color": "#06b6d4",
    },
    {
        "key": "physics",
        "name": "物理",
        "emoji": "🧲",
        "desc": "力与运动 · 声光热 · 电学与能量",
        "grades": "初中",
        "color": "#8b5cf6",
    },
    {
        "key": "chemistry",
        "name": "化学",
        "emoji": "⚗️",
        "desc": "物质构成 · 化学方程式 · 酸碱盐",
        "grades": "初中",
        "color": "#06b6d4",
    },
    {
        "key": "biology",
        "name": "生物",
        "emoji": "🧬",
        "desc": "细胞与生物体 · 绿色植物 · 人体与健康",
        "grades": "初中",
        "color": "#10b981",
    },
    {
        "key": "politics",
        "name": "政治",
        "emoji": "🏛️",
        "desc": "思想品德 · 时事政治 · 法治与责任",
        "grades": "初中",
        "color": "#ef4444",
    },
    {
        "key": "morality",
        "name": "道德与法治",
        "emoji": "⚖️",
        "desc": "成长的节拍 · 友谊与师长 · 法治教育",
        "grades": "小学 · 初中",
        "color": "#eab308",
    },
    {
        "key": "geography",
        "name": "地理",
        "emoji": "🗺️",
        "desc": "地球与地图 · 气候与人口 · 中国地理",
        "grades": "小学 · 初中",
        "color": "#3b82f6",
    },
    {
        "key": "history",
        "name": "历史",
        "emoji": "📜",
        "desc": "中国古代史 · 中国近现代史 · 世界史",
        "grades": "初中",
        "color": "#f97316",
    },
]

# 名称 -> 元数据 / key -> 元数据
SUBJECT_BY_NAME: dict[str, dict[str, Any]] = {s["name"]: s for s in SUBJECTS}
SUBJECT_BY_KEY: dict[str, dict[str, Any]] = {s["key"]: s for s in SUBJECTS}

# 提示词里可用的学科名列表（便于校验/兜底）
SUBJECT_NAMES: list[str] = [s["name"] for s in SUBJECTS]

# 学科关键词：用于「学科范围护栏」——学生提问点名了其它学科时礼貌引导回本学科。
# 只需覆盖高频/特征词，不求穷尽；检测逻辑见 detect_other_subject。
SUBJECT_KEYWORDS: dict[str, list[str]] = {
    "语文": ["语文", "作文", "古诗", "古诗文", "成语", "拼音", "汉字", "造句", "病句", "修辞", "阅读理解", "文言文", "笔顺", "偏旁", "李白", "杜甫", "诗人", "词人", "作家", "课文", "作者"],
    "数学": ["数学", "数学题", "加法", "减法", "乘法", "除法", "方程", "几何", "分数", "小数", "百分数", "比例", "函数", "面积", "周长", "三角形", "圆", "计算", "应用题", "约分", "通分"],
    "英语": ["英语", "英文", "单词", "语法", "时态", "完形填空", "音标", "冠词", "介词", "听力", "口语", "be动词", "英语作文"],
    # 科学只保留「专属特征词」：实验/观察/探究/猜想 之类是跨学科通用词，
    # 一旦写进来，物理/化学/生物学生一提「实验」就会被误判成科学（见 _GENERIC_TERMS）。
    # 不含「光合作用」：它是生物的核心特征词，留给生物独占（小学科学讲这块也归生命科学领域，
    # 但判重会让「光合作用」在科学/生物之间摇摆，索性交给生物唯一持有）。
    "科学": ["科学", "磁铁", "植物生长", "水和空气"],
    "物理": ["物理", "速度", "加速度", "压强", "浮力", "杠杆", "电路", "电阻", "电流", "电压", "磁场", "折射", "惯性", "功率"],
    "化学": ["化学", "元素", "分子", "原子", "氧化", "还原", "溶液", "酸碱", "化学方程式", "催化剂", "化学式"],
    "生物": ["生物", "细胞", "光合作用", "基因", "遗传", "植物", "动物", "血液", "消化", "神经系统", "生态系统"],
    "政治": ["政治", "思想品德", "时事", "人民代表大会", "社会主义"],
    "道德与法治": ["道德与法治", "道德", "法治", "宪法", "法律", "公民", "未成年人保护"],
    "地理": ["地理", "地图", "气候", "经度", "纬度", "经纬", "地形", "板块", "季风", "人口分布"],
    "历史": ["历史", "朝代", "古代史", "近代史", "世界大战", "革命", "变法", "文明", "丝绸之路", "辛亥革命"],
}

_DEFAULT = "综合"


# 跨学科通用词：几乎每个学科都会出现（做实验、观察现象、提出猜想、分析数据…）。
# 这些词可以用来判定「属于本学科」，但**绝不能**用来判定「跑到别的学科去了」——
# 否则物理/化学/生物学生一提「实验」就会被判成科学，触发教学护栏误拦。
# 历史坑：合并新增「科学」学科时把「实验/观察/探究」写进了科学的特征词，
# 导致物理/化学/生物学生问「这个实验怎么做」被误判为跨学科。
_GENERIC_TERMS: frozenset[str] = frozenset({
    "实验", "观察", "探究", "猜想", "假设", "分析", "数据", "结论", "规律",
    "方法", "过程", "原因", "原理", "现象", "应用", "特点", "区别", "关系",
    "力的作用", "天气",
})


# 数字人素材与学科的对应关系**不再在这里维护**：
# 学科 -> 老师 -> 底片 由 `config/teachers.py` 按 `data/teachers/<teacher_key>/teacher.json`
# 的 `subject` 字段解析（同科可挂多位老师），引擎侧按 `teacher_id` 取 `<key>/talk.mp4`。


def resolve_subject(value: str | None) -> str:
    """把前端传来的学科（key 或中文名）归一化为中文名；不认识则原样返回。"""
    raw = (value or "").strip()
    if not raw:
        return _DEFAULT
    if raw in SUBJECT_BY_NAME:
        return raw
    if raw in SUBJECT_BY_KEY:
        return str(SUBJECT_BY_KEY[raw]["name"])
    return raw


def detect_other_subject(text: str, current_subject: str) -> str | None:
    """检测提问是否明确指向**其它**学科；返回命中的学科名，未命中返回 None。

    规则（避免误伤）：
    * 问题里包含**本学科**关键词 → 视为在范围内，直接放行；
    * 问题里不含本学科关键词、但点名了**其它**学科（学科名或其特征词）→ 返回该学科；
    * 两者都不含（如「这道题怎么解」）→ 不拦，交给 LLM 正常答疑。

    跨学科通用词（见 :data:`_GENERIC_TERMS`）在第二步会被剔除：它们几乎每个学科
    都会出现，用它判「跑到别的学科」会大量误伤（如物理课上的「实验」）。
    """
    question = (text or "").strip()
    if not question:
        return None
    current = (current_subject or "").strip()
    current_kws = SUBJECT_KEYWORDS.get(current, [])
    if any(kw in question for kw in ([current] if current else []) + current_kws):
        return None                      # 涉及本学科，放行
    # 命中判据取「最长的那条特征词」，而不是「学科列表里排最前的那个学科」：
    # 例：「化学方程式」同时能被数学的「方程」和化学的「化学方程式」命中，
    # 按学科顺序会误判成数学，按最长匹配才会判到化学。
    best_name: str | None = None
    best_len = 0
    for name in SUBJECT_NAMES:
        if name == current:
            continue
        kws = [name] + SUBJECT_KEYWORDS.get(name, [])
        for kw in kws:
            # 学科名本身（如「科学」）始终参与判定；通用词一律不参与
            if kw in _GENERIC_TERMS and kw != name:
                continue
            if kw in question and len(kw) > best_len:
                best_name, best_len = name, len(kw)
    return best_name
