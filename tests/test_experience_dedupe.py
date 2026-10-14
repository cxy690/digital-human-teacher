"""L5 语义归并的测试：模型给的分组建议，必须能被安全地收敛成「只读的建议」。

这里钉的重点不是「模型判得准不准」（那取决于模型），而是**模型判错/乱说时会不会
造成损失**。归并错了会永久丢掉一个独立的教学要点，所以对模型输出的每一条
都要有防线：id 不存在、只给一条、``keep`` 指向组外、同一条出现在两组 —— 四种
真实会发生的失误，逐个钉住。

另外钉住 ``core/llm_json.parse_json_object``：它替代了 ``failure_cluster`` 里的
私有实现，行为必须**兼容且更强**（多一种噪声容忍）。
"""
from __future__ import annotations

import json

import pytest

from core.llm_json import parse_json_object
from core.skills.experience import find_duplicate_groups
from core.skills.experience.store import ExperienceStore


class _StubLLM:
    """桩模型：固定回一段文本，记录被调用了几次。"""

    def __init__(self, reply: str, *, boom: bool = False) -> None:
        self.reply = reply
        self.boom = boom
        self.calls = 0
        self.prompts: list[str] = []

    async def chat(self, messages, **kwargs):  # noqa: ARG002 - 桩只关心内容与次数
        self.calls += 1
        self.prompts.append("\n".join(str(m.get("content")) for m in messages))
        if self.boom:
            raise RuntimeError("假的模型故障")
        return self.reply


def _seed(tmp_path, notes: list[str]) -> ExperienceStore:
    store = ExperienceStore(root=tmp_path)
    for note in notes:
        assert store.observe("math", "pitfall", note, confidence=0.7) is not None
    return store


# ---------------------------------------------------------------- 抓得到
async def test_finds_semantically_same_items(tmp_path) -> None:
    """核心场景：**字面看不出来、意思却相同**的两条，能被模型判成一组。

    这两句来自云端真实数据 —— 字面相似度 0.348（无关条目是 0.07），
    字面方法够不着，本模块就是为它们写的。
    """
    store = _seed(tmp_path, [
        "学生常只对一边运算，或把未跨等号的项也变号，需强调只有跨过等号的项才变号",
        "学生常只对含未知数的一边操作，忘记另一边同步变化，需强调等式如天平两边同动",
    ])
    ids = [it.id for it in store.all_items("math")]
    llm = _StubLLM(json.dumps({"groups": [
        {"ids": ids, "keep": ids[0], "drop": [ids[1]], "why": "都在讲「只对一边操作」"},
    ]}, ensure_ascii=False))

    result = await find_duplicate_groups("math", store=store, llm=llm)

    assert result["ok"] is True
    assert len(result["groups"]) == 1
    group = result["groups"][0]
    assert group["keep"] == ids[0] and group["drop"] == [ids[1]]
    assert group["why"] and set(group["texts"]) == set(ids), "建议里要带上原文，人才好判断"


async def test_scanned_counts_what_was_compared(tmp_path) -> None:
    """如实报告「比较了几条」—— 条数被截断时人要看得出来。

    文本刻意取得彼此无关：写「第 1 条 / 第 2 条…」这种只差一个数字的，
    实测相似度 0.929，会被 ``observe`` 的去重合并成一条，条数就对不上了。
    """
    store = _seed(tmp_path, [
        "移项要变号，两边同时加减同一个数",
        "讲数轴要强调原点、正方向、单位长度",
        "负号看错会导致整题全错",
        "解完方程记得代回原式检验",
    ])
    llm = _StubLLM('{"groups": []}')

    result = await find_duplicate_groups("math", store=store, llm=llm)

    assert result["scanned"] == {"math": 4}
    assert llm.calls == 1


async def test_no_duplicates_returns_empty(tmp_path) -> None:
    store = _seed(tmp_path, ["移项要变号", "讲数轴要强调三要素"])
    result = await find_duplicate_groups("math", store=store, llm=_StubLLM('{"groups": []}'))
    assert result["ok"] is True and result["groups"] == []


# ---------------------------------------------------------------- 模型的四类失误
async def test_hallucinated_ids_are_dropped(tmp_path) -> None:
    """① 模型编 id（把编号拼错）—— 必须丢掉，绝不能凭它去停用别的条目。"""
    store = _seed(tmp_path, ["移项要变号", "讲数轴要强调三要素"])
    llm = _StubLLM(json.dumps({"groups": [
        {"ids": ["exp-编造的", "exp-也是编的"], "keep": "exp-编造的", "why": "x"},
    ]}, ensure_ascii=False))

    result = await find_duplicate_groups("math", store=store, llm=llm)

    assert result["groups"] == [], "编造的 id 不该产出任何建议"


async def test_single_item_group_is_ignored(tmp_path) -> None:
    """② 一组只有一条 —— 那不是「重复」。"""
    store = _seed(tmp_path, ["移项要变号", "讲数轴要强调三要素"])
    real = store.all_items("math")[0].id
    llm = _StubLLM(json.dumps({"groups": [{"ids": [real], "keep": real, "why": "x"}]}))

    result = await find_duplicate_groups("math", store=store, llm=llm)

    assert result["groups"] == []


async def test_keep_outside_group_falls_back_to_first(tmp_path) -> None:
    """③ ``keep`` 指向组外 —— 保守取第一条，不能让模型把两条都丢掉。"""
    store = _seed(tmp_path, ["移项要变号", "讲数轴要强调三要素"])
    ids = [it.id for it in store.all_items("math")]
    llm = _StubLLM(json.dumps({"groups": [
        {"ids": ids, "keep": "exp-不存在", "why": "x"},
    ]}))

    result = await find_duplicate_groups("math", store=store, llm=llm)

    assert len(result["groups"]) == 1
    assert result["groups"][0]["keep"] == ids[0]
    assert result["groups"][0]["drop"] == [ids[1]] or result["groups"][0]["drop"] == ids[1:]


async def test_same_item_cannot_appear_in_two_groups(tmp_path) -> None:
    """④ 一条同时出现在两组 —— 「留哪条」会自相矛盾，后面的组必须让位。"""
    store = _seed(tmp_path, ["甲条经验内容", "乙条经验内容", "丙条经验内容"])
    a, b, c = [it.id for it in store.all_items("math")]
    llm = _StubLLM(json.dumps({"groups": [
        {"ids": [a, b], "keep": a, "why": "第一组"},
        {"ids": [b, c], "keep": c, "why": "第二组"},
    ]}, ensure_ascii=False))

    result = await find_duplicate_groups("math", store=store, llm=llm)

    assert len(result["groups"]) == 1, "第二组里 b 已被用掉，整组不该成立"
    assert result["groups"][0]["ids"] == [a, b]


async def test_drop_is_completed_when_model_omits_it(tmp_path) -> None:
    """模型只给了 ids 和 keep、忘了 drop —— 补成「除了 keep 的其余」，不能空手而归。"""
    store = _seed(tmp_path, ["甲条经验内容", "乙条经验内容"])
    a, b = [it.id for it in store.all_items("math")]
    llm = _StubLLM(json.dumps({"groups": [{"ids": [a, b], "keep": a, "why": "x"}]}))

    group = (await find_duplicate_groups("math", store=store, llm=llm))["groups"][0]

    assert group["drop"] == [b]


# ---------------------------------------------------------------- 失败时不炸
async def test_model_failure_does_not_raise(tmp_path) -> None:
    """模型调用失败：如实报错、不抛 —— 排障工具自己崩会让人以为系统坏了。"""
    store = _seed(tmp_path, ["甲条经验内容", "乙条经验内容"])
    result = await find_duplicate_groups("math", store=store, llm=_StubLLM("", boom=True))

    assert result["ok"] is True, "一位老师失败不该让整轮失败"
    assert result["groups"][0].get("error")


async def test_garbage_reply_does_not_raise(tmp_path) -> None:
    """模型回了一堆不是 JSON 的话：当作没发现重复，不抛。"""
    store = _seed(tmp_path, ["甲条经验内容", "乙条经验内容"])
    result = await find_duplicate_groups(
        "math", store=store, llm=_StubLLM("我觉得这两条挺像的，但我不打算输出 JSON。")
    )
    assert result["ok"] is True and result["groups"] == []


async def test_retired_items_are_not_compared(tmp_path) -> None:
    """已停用的不占注入名额，没必要参与归并（否则会建议「去掉」一条本来就停用的）。

    这里 seed **三条**再停用一条：只剩两条才会真的调模型 —— seed 两条再停用一条的话
    会掉到「一条无从比较」的分支，模型压根不会被调用，用例就测不到想测的东西了。
    """
    store = _seed(tmp_path, [
        "移项要变号，两边同时加减同一个数",
        "讲数轴要强调原点、正方向、单位长度",
        "负号看错会导致整题全错",
    ])
    retired = next(it for it in store.all_items("math") if "移项" in it.text)
    store.retire("math", retired.id)
    llm = _StubLLM('{"groups": []}')

    await find_duplicate_groups("math", store=store, llm=llm)

    assert llm.calls == 1, "停用后仍应剩两条参与比较"
    assert retired.text not in llm.prompts[0], "已停用的条目不该出现在提示词里"


async def test_empty_teacher_list_returns_ok(tmp_path) -> None:
    """一个老师都没有：正常返回空结果，不该去调模型。"""
    store = ExperienceStore(root=tmp_path)
    llm = _StubLLM('{"groups": []}')
    result = await find_duplicate_groups("", store=store, llm=llm)
    assert result["ok"] is True and llm.calls == 0


# ---------------------------------------------------------------- 公共 JSON 解析
@pytest.mark.parametrize("raw,expect_key", [
    ('{"a": 1}', "a"),
    ('```json\n{"a": 1}\n```', "a"),
    ('好的，分析如下：{"a": 1}', "a"),
    ('{"a": {"b": "用 {x} 表示"}}', "a"),          # 字符串里有花括号，计数不能错位
    ('先给一个例子 {"ignore": 1} 再看结论 {"a": 1}', "ignore"),  # 第一个配平的对象
])
def test_parse_json_object_tolerates_model_noise(raw: str, expect_key: str) -> None:
    parsed = parse_json_object(raw)
    assert isinstance(parsed, dict) and expect_key in parsed


@pytest.mark.parametrize("raw", ["", "没有 JSON", "[1, 2, 3]", "{坏掉的", None])
def test_parse_json_object_returns_none(raw) -> None:
    assert parse_json_object(raw) is None


def test_parse_json_object_prefers_whole_object_first() -> None:
    """兼容既有行为：整体能解出来时，就用整体（与改动前的 ``find/rfind`` 一致）。"""
    raw = '{"a": 1}'
    assert parse_json_object(raw) == {"a": 1}
