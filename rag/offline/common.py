#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
公共工具模块（rag/offline/common.py）
====================================
本模块是「教研数据接入流水线」所有 stage 的共享底座，只做四类事：

1. 受控词表读取   —— ``load_taxonomy()``：``schema/taxonomy.yaml`` 是唯一标签真源；
2. YAML 写出       —— ``write_yaml_list()``：每项前加中文注释块，方便教研 review diff；
3. 命名与哈希      —— ``slug()`` / ``stable_hash()``：ID 与幂等去重用；
4. JSON Schema 校验 —— ``load_json_schema()`` / ``validate()``：直接复用 ``schema/json/``。

依赖：仅标准库 + pyyaml + jsonschema（本仓库已安装，不需要 pip install）。

目录约定（相对仓库根目录）：::

    schema/            受控词表 + JSON Schema（只读，流水线不修改）
    samples/           教研已人工写好的成品课（只读，是流水线的目标格式基准）
    tools/lint.py      教研资产 Lint（只读，被 s6_export 调用做导出门禁）
    rag/offline/       本流水线代码（接入链路 s1~s6 + 共享底座 common）
    data/              流水线中间产物（parsed/clean/chunks/labels/qa_candidates/export）
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    import yaml
except ImportError:  # pragma: no cover - 环境缺失时给出明确指引
    sys.exit("缺少依赖 pyyaml，请先执行：pip install pyyaml")
try:
    from jsonschema import Draft202012Validator
except ImportError:  # pragma: no cover
    sys.exit("缺少依赖 jsonschema，请先执行：pip install jsonschema")


def _force_utf8_when_piped() -> None:
    """被管道/子进程捕获时强制 UTF-8 输出（在模块导入时生效，覆盖所有 stage）。

    Windows 上 stdout 一旦不是终端（被 CI、``| Select-String``、另一个 Python 进程捕获），
    Python 就按 locale 编码（中文机器上是 GBK）写字节。而各 stage 会打印 ``✓``/``✗``/``→``，
    其中 ``✓``(U+2713)、``✗``(U+2717) **GBK 无法表示** → UnicodeEncodeError →
    工具崩掉、调用方误判为流程失败。

    放在模块顶层是刻意的：每个 stage 都在打印之前 ``import common``，
    一处生效即可覆盖 s1~s6，不必逐个 main() 去加。
    只在**非交互**场景切换；交互式控制台保持原样（Python 在真控制台下走 UTF-16 写终端）。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            if stream is not None and not stream.isatty():
                stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001 - 拿不到 isatty/reconfigure 就算了，不能因此崩
            pass


_force_utf8_when_piped()

# --------------------------------------------------------------------------- #
# 采集约定（schema/source_conventions.yaml）—— 角色与格式的唯一真源
# --------------------------------------------------------------------------- #
#: 约定：PPTX→课件 / PDF→讲义·教材扫描件 / DOCX→教案
#: 这里放**公共**读取与判定，供三方共用，避免各写一份常量慢慢漂移：
#:   · rag/offline/s1_parse.py    → 解析时给每条素材打上 source_role
#:   · rag/offline/s3_chunk.py     → 按 routing 决定该不该切分
#:   · tools/audit_sources.py   → 盘点时归类与检出格式/角色不符
#:   · tools/lint.py            → 校验课程挂载素材的 role/type 是否匹配
#: 路径常量 CONVENTIONS_PATH 定义在本文件下方的「路径常量」段（那里才算得准 ROOT）。
_CONVENTIONS_CACHE: dict | None = None


def load_source_conventions() -> dict:
    """读取采集约定（带缓存）。

    契约文件读不到时**抛异常**，不静默退回默认值——约定坏了，
    角色归类与解析路由都会失准，产物不可信。这条教训来自一次真实事故：
    约定文件里的 YAML 语法错误被吞掉后，工具照常出报告、结论全错。
    """
    global _CONVENTIONS_CACHE
    if _CONVENTIONS_CACHE is None:
        try:
            with open(CONVENTIONS_PATH, encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"采集约定文件解析失败：{CONVENTIONS_PATH}\n{type(exc).__name__}: {exc}\n"
                f"提示：YAML 里以 * 开头的裸标量会被当作别名，需加引号，例如 - \"**加粗**\""
            ) from exc
        if not isinstance(data, dict) or "role_hints" not in data:
            raise RuntimeError(f"采集约定文件结构不完整（缺 role_hints）：{CONVENTIONS_PATH}")
        _CONVENTIONS_CACHE = data
    return _CONVENTIONS_CACHE


def role_expected_formats() -> dict[str, set[str]]:
    """角色 → 允许的扩展名集合（不含点，小写）。"""
    conv = load_source_conventions()
    return {k: {str(x).lower().lstrip(".") for x in v}
            for k, v in (conv.get("role_expected_formats") or {}).items()}


def ext_default_role() -> dict[str, str]:
    """扩展名 → 兜底角色。"""
    conv = load_source_conventions()
    return {str(k).lower().lstrip("."): v for k, v in (conv.get("ext_default_role") or {}).items()}


def ocr_required_roles() -> set[str]:
    conv = load_source_conventions()
    return set(conv.get("ocr_required_roles") or [])


def route_of(role: str) -> dict:
    """取某角色的流水线路由（chunk_unit / next / needs_ocr）。"""
    conv = load_source_conventions()
    return (conv.get("pipeline_routing") or {}).get(role) or {}


def s3_chunk_unit() -> str:
    """s3 只处理这个切分单位的角色（即「课堂语言」）。默认「教学环节」。"""
    conv = load_source_conventions()
    return conv.get("s3_chunk_unit") or "教学环节"


def classify_role(path: str | Path) -> str:
    """按「文件名语义关键词优先，扩展名兜底」判定素材角色。

    关键词里**不含扩展名本身**（见 source_conventions.yaml 的说明）：
    ``教案_U03_误存为pptx.pptx`` 的词干含 "pptx"，用 "ppt" 当关键词会把它误判成课件，
    而它恰恰是最该被检出"格式放错"的那一个。
    """
    p = Path(path)
    conv = load_source_conventions()
    stem = p.stem
    for role, kws in (conv.get("role_hints") or {}).items():
        for kw in kws:
            if kw and kw in stem:
                return role
    return ext_default_role().get(p.suffix.lower().lstrip("."), "unknown")


def role_names() -> dict[str, str]:
    """角色 id → 中文名（课件/讲义/教案…）。"""
    conv = load_source_conventions()
    names: dict[str, str] = {}
    for c in (conv.get("conventions") or []) + (conv.get("other_roles") or []):
        names.setdefault(c.get("role"), c.get("name"))
    return names

# --------------------------------------------------------------------------- #
# 路径常量
# --------------------------------------------------------------------------- #
RAG_DIR = Path(__file__).resolve().parent.parent       # .../rag（本文件在 rag/offline/ 下）
ROOT = RAG_DIR.parent                                   # 仓库根目录
SCHEMA_DIR = ROOT / "schema"
JSON_SCHEMA_DIR = SCHEMA_DIR / "json"
TAXONOMY_PATH = SCHEMA_DIR / "taxonomy.yaml"
CONVENTIONS_PATH = SCHEMA_DIR / "source_conventions.yaml"
SAMPLES_DIR = ROOT / "teaching_assets" / "assets"   # 教研资产（原 samples/，2026-09-18 收敛）
LINT_SCRIPT = ROOT / "tools" / "lint.py"
DATA_DIR = ROOT / "data"

#: 流水线各 stage 的输入/输出目录（run_all 与 README 共用这一份定义）
#: 2026-09-18 路径收敛：原始素材 → teaching_assets/raw_materials；人工审核 → teaching_assets/review；
#: 其余中间产物留在 data/（流水线工作区）。真实资产落位见 docs/08。
STAGE_DIRS: dict[str, str] = {
    "raw": "teaching_assets/raw_materials",  # 人工投递的原始素材（ppt/pdf/docx/音频/转写 json）
    "parsed": "data/parsed",      # s1 输出
    "clean": "data/clean",        # s2 输出
    "chunks": "data/chunks",      # s3 输出
    "labels": "data/labels",      # s4 输出
    "qa_candidates": "data/qa_candidates",  # s5 输出
    "review": "teaching_assets/review",      # 人工审核回填（教研手工维护）
    "export": "data/export",      # s6 输出（交付下游）
}

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


# --------------------------------------------------------------------------- #
# 日志
# --------------------------------------------------------------------------- #
def get_logger(name: str = "rag.offline", level: int = logging.INFO) -> logging.Logger:
    """返回带统一格式的 logger；重复调用不会叠加 handler。"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(stream=sys.stderr)
        handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt="%H:%M:%S"))
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def ensure_dir(path: str | Path) -> Path:
    """创建目录（含父级）并返回 Path。"""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# --------------------------------------------------------------------------- #
# 受控词表
# --------------------------------------------------------------------------- #
_TAXONOMY_CACHE: dict[str, Any] | None = None


def load_taxonomy(path: str | Path | None = None, use_cache: bool = True) -> dict:
    """读取 ``schema/taxonomy.yaml``（受控词表）。

    任何标签值都必须能在本文件中找到，流水线里的规则标注全部基于它，
    避免把 ``TI_xxx`` / ``E_xxx`` 之类的枚举写死在代码里造成漂移。
    """
    global _TAXONOMY_CACHE
    if path is None and use_cache and _TAXONOMY_CACHE is not None:
        return _TAXONOMY_CACHE
    target = Path(path) if path is not None else TAXONOMY_PATH
    if not target.exists():
        raise FileNotFoundError(f"受控词表不存在：{target}")
    with open(target, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"受控词表格式非法（应为映射）：{target}")
    if path is None:
        _TAXONOMY_CACHE = data
    return data


def taxonomy_ids(section: str) -> list[str]:
    """取某一段词表的所有合法 id，例如 ``taxonomy_ids("teacher_intent")``。"""
    tax = load_taxonomy()
    items = tax.get(section) or []
    return [str(e["id"]) for e in items if isinstance(e, dict) and "id" in e]


def taxonomy_entries(section: str) -> list[dict]:
    """取某一段词表的原始条目列表（每项是 dict）。"""
    tax = load_taxonomy()
    return [e for e in (tax.get(section) or []) if isinstance(e, dict)]


# --------------------------------------------------------------------------- #
# ID / 哈希
# --------------------------------------------------------------------------- #
_SLUG_BAD = re.compile(r"[^0-9A-Za-z\u4e00-\u9fff]+")


def slug(text: str, max_len: int = 60) -> str:
    """把任意文本变成可安全用作文件名/ID 片段的 slug。

    - 保留中文与字母数字（改成小写），其余（空格/标点/路径分隔符）压成 ``-``；
    - 连续分隔符合并，首尾分隔符去掉；
    - 空结果退化为 ``item``，便于批量处理时不炸。
    """
    s = _SLUG_BAD.sub("-", str(text or "")).strip("-").lower()
    s = re.sub(r"-{2,}", "-", s)
    return (s[:max_len].strip("-")) or "item"


def stable_hash(text: str, length: int = 16) -> str:
    """文本的稳定哈希（sha256 前 N 位）。

    用途：a) 中间产物幂等（同样的输入产出同样的 key）；b) 去重用；
    c) manifest 里记录原始素材指纹，便于定位"这批数据到底来自哪一版素材"。
    """
    h = hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()
    return h[: max(4, int(length))]


def sha256_file(path: str | Path) -> str:
    """计算文件的 sha256（供 s6_export 的 manifest 使用）。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# JSON Schema
# --------------------------------------------------------------------------- #
_SCHEMA_CACHE: dict[str, dict] = {}


def load_json_schema(name: str) -> dict:
    """按文件名或短名读取 ``schema/json/`` 下的 JSON Schema。

    ``"kp"`` 与 ``"kp.schema.json"`` 等价。找不到时抛 ``FileNotFoundError``，
    并在报错里列出可用 schema，避免猜名字。
    """
    key = name if str(name).endswith(".json") else f"{name}.schema.json"
    if key in _SCHEMA_CACHE:
        return _SCHEMA_CACHE[key]
    path = JSON_SCHEMA_DIR / key
    if not path.exists():
        avail = sorted(p.name for p in JSON_SCHEMA_DIR.glob("*.schema.json"))
        raise FileNotFoundError(f"未找到 schema {key}；可用：{avail}")
    with open(path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    _SCHEMA_CACHE[key] = schema
    return schema


def validate(record: Any, schema_name: str) -> list[str]:
    """校验单条记录，返回错误信息列表（空列表 = 通过）。

    刻意「返回错误」而不是抛异常：流水线需要把错误汇总成报告给教研看，
    而不是第一处错误就中断。s6_export 会把非空结果当作导出阻断条件。
    """
    schema = load_json_schema(schema_name)
    validator = Draft202012Validator(schema)
    errors = []
    for err in sorted(validator.iter_errors(record), key=lambda e: list(e.path)):
        loc = "/".join(str(p) for p in err.path) or "<root>"
        errors.append(f"{loc}: {err.message}")
    return errors


def validate_many(records: Iterable[Any], schema_name: str) -> dict[int, list[str]]:
    """批量校验，返回 ``{下标: [错误...]}``，只包含有问题的下标。"""
    bad: dict[int, list[str]] = {}
    for i, rec in enumerate(records):
        errs = validate(rec, schema_name)
        if errs:
            bad[i] = errs
    return bad


# --------------------------------------------------------------------------- #
# YAML 读写（带中文注释块）
# --------------------------------------------------------------------------- #
class _LiteralStr(str):
    """标记"这个字符串请用字面块风格输出"，避免 YAML 转义把中文长句折断。"""


class _Dumper(yaml.SafeDumper):
    pass


def _literal_representer(dumper: yaml.Dumper, data: "_LiteralStr"):
    return dumper.represent_scalar("tag:yaml.org,2002:str", str(data), style="|")


_Dumper.add_representer(_LiteralStr, _literal_representer)


def _wrap_long_text(obj: Any, threshold: int = 60) -> Any:
    """把过长或含换行的字符串标记为字面块，让 YAML diff 可读。"""
    if isinstance(obj, str):
        if "\n" in obj or len(obj) > threshold:
            return _LiteralStr(obj)
        return obj
    if isinstance(obj, dict):
        return {k: _wrap_long_text(v, threshold) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_wrap_long_text(v, threshold) for v in obj]
    return obj


def dump_yaml(data: Any, header: str | None = None) -> str:
    """序列化为 YAML 文本；``header`` 会被写成顶部中文注释块。"""
    body = yaml.dump(
        _wrap_long_text(data),
        Dumper=_Dumper,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
        width=100,
    )
    if not header:
        return body
    lines = [ln if ln.startswith("#") or not ln.strip() else f"# {ln}" for ln in header.strip().splitlines()]
    return "\n".join(lines) + "\n" + body


def write_yaml_list(path: str | Path, items: Sequence[Any], header: str | None = None) -> Path:
    """写出一个 YAML 列表，并在每项前插入中文注释块。

    ``items`` 的每一项可以是：

    - ``dict``：正常写出，可带保留键 ``_comment``（字符串或字符串列表），
      该键不会写进 YAML，而是变成该项上方的中文注释；
    - ``(comment, payload)`` 二元组：等价于给 payload 带上 ``_comment``。

    这样产出的草稿文件对教研是"可读的"：看到工件就知道机器为什么这么切、哪里必须人工改。
    """
    out = Path(path)
    ensure_dir(out.parent)
    parts: list[str] = []
    if header:
        parts.append("\n".join(f"# {ln}" if ln.strip() else "#" for ln in header.strip().splitlines()))
    for item in items:
        comment: Any = None
        payload = item
        if isinstance(item, tuple) and len(item) == 2:
            comment, payload = item
        elif isinstance(item, dict) and "_comment" in item:
            comment = item["_comment"]
            payload = {k: v for k, v in item.items() if k != "_comment"}
        if comment:
            c_lines = comment if isinstance(comment, (list, tuple)) else str(comment).splitlines()
            parts.append("\n".join(f"# {c}" if str(c).strip() else "#" for c in c_lines))
        # 列表必须"逐项"dump：整体 dump 一个 list 会得到带 "--- " 前缀的文档串，
        # 那样既没法在项之间插注释，写进文件后还会变成 YAML 多文档而解析失败。
        parts.append(
            yaml.dump(
                [_wrap_long_text(payload)],
                Dumper=_Dumper,
                allow_unicode=True,
                sort_keys=False,
                default_flow_style=False,
                width=100,
            ).rstrip("\n")
        )
    out.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return out


def read_yaml(path: str | Path) -> Any:
    """读取 YAML；文件不存在返回 ``None``（调用方自行决定是缺省还是报错）。"""
    p = Path(path)
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_yaml_list(path: str | Path) -> list[dict]:
    """读取"列表型"YAML；单条映射会被包装成单元素列表，便于容错。"""
    data = read_yaml(path)
    if data is None:
        return []
    if isinstance(data, dict):
        return [data]
    return [x for x in data if isinstance(x, dict)]


# --------------------------------------------------------------------------- #
# JSON 中间产物
# --------------------------------------------------------------------------- #
def write_json(path: str | Path, data: Any) -> Path:
    """写出 UTF-8 JSON（``ensure_ascii=False``，中文不转义，方便人看）。"""
    out = Path(path)
    ensure_dir(out.parent)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def read_json(path: str | Path) -> Any:
    """读取 UTF-8 JSON。"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: str | Path, records: Iterable[Any]) -> int:
    """写出 JSONL（一行一条），返回条数。下游检索/入库按行流式消费。"""
    out = Path(path)
    ensure_dir(out.parent)
    n = 0
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


# --------------------------------------------------------------------------- #
# CLI 小工具
# --------------------------------------------------------------------------- #
def add_in_out_args(parser, default_in: str, default_out: str) -> None:
    """统一 ``--in`` / ``--out`` 参数。

    注意：``in`` 是 Python 关键字，不能写成 ``dest="in"``，
    因此这里把 dest 设为 ``in_dir`` / ``out_dir``，命令行上仍然是 ``--in/--out``。
    """
    parser.add_argument("--in", dest="in_dir", default=default_in,
                        help=f"输入目录（默认 {default_in}）")
    parser.add_argument("--out", dest="out_dir", default=default_out,
                        help=f"输出目录（默认 {default_out}）")


def iter_input_files(in_dir: str | Path, suffixes: Iterable[str]) -> list[Path]:
    """列出输入目录下指定后缀的文件（不递归），按文件名排序保证可复现。"""
    d = Path(in_dir)
    if not d.exists():
        return []
    wanted = {s.lower() for s in suffixes}
    return sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in wanted)


def rel_to_root(path: str | Path) -> str:
    """尽量返回相对仓库根的路径（日志与 manifest 里更短、可移植）。"""
    p = Path(path).resolve()
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def fullwidth_to_halfwidth(text: str) -> str:
    """全角 → 半角（NFKC 的更保守版本）。

    只处理 ASCII 可见区（！～ 与全角空格），中文标点（，。；：？！"）保持全角——
    它们是中文排版正确形态，TTS 也需要它们来做断句，不能一起转掉。
    """
    out = []
    for ch in str(text or ""):
        code = ord(ch)
        if code == 0x3000:               # 全角空格
            out.append(" ")
        elif 0xFF01 <= code <= 0xFF5E:   # 全角 ASCII
            out.append(chr(code - 0xFEE0))
        else:
            out.append(ch)
    return "".join(out)


def normalize_unicode(text: str) -> str:
    """Unicode 归一（兼容字符 → 标准形式），不改变中文语义。"""
    return unicodedata.normalize("NFKC", str(text or ""))


def main(argv: list[str] | None = None) -> int:
    """``python rag/offline/common.py`` —— 自检：打印词表版本与可用 schema，便于排障。"""
    log = get_logger("common")
    tax = load_taxonomy()
    log.info("仓库根目录：%s", ROOT)
    log.info("受控词表版本：%s（生效日期 %s，owner %s）",
             tax.get("version"), tax.get("effective_from"), tax.get("owner"))
    for section in ("teacher_intent", "style_emotion", "student_intent", "safety_policy", "source_type", "status"):
        log.info("  taxonomy.%-16s 共 %d 个合法取值", section, len(taxonomy_ids(section)))
    schemas = sorted(p.name for p in JSON_SCHEMA_DIR.glob("*.schema.json"))
    log.info("可用 JSON Schema（%d 个）：%s", len(schemas), ", ".join(schemas))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
