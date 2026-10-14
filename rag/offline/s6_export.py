#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 6 · 导出门禁与编排（rag/offline/rag/offline/s6_export.py）
================================================
**本模块不实现导出逻辑。** 仓库里 `tools/build_bundle.py` 才是唯一的导出实现
（YAML 真源 → 运行时交付包），它产出的东西比"把 YAML 转成 jsonl"完整得多：
`lesson.bundle.json` / `tts_script.jsonl` / `milvus_upsert.jsonl` / `safety_matrix.json` /
`kp.jsonl` / `qa.jsonl` / `utterances.jsonl` / `interaction_templates.jsonl` /
`manifest.json` / `excluded.jsonl`。

如果在这里再写一套「YAML → jsonl」，就会出现**两套导出标准**，早晚漂移，
下游拿到的包取决于谁先跑——这是数据工程里最贵的一类 bug。所以 s6 的职责被收窄成两件事：

1. **前置门禁（本模块自己的代码）**
   - ``schema/json/*.schema.json`` 逐条校验（结构合法性）；
   - 以子进程调用 ``tools/lint.py --dir <课目录>``，**只有 0 个 ERROR 才继续**。
2. **编排导出（子进程调用唯一实现）**::

       python tools/build_bundle.py --lesson-dir <课目录> --out <out> --strict

   把它的 stdout / 退出码 / 产物目录里的 ``manifest.json`` 汇总进本模块的报告结构，
   供 CI 与人工一次性看清"为什么出包 / 为什么没出包"。

退出码约定：``0`` 导出成功；``1`` 入参或工具缺失；``2`` 门禁未通过（schema / lint / build_bundle 报错）。

CLI::

    python rag/offline/rag/offline/s6_export.py --lesson-dir samples/LESSON-MATH-P3A-RJ-U03 --out data/export
    python rag/offline/rag/offline/s6_export.py --lesson-dir samples/LESSON-MATH-P3A-RJ-U03 --out data/export --dry-run
    python rag/offline/rag/offline/s6_export.py --lesson-dir samples/LESSON-MATH-P3A-RJ-U03 --out data/export --json
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

try:  # pragma: no cover
    from . import common
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import common  # type: ignore

log = common.get_logger("s6_export")

#: 唯一导出实现（本模块只调用它，不复制它的逻辑）
BUILD_BUNDLE = common.ROOT / "tools" / "build_bundle.py"
LINT = common.LINT_SCRIPT

#: 课目录文件名 → schema 短名（用于本模块的 A 层结构校验）
SCHEMA_SPECS: list[tuple[str, str | None, bool]] = [
    ("lesson.yaml", "lesson", False),
    ("kp.yaml", "kp", True),
    ("segments.yaml", "segment", True),
    ("qa.yaml", "qa_item", True),
    ("utterances.yaml", "utterance", True),
    ("interaction_templates.yaml", "interaction_template", True),
    ("chains.yaml", "followup_chain", True),
]


# --------------------------------------------------------------------------- #
# 子进程（统一 UTF-8）
# --------------------------------------------------------------------------- #
def run_utf8(cmd: list[str], timeout: int = 300,
             cwd: str | Path | None = None) -> subprocess.CompletedProcess:
    """统一的子进程调用：强制 UTF-8 读写。

    Windows 上 ``subprocess`` 默认按 GBK 解码子进程输出，而 lint.py / build_bundle.py 打印的是
    中文 UTF-8，不加这个参数会直接抛 ``UnicodeDecodeError``。显式指定编码 +
    ``errors="replace"``，跨平台行为一致。
    """
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    return subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=timeout, cwd=str(cwd or common.ROOT), env=env,
    )


def git_commit(cwd: str | Path | None = None, timeout: int = 10) -> tuple[str, str | None]:
    """取 ``git rev-parse HEAD``；失败返回 ``("unknown", 原因)``，绝不抛异常。"""
    if shutil.which("git") is None:
        return "unknown", "系统未安装 git"
    try:
        proc = run_utf8(["git", "rev-parse", "HEAD"], timeout=timeout, cwd=cwd or common.ROOT)
    except Exception as exc:  # noqa: BLE001 - 超时/权限/编码都归为"取不到"
        return "unknown", f"{type(exc).__name__}: {exc}"
    if proc.returncode != 0:
        lines = (proc.stderr or proc.stdout or "").strip().splitlines()
        return "unknown", lines[0] if lines else f"git 退出码 {proc.returncode}"
    return proc.stdout.strip() or "unknown", None


# --------------------------------------------------------------------------- #
# 门禁 1：schema 校验
# --------------------------------------------------------------------------- #
def validate_lesson_dir(lesson_dir: str | Path) -> tuple[dict[str, list], list[str]]:
    """按 ``schema/json/*.schema.json`` 逐条校验课目录内 YAML，返回 ``(数据, 错误列表)``。"""
    ld = Path(lesson_dir)
    data: dict[str, list] = {}
    errors: list[str] = []
    for fname, schema, is_list in SCHEMA_SPECS:
        fp = ld / fname
        if not fp.exists():
            if fname in ("lesson.yaml", "kp.yaml"):
                errors.append(f"缺少必需文件 {fname}")
            else:
                log.info("课目录无 %s（可选），跳过", fname)
            data[fname] = []
            continue
        payload = common.read_yaml(fp)
        if is_list:
            if not isinstance(payload, list):
                errors.append(f"{fname} 应为列表，实际是 {type(payload).__name__}")
                data[fname] = []
                continue
            items = payload
        else:
            items = [payload] if isinstance(payload, dict) else []
            if not items:
                errors.append(f"{fname} 应为映射对象")
                continue
        data[fname] = items
        if schema:
            for i, item in enumerate(items):
                for msg in common.validate(item, schema):
                    errors.append(f"{fname}[{i}] {msg}")
    # 内联 episodes 也要过 schema（lesson.schema.json 里只约束成 object）
    lesson = (data.get("lesson.yaml") or [None])[0]
    if isinstance(lesson, dict):
        for i, ep in enumerate(lesson.get("episodes") or []):
            for msg in common.validate(ep, "episode"):
                errors.append(f"lesson.yaml/episodes[{i}] {msg}")
    return data, errors


# --------------------------------------------------------------------------- #
# 门禁 2：tools/lint.py
# --------------------------------------------------------------------------- #
def run_lint(lesson_dir: str | Path, timeout: int = 180) -> tuple[int, int, str]:
    """调用 ``tools/lint.py --dir <课目录> --json``，返回 ``(errors, warns, 原始输出)``。

    走子进程而不是 import：lint 是仓库既有资产，命令行调用才能保证**导出门禁与 CI
    用的是同一份规则**；lint 升级时这里自动跟着升级，不会出现"两套标准"。
    """
    if not LINT.exists():
        return 1, 0, f"未找到 {LINT}"
    proc = run_utf8([sys.executable, str(LINT), "--dir", str(lesson_dir), "--json"],
                    timeout=timeout)
    out = (proc.stdout or "").strip()
    try:
        report = json.loads(out)
        return int(report.get("errors", 0)), int(report.get("warns", 0)), out
    except json.JSONDecodeError:
        return 1, 0, (out or proc.stderr or "lint 未返回可解析的 JSON")


# --------------------------------------------------------------------------- #
# 门禁 3（非阻塞）：草稿状态提示
# --------------------------------------------------------------------------- #
def collect_gate_warnings(data: dict[str, list]) -> list[str]:
    """提示"这版包里还有未过审内容"，避免把半成品当成品发出去（不阻塞导出）。"""
    warns: list[str] = []
    pairs = [("知识点", data.get("kp.yaml"), "kp_id"),
             ("切片", data.get("segments.yaml"), "segment_id"),
             ("问答对", data.get("qa.yaml"), "qa_id"),
             ("话术", data.get("utterances.yaml"), "utterance_id")]
    for label, items, key in pairs:
        draft = [it.get(key) for it in (items or []) if isinstance(it, dict)
                 and it.get("status") in ("draft", "rejected")]
        if draft:
            warns.append(f"{label}中存在 {len(draft)} 条 draft/rejected，"
                         f"build_bundle 会把它们排除在运行时包之外：{draft[:5]}")
    return warns


# --------------------------------------------------------------------------- #
# 编排导出：调用唯一的导出实现
# --------------------------------------------------------------------------- #
def run_build_bundle(lesson_dir: str | Path, out_root: str | Path,
                     strict: bool = True, timeout: int = 600) -> dict:
    """以子进程调用 ``tools/build_bundle.py``（唯一导出实现），返回调用报告。

    ``--strict`` 会让 build_bundle 在**存在 warning 时**以非零码退出。这对 CI 是好事，
    但对本流水线来说 warning 不是"包不可用"——真正的硬失败是它 manifest 里的 ``errors``
    （引用完整性错误：环节会缺内容）。因此这里把两者分开判断：
    ``returncode != 0`` 但 ``errors == []`` → 只是有 warning，包仍然有效；
    ``errors != []`` → 硬失败，报告 ``ok=False``。
    """
    if not BUILD_BUNDLE.exists():
        return {"ok": False, "returncode": None, "stdout": "", "stderr": f"未找到 {BUILD_BUNDLE}",
                "errors": [f"未找到导出实现 {BUILD_BUNDLE}"], "warnings": [], "manifest": None,
                "out_dir": None}
    cmd = [sys.executable, str(BUILD_BUNDLE), "--lesson-dir", str(lesson_dir),
           "--out", str(out_root)]
    if strict:
        cmd.append("--strict")
    log.info("调用唯一导出实现：%s", " ".join(cmd))
    try:
        proc = run_utf8(cmd, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "returncode": None, "stdout": "", "stderr": "build_bundle 超时",
                "errors": ["build_bundle 执行超时"], "warnings": [], "manifest": None,
                "out_dir": None, "cmd": cmd}
    report: dict[str, Any] = {
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout": (proc.stdout or "").strip(),
        "stderr": (proc.stderr or "").strip(),
        "manifest_path": None,
        "manifest": None,
        "out_dir": None,
        "errors": [],
        "warnings": [],
    }
    # 定位产物目录：优先相信 manifest（build_bundle 自己做目录命名），否则按 lesson_id 猜
    candidate = Path(out_root)
    found: Path | None = None
    if candidate.is_dir():
        subs = sorted(p for p in candidate.iterdir() if p.is_dir() and (p / "manifest.json").exists())
        if len(subs) == 1:
            found = subs[0]
        elif subs:
            found = max(subs, key=lambda p: (p / "manifest.json").stat().st_mtime)
    if found is not None:
        report["out_dir"] = str(found)
        report["manifest_path"] = str(found / "manifest.json")
        try:
            report["manifest"] = json.loads((found / "manifest.json").read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            report["errors"].append(f"manifest.json 不可解析：{type(exc).__name__}: {exc}")
    manifest = report["manifest"] or {}
    report["errors"] += list(manifest.get("errors") or [])
    report["warnings"] += list(manifest.get("warnings") or [])
    if proc.returncode != 0 and not report["errors"]:
        report["warnings"].append(
            f"build_bundle 退出码 {proc.returncode}（--strict 下通常意味着有 warning），"
            f"manifest.errors 为空，视为包仍可用")
    report["ok"] = (not report["errors"]) and found is not None
    return report


def export_lesson(lesson_dir: str | Path, out_dir: str | Path,
                  skip_lint: bool = False, dry_run: bool = False,
                  skip_schema: bool = False) -> tuple[Path | None, dict]:
    """执行一次完整导出：门禁 → 调用 build_bundle → 汇总报告。

    返回 ``(产物目录 或 None, 报告)``；报告结构固定，便于 CI 直接消费::

        {ok, stage, lesson_dir, errors, warnings, schema, lint, build_bundle, manifest}
    """
    ld = Path(lesson_dir).resolve()
    out_root = Path(out_dir)
    report: dict[str, Any] = {
        "ok": False, "stage": "start", "lesson_dir": str(ld), "out_dir": str(out_root),
        "errors": [], "warnings": [], "schema": None, "lint": None,
        "build_bundle": None, "manifest": None, "manifest_path": None,
        "source_commit": None,
    }
    if not BUILD_BUNDLE.exists():
        report["errors"].append(f"未找到导出实现 {BUILD_BUNDLE}（build_bundle.py 是唯一导出实现）")
        return None, report
    if not ld.exists():
        report["errors"].append(f"课目录不存在：{ld}")
        return None, report
    if not (ld / "lesson.yaml").exists():
        report["errors"].append(f"课目录缺少 lesson.yaml：{ld}")
        return None, report

    # ---- 门禁 1：schema ----
    report["stage"] = "schema"
    if skip_schema:
        report["warnings"].append("已跳过 schema 校验（--skip-schema，仅限排障）")
    else:
        data, schema_errors = validate_lesson_dir(ld)
        report["schema"] = {"errors": len(schema_errors), "details": schema_errors[:50],
                            "files": {k: len(v) for k, v in data.items() if v}}
        report["errors"].extend(schema_errors)
        if schema_errors:
            log.error("Schema 校验失败 %d 项，终止导出（未调用 build_bundle）", len(schema_errors))
            for e in schema_errors[:20]:
                log.error("  %s", e)
            return None, report
        log.info("Schema 校验通过：%s", report["schema"]["files"])
        report["warnings"].extend(collect_gate_warnings(data))
        for w in report["warnings"]:
            log.warning("状态提示：%s", w)

    # ---- 门禁 2：tools/lint.py ----
    report["stage"] = "lint"
    lint_errors = lint_warns = 0
    if skip_lint:
        report["warnings"].append("已按 --skip-lint 跳过 tools/lint.py 门禁（仅限本地调试，禁止用于交付）")
        log.warning(report["warnings"][-1])
    else:
        lint_errors, lint_warns, raw = run_lint(ld)
        report["lint"] = {"errors": lint_errors, "warnings": lint_warns, "raw": raw[:4000]}
        if lint_errors:
            report["errors"].append(f"tools/lint.py 报告 {lint_errors} 个 ERROR，拒绝导出")
            log.error("Lint 门禁未通过：%d 个 ERROR / %d 个 WARN，终止导出（未调用 build_bundle）",
                      lint_errors, lint_warns)
            return None, report
        log.info("Lint 门禁通过：0 个 ERROR / %d 个 WARN", lint_warns)

    # ---- 编排导出 ----
    if dry_run:
        report.update({"ok": True, "stage": "dry-run"})
        log.info("dry-run：门禁全部通过，未调用 build_bundle")
        return None, report

    report["stage"] = "build_bundle"
    bb = run_build_bundle(ld, out_root, strict=not skip_lint)
    report["build_bundle"] = bb
    report["manifest"] = bb.get("manifest")
    report["manifest_path"] = bb.get("manifest_path")
    if bb.get("stdout"):
        for line in str(bb["stdout"]).splitlines():
            log.info("  build_bundle | %s", line)
    if bb.get("stderr"):
        log.warning("  build_bundle(stderr) | %s", bb["stderr"].strip())
    report["errors"].extend(bb.get("errors") or [])
    report["warnings"].extend(bb.get("warnings") or [])
    if not bb.get("ok"):
        report["stage"] = "build_bundle_failed"
        if not report["errors"]:
            report["errors"].append(f"build_bundle 未产出可用交付包（退出码 {bb.get('returncode')}）")
        log.error("导出失败：%s", "；".join(report["errors"][:5]))
        return None, report

    # ---- 汇总 manifest（build_bundle 的 manifest 是权威来源，这里只做透传与补充）----
    target = Path(bb["out_dir"])
    commit, commit_reason = git_commit(ld)
    report["source_commit"] = commit
    report["source_commit_note"] = commit_reason
    report["stage"] = "done"
    report["ok"] = True
    counts = (bb.get("manifest") or {}).get("counts") or {}
    files = (bb.get("manifest") or {}).get("files") or {}
    log.info("导出完成 → %s（由 tools/build_bundle.py 产出）", common.rel_to_root(target))
    if counts:
        log.info("  条数：%s", counts)
    for name in sorted(files):
        meta = files[name] or {}
        log.info("  %-28s %5s 条  sha256=%s…", name, meta.get("count"),
                 str(meta.get("sha256") or "")[:12])
    if commit == "unknown":
        log.warning("  source_commit 取不到（%s），报告里记为 unknown", commit_reason)
    else:
        log.info("  source_commit=%s", commit[:12])
    report["summary"] = {
        "lesson_id": (bb.get("manifest") or {}).get("lesson_id"),
        "built_at": (bb.get("manifest") or {}).get("built_at"),
        "git_commit_from_bundle": (bb.get("manifest") or {}).get("git_commit"),
        "counts": counts,
        "file_count": len(files),
        "target_dir": str(target),
        "lint": {"errors": lint_errors, "warnings": lint_warns},
    }
    return target, report


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="s6_export",
        description="Stage6 导出门禁与编排：schema + lint 双重门禁通过后，调用 tools/build_bundle.py 产出交付包",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "退出码：0=导出成功；1=入参/工具缺失；2=门禁未通过（schema/lint/build_bundle 报错）。\n"
            "说明：本脚本**不实现导出逻辑**，导出由 tools/build_bundle.py 完成（唯一实现，避免双标准）。\n"
            "示例：\n"
            "  python rag/offline/rag/offline/s6_export.py --lesson-dir samples/LESSON-MATH-P3A-RJ-U03 --out data/export\n"
            "  python rag/offline/rag/offline/s6_export.py --lesson-dir samples/LESSON-MATH-P3A-RJ-U03 --out data/export --dry-run\n"
            "  python rag/offline/rag/offline/s6_export.py --lesson-dir samples/LESSON-MATH-P3A-RJ-U03 --out data/export --json\n"
        ),
    )
    ap.add_argument("--lesson-dir", required=True,
                    help="人工审核后的课目录，如 samples/LESSON-MATH-P3A-RJ-U03")
    ap.add_argument("--out", dest="out_dir", default="data/export",
                    help="导出根目录（默认 data/export）")
    ap.add_argument("--skip-lint", action="store_true",
                    help="跳过 tools/lint.py 门禁（仅本地调试；同时不再给 build_bundle 传 --strict）")
    ap.add_argument("--skip-schema", action="store_true",
                    help="跳过 schema 校验（仅本地排障，禁止用于交付）")
    ap.add_argument("--dry-run", action="store_true", help="只跑门禁、不调用 build_bundle")
    ap.add_argument("--json", dest="as_json", action="store_true",
                    help="以 JSON 打印完整导出报告（CI 用）")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    target, report = export_lesson(args.lesson_dir, args.out_dir,
                                   args.skip_lint, args.dry_run, args.skip_schema)
    if args.as_json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["ok"]:
        return 0
    if report["errors"] and report["stage"] in ("schema", "lint", "build_bundle_failed"):
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
