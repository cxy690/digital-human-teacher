"""渲染「4 个 Agent / 9 条流程线 / 40 个节点」总览图。

产物：``docs/agent_flow.png``（讲解、答辩、投屏用）。

为什么用脚本画而不是手画：节点清单是**从 ``agents/nodes.py`` 读出来的**
（``describe_agent_nodes()``），所以代码里改了节点，重跑一次图就同步了，
不会出现「图上是旧的、代码是新的」。

用法（项目根目录）：
    python scripts/render_agent_flow.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents import nodes as agent_nodes  # noqa: E402
from agents.protocol import (  # noqa: E402
    ROLE_LECTURER,
    ROLE_STUDENT,
    ROLE_SUPERVISOR,
    ROLE_TEACHER,
)

# 中文字体：Windows 自带雅黑；缺了就退黑体
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
matplotlib.rcParams["axes.unicode_minus"] = False

W, H = 22.0, 12.4
DPI = 120

# ---------------------------------------------------------------- 配色（按角色）
ROLE_STYLE = {
    ROLE_STUDENT: ("#EBF3FF", "#3B82F6", "#1D4DB8"),      # 底 / 边 / 标题
    ROLE_SUPERVISOR: ("#FFF6E8", "#F59E0B", "#B45309"),
    ROLE_TEACHER: ("#EAF9EF", "#22C55E", "#15803D"),
    ROLE_LECTURER: ("#F5EBFF", "#A855F7", "#7E22CE"),
}
ROLE_LABEL = {
    ROLE_STUDENT: "学生端 Agent（Student）",
    ROLE_SUPERVISOR: "总控 Agent（Supervisor）",
    ROLE_TEACHER: "老师 Agent（Teacher）",
    ROLE_LECTURER: "讲师 Agent（Lecturer）",
}
INK = "#1F2937"
GRAY = "#6B7280"
LINE = "#4B5563"

# ---------------------------------------------------------------- 版面坐标
COL_HEAD_Y = 11.2
# 图名 -> (编号, 列 x0, 列宽, 底边 y, 高, 标题, 副标题, 自定义正文行 or None)
BOXES: dict[str, tuple] = {
    "student.voice": (
        "①", 0.6, 3.7, 8.25, 1.9, "student.voice", "学生说一句话（入口线）", None,
    ),
    "student.control": (
        "⑨", 0.6, 3.7, 6.05, 1.6, "student.control", "设备 / 白板指令（同步图）", None,
    ),
    "supervisor.turn": (
        "②", 5.4, 5.8, 6.05, 4.1, "supervisor.turn", "总控分诊：一句话该走哪条路", [
            "route（分诊台，只判断不讲内容）",
            "· 在答上一题 → ④ 判对错",
            "· 一次性问 ≥2 件事 → 逐条讲（控制先办）",
            "· 纯指令 → ⑨ 设备控制（不出声）",
            "· 闲聊 → 直接回应（不进讲题线）",
            "· 讲题 → ③ 老师流水线（到点则 → ⑤ 出题）",
            "（未命中的分支自动跳过，只执行命中的那一条）",
        ],
    ),
    "teacher.answer": (
        "③", 12.6, 5.0, 8.45, 1.9, "teacher.answer", "讲一道题：想内容 + 自检", None,
    ),
    "teacher.judge": (
        "④", 12.6, 5.0, 6.85, 1.4, "teacher.judge", "判学生答得对不对", None,
    ),
    "teacher.quiz": (
        "⑤", 12.6, 5.0, 5.25, 1.4, "teacher.quiz", "出一道模拟题", None,
    ),
    "supervisor.slide": (
        "⑥", 5.4, 5.8, 3.05, 1.8, "supervisor.slide", "打开课件 / 点页码 / 单页翻页", None,
    ),
    "lecturer.slide": (
        "⑦", 18.5, 2.9, 3.05, 1.8, "lecturer.slide", "讲这一页 + 三级降级", None,
    ),
    "supervisor.slide_prepared": (
        "⑧", 5.4, 5.8, 0.8, 1.7, "supervisor.slide_prepared", "连讲自动翻页（页面与声音同到）", None,
    ),
}


def _width(text: str) -> float:
    """粗略估算文本宽度（数据单位：1 单位 = 1 英寸）。"""
    units = sum(1.0 if ord(ch) > 0x2E80 else 0.55 for ch in text)
    return units * 9.8 / 72.0


def _chain_lines(order: list[str], limit: float) -> list[str]:
    """把节点顺序折行，保证每行不超过 ``limit`` 宽度。"""
    lines: list[str] = []
    current = ""
    for name in order:
        piece = name if not current else f"{current} → {name}"
        if current and _width(piece) > limit:
            lines.append(f"{current} →")
            current = name
        else:
            current = piece
    if current:
        lines.append(current)
    return lines


def _box(ax, x, y, w, h, role, title, subtitle, body, *, spacing: float = 0.30) -> None:
    face, edge, title_color = ROLE_STYLE[role]
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.12",
        linewidth=1.8, edgecolor=edge, facecolor=face, zorder=2,
    ))
    ax.text(x + 0.16, y + h - 0.24, title, fontsize=12.5, fontweight="bold",
            color=title_color, ha="left", va="center", zorder=3)
    if subtitle:
        ax.text(x + 0.16, y + h - 0.52, subtitle, fontsize=9.6, color=GRAY,
                ha="left", va="center", zorder=3)
    top = y + h - 0.84
    for i, text in enumerate(body):
        ax.text(x + 0.16, top - i * spacing, text, fontsize=9.8, color=INK,
                ha="left", va="center", zorder=3)


def _arrow(ax, p1, p2, label="", *, dashed=False, label_at=None, color=LINE, lw=1.7):
    ax.add_patch(FancyArrowPatch(
        p1, p2, arrowstyle="-|>", mutation_scale=14, linewidth=lw, color=color,
        linestyle="--" if dashed else "-", shrinkA=2, shrinkB=2, zorder=4,
    ))
    if label:
        lx, ly = label_at if label_at else ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2 + 0.16)
        ax.text(lx, ly, label, fontsize=9.4, color="#374151", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.92),
                zorder=5)


def _poly_arrow(ax, points, *, dashed=False, color=LINE, lw=1.7):
    """折线箭头（多段路由，用来绕开其他方框）。"""
    for a, b in zip(points[:-2], points[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, zorder=4,
                linestyle="--" if dashed else "-")
    ax.add_patch(FancyArrowPatch(
        points[-2], points[-1], arrowstyle="-|>", mutation_scale=14, linewidth=lw,
        color=color, linestyle="--" if dashed else "-", shrinkA=0, shrinkB=2, zorder=4,
    ))


def main() -> int:
    described = agent_nodes.describe_agent_nodes()
    role_of: dict[str, str] = {}
    for role, graphs in described.items():
        for graph in graphs:
            role_of[graph] = role
    missing = [g for g in BOXES if g not in role_of]
    extra = [g for g in role_of if g not in BOXES]
    if missing or extra:
        raise SystemExit(f"版面与代码不同步：缺少 {missing}，多余 {extra}")

    fig = plt.figure(figsize=(W, H), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    ax.text(0.35, H - 0.38, "AI 教学系统 · 4 个 Agent / 9 条流程线 / 40 个节点",
            fontsize=17, fontweight="bold", color="#111827", ha="left", va="center")
    ax.text(0.35, H - 0.72,
            "实线＝正向流程　虚线＝回到总控 / 降级回退　每行文字＝该条线的节点顺序（怎么走）",
            fontsize=10.5, color=GRAY, ha="left", va="center")

    # 角色列标题 + 背景条
    columns = {ROLE_STUDENT: (0.6, 3.7), ROLE_SUPERVISOR: (5.4, 5.8),
               ROLE_TEACHER: (12.6, 5.0), ROLE_LECTURER: (18.5, 2.9)}
    for role, (x, w) in columns.items():
        face, edge, _ = ROLE_STYLE[role]
        ax.add_patch(FancyBboxPatch(
            (x - 0.12, 0.45), w + 0.24, 10.25, boxstyle="round,pad=0.02,rounding_size=0.1",
            linewidth=1.0, edgecolor=edge, facecolor=face, alpha=0.35, zorder=0,
        ))
        ax.text(x + w / 2, COL_HEAD_Y, ROLE_LABEL[role], fontsize=12, fontweight="bold",
                color=ROLE_STYLE[role][2], ha="center", va="center")

    # 各方框（正文：自动从节点声明里取连接顺序）
    for graph, (num, x, w, y, h, title, subtitle, body) in BOXES.items():
        role = role_of[graph]
        spacing = 0.30
        if body is None:
            order = described[role][graph]["order"]
            body = _chain_lines(order, limit=w - 0.45)
        elif len(body) > 5:
            spacing = 0.42          # 分支清单行多，撑开行距把方框填满
        _box(ax, x, y, w, h, role, f"{num}  {title}", subtitle, body, spacing=spacing)

    # ---- 箭头：入口 -> 图 ----
    _arrow(ax, (2.45, 10.62), (2.45, 10.17), "学生语音", label_at=(2.45, 10.82))
    _arrow(ax, (8.4, 10.62), (8.4, 10.17), "学生打字 / 图片", label_at=(8.4, 10.82))
    _arrow(ax, (2.45, 4.35), (5.28, 4.35), "课件打开 / 点页码 / 翻页", label_at=(3.9, 4.6))
    _arrow(ax, (13.4, 1.65), (11.32, 1.65), "连讲自动翻页", label_at=(14.6, 1.65))

    # ---- 图与图之间 ----
    _arrow(ax, (4.3, 9.2), (5.28, 9.2), "dispatch 交回总控", label_at=(4.8, 9.47))
    _arrow(ax, (11.2, 9.4), (12.6, 9.4), "teach 讲题", label_at=(11.9, 9.65))
    _arrow(ax, (11.2, 7.55), (12.6, 7.55), "answer_check 判题", label_at=(11.9, 7.8))
    _arrow(ax, (11.2, 6.75), (12.6, 5.95), "到点 / 巡检 出题", label_at=(11.9, 6.2))
    _arrow(ax, (5.4, 7.05), (4.3, 7.05), "control → 设备指令", label_at=(4.8, 7.3))
    _arrow(ax, (11.2, 3.95), (18.5, 3.95), "explain 讲这一页", label_at=(14.85, 4.2))
    # 讲课结果回到呈现（失败：fallback → degrade 兜底）
    _poly_arrow(ax, [(19.95, 3.0), (19.95, 2.7), (6.6, 2.7), (6.6, 3.0)], dashed=True)
    ax.text(13.3, 2.88, "讲解结果回到呈现（失败：fallback → degrade 兜底）",
            fontsize=9.4, color="#374151", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.92), zorder=5)
    # 出题 -> 学生作答 -> 回到分诊（沿右侧上行，避开列内方框）
    _poly_arrow(ax, [(17.6, 5.35), (17.78, 5.35), (17.78, 10.85), (11.6, 10.85),
                     (11.6, 10.3), (11.24, 10.3)], dashed=True)
    ax.text(18.1, 8.0, "学生作答后回到分诊（→ ④ 判题）", fontsize=9.4, color="#374151",
            ha="center", va="center", rotation=90,
            bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.92), zorder=5)

    # ---- 页脚 ----
    ax.text(0.35, 0.15,
            "节点清单自动取自 agents/nodes.py（文字版：python -m agents.nodes；自检："
            "python scripts/selfcheck.py 第 [7] 步）",
            fontsize=9.6, color=GRAY, ha="left", va="center")

    out = ROOT / "docs" / "agent_flow.png"
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    print(f"已生成 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
