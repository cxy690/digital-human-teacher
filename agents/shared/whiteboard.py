"""共享白板：AI 教师在白板上的所有书写/批注/画图操作都在此构造。

白板指令是前后端同步的唯一载体（见 ``WhiteboardOp``），AI 自主答疑与
图片题目圈画都通过这里生成标准化操作序列。
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from config.constants import BoardOp
from core.state.models import WhiteboardOp

if TYPE_CHECKING:  # 仅类型标注，避免运行时循环导入
    from core.courseware.models import SlideElement


# ---------------------------------------------------------------- 标注样式
# 前端按 ``meta.style`` 分支渲染：画圈 / 划横线 / 虚线框
ANNOTATE_ELLIPSE = "ellipse"      # 单行文字：圈起来（老师划重点的经典动作）
ANNOTATE_UNDERLINE = "underline"  # 文字：下方一条横线
ANNOTATE_BOX = "box"              # 区域：虚线框（跨行内容、图片、表格）
ANNOTATE_CROSS = "cross"          # 判错标记：画一个红叉（表示「这一步错了」）
                                  # 注意用叉而不用字母 X —— 后者朗读时会被念成「乘」

# 元素框高度（归一化）不超过它，视为「单行文字」——可以画圈
ANNOTATE_SINGLE_LINE_MAX_H = 0.10
# 单行文字最长多少字仍画圈：再长会横跨大半页，圈出来像「大鸭蛋」，退回避线框
ANNOTATE_ELLIPSE_MAX_CHARS = 24


class Whiteboard:
    """白板操作工厂：把语义化动作翻译成前端可渲染的指令。"""

    DEFAULT_COLOR = "#1677ff"

    # 归一化坐标合法区间（与 core/validator.py 的判据一致）：越界会被判
    # board_text_x/y_out_of_range，历史上会触发「整块白板被清空」。
    MIN_XY = 0.0
    MAX_XY = 1.0
    # 排版安全下沿：最后一行不超过这里，留一点底边距
    SAFE_MAX_Y = 0.94
    MIN_LINE_GAP = 0.035

    @classmethod
    def _clamp(cls, value: float) -> float:
        """把归一化坐标夹回 [0, 1]（越界的兜底，正常排版不会走到）。"""
        return max(cls.MIN_XY, min(cls.MAX_XY, float(value)))

    def write_text(
        self,
        text: str,
        x: float,
        y: float,
        *,
        color: str = "#1f1f1f",
        size: float = 18.0,
    ) -> WhiteboardOp:
        """在白板指定位置写入文字（坐标自动夹回 [0,1]，保证校验不越界）。"""
        return WhiteboardOp(
            op=BoardOp.TEXT, text=text,
            x=self._clamp(x), y=self._clamp(y),
            color=color, size=size,
        )

    def draw_path(
        self,
        points: list[list[float]],
        *,
        color: str = DEFAULT_COLOR,
        size: float = 3.0,
    ) -> WhiteboardOp:
        """自由画笔轨迹（点坐标为 0~1 归一化）。"""
        return WhiteboardOp(op=BoardOp.DRAW, points=points, color=color, size=size)

    def draw_shape(
        self,
        kind: str,
        x: float,
        y: float,
        width: float,
        height: float,
        *,
        color: str = DEFAULT_COLOR,
        size: float = 2.0,
    ) -> WhiteboardOp:
        """绘制几何图形（直线/矩形/圆等）。"""
        return WhiteboardOp(
            op=BoardOp.SHAPE,
            x=x,
            y=y,
            width=width,
            height=height,
            color=color,
            size=size,
            meta={"kind": kind},
        )

    def annotate_region(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        *,
        image_id: str | None = None,
        color: str = "#ff4d4f",
        label: str | None = None,
    ) -> WhiteboardOp:
        """在图片上层圈画题目区域 / 重难点。"""
        return WhiteboardOp(
            op=BoardOp.ANNOTATE,
            x=x,
            y=y,
            width=width,
            height=height,
            color=color,
            size=3.0,
            image_id=image_id,
            meta={"label": label} if label else {},
        )

    def highlight(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        *,
        color: str = "#ffe58f",
    ) -> WhiteboardOp:
        return WhiteboardOp(op=BoardOp.HIGHLIGHT, x=x, y=y, width=width, height=height, color=color)

    def erase(self, points: list[list[float]], *, size: float = 16.0) -> WhiteboardOp:
        return WhiteboardOp(op=BoardOp.ERASE, points=points, size=size)

    def clear(self) -> WhiteboardOp:
        return WhiteboardOp(op=BoardOp.CLEAR)

    # ---------------------------------------------------------------- 组合
    # 讲解排版常量：**增量上板与整体上板共用同一套坐标公式**，
    # 这样「边流边推」出来的每一行，位置与最终一次性渲染的结果逐像素一致。
    EXPLAIN_X = 0.08
    EXPLAIN_START_Y = 0.12
    EXPLAIN_LINE_GAP = 0.08

    def explain_title(
        self, title: str, *, start_y: float = EXPLAIN_START_Y, x: float = EXPLAIN_X
    ) -> WhiteboardOp:
        """讲解标题（白板第一行）。"""
        return self.write_text(title, x, start_y, color="#d4380d", size=22.0)

    def explain_step(
        self, index: int, step: str, *, start_y: float = EXPLAIN_START_Y,
        line_gap: float = EXPLAIN_LINE_GAP, x: float = EXPLAIN_X,
    ) -> WhiteboardOp:
        """第 ``index`` 步（从 1 开始）：位置由序号决定，因此可以「生成一步推一步」。"""
        return self.write_text(f"{index}. {step}", x, round(start_y + index * line_gap, 3))

    def explain_section(
        self, label: str, items: list[str], *, after_steps: int,
        start_y: float = EXPLAIN_START_Y, line_gap: float = EXPLAIN_LINE_GAP, x: float = EXPLAIN_X,
        color: str = "#0958d9",
    ) -> list[WhiteboardOp]:
        """步骤之后的小区块（【知识点】/【易错点】）。

        区块位置取决于**步骤总数**，所以必须在 steps 全部到齐后才能定位——
        这也是「步骤可以先推、知识点/易错点等最后一起推」的原因。
        """
        y = start_y + (after_steps + 1) * line_gap
        ops = [self.write_text(label, x, round(y, 3), color=color, size=18.0)]
        y += line_gap * 0.9
        for item in items:
            ops.append(self.write_text(f"· {item}", x + 0.02, round(y, 3)))
            y += line_gap * 0.9
        return ops

    def build_explain(
        self,
        *,
        title: str,
        steps: list[str],
        knowledge_points: list[str] | None = None,
        pitfalls: list[str] | None = None,
        start_y: float = EXPLAIN_START_Y,
        line_gap: float = EXPLAIN_LINE_GAP,
        x: float = EXPLAIN_X,
    ) -> list[WhiteboardOp]:
        """把「标题 + 分步讲解 + 知识点 + 易错点」渲染成一串白板指令。

        知识点与易错点作为独立小区块追加在步骤之后（各带小标题）——
        学生听完还能在板上回看「这题考什么、容易错在哪」，白板不再是干巴巴几行步骤。
        """
        # 白板高度有限：先数清总共要写多少行，再按需压缩行距，保证最后一行不越界。
        # 历史 bug：行距固定 0.08，步骤/知识点/易错点一多（≥12 行）y 就 >1.0，
        # validator 判 board_text_y_out_of_range，降级逻辑把整块板清空 -> 白板全空。
        total_lines = 1 + len(steps)
        if knowledge_points:
            total_lines += 1 + len(knowledge_points)
        if pitfalls:
            total_lines += 1 + len(pitfalls)
        gap = self._fit_line_gap(total_lines, start_y=start_y, line_gap=line_gap)

        ops: list[WhiteboardOp] = [
            self.explain_title(title, start_y=start_y, x=x)
        ]
        for i, step in enumerate(steps, start=1):
            ops.append(self.explain_step(i, step, start_y=start_y, line_gap=gap, x=x))
        if knowledge_points:
            ops.extend(self.explain_section(
                "【知识点】", knowledge_points, after_steps=len(steps),
                start_y=start_y, line_gap=gap, x=x, color="#0958d9"))
        if pitfalls:
            kp_extra = 1 + len(knowledge_points or []) if knowledge_points else 0
            ops.extend(self.explain_section(
                "【易错点】", pitfalls, after_steps=len(steps) + kp_extra,
                start_y=start_y, line_gap=gap, x=x, color="#cf1322"))
        return ops

    # ---------------------------------------------------------------- 课件
    def slide_page(
        self,
        deck_id: str,
        index: int,
        *,
        image_url: str = "",
        title: str = "",
        total: int = 0,
        text: str = "",
    ) -> WhiteboardOp:
        """切换课件页：白板背景换成该页，并清空上一页的标注。

        这是 PPT 讲解的**第一条指令**：先上页，再上标注/讲解。
        页面上没有图（纯文字页）时 ``image_url`` 为空，前端按纯文字页渲染。
        """
        return WhiteboardOp(
            op=BoardOp.SLIDE,
            deck_id=deck_id,
            page_index=index,
            text=text or None,
            meta={
                "title": title,
                "image_url": image_url,
                "total": total,
                "page_no": index + 1,      # 1-based，供前端显示「第 3 / 12 页」
            },
        )

    def annotate_element(
        self,
        element: "SlideElement",
        *,
        deck_id: str,
        page_index: int,
        kind: str = "annotate",
        label: str | None = None,
        color: str | None = None,
        style: str | None = None,
    ) -> WhiteboardOp:
        """把课件元素框翻译成标注指令（圈画 / 高亮）。

        PPT 标注的关键：**不让 LLM 凭空猜坐标**。模型只需说出「圈住标题」
        或给出元素 id，这里把它映射到元素真实的 bbox 上。

        样式（``meta.style``）交给前端决定怎么画：单行短文字**画圈**最像老师
        划重点，跨行内容圈不住、图片表格没有「一行」的概念，都用虚线框。
        """
        if kind == "highlight":
            return WhiteboardOp(
                op=BoardOp.HIGHLIGHT,
                deck_id=deck_id, page_index=page_index,
                x=self._clamp(element.x), y=self._clamp(element.y),
                width=self._clamp(element.width), height=self._clamp(element.height),
                color=color or "#ffe58f",
                meta={"label": label or element.text, "element_id": element.id},
            )
        return WhiteboardOp(
            op=BoardOp.ANNOTATE,
            deck_id=deck_id, page_index=page_index,
            x=self._clamp(element.x), y=self._clamp(element.y),
            width=self._clamp(element.width), height=self._clamp(element.height),
            color=color or "#ff4d4f", size=3.0,
            meta={
                "label": label or element.text,
                "element_id": element.id,
                "element_kind": element.kind,
                "style": style or self.annotate_style(element),
            },
        )

    @staticmethod
    def annotate_style(element: "SlideElement") -> str:
        """挑标注样式：单行短文字**画圈**，其余用虚线框。

        画圈指向最明确，但只适合**一行**文字——椭圆圈跨行的段落会切到别的行；
        图片 / 表格 / 图表也没有「一行文字」可圈。所以按元素高度与字数决定。
        """
        if element.kind and element.kind != "text":
            return ANNOTATE_BOX
        text = (element.text or "").strip()
        if element.height <= ANNOTATE_SINGLE_LINE_MAX_H and len(text) <= ANNOTATE_ELLIPSE_MAX_CHARS:
            return ANNOTATE_ELLIPSE
        return ANNOTATE_BOX

    @classmethod
    def _fit_line_gap(cls, total_lines: int, *, start_y: float, line_gap: float) -> float:
        """按总行数压缩行距，使最后一行不超过 ``SAFE_MAX_Y``。"""
        if total_lines <= 1:
            return line_gap
        last_y = start_y + (total_lines - 1) * line_gap
        if last_y <= cls.SAFE_MAX_Y:
            return line_gap
        # 内容多时挤一挤，而不是让后面的行掉出白板之外
        return max(cls.MIN_LINE_GAP, (cls.SAFE_MAX_Y - start_y) / (total_lines - 1))
