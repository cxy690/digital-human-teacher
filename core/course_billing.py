"""课程倒计时与计费服务：多课程并行 / 退出后继续 / 到点自动结算。

设计要点
--------
* **课时状态放服务端**：需求要求「中途退出后倒计时仍在跑」「同时开多门课各自独立倒计时」——
  前端计时器在页面关闭/刷新后即失效，因此课时必须由后端持有；
* **结算规则在服务端判定**：前 ``course_free_seconds`` 秒（默认 120s）内结束不收费，
  超过则收取整节课费用 —— 规则若放前端可被篡改；
* **持久化**：状态写入 ``data/courses.json``，服务重启后未结算课程继续倒计时。

与 :mod:`core.course_session` 的分工：后者负责「已用时长 → 阶段」映射与话术模板，
本模块负责「课时、续费、计费」，两者互不依赖。
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config.logging import get_logger
from config.settings import settings

log = get_logger("core.course_billing")

_STORE = Path("data/courses.json")
_MAX_BILLS = 500          # 账单最多保留条数（防文件无限增长）


def charge_for(elapsed_seconds: float) -> float:
    """计费规则：前 ``course_free_seconds`` 秒（含）免费，超过则收整节课费用。"""
    free = float(settings.course_free_seconds)
    if elapsed_seconds <= free:
        return 0.0
    return float(settings.course_price)


@dataclass
class Course:
    """一门课的课时与计费状态。"""

    course_id: str
    student_id: str
    subject: str
    start_at: float
    end_at: float
    duration_minutes: int
    renew_count: int = 0
    prompt_acked: bool = False        # 学生是否已处理过「剩 5 分钟」提示
    settled: bool = False
    settled_at: float | None = None
    reason: str = ""                  # manual / auto_expired
    charge: float = 0.0
    elapsed_seconds: float = 0.0

    def remaining(self, now: float | None = None) -> float:
        return max(0.0, self.end_at - (now or time.time()))

    def elapsed(self, now: float | None = None) -> float:
        return max(0.0, (now or time.time()) - self.start_at)

    def to_dict(self, now: float | None = None) -> dict[str, Any]:
        now = now or time.time()
        remaining = self.remaining(now)
        prompt_window = float(settings.course_renew_prompt_seconds)
        return {
            "course_id": self.course_id,
            "student_id": self.student_id,
            "subject": self.subject,
            "duration_minutes": self.duration_minutes,
            "renew_count": self.renew_count,
            "remaining_seconds": int(remaining),
            "elapsed_seconds": int(self.elapsed(now)),
            "expired": self.settled or remaining <= 0,
            "settled": self.settled,
            # 剩 N 秒内且尚未处理过提示 → 前端弹「续费 / 知道了」
            "need_renew_prompt": (not self.settled) and (not self.prompt_acked)
            and remaining <= prompt_window,
            "charge": self.charge,
            "reason": self.reason,
        }


class CourseManager:
    """全局课程管理器（内存态 + 磁盘持久化，单进程使用）。"""

    def __init__(self, store: Path | None = None) -> None:
        self._store = store or _STORE
        self._courses: dict[str, Course] = {}
        self._bills: list[dict[str, Any]] = []
        self._load()

    # ---------------------------------------------------------------- 生命周期
    def start(self, student_id: str, subject: str) -> Course:
        """开一门新课。同一学生可同时存在多门课，各自独立倒计时。"""
        now = time.time()
        minutes = max(1, int(settings.course_duration_minutes))
        course = Course(
            course_id=uuid.uuid4().hex[:12],
            student_id=(student_id or "").strip() or "anonymous",
            subject=(subject or "").strip(),
            start_at=now,
            end_at=now + minutes * 60,
            duration_minutes=minutes,
        )
        self._courses[course.course_id] = course
        self._save()
        log.info("课程开始：{}（学生 {}，{}，{} 分钟）",
                 course.course_id, course.student_id, course.subject or "未指定学科", minutes)
        return course

    def renew(self, course_id: str, minutes: int | None = None) -> Course | None:
        """续费：在**剩余时间上再加一节课时长**，并重置「剩 5 分钟」提示。"""
        course = self._courses.get(course_id)
        if course is None or course.settled:
            return None
        add = int(minutes or course.duration_minutes)
        course.end_at += add * 60
        course.renew_count += 1
        course.prompt_acked = False
        self._save()
        log.info("课程续费：{} 增加 {} 分钟（累计续费 {} 次）", course_id, add, course.renew_count)
        return course

    def ack_prompt(self, course_id: str) -> bool:
        """学生点了「知道了」：本次不再重复弹窗。"""
        course = self._courses.get(course_id)
        if course is None:
            return False
        course.prompt_acked = True
        self._save()
        return True

    def close(self, course_id: str, *, reason: str = "manual") -> dict[str, Any] | None:
        """结束课程并结算。"""
        course = self._courses.get(course_id)
        if course is None:
            return None
        if not course.settled:
            self._settle(course, reason)
        return course.to_dict()

    def _settle(self, course: Course, reason: str) -> None:
        now = time.time()
        course.settled = True
        course.settled_at = now
        course.reason = reason
        course.elapsed_seconds = course.elapsed(now)
        # 幂等：同一课程只计一次账（服务重启后可能对同一课程再次触发结算）
        already = any(b.get("course_id") == course.course_id for b in self._bills)
        if already:
            course.charge = 0.0
        else:
            course.charge = charge_for(course.elapsed_seconds)
            self._bills.append(
                {
                    "course_id": course.course_id,
                    "student_id": course.student_id,
                    "subject": course.subject,
                    "started_at": course.start_at,
                    "settled_at": now,
                    "elapsed_seconds": int(course.elapsed_seconds),
                    "renew_count": course.renew_count,
                    "charge": course.charge,
                    "free": course.charge <= 0,
                    "reason": reason,
                }
            )
            if len(self._bills) > _MAX_BILLS:
                self._bills = self._bills[-_MAX_BILLS:]
        # 结算后移出活跃表：持久化文件里不再残留该课程，避免重启后重复结算。
        # 注意必须立刻落盘：``_save()`` 只写未结算课程，若这里不写，磁盘上仍留着
        # 这门已结算的课 —— 服务一重启就会被 ``_load()`` 复活并继续倒计时。
        self._courses.pop(course.course_id, None)
        self._save()
        free = float(settings.course_free_seconds)
        if course.charge > 0:
            tip = f"收费 {course.charge}"
        elif course.elapsed_seconds <= free:
            tip = "免费（未超过免费时长）"
        else:
            tip = "不收费（未配置课程价格）"
        log.info("课程结算：{} 用时 {}s，{}（{}）",
                 course.course_id, int(course.elapsed_seconds), tip, reason)

    # ---------------------------------------------------------------- 巡检
    def tick(self) -> list[dict[str, Any]]:
        """后台巡检：把已到时的课程**自动结算并关闭**（「到时间后自动关闭」）。"""
        now = time.time()
        closed: list[dict[str, Any]] = []
        for course in list(self._courses.values()):
            if not course.settled and now >= course.end_at:
                self._settle(course, "auto_expired")
                closed.append(course.to_dict(now))
        return closed

    # ---------------------------------------------------------------- 查询
    def snapshot(self, student_id: str) -> list[dict[str, Any]]:
        """某学生**进行中**的课程列表（已结算的不再返回）。"""
        now = time.time()
        sid = (student_id or "").strip() or "anonymous"
        return [c.to_dict(now) for c in self._courses.values()
                if c.student_id == sid and not c.settled]

    def bills(self, student_id: str, limit: int = 50) -> list[dict[str, Any]]:
        sid = (student_id or "").strip() or "anonymous"
        items = [b for b in self._bills if b.get("student_id") == sid]
        return items[-limit:]

    # ---------------------------------------------------------------- 持久化
    def _save(self) -> None:
        try:
            self._store.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                # 只持久化未结算课程：已结算的进账单，不必再跟时
                "courses": [c.__dict__ for c in self._courses.values() if not c.settled],
                "bills": self._bills,
            }
            self._store.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as err:  # noqa: BLE001 - 落盘失败不影响内存态功能
            log.warning("课程状态保存失败：{}", err)

    def _load(self) -> None:
        if not self._store.is_file():
            return
        try:
            data = json.loads(self._store.read_text(encoding="utf-8"))
        except Exception as err:  # noqa: BLE001
            log.warning("课程状态读取失败（忽略）：{}", err)
            return
        # 顺序很关键：先恢复账单，再据此过滤已结算课程。
        # 反过来的话 settled_ids 会取到尚未加载的空 _bills，"已结算不恢复"的保护失效，
        # 已被结算的课程会在重启后重新出现并继续倒计时。
        self._bills = list(data.get("bills") or [])
        settled_ids = {b.get("course_id") for b in self._bills}
        for item in data.get("courses") or []:
            try:
                course = Course(**item)
            except Exception:  # noqa: BLE001 - 脏数据跳过
                continue
            if course.course_id in settled_ids:
                continue          # 已经计过账的课程不再恢复（防重启后重复计费）
            self._courses[course.course_id] = course
        self._save()               # 落盘清理：把文件中残留的已结算课程一并剔除
        if self._courses:
            log.info("恢复 {} 门进行中的课程", len(self._courses))


_manager: CourseManager | None = None


def get_course_manager() -> CourseManager:
    """进程内单例（延迟创建，避免导入期依赖 data 目录）。"""
    global _manager
    if _manager is None:
        _manager = CourseManager()
    return _manager
