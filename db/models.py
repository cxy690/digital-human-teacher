"""ORM 模型：平台用户 / 监护关系 + 学生 / 会话 / 消息 / 错题本。

两个来源合并为一个模型集：
* ``User`` / ``GuardianLink``（A 线）：平台身份体系（core/auth 使用），
  ``student_id`` 取自 ``users.id``，``to_public()`` 是对外安全视图（不含密码哈希）；
* ``Student`` / ``SessionRecord`` / ``MessageRecord`` / ``WrongBook``（B 线）：答疑会话与
  学情落库（db/runtime_sync.py、db/importer.py、scripts/pull_chat_to_mysql.py 使用）。

两套表名不冲突（users / guardian_links / students / sessions / messages / wrong_book），
``Base.metadata.create_all`` 会一次性建全。
"""
from __future__ import annotations

import time

from sqlalchemy import Boolean, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.base import Base

#: 平台角色
ROLES: tuple[str, ...] = ("student", "parent", "teacher", "admin")


def _now() -> float:
    return time.time()


class User(Base):
    """平台用户。"""

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(16), default="student", index=True)
    display_name: Mapped[str] = mapped_column(String(64), default="")
    grade: Mapped[str] = mapped_column(String(32), default="")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[float] = mapped_column(Float, default=_now)
    last_login_at: Mapped[float | None] = mapped_column(Float, nullable=True)

    def to_public(self) -> dict:
        """对外安全视图（不含密码哈希）。"""
        return {
            "id": self.id,
            "username": self.username,
            "role": self.role,
            "display_name": self.display_name or self.username,
            "grade": self.grade,
            "is_active": bool(self.is_active),
            "created_at": self.created_at,
        }


class GuardianLink(Base):
    """家长 ↔ 学生 监护关系（未成年人数据授权的基础）。"""

    __tablename__ = "guardian_links"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    guardian_id: Mapped[str] = mapped_column(String(64), index=True)
    student_id: Mapped[str] = mapped_column(String(64), index=True)
    relation: Mapped[str] = mapped_column(String(16), default="parent")
    created_at: Mapped[float] = mapped_column(Float, default=_now)


class Student(Base):
    """学生。"""

    __tablename__ = "students"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(64), default="")
    grade: Mapped[str] = mapped_column(String(32), default="")
    created_at: Mapped[float] = mapped_column(Float, default=_now)

    sessions: Mapped[list["SessionRecord"]] = relationship(back_populates="student")


class SessionRecord(Base):
    """一次答疑会话。"""

    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    student_id: Mapped[str] = mapped_column(ForeignKey("students.id"), index=True)
    started_at: Mapped[float] = mapped_column(Float, default=_now)
    ended_at: Mapped[float | None] = mapped_column(Float, nullable=True)

    student: Mapped[Student] = relationship(back_populates="sessions")
    messages: Mapped[list["MessageRecord"]] = relationship(back_populates="session")


class MessageRecord(Base):
    """会话消息。"""

    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id"), index=True)
    role: Mapped[str] = mapped_column(String(16))
    text: Mapped[str] = mapped_column(Text, default="")
    ts: Mapped[float] = mapped_column(Float, default=_now)

    session: Mapped[SessionRecord] = relationship(back_populates="messages")


class WrongBook(Base):
    """错题本。"""

    __tablename__ = "wrong_book"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    student_id: Mapped[str] = mapped_column(String(64), index=True)
    subject: Mapped[str] = mapped_column(String(32), default="")
    question: Mapped[str] = mapped_column(Text, default="")
    analysis: Mapped[str] = mapped_column(Text, default="")
    ts: Mapped[float] = mapped_column(Float, default=_now)
