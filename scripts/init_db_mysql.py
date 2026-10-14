"""初始化数据库：建库 + 建表 + 连接自检（MySQL / SQLite 通用）。

用法（项目根执行）：
    python scripts/init_db_mysql.py

MySQL 场景（先在 .env 填好密码）：
    DATABASE_URL=mysql+pymysql://root:你的密码@127.0.0.1:3306/education?charset=utf8mb4
    —— 密码含 @ : # / 等字符时需 URL 编码（如 @ -> %40），否则会被当作地址分隔符。

产物：目标库中创建 4 张表（students / sessions / messages / wrong_book）。
     Navicat 里刷新连接即可看到库与表；重复执行幂等，不会清空已有数据。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # 相对路径（如 sqlite 文件）以项目根为准
try:  # Windows 控制台中文输出
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from sqlalchemy import create_engine, inspect, text  # noqa: E402
from sqlalchemy.engine import URL, make_url  # noqa: E402
from sqlalchemy.exc import OperationalError, SQLAlchemyError  # noqa: E402

from config.settings import settings  # noqa: E402


def ensure_mysql_database(url: URL) -> None:
    """MySQL：确保目标库存在（先连 server 再 CREATE DATABASE IF NOT EXISTS）。"""
    if url.get_backend_name() != "mysql":
        return
    dbname = url.database
    if not dbname:
        print("[x] DATABASE_URL 里缺少库名，应为 .../3306/<库名>?charset=utf8mb4")
        raise SystemExit(1)
    # 连 MySQL server 建库（不能带库名）。
    # 不能用 url.set(database=None)：SQLAlchemy 的 URL.set() 对 None 是「保持原值」，
    # 库名清不掉，会带着 education 去连接 -> 1049 Unknown database。
    server_url = URL.create(
        drivername=url.drivername,
        username=url.username,
        password=url.password,
        host=url.host,
        port=url.port,
        query=url.query,
    )
    server_engine = create_engine(server_url, future=True)
    try:
        with server_engine.connect() as conn:
            conn.execute(text(
                f"CREATE DATABASE IF NOT EXISTS `{dbname}` "
                "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            ))
            conn.commit()
    finally:
        server_engine.dispose()
    print(f"[1] 数据库 `{dbname}` 已就绪（不存在则已创建）")


def main() -> None:
    url: URL = make_url(settings.database_url)
    backend = url.get_backend_name()
    print(f"[0] DATABASE_URL：{url.render_as_string(hide_password=True)}（后端 {backend}）")

    try:
        ensure_mysql_database(url)

        from db.base import engine, init_db  # 延迟导入：读的就是 .env 里当前的 URL

        init_db()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        names = inspect(engine).get_table_names()
        print(f"[2] 表已创建：{', '.join(names) or '(无)'}")
        with engine.connect() as conn:
            for table in names:
                count = conn.execute(text(f"SELECT COUNT(*) FROM `{table}`")).scalar()
                print(f"    - {table}: {count} 行")
        print("[3] 完成：Navicat 里刷新该连接即可看到库与表（重复执行幂等，不清数据）")
    except OperationalError as err:
        print(f"[x] 连接失败：{err.orig}")
        print("    排查清单：")
        print("      ① .env 里 DATABASE_URL 的密码是否已替换占位符（含 @ : # 等字符需 URL 编码）")
        print("      ② MySQL 服务是否在运行（本机端口 3306 可达）")
        print("      ③ 账号是否有建库权限（root 一般有）")
        raise SystemExit(1)
    except SQLAlchemyError as err:
        print(f"[x] 数据库操作失败：{err}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
