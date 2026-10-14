# db — 数据层（MySQL / SQLAlchemy 2.0）

平台用户与监护关系的存储层。数据库不可用时**登录/注册降级不可用，教学链路完全不受影响**（启动阶段只记 WARNING）。

## 文件说明

| 文件 | 用途 |
| --- | --- |
| `__init__.py` | 包门户。 |
| `base.py` | 基础设施：MySQL 优先的引擎与会话工厂；`init_db()` 建库建表。 |
| `models.py` | ORM 模型：`User`（student / parent / teacher / admin）、`GuardianLink`（监护关系）。 |

## 设计要点

- **引擎惰性创建**：只有真正访问数据库时才建连，避免「仅 import 就要求 pymysql 且必须连得上」。
- **自动建库**：MySQL 下目标库不存在时自动 `CREATE DATABASE`（库名走 `^[A-Za-z0-9_]+$` 白名单校验防注入），做到开箱即跑。
- **地址可覆盖**：`set_database_url()` 供单测（SQLite）与多环境切换。
- **用户是唯一身份事实源**：`student_id` 一律取自 `users.id`，不再信任前端自报；`to_public()` 是对外安全视图，**绝不包含密码哈希**。
- **角色枚举**：`ROLES = (student, parent, teacher, admin)`，授权判定由调用方按 `role` 完成（见 `core/auth/deps.py`）。
- **演示账号**：`AUTH_SEED_DEMO=true` 时启动预置 student/parent/teacher/admin（密码 123456），仅用于本地演示，生产应关闭。
