# core/auth — 身份认证

只解决「你是谁」（认证）；「你能做什么」（授权）由调用方按 `role` / `id` 判定。用户数据落 `db/models.py` 的 MySQL（`AUTH_ENABLE=false` 时整个体系关闭）。

## 文件说明

| 文件 | 用途 |
| --- | --- |
| `__init__.py` | 包门户：导出用户存储 + 密码哈希 + JWT + FastAPI 依赖。 |
| `security.py` | 安全原语：PBKDF2-SHA256 密码哈希 + 自实现 HS256 JWT。 |
| `service.py` | 用户仓储与认证业务（注册 / 登录 / 按 id 取用户）；`AuthError` 带 HTTP 语义 `code`。 |
| `deps.py` | FastAPI 依赖：`user_from_token`（失败返回 None 不抛，供 WS 复用）、`current_user` / `require_role`（REST 侧转 401/403）、`require_developer`（admin 门禁）。 |

## 设计要点

- **刻意只用标准库**：免 bcrypt 原生编译（Windows 友好）、免 pyjwt 依赖；存库格式 `algo$iterations$salt$digest`，迭代 20 万次（OWASP 2023 量级），比对用 `hmac.compare_digest` 常量时间。
- **失败不抛异常的一致性设计**：`user_from_token` 返回 `None` 而非抛异常——WebSocket 握手阶段要复用同一函数，抛异常会打断连接流程。
- **`AUTH_ENABLE=false` 时开门即拒**：`require_developer` 等门禁直接拒绝，不会因关闭开关而意外放行内部接口。
- **演示账号**：启动时可预置 student/parent/teacher/admin（密码 123456），`AUTH_SEED_DEMO` 控制。
