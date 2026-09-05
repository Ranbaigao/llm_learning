# LLM Learning · 星尘知识库

个人 AI（LLM/VLM）学习知识库 + 双端博客：`content/` 下的 Markdown / Jupyter 笔记即内容源，
后端实时渲染并索引，前端以「知识星图」首页 + 文章页呈现，另含微信小程序端（`miniprogram/`）。

这是一个 AI 与我共创的仓库，需要 AI 与我一同维护。AI 的行动准则参考 [AGENT.md](AGENT.md)。

## 架构

```
                        ┌────────────┐
        浏览器/小程序 ──▶│   nginx    │ :80
                        └─────┬──────┘
              ┌───────────────┼────────────────┐
              │ /             │ /api/          │ /api/assets/（长缓存）
              ▼               ▼                ▼
       ┌────────────┐  ┌────────────┐   ┌────────────┐
       │  frontend  │  │  backend   │   │  content/  │
       │ Nuxt 4 SSR │─▶│  FastAPI   │──▶│ md/ipynb   │
       │   :3000    │  │   :8000    │   │ （只读挂载） │
       └────────────┘  └─────┬──────┘   └────────────┘
                             │
                  ┌──────────┼──────────┐
                  ▼          ▼          ▼
               MySQL 5.7/8  Redis     微信 code2session
              （文章索引/    （可选    （小程序登录，
               浏览/点赞/     缓存）    个人主体限制见
               评论/用户）              thinking/ 文档）
```

## 技术栈

| 层 | 技术 | 说明 |
| --- | --- | --- |
| 前端 | Nuxt 4.5（Vue 3 SSR） | 星图 Canvas 引擎 + 文章渲染（MathJax / Pygments） |
| 后端 | FastAPI + SQLAlchemy 2 | 内容索引、图谱、浏览/点赞/评论、搜索、微信登录 |
| 数据库 | MySQL 5.7+（utf8mb4） | 文章索引与互动数据；内容本体仍在 Git |
| 缓存 | Redis 7（可选） | 后端 `REDIS_URL` 留空即降级不用 |
| 小程序 | uni-app（`miniprogram/`） | 微信端博客 |
| 部署 | Docker Compose + nginx | 见下文「生产部署」 |

## 目录结构

```
backend/     FastAPI 后端（app/ 入口 main.py，scripts/smoke_test.py 冒烟）
frontend/    Nuxt 4 前端（SSR；app/components/starmap/ 为星图）
miniprogram/ uni-app 微信小程序端
content/     知识库内容（md + ipynb + .assets 资产），内容的唯一事实源，随 Git 管理
nginx/       生产反代配置
site/        旧 MkDocs 构建产物（已弃用，仅留档）
feedbacks/   踩坑与修复记录（Harness 经验层）
thinking/    方案对比与决策记录
```

## 本地开发

前置：Python 3.11、Node 22、MySQL 5.7+（venv 与前端依赖脚本会自动初始化）。

### 一键启动（推荐）

```bat
:: Windows（双击 start.bat 或在终端执行）
start.bat
```

```bash
# Linux（Ctrl+C 停止全部服务，日志在 .cache/logs/）
bash start.sh
```

脚本自动完成：端口与 MySQL 连通性检查（默认本地 root/root 连接成功时会顺带自动建库 `llm_kb`）→ 缺失依赖初始化（`.cache/venv` / `frontend/node_modules`）→ 启动后端（8000）与前端（3000）→ 前端就绪后打开浏览器。后端带 `--reload`，改后端代码或 `content/` 笔记都会自动重载并重建内容索引。

### 手动分步启动

```sql
-- 1. 建库（utf8mb4；使用一键脚本且默认本地连接时可跳过，脚本会自动建）
CREATE DATABASE llm_kb CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

```bash
# 2. 启动后端（项目根目录；首次启动自动建表并同步内容索引）
export PYTHONUTF8=1
.cache/venv/Scripts/python.exe -m uvicorn --app-dir backend app.main:app --host 127.0.0.1 --port 8000

# 3. 启动前端（另开终端）
cd frontend && npm install   # 若崩溃改用 npx npm@12.0.2 install
npm run dev -- --port 3000
```

访问：

- 星图首页/博客：<http://localhost:3000/>（Windows 上 dev 绑定 localhost(IPv6)，**不要用 127.0.0.1:3000**）
- 后端 API 文档：<http://127.0.0.1:8000/docs>
- 后端配置项见 `backend/app/core/config.py`，全部可用环境变量覆盖（参考 `.env.example`）

## 生产部署（VPS + Docker Compose）

```bash
# 1. 配置环境变量（改密码和密钥！）
cp .env.example .env && $EDITOR .env

# 2. 构建并启动（MySQL 8.4 随项目一起容器化，数据存 mysql_data 卷）
docker compose up -d --build

# 可选：启用 Redis（.env 设置 REDIS_URL=redis://redis:6379/0）
docker compose --profile redis up -d
```

> **服务器拉不动外网镜像？** 用离线镜像包部署：本机构建 → `docker save` 导出 → 上传 →
> `docker load` → `compose up -d`，完整命令与更新/备份/排查流程见
> `thinking/deploy-to-production.md`（注意：thinking/ 在 .gitignore 中，clone 前先看本地的）。

**数据库凭证是单源的**：`.env` 里的 `MYSQL_USER` / `MYSQL_PASSWORD` / `MYSQL_DATABASE` / `DB_HOST` / `DB_PORT`
同时决定 MySQL 容器初始化和后端 `DATABASE_URL`（由 compose 自动拼装），改一处两边生效。
后端用应用账号（`MYSQL_USER`）连接，不用 root。

想用外部/宿主机 MySQL 而不是内置容器：`.env` 里把 `DB_HOST` 改为 `host.docker.internal`（或实际地址），
启动时关闭内置容器：`docker compose up -d --scale mysql=0`。

注意：MySQL 容器**只在首次启动**（数据卷为空）时用 `.env` 的值建库建号；之后改密码不会回写，
需 `docker compose down -v` 清卷重建（会清空数据，谨慎）。

**网络说明**：构建涉及的外网拉取——Docker 基础镜像走 Docker Desktop 代理（本机已配
`127.0.0.1:7897`，Registry 拉取实测通畅）；构建期 pip 默认官方源、npm 默认 npmmirror
（两者在本机代理环境与国内服务器实测可达；国内无代理服务器可在 `.env` 设
`PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn`、`NPM_REGISTRY=https://registry.npmjs.org` 等覆盖）。
备选：Docker 镜像也可用 registry-mirror（如 `docker.1ms.run`，本机实测大层偏慢，不如代理）。

部署后站点在 80 端口。注意事项：

- **HTTPS**：生产环境务必上证书。可在 nginx 层加 443 server（Let's Encrypt / acme.sh），
  或把本栈挂在已有网关（如 Caddy / 宝塔）后面。
- **小程序合法域名**：微信要求 request 合法域名必须是**已备案的 HTTPS 域名**；
  在小程序后台「开发管理 → 服务器域名」中加入站点域名，并把 `miniprogram/` 里的
  API 地址指向它。个人主体小程序无法开通微信登录的部分能力，双通道方案见
  `thinking/refactor-mkdocs-to-fullstack.md`。
- **content/ 更新**：内容以只读卷挂载进后端容器，`git pull` 后重启 backend 即完成索引同步。

## API 概览（前缀 `/api`）

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/articles/tree` | 目录树 |
| GET | `/articles/latest` · `/articles/hot` | 最新 / 热门文章 |
| GET | `/articles?category=blog` | 按一级分类列文章 |
| GET | `/articles/{slug:path}` | 文章详情（md/ipynb 渲染为 HTML） |
| GET | `/graph` | 知识星图 {nodes, links, stats} |
| GET | `/search?q=` | 全文搜索 |
| GET | `/stats/site` | 站点级 PV/UV |
| POST | `/articles/{id}/view` | 记录浏览（同访客同天去重） |
| POST/DELETE | `/articles/{id}/like` | 点赞 / 取消点赞 |
| GET/POST | `/articles/{id}/comments` | 评论列表 / 发表评论 |
| DELETE | `/comments/{id}` | 删除评论（需管理令牌） |
| POST | `/auth/wx-mini` | 微信小程序登录 |
| GET | `/assets/{path}` | content/.assets 静态资产 |
| GET | `/health` | 健康检查 |

## 环境变量

`.env`（项目根，docker compose 读取；本地开发用 `backend/.env` 或内置默认值）：

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `MYSQL_ROOT_PASSWORD` | 无（必填） | 内置 MySQL 容器的 root 密码，仅容器内管理用 |
| `MYSQL_DATABASE` | `llm_kb` | 数据库名 |
| `MYSQL_USER` / `MYSQL_PASSWORD` | 无（必填） | 应用账号，后端以此连接；容器首启自动创建 |
| `DB_HOST` / `DB_PORT` | `mysql` / `3306` | 后端连库地址；外部数据库改这里并 `--scale mysql=0` |
| `ADMIN_TOKEN` | `dev-admin-token` | 管理接口令牌，生产必改 |
| `WX_MINI_APPID` / `WX_MINI_SECRET` | 空 | 微信小程序登录凭据，可留空 |
| `REDIS_URL` | 空 | 可选缓存，留空则不用 |
| `IP_SALT` | `dev-ip-salt` | 评论 IP 脱敏盐，生产必改 |

compose 相关：`CONTENT_DIR`（容器内固定 `/app/content`）、`NUXT_API_SERVER`（前端 SSR 直连后端，编排内固定 `http://backend:8000`）已在 compose 里设好，一般不用动。本地开发直连本机 MySQL 时的后端配置见 `backend/app/core/config.py`。

效果预览（旧 MkDocs 版留档）：[LLM Learning](https://ranbaigao.github.io/llm_learning/)
