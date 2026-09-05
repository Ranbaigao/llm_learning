"""FastAPI 入口：CORS（开发期全放开）、路由挂载、/api/assets 静态服务、
启动时建表 + 内容索引同步。"""
from __future__ import annotations

import logging
import mimetypes
import time
from contextlib import asynccontextmanager
from urllib.parse import unquote

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.core.config import settings
from app.db import Base, SessionLocal, engine
from app import models  # noqa: F401  # 确保所有表已注册到 metadata
from app.routers import articles, auth, comments, graph, likes, search, stats, views
from app.services.content_index import sync_content_index

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Windows 注册表可能缺这些 MIME，显式补上（字体类型不对会被浏览器拒载）
for ext, mime in {
    ".woff2": "font/woff2",
    ".woff": "font/woff",
    ".ttf": "font/ttf",
    ".otf": "font/otf",
    ".svg": "image/svg+xml",
    ".webp": "image/webp",
    ".avif": "image/avif",
}.items():
    mimetypes.add_type(mime, ext)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 等待数据库就绪：compose 编排下 MySQL 容器首次初始化需要数秒，
    # 后端自己重试比依赖 depends_on 条件更稳（外部数据库/本地开发同样受益）
    for attempt in range(1, 21):
        try:
            Base.metadata.create_all(bind=engine)
            break
        except Exception as exc:
            if attempt == 20:
                raise RuntimeError("数据库连接失败（已重试 20 次），请检查 DATABASE_URL") from exc
            logger.warning("数据库未就绪（第 %s/20 次，3 秒后重试）：%s", attempt, type(exc).__name__)
            time.sleep(3)
    logger.info("数据表已就绪")
    db = SessionLocal()
    try:
        stats = sync_content_index(db)
        logger.info("内容索引: %s", stats)
    except Exception:
        logger.exception("内容索引同步失败（不阻塞服务启动）")
    finally:
        db.close()
    yield


app = FastAPI(title="LLM KB API", version="0.1.0", lifespan=lifespan)

# 开发期全放开，生产再收紧
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注意挂载顺序：articles.router 里有 GET /articles/{slug:path}（path 转换器会
# 吞掉多级路径），必须最后挂载，否则会抢先匹配 /articles/{id}/comments 这类路由；
# stats 虽不在 /articles 前缀下，同样挂在 articles 之前保持防御性顺序
for r in (
    views.router,
    likes.router,
    comments.router,
    search.router,
    graph.router,
    auth.router,
    stats.router,
    articles.router,
):
    app.include_router(r, prefix="/api")


@app.get("/api/assets/{file_path:path}")
def serve_asset(file_path: str):
    """服务 content/.assets/ 下的静态文件（图片/字体/html），防路径穿越。"""
    base = settings.assets_dir.resolve()
    target = (base / unquote(file_path)).resolve()
    if target != base and base not in target.parents:
        raise HTTPException(status_code=403, detail="非法路径")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(target)


@app.get("/api/health")
def health():
    return {"status": "ok"}
