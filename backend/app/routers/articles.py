"""文章路由：目录树、最新、热门、详情（md/ipynb 渲染）。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db import get_db
from app.models.article import Article
from app.schemas import ArticleDetail, ArticleListItem
from app.services import content_index
from app.services.markdown import render_article
from app.services.notebook import render_notebook

router = APIRouter(tags=["articles"])


def _article_path(article: Article) -> Path:
    return settings.content_dir / article.path


@router.get("/articles/tree")
def article_tree(db: Session = Depends(get_db)) -> dict[str, Any]:
    """目录树（嵌套）。排除 .assets、blog、隐藏项及 index.md，规则同旧钩子。"""
    content_dir = settings.content_dir
    titles = {a.slug: a.title for a in db.query(Article.slug, Article.title)}

    def build(directory: Path, depth: int) -> list[dict[str, Any]]:
        try:
            entries = sorted(
                directory.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())
            )
        except (PermissionError, OSError):
            return []
        nodes: list[dict[str, Any]] = []
        for entry in entries:
            if content_index._is_excluded(entry.name):
                continue
            rel = entry.relative_to(content_dir).as_posix()
            if entry.is_dir():
                nodes.append(
                    {
                        "name": content_index._humanize(entry.name),
                        "slug": rel,
                        "type": "category" if depth == 0 else "subcategory",
                        "note_count": _dir_note_count(entry),
                        "children": build(entry, depth + 1),
                    }
                )
            elif entry.is_file() and entry.suffix.lower() in content_index.NOTEBOOK_SUFFIXES:
                if entry.name in content_index.EXCLUDE_FILES:
                    continue
                slug = rel[: -len(entry.suffix)]
                nodes.append(
                    {
                        "name": titles.get(slug) or content_index._humanize(entry.name),
                        "slug": slug,
                        "type": "note",
                        "note_count": 1,
                        "children": [],
                    }
                )
        return nodes

    def _dir_note_count(directory: Path) -> int:
        count = 0
        for suffix in content_index.NOTEBOOK_SUFFIXES:
            for p in directory.rglob(f"*{suffix}"):
                try:
                    parts = p.relative_to(content_dir).parts
                except ValueError:
                    continue
                if p.is_file() and not any(content_index._is_excluded(x) for x in parts):
                    count += 1
        return count

    children = build(content_dir, 0) if content_dir.is_dir() else []
    return {
        "name": "LLM Learning",
        "slug": "",
        "type": "root",
        "note_count": sum(n["note_count"] for n in children),
        "children": children,
    }


@router.get("/articles/latest", response_model=list[ArticleListItem])
def latest_articles(n: int = 10, db: Session = Depends(get_db)):
    # MySQL 5.7 不支持 NULLS LAST 语法，用 (col IS NULL) 升序达到同样效果
    return (
        db.query(Article)
        .order_by(Article.content_updated_at.is_(None), Article.content_updated_at.desc(), Article.id.desc())
        .limit(max(1, min(n, 50)))
        .all()
    )


@router.get("/articles/hot", response_model=list[ArticleListItem])
def hot_articles(n: int = 10, db: Session = Depends(get_db)):
    return (
        db.query(Article)
        .order_by(Article.views.desc(), Article.id.asc())
        .limit(max(1, min(n, 50)))
        .all()
    )


def _meta_date(article: Article) -> str:
    """读 frontmatter 的 date（博客有），失败回退 content_updated_at，用于排序。"""
    try:
        import frontmatter

        post = frontmatter.loads(_article_path(article).read_text(encoding="utf-8"))
        date_val = post.metadata.get("date")
        if date_val:
            return str(date_val)
    except Exception:
        pass
    if article.content_updated_at is not None:
        return article.content_updated_at.isoformat()
    return ""


# 注意：本路由必须位于 /articles/{slug:path} 之前，否则会被 path 转换器吞掉
@router.get("/articles", response_model=list[ArticleListItem])
def list_articles(category: str, n: int = 50, db: Session = Depends(get_db)):
    """按一级分类列文章（如 category=blog），按 meta date / 内容时间倒序。"""
    articles = db.query(Article).filter(Article.category == category).all()
    articles.sort(key=_meta_date, reverse=True)
    return articles[: max(1, min(n, 100))]


@router.get("/articles/{slug:path}", response_model=ArticleDetail)
def article_detail(slug: str, db: Session = Depends(get_db)):
    article = db.query(Article).filter(Article.slug == slug).first()
    if article is None:
        raise HTTPException(status_code=404, detail="文章不存在")

    path = _article_path(article)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="文章源文件缺失")

    meta: dict[str, Any] = {}
    if article.format == "jupyter":
        html = render_notebook(path, article.slug)
    else:
        html, meta = render_article(path)

    return ArticleDetail(
        id=article.id,
        slug=article.slug,
        title=article.title,
        category=article.category,
        format=article.format,
        html=html,
        views=article.views,
        like_count=article.like_count,
        comment_count=article.comment_count,
        content_updated_at=article.content_updated_at,
        meta=meta,
    )
