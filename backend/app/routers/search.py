"""搜索：title/text_content LIKE，返回前 20 条带摘要片段。"""
from __future__ import annotations

import re

from fastapi import APIRouter, Depends, Query
from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.db import get_db
from app.models.article import Article
from app.schemas import SearchResult

router = APIRouter(tags=["search"])

_WS_RE = re.compile(r"\s+")


def _escape_like(q: str) -> str:
    return q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _make_snippet(text: str, q: str, width: int = 80) -> str:
    plain = _WS_RE.sub(" ", text or "").strip()
    idx = plain.lower().find(q.lower())
    if idx < 0:
        return plain[: width * 2]
    start = max(0, idx - width)
    end = min(len(plain), idx + len(q) + width)
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(plain) else ""
    return f"{prefix}{plain[start:end]}{suffix}"


@router.get("/search", response_model=list[SearchResult])
def search(q: str = Query(min_length=1, max_length=100), db: Session = Depends(get_db)):
    pattern = f"%{_escape_like(q)}%"
    rows = (
        db.query(Article)
        .filter(
            or_(
                Article.title.like(pattern, escape="\\"),
                Article.text_content.like(pattern, escape="\\"),
            )
        )
        .order_by(Article.views.desc(), Article.id.asc())
        .limit(20)
        .all()
    )
    return [
        SearchResult(
            slug=a.slug,
            title=a.title,
            category=a.category,
            snippet=_make_snippet(a.text_content, q),
        )
        for a in rows
    ]
