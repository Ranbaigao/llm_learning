"""浏览量：同 visitor+article+当天 只计 1 次（page_view 唯一约束兜底）。"""
from __future__ import annotations

from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.security import hash_ip
from app.db import get_db
from app.models.article import Article
from app.models.page_view import PageView
from app.schemas import ViewIn, ViewOut

router = APIRouter(tags=["views"])


@router.post("/articles/{article_id}/view", response_model=ViewOut)
def record_view(article_id: int, body: ViewIn, request: Request, db: Session = Depends(get_db)):
    article = db.get(Article, article_id)
    if article is None:
        raise HTTPException(status_code=404, detail="文章不存在")

    ip = request.client.host if request.client else None
    db.add(
        PageView(
            article_id=article.id,
            visitor_id=body.visitor_id,
            ip_hash=hash_ip(ip),
            view_date=date.today(),
        )
    )
    try:
        db.commit()
        counted = True
    except IntegrityError:
        # 唯一约束冲突 = 同访客当天已计过，忽略
        db.rollback()
        counted = False

    # article.views 与 page_view 保持一致：直接以 COUNT 回写
    views = (
        db.query(PageView).filter(PageView.article_id == article.id).count()
    )
    if article.views != views:
        article.views = views
        db.commit()
    return ViewOut(counted=counted, views=views)
