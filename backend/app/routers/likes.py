"""点赞：唯一约束 (article_id, visitor_id) 保证幂等，重复操作返回当前状态。"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db import get_db
from app.models.article import Article
from app.models.article_like import ArticleLike
from app.schemas import LikeIn, LikeOut

router = APIRouter(tags=["likes"])


def _sync_like_count(db: Session, article: Article) -> int:
    count = (
        db.query(ArticleLike).filter(ArticleLike.article_id == article.id).count()
    )
    if article.like_count != count:
        article.like_count = count
        db.commit()
    return count


@router.post("/articles/{article_id}/like", response_model=LikeOut)
def like_article(article_id: int, body: LikeIn, db: Session = Depends(get_db)):
    article = db.get(Article, article_id)
    if article is None:
        raise HTTPException(status_code=404, detail="文章不存在")

    db.add(ArticleLike(article_id=article.id, visitor_id=body.visitor_id))
    try:
        db.commit()
    except IntegrityError:
        # 已赞过：返回当前状态，不报错
        db.rollback()
    return LikeOut(liked=True, like_count=_sync_like_count(db, article))


@router.delete("/articles/{article_id}/like", response_model=LikeOut)
def unlike_article(article_id: int, body: LikeIn, db: Session = Depends(get_db)):
    article = db.get(Article, article_id)
    if article is None:
        raise HTTPException(status_code=404, detail="文章不存在")

    row = (
        db.query(ArticleLike)
        .filter(
            ArticleLike.article_id == article.id,
            ArticleLike.visitor_id == body.visitor_id,
        )
        .first()
    )
    if row is not None:
        db.delete(row)
        db.commit()
    return LikeOut(liked=False, like_count=_sync_like_count(db, article))
