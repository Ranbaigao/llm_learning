"""评论：树形查询、发帖（昵称/长度校验 + 同 IP 限频）、管理员软删。"""
from __future__ import annotations

from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import hash_ip
from app.db import get_db
from app.models.article import Article
from app.models.comment import Comment
from app.schemas import CommentIn, CommentOut

router = APIRouter(tags=["comments"])

# 同 IP 1 分钟最多 3 条
RATE_LIMIT_WINDOW = timedelta(minutes=1)
RATE_LIMIT_MAX = 3


def _sync_comment_count(db: Session, article: Article) -> int:
    count = (
        db.query(Comment)
        .filter(Comment.article_id == article.id, Comment.status == "visible")
        .count()
    )
    if article.comment_count != count:
        article.comment_count = count
        db.commit()
    return count


def _to_out(comment: Comment) -> CommentOut:
    return CommentOut(
        id=comment.id,
        article_id=comment.article_id,
        nickname=comment.nickname,
        content=comment.content,
        parent_id=comment.parent_id,
        created_at=comment.created_at,
        children=[],
    )


@router.get("/articles/{article_id}/comments", response_model=list[CommentOut])
def list_comments(article_id: int, db: Session = Depends(get_db)):
    """树形返回（任意层嵌套，按时间正序），软删的不返回。"""
    comments = (
        db.query(Comment)
        .filter(Comment.article_id == article_id, Comment.status == "visible")
        .order_by(Comment.created_at.asc(), Comment.id.asc())
        .all()
    )
    nodes = {c.id: _to_out(c) for c in comments}
    roots: list[CommentOut] = []
    for c in comments:
        node = nodes[c.id]
        if c.parent_id is not None and c.parent_id in nodes:
            nodes[c.parent_id].children.append(node)
        else:
            roots.append(node)
    return roots


@router.post("/articles/{article_id}/comments", response_model=CommentOut, status_code=201)
def create_comment(
    article_id: int, body: CommentIn, request: Request, db: Session = Depends(get_db)
):
    article = db.get(Article, article_id)
    if article is None:
        raise HTTPException(status_code=404, detail="文章不存在")

    nickname = body.nickname.strip()
    content = body.content.strip()
    if not nickname:
        raise HTTPException(status_code=422, detail="昵称不能为空")
    if not content:
        raise HTTPException(status_code=422, detail="评论内容不能为空")

    ip = request.client.host if request.client else None
    ip_h = hash_ip(ip)

    # 限频：同 IP 1 分钟最多 3 条
    cutoff = datetime.now() - RATE_LIMIT_WINDOW
    recent = (
        db.query(Comment)
        .filter(Comment.ip_hash == ip_h, Comment.created_at >= cutoff)
        .count()
    )
    if recent >= RATE_LIMIT_MAX:
        raise HTTPException(status_code=429, detail="发言太频繁，请稍后再试")

    if body.parent_id is not None:
        parent = db.get(Comment, body.parent_id)
        if (
            parent is None
            or parent.article_id != article.id
            or parent.status != "visible"
        ):
            raise HTTPException(status_code=400, detail="父评论不存在")

    comment = Comment(
        article_id=article.id,
        user_id=None,
        nickname=nickname,
        content=content,
        parent_id=body.parent_id,
        status="visible",
        ip_hash=ip_h,
    )
    db.add(comment)
    db.commit()
    db.refresh(comment)
    _sync_comment_count(db, article)
    return _to_out(comment)


@router.delete("/comments/{comment_id}")
def delete_comment(
    comment_id: int,
    db: Session = Depends(get_db),
    x_admin_token: str | None = Header(default=None),
):
    if x_admin_token != settings.ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="管理令牌无效")

    comment = db.get(Comment, comment_id)
    if comment is None:
        raise HTTPException(status_code=404, detail="评论不存在")
    if comment.status != "deleted":
        comment.status = "deleted"
        db.commit()
        article = db.get(Article, comment.article_id)
        if article is not None:
            _sync_comment_count(db, article)
    return {"ok": True, "id": comment_id, "status": "deleted"}
