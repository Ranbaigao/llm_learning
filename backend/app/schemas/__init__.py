"""Pydantic 请求/响应模型。"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---- 请求 ------------------------------------------------------------------
class ViewIn(BaseModel):
    visitor_id: str = Field(min_length=1, max_length=64)


class LikeIn(BaseModel):
    visitor_id: str = Field(min_length=1, max_length=64)


class CommentIn(BaseModel):
    visitor_id: str = Field(min_length=1, max_length=64)
    nickname: str = Field(min_length=1, max_length=20)
    content: str = Field(min_length=1, max_length=2000)
    parent_id: Optional[int] = None


class WxMiniLoginIn(BaseModel):
    code: str = Field(min_length=1, max_length=128)
    nickname: Optional[str] = Field(default=None, max_length=64)
    avatar: Optional[str] = Field(default=None, max_length=512)


# ---- 响应 ------------------------------------------------------------------
class ArticleListItem(BaseModel):
    id: int
    slug: str
    title: str
    category: str
    format: str
    views: int
    like_count: int
    comment_count: int
    content_updated_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class ArticleDetail(BaseModel):
    id: int
    slug: str
    title: str
    category: str
    format: str
    html: str
    views: int
    like_count: int
    comment_count: int
    content_updated_at: Optional[datetime] = None
    meta: dict[str, Any] = {}  # frontmatter 元数据（title/date/categories 等）


class CommentOut(BaseModel):
    id: int
    article_id: int
    nickname: str
    content: str
    parent_id: Optional[int] = None
    created_at: datetime
    children: list["CommentOut"] = []


class SearchResult(BaseModel):
    slug: str
    title: str
    category: str
    snippet: str


class LikeOut(BaseModel):
    liked: bool
    like_count: int


class ViewOut(BaseModel):
    counted: bool
    views: int
