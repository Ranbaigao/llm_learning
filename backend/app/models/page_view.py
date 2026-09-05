from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import Date, DateTime, ForeignKey, Integer, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class PageView(Base):
    __tablename__ = "page_view"
    __table_args__ = (
        UniqueConstraint(
            "article_id", "visitor_id", "view_date", name="uq_view_article_visitor_date"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    article_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("article.id"), index=True
    )
    visitor_id: Mapped[str] = mapped_column(String(64))
    ip_hash: Mapped[str] = mapped_column(String(64), default="", server_default="")
    view_date: Mapped[date] = mapped_column(Date)  # 按天去重
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
