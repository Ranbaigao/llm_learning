from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Integer, String, func
from sqlalchemy.dialects.mysql import LONGTEXT
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class Article(Base):
    __tablename__ = "article"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # content 相对路径去扩展名，posix 风格，如 NLP/LLM模型架构/xxx
    slug: Mapped[str] = mapped_column(String(512), unique=True, index=True)
    title: Mapped[str] = mapped_column(String(512))
    path: Mapped[str] = mapped_column(String(768))  # 含扩展名的相对路径
    format: Mapped[str] = mapped_column(String(16))  # markdown / jupyter
    category: Mapped[str] = mapped_column(String(128), index=True)  # 一级目录名
    text_content: Mapped[str] = mapped_column(LONGTEXT)  # 纯文本快照，供搜索
    views: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    like_count: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    comment_count: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    content_updated_at: Mapped[datetime | None] = mapped_column(
        DateTime, nullable=True, index=True
    )  # 内容时间戳（git log -M 取内容修改时间，回退文件 mtime）
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )
