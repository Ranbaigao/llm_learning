"""ORM 模型汇总（供 main.py create_all 一次性导入）。"""
from app.models.article import Article
from app.models.article_like import ArticleLike
from app.models.comment import Comment
from app.models.page_view import PageView
from app.models.user import User

__all__ = ["Article", "ArticleLike", "Comment", "PageView", "User"]
