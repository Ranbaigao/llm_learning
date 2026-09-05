"""站点级统计：总浏览量（PV）与独立访客数（UV）。"""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.db import get_db
from app.models.article import Article
from app.models.page_view import PageView

router = APIRouter(tags=["stats"])


@router.get("/stats/site")
def site_stats(db: Session = Depends(get_db)) -> dict[str, int]:
    # site_pv = 全部文章浏览量求和；site_uv = page_view 去重访客数
    site_pv = db.query(func.coalesce(func.sum(Article.views), 0)).scalar() or 0
    site_uv = db.query(func.count(func.distinct(PageView.visitor_id))).scalar() or 0
    return {"site_pv": int(site_pv), "site_uv": int(site_uv)}
