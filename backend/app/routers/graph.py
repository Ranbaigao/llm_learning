"""知识星图：实时扫描 content/ 输出 {nodes, links, stats}。"""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db import get_db
from app.services.graph import build_graph

router = APIRouter(tags=["graph"])


@router.get("/graph")
def graph(db: Session = Depends(get_db)):
    return build_graph(db)
