"""Jupyter Notebook → HTML 片段渲染服务（nbconvert basic 模板，磁盘缓存）。

绝不执行 notebook 代码：只做格式转换，不挂 ExecutePreprocessor。
"""
from __future__ import annotations

import logging
from pathlib import Path
from urllib.parse import quote

import nbformat
from nbconvert import HTMLExporter

from app.core.config import settings
from app.services.markdown import rewrite_links

logger = logging.getLogger(__name__)


def _cache_paths(slug: str) -> tuple[Path, Path]:
    cache_dir = settings.nb_cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe = quote(slug, safe="")  # 全量转义，避免路径分隔符与中文问题
    return cache_dir / f"{safe}.html", cache_dir / f"{safe}.meta"


def render_notebook(path: Path, slug: str) -> str:
    """渲染 .ipynb 为 body HTML 片段。按 (path, mtime) 命中磁盘缓存。"""
    try:
        mtime = path.stat().st_mtime_ns
    except OSError:
        mtime = -1
    signature = f"{path}|{mtime}"

    cache_html, cache_meta = _cache_paths(slug)
    if cache_html.exists() and cache_meta.exists():
        try:
            if cache_meta.read_text(encoding="utf-8") == signature:
                return cache_html.read_text(encoding="utf-8")
        except OSError:
            pass

    nb = nbformat.read(path, as_version=4)
    exporter = HTMLExporter(template_name="basic")
    body, _resources = exporter.from_notebook_node(nb)
    body = rewrite_links(body, path)

    try:
        cache_html.write_text(body, encoding="utf-8")
        cache_meta.write_text(signature, encoding="utf-8")
    except OSError as exc:
        logger.warning("notebook 缓存写入失败 %s: %s", slug, exc)
    return body
