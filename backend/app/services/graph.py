"""知识星图数据：移植自旧 MkDocs 钩子 hooks/generate_graph_data.py 的扫描逻辑。

差异：
- 内容目录 notebooks/ → content/（通过 settings.content_dir）
- note 节点 url 改为新站路由 /notes/{slug 去扩展名}
- note 节点 pv 来自 article.views（page_view 冗余计数），不再走 Vercount
- 不写 JSON 落盘，每次请求实时返回
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlparse

from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.article import Article
from app.services.content_index import (
    EXCLUDE_DIRS,
    EXCLUDE_FILES,
    EXCLUDE_HIDDEN,
    NOTEBOOK_SUFFIXES,
    git_last_commit_times,
)

MARKDOWN_LINK_RE = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
WIKI_LINK_RE = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|[^\]]+)?\]\]")
# 引用式链接定义, 例如: [5]: ../path/note.md "标题" 或 [5]: <../path with space/note.md>
REF_LINK_DEF_RE = re.compile(r"(?m)^[ \t]{0,3}\[([^\]]+)\]:[ \t]*(?:<([^>]+)>|(\S+))")
# 引用式链接使用, 例如: [文字][5] 或折叠形式 [5][]
REF_LINK_USE_RE = re.compile(r"(?<!!)\[([^\]]+)\]\[([^\]]*)")


def _is_excluded(name: str) -> bool:
    if EXCLUDE_HIDDEN and name.startswith("."):
        return True
    if name in EXCLUDE_DIRS or name in EXCLUDE_FILES:
        return True
    return False


def _humanize(name: str) -> str:
    stem = Path(name).stem
    return stem.replace("_", " ")


def _determine_type(depth: int) -> str:
    if depth == 0:
        return "category"
    return "subcategory"


def _is_visible_note(path: Path) -> bool:
    content_dir = settings.content_dir
    try:
        rel_parts = path.relative_to(content_dir).parts
    except ValueError:
        return False
    return (
        path.is_file()
        and path.suffix.lower() in NOTEBOOK_SUFFIXES
        and not any(_is_excluded(part) for part in rel_parts)
    )


def _note_count(directory: Path) -> int:
    count = 0
    for suffix in NOTEBOOK_SUFFIXES:
        for path in directory.rglob(f"*{suffix}"):
            if _is_visible_note(path):
                count += 1
    return count


def _node_size(entry: Path, depth: int, is_file: bool) -> int:
    if is_file:
        return 7
    notes = _note_count(entry)
    if depth == 0:
        return min(30, 18 + notes // 4)
    return min(24, max(12, 10 + notes // 3))


def _as_text(value: Any) -> str:
    if isinstance(value, list):
        return "".join(str(item) for item in value)
    if value is None:
        return ""
    return str(value)


def _note_url(rel_path: str) -> str:
    """笔记节点 URL：/notes/{slug 去扩展名}，按段 URL 编码。"""
    slug = rel_path
    for suffix in (".md", ".ipynb"):
        if slug.lower().endswith(suffix):
            slug = slug[: -len(suffix)]
            break
    return "/notes/" + quote(slug, safe="/")


def _resolve_markdown_target(source: Path, raw_target: str) -> str | None:
    """解析本地 Markdown 链接目标为 content 相对路径；外部链接/锚点/不存在则忽略。"""
    content_dir = settings.content_dir
    raw_target = raw_target.strip()
    if not raw_target or raw_target.startswith("#"):
        return None

    parsed = urlparse(raw_target)
    if parsed.scheme or parsed.netloc:
        return None

    target_path = unquote(parsed.path)
    if not target_path:
        return None
    if Path(target_path).suffix.lower() not in NOTEBOOK_SUFFIXES:
        target_path += ".md"

    candidate = (source.parent / target_path).resolve()
    try:
        candidate.relative_to(content_dir.resolve())
    except ValueError:
        return None
    if not _is_visible_note(candidate):
        return None
    return candidate.relative_to(content_dir).as_posix()


def _note_text_for_links(source: Path) -> str:
    """返回 Markdown/Jupyter 笔记的 Markdown 文本（ipynb 只取 markdown/raw 单元格）。"""
    if source.suffix.lower() == ".ipynb":
        import json

        try:
            notebook = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            return ""
        return "\n".join(
            _as_text(cell.get("source"))
            for cell in notebook.get("cells", [])
            if cell.get("cell_type") in {"markdown", "raw"}
        )

    try:
        return source.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return source.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def _reference_links(note_files: list[Path]) -> list[dict[str, str]]:
    """从笔记正文中提取笔记间的引用边（Markdown/Wiki/引用式链接）。"""
    content_dir = settings.content_dir
    links: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for source in note_files:
        source_id = source.relative_to(content_dir).as_posix()
        text = _note_text_for_links(source)
        if not text:
            continue

        raw_targets = [match.group(1) for match in MARKDOWN_LINK_RE.finditer(text)]
        raw_targets.extend(match.group(1) for match in WIKI_LINK_RE.finditer(text))

        # 引用式链接: 先收集文末的 [label]: target 定义, 再解析 [文字][label] 使用
        ref_defs: dict[str, str] = {}
        for def_match in REF_LINK_DEF_RE.finditer(text):
            label = def_match.group(1).strip().casefold()
            target = def_match.group(2) or def_match.group(3)
            if label and target:
                ref_defs[label] = target
        if ref_defs:
            for use_match in REF_LINK_USE_RE.finditer(text):
                label = (use_match.group(2) or use_match.group(1)).strip().casefold()
                target = ref_defs.get(label)
                if target:
                    raw_targets.append(target)

        for raw_target in raw_targets:
            target_id = _resolve_markdown_target(source, raw_target)
            if not target_id or target_id == source_id:
                continue
            key = (source_id, target_id)
            if key in seen:
                continue
            seen.add(key)
            links.append({"source": source_id, "target": target_id, "type": "reference"})

    return links


def build_graph(db: Session) -> dict[str, Any]:
    """扫描 content/ 生成 {nodes, links, stats}。note 节点附 mtime/pv。"""
    content_dir = settings.content_dir
    nodes: list[dict[str, Any]] = []
    links: list[dict[str, Any]] = []
    note_files: list[Path] = []
    # 每篇笔记的最近内容修改时间（秒），取不到时回退文件 mtime
    commit_times = git_last_commit_times()
    # slug -> views，用于 note 节点 pv
    views_map = {a.slug.casefold(): a.views for a in db.query(Article.slug, Article.views)}

    # 1) 虚拟 Root 节点，作为所有顶层分类的汇聚点
    nodes.append(
        {
            "id": "__root__",
            "name": "LLM Learning",
            "type": "root",
            "depth": -1,
            "url": "",
            "size": 28,
            "note_count": _note_count(content_dir) if content_dir.is_dir() else 0,
        }
    )

    if not content_dir.is_dir():
        return {"nodes": nodes, "links": links, "stats": {"nodes": 0, "links": 0}}

    def walk(directory: Path, parent_id: str, depth: int) -> None:
        try:
            entries = sorted(
                directory.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())
            )
        except (PermissionError, OSError):
            return

        for entry in entries:
            if _is_excluded(entry.name):
                continue

            rel_path = entry.relative_to(content_dir).as_posix()

            if entry.is_dir():
                dir_id = rel_path
                notes_below = _note_count(entry)
                nodes.append(
                    {
                        "id": dir_id,
                        "name": _humanize(entry.name),
                        "type": _determine_type(depth),
                        "depth": depth,
                        "url": "",  # 目录在新站没有独立路由页
                        "size": _node_size(entry, depth, is_file=False),
                        "note_count": notes_below,
                    }
                )
                links.append({"source": parent_id, "target": dir_id, "type": "contains"})
                walk(entry, dir_id, depth + 1)

            elif entry.is_file() and entry.suffix.lower() in NOTEBOOK_SUFFIXES:
                file_id = rel_path
                note_files.append(entry)
                file_format = (
                    "jupyter" if entry.suffix.lower() == ".ipynb" else "markdown"
                )
                try:
                    file_mtime = int(entry.stat().st_mtime)
                except OSError:
                    file_mtime = 0
                slug = rel_path[: -len(entry.suffix)]
                nodes.append(
                    {
                        "id": file_id,
                        "name": _humanize(entry.name),
                        "type": "note",
                        "format": file_format,
                        "depth": depth,
                        "url": _note_url(rel_path),
                        "size": _node_size(entry, depth, is_file=True),
                        "note_count": 1,
                        "mtime": commit_times.get(rel_path.casefold(), file_mtime),
                        "pv": views_map.get(slug.casefold(), 0),
                    }
                )
                links.append({"source": parent_id, "target": file_id, "type": "contains"})
            # 忽略其他文件

    walk(content_dir, "__root__", depth=0)
    links.extend(_reference_links(note_files))

    return {
        "nodes": nodes,
        "links": links,
        "stats": {
            "nodes": len(nodes),
            "links": len(links),
            "notes": len(note_files),
            "jupyter": sum(
                1 for path in note_files if path.suffix.lower() == ".ipynb"
            ),
            "contains": sum(1 for link in links if link["type"] == "contains"),
            "references": sum(1 for link in links if link["type"] == "reference"),
        },
    }
