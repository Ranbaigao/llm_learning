"""扫描 content/ 目录，同步 article 表。

目录扫描与内容时间戳逻辑移植自已删除的 MkDocs 钩子
hooks/generate_graph_data.py（见 git 历史），去掉了 Vercount 浏览量烘焙
和 MkDocs 耦合。
"""
from __future__ import annotations

import json
import logging
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import frontmatter
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.article import Article

logger = logging.getLogger(__name__)

# ---- 扫描配置（与旧钩子一致）------------------------------------------------
EXCLUDE_DIRS = {
    ".assets",
    ".git",
    "_generated",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "blog",  # 博客是独立板块：不进星图与笔记目录树（但仍进 article 索引）
}
EXCLUDE_FILES = {"index.md", "knowledge_graph.md"}
EXCLUDE_HIDDEN = True
NOTEBOOK_SUFFIXES = {".md", ".ipynb"}

# 索引（article 表）不排除 blog，只排除资产/垃圾目录
_INDEX_EXCLUDE_DIRS = EXCLUDE_DIRS - {"blog"}

# git 历史里内容目录曾叫 notebooks/，现名 content/。键统一归一化为
# 「内容目录相对路径」，两种前缀都能命中（改名提交与否都兼容）。
_LEGACY_ROOT_PREFIXES = ("notebooks/",)


def _is_excluded(name: str, exclude_dirs: set[str] | frozenset[str] | None = None) -> bool:
    """exclude_dirs 为整组替换的目录排除集；None 时用默认 EXCLUDE_DIRS。"""
    if EXCLUDE_HIDDEN and name.startswith("."):
        return True
    if name in (EXCLUDE_DIRS if exclude_dirs is None else exclude_dirs):
        return True
    if name in EXCLUDE_FILES:
        return True
    return False


def _humanize(name: str) -> str:
    """目录/文件名转展示名：去扩展名、下划线/连字符转空格、中文原样保留。"""
    stem = Path(name).stem
    return stem.replace("_", " ")


def iter_note_files(
    exclude_dirs: set[str] | frozenset[str] | None = None,
    exclude_files: bool = False,
) -> Iterator[Path]:
    """遍历 content/ 下所有可见笔记文件（.md/.ipynb）。

    - exclude_dirs: 整组替换的目录排除集；索引用 _INDEX_EXCLUDE_DIRS（含 blog），
      树/星图不传（默认 EXCLUDE_DIRS，排除 blog）
    - exclude_files: 是否应用 EXCLUDE_FILES（index.md 等），树/星图用
    """
    content_dir = settings.content_dir
    if not content_dir.is_dir():
        return

    def walk(directory: Path) -> Iterator[Path]:
        try:
            entries = sorted(
                directory.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())
            )
        except (PermissionError, OSError):
            return
        for entry in entries:
            if _is_excluded(entry.name, exclude_dirs):
                continue
            if entry.is_dir():
                yield from walk(entry)
            elif entry.is_file() and entry.suffix.lower() in NOTEBOOK_SUFFIXES:
                if exclude_files and entry.name in EXCLUDE_FILES:
                    continue
                yield entry

    yield from walk(content_dir)


# ---- git 内容时间戳（移植自旧钩子 _git_last_commit_times）--------------------
def git_last_commit_times() -> dict[str, int]:
    """一次 git log 调用，获取每个笔记文件最近一次【内容修改】的提交时间戳（秒）。

    返回键为 content/ 相对路径（posix、casefold、含扩展名）。
    --name-status -M 开启重命名检测：纯移动（相似度 >=90%）沿用改名前时间；
    移动同时大改（<90%）才算内容更新。失败返回 {}，调用方回退文件 mtime。
    """
    content_dir = settings.content_dir
    project_root = content_dir.parent
    root_prefixes = tuple(
        dict.fromkeys([content_dir.name.casefold() + "/", *_LEGACY_ROOT_PREFIXES])
    )

    def _strip_root(path_cf: str) -> str | None:
        """去掉内容根前缀，归一化为内容目录相对路径。"""
        for prefix in root_prefixes:
            if path_cf.startswith(prefix):
                return path_cf[len(prefix) :]
        return None

    try:
        result = subprocess.run(
            ["git", "-c", "core.quotepath=false", "log", "--format=@%ct", "--name-status", "-M"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=project_root,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("git log 调用失败(%s)，内容时间将回退为文件 mtime", exc)
        return {}
    if result.returncode != 0:
        logger.warning("git log 退出码 %s，内容时间将回退为文件 mtime", result.returncode)
        return {}

    def _current_path(path_cf: str) -> str:
        """沿改名链解析到最新路径（git log 倒序遍历，alias 只指向更晚的名字）。"""
        seen = set()
        while path_cf in alias and path_cf not in seen:
            seen.add(path_cf)
            path_cf = alias[path_cf]
        return path_cf

    times: dict[str, int] = {}
    alias: dict[str, str] = {}  # 改名前路径 -> 改名后路径（仓库相对，casefold）
    current = 0
    for line in result.stdout.splitlines():
        if line.startswith("@"):
            try:
                current = int(line[1:])
            except ValueError:
                current = 0
        elif line and current:
            parts = line.split("\t")
            status = parts[0]
            if status.startswith("R") and len(parts) == 3:
                old_cf = parts[1].casefold()
                new_cf = parts[2].casefold()
                # 纯大小写改名在 casefold 语义下是同一个键，写入 alias 会断链
                if old_cf != new_cf:
                    alias[old_cf] = new_cf
                similarity = int(status[1:]) if status[1:].isdigit() else 100
                if similarity < 90:
                    key = _strip_root(_current_path(new_cf))
                    if key:
                        times.setdefault(key, current)
            elif status[:1] in ("A", "M", "T") and len(parts) == 2:
                # git log 按时间倒序输出，首次出现即为最近一次内容修改
                key = _strip_root(_current_path(parts[1].casefold()))
                if key:
                    times.setdefault(key, current)
    if not times:
        logger.warning("未从 git 历史解析到任何提交时间，内容时间将回退为文件 mtime")
    return times


# ---- 标题与纯文本提取 -------------------------------------------------------
_H1_RE = re.compile(r"(?m)^#[ \t]+(.+?)[ \t]*#*$")
_CODE_FENCE_RE = re.compile(r"(?ms)^```[^\n]*\n(.*?)^```[ \t]*$")
_IMG_RE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_LINK_RE = re.compile(r"(?<!!)\[([^\]]+)\]\([^)]*\)")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_BLANK_LINES_RE = re.compile(r"\n{3,}")


def _markdown_to_plain(text: str) -> str:
    """Markdown 源码转纯文本快照（供 LIKE 搜索），保留代码块内容。"""
    text = _CODE_FENCE_RE.sub(lambda m: "\n" + m.group(1) + "\n", text)
    text = _IMG_RE.sub(lambda m: m.group(1), text)
    text = _LINK_RE.sub(lambda m: m.group(1), text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = re.sub(r"(?m)^#{1,6}[ \t]*", "", text)
    text = _BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()


def _notebook_cells(path: Path) -> list[dict[str, Any]]:
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return []
    return notebook.get("cells", [])


def _as_text(value: Any) -> str:
    if isinstance(value, list):
        return "".join(str(item) for item in value)
    if value is None:
        return ""
    return str(value)


def extract_title_and_text(path: Path) -> tuple[str, str]:
    """返回 (title, text_content)。

    title 优先级：frontmatter title > 首个一级标题 > 文件名（humanize）。
    """
    fallback_title = _humanize(path.name)
    if path.suffix.lower() == ".ipynb":
        cells = _notebook_cells(path)
        parts = [_as_text(c.get("source")) for c in cells]
        title = fallback_title
        for cell in cells:
            if cell.get("cell_type") == "markdown":
                m = _H1_RE.search(_as_text(cell.get("source")))
                if m:
                    title = m.group(1).strip()
                    break
        return title, "\n\n".join(p for p in parts if p).strip()

    try:
        raw = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raw = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return fallback_title, ""

    try:
        post = frontmatter.loads(raw)
        body = post.content
        meta_title = post.metadata.get("title")
    except Exception:  # frontmatter 解析失败不应打断索引
        body = raw
        meta_title = None

    title = None
    if isinstance(meta_title, str) and meta_title.strip():
        title = meta_title.strip()
    if not title:
        m = _H1_RE.search(body)
        if m:
            title = m.group(1).strip()
    return title or fallback_title, _markdown_to_plain(body)


def slug_for_path(path: Path) -> str:
    """content 相对路径去扩展名，posix 风格。"""
    rel = path.relative_to(settings.content_dir).as_posix()
    return rel[: -len(path.suffix)] if path.suffix else rel


# ---- 同步入口 ---------------------------------------------------------------
def sync_content_index(db: Session) -> dict[str, int]:
    """扫描 content/ 全量同步 article 表：新增/更新/删除。返回统计。"""
    files = list(iter_note_files(exclude_dirs=_INDEX_EXCLUDE_DIRS))
    commit_times = git_last_commit_times()

    seen_paths: set[str] = set()
    added = updated = 0
    for path in files:
        rel = path.relative_to(settings.content_dir).as_posix()
        seen_paths.add(rel)
        slug = slug_for_path(path)
        fmt = "jupyter" if path.suffix.lower() == ".ipynb" else "markdown"
        category = rel.split("/", 1)[0]
        try:
            title, text = extract_title_and_text(path)
        except Exception as exc:  # 单文件失败不阻塞整体同步
            logger.warning("提取失败 %s: %s", rel, exc)
            continue
        ts = commit_times.get(rel.casefold())
        if ts is None:
            try:
                ts = int(path.stat().st_mtime)
            except OSError:
                ts = 0
        content_updated_at = datetime.fromtimestamp(ts) if ts else None

        row = db.query(Article).filter(Article.slug == slug).first()
        if row is None:
            db.add(
                Article(
                    slug=slug,
                    title=title,
                    path=rel,
                    format=fmt,
                    category=category,
                    text_content=text,
                    content_updated_at=content_updated_at,
                )
            )
            added += 1
        else:
            row.title = title
            row.path = rel
            row.format = fmt
            row.category = category
            row.text_content = text
            row.content_updated_at = content_updated_at
            updated += 1

    removed = 0
    if seen_paths:
        stale_rows = db.query(Article).filter(~Article.path.in_(seen_paths)).all()
    else:
        stale_rows = db.query(Article).all()
    for row in stale_rows:
        db.delete(row)
        removed += 1

    db.commit()
    stats = {"added": added, "updated": updated, "removed": removed, "total": len(seen_paths)}
    logger.info("内容索引同步完成: %s", stats)
    return stats
