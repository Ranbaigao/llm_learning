"""
MkDocs hook: scan notebooks/ and generate knowledge graph data.

The hook runs before each MkDocs build. It turns the real directory tree under
notebooks/ into graph nodes and edges for notebooks/.assets/knowledge_graph.html.

Node types:
  - root: virtual root node
  - category: direct child directory of notebooks/
  - subcategory: nested directory
  - note: Markdown note file or Jupyter notebook

Edge types:
  - contains: directory/file containment
  - reference: local Markdown links between notes

New notes and directories are picked up automatically on the next MkDocs build
or serve rebuild.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

# ---- 配置 ----------------------------------------------------------------
# 项目根目录 (相对于 mkdocs.yml)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# 笔记根目录
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
# 输出位置 (knowledge_graph.html 位于 .assets/ 下, 通过 data/knowledge_graph.json 读取)
OUTPUT_PATH = NOTEBOOKS_DIR / ".assets" / "data" / "knowledge_graph.json"
# 排除目录
EXCLUDE_DIRS = {
    ".assets",
    ".git",
    "_generated",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
}
# 排除文件
EXCLUDE_FILES = {"index.md", "knowledge_graph.md"}
# 排除隐藏文件/目录
EXCLUDE_HIDDEN = True
NOTEBOOK_SUFFIXES = {".md", ".ipynb"}
MARKDOWN_LINK_RE = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
WIKI_LINK_RE = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|[^\]]+)?\]\]")
TOP_LEVEL_LABELS = {
    "llm": "LLM",
    "computer_vision": "Computer Vision",
    "llm应用开发": "LLM 应用开发",
}


def _is_excluded(name: str) -> bool:
    """判断是否应该排除该名称。"""
    if EXCLUDE_HIDDEN and name.startswith("."):
        return True
    if name in EXCLUDE_DIRS or name in EXCLUDE_FILES:
        return True
    return False


def _humanize(name: str) -> str:
    """
    将目录/文件名转换为更友好的展示名。

    - 去除文件扩展名
    - 将下划线/连字符替换为空格
    - 中文文件名保持原样
    """
    stem = Path(name).stem
    return TOP_LEVEL_LABELS.get(stem, stem.replace("_", " "))


def _to_url(rel_path: str, is_file: bool) -> str:
    """
    将相对路径转换为 MkDocs 可访问的 URL。
    - 目录 -> 以 / 结尾
    - .md 文件 -> 移除扩展名
    """
    if is_file:
        if rel_path.endswith(".md"):
            return rel_path[:-3] + "/"
        if rel_path.endswith(".ipynb"):
            return rel_path[:-6] + "/"
        return rel_path
    return rel_path + "/"


def _determine_type(depth: int) -> str:
    """根据深度决定节点类型。depth=0 表示根目录的直接子项。"""
    if depth == 0:
        return "category"
    return "subcategory"


def _is_visible_note(path: Path) -> bool:
    """Return whether a file should be represented as a note node."""
    try:
        rel_parts = path.relative_to(NOTEBOOKS_DIR).parts
    except ValueError:
        return False
    return (
        path.is_file()
        and path.suffix.lower() in NOTEBOOK_SUFFIXES
        and not any(_is_excluded(part) for part in rel_parts)
    )


def _note_count(directory: Path) -> int:
    """Count visible Markdown/Jupyter notes below a directory."""
    count = 0
    for suffix in NOTEBOOK_SUFFIXES:
        for path in directory.rglob(f"*{suffix}"):
            if _is_visible_note(path):
                count += 1
    return count


def _node_size(entry: Path, depth: int, is_file: bool) -> int:
    """Choose a stable visual size from the notebook structure."""
    if is_file:
        return 7
    notes = _note_count(entry)
    if depth == 0:
        return min(30, 18 + notes // 4)
    return min(24, max(12, 10 + notes // 3))


def _max_source_mtime(paths: list[Path]) -> int:
    """Stable freshness marker used by the frontend and write-if-changed."""
    mtimes = []
    for path in paths:
        try:
            mtimes.append(path.stat().st_mtime_ns)
        except OSError:
            continue
    return max(mtimes, default=0)


def _as_text(value: Any) -> str:
    """Normalize a Jupyter string/list source field to text."""
    if isinstance(value, list):
        return "".join(str(item) for item in value)
    if value is None:
        return ""
    return str(value)


def _resolve_markdown_target(source: Path, raw_target: str) -> str | None:
    """
    Resolve a local Markdown link target to a notebooks/-relative path.

    External URLs, anchors, missing files, and links outside notebooks/ are
    ignored so the generated graph stays deterministic.
    """
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
        candidate.relative_to(NOTEBOOKS_DIR.resolve())
    except ValueError:
        return None
    if not _is_visible_note(candidate):
        return None
    return candidate.relative_to(NOTEBOOKS_DIR).as_posix()


def _note_text_for_links(source: Path) -> str:
    """Return Markdown-like text from a Markdown or Jupyter note."""
    if source.suffix.lower() == ".ipynb":
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
    """Extract note-to-note links from local Markdown/Jupyter references."""
    links: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for source in note_files:
        source_id = source.relative_to(NOTEBOOKS_DIR).as_posix()
        text = _note_text_for_links(source)
        if not text:
            continue

        raw_targets = [match.group(1) for match in MARKDOWN_LINK_RE.finditer(text)]
        raw_targets.extend(match.group(1) for match in WIKI_LINK_RE.finditer(text))

        for raw_target in raw_targets:
            target_id = _resolve_markdown_target(source, raw_target)
            if not target_id or target_id == source_id:
                continue
            key = (source_id, target_id)
            if key in seen:
                continue
            seen.add(key)
            links.append(
                {
                    "source": source_id,
                    "target": target_id,
                    "type": "reference",
                }
            )

    return links


def scan_notebooks() -> dict[str, Any]:
    """
    递归扫描 NOTEBOOKS_DIR,生成 {nodes, links} 格式的图数据。
    """
    nodes: list[dict[str, Any]] = []
    links: list[dict[str, Any]] = []
    note_files: list[Path] = []
    source_paths: list[Path] = []

    # 1) 添加一个虚拟的 "Root" 节点,作为所有顶层分类的汇聚点
    nodes.append(
        {
            "id": "__root__",
            "name": "LLM Learning",
            "type": "root",
            "depth": -1,
            "url": "",
            "size": 28,
            "note_count": _note_count(NOTEBOOKS_DIR) if NOTEBOOKS_DIR.is_dir() else 0,
        }
    )

    if not NOTEBOOKS_DIR.is_dir():
        return {"nodes": nodes, "links": links, "stats": {"nodes": 0, "links": 0}}

    def walk(directory: Path, parent_id: str, depth: int) -> None:
        """递归遍历目录。"""
        try:
            entries = sorted(
                directory.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())
            )
        except (PermissionError, OSError):
            return

        for entry in entries:
            if _is_excluded(entry.name):
                continue

            rel_path = entry.relative_to(NOTEBOOKS_DIR).as_posix()

            if entry.is_dir():
                # ----- 目录节点 -----
                dir_id = rel_path  # 使用相对路径作为唯一 ID
                notes_below = _note_count(entry)
                nodes.append(
                    {
                        "id": dir_id,
                        "name": _humanize(entry.name),
                        "type": _determine_type(depth),
                        "depth": depth,
                        "url": _to_url(rel_path, is_file=False),
                        "size": _node_size(entry, depth, is_file=False),
                        "note_count": notes_below,
                    }
                )
                links.append(
                    {
                        "source": parent_id,
                        "target": dir_id,
                        "type": "contains",
                    }
                )
                # 递归进入子目录
                source_paths.append(entry)
                walk(entry, dir_id, depth + 1)

            elif entry.is_file() and entry.suffix.lower() in NOTEBOOK_SUFFIXES:
                # ----- 笔记文件节点 -----
                file_id = rel_path
                note_files.append(entry)
                source_paths.append(entry)
                file_format = "jupyter" if entry.suffix.lower() == ".ipynb" else "markdown"
                nodes.append(
                    {
                        "id": file_id,
                        "name": _humanize(entry.name),
                        "type": "note",
                        "format": file_format,
                        "depth": depth,
                        "url": _to_url(rel_path, is_file=True),
                        "size": _node_size(entry, depth, is_file=True),
                        "note_count": 1,
                    }
                )
                links.append(
                    {
                        "source": parent_id,
                        "target": file_id,
                        "type": "contains",
                    }
                )
            # 忽略其他文件

    walk(NOTEBOOKS_DIR, "__root__", depth=0)
    links.extend(_reference_links(note_files))

    return {
        "nodes": nodes,
        "links": links,
        "stats": {
            "nodes": len(nodes),
            "links": len(links),
            "notes": len(note_files),
            "jupyter": sum(1 for path in note_files if path.suffix.lower() == ".ipynb"),
            "contains": sum(1 for link in links if link["type"] == "contains"),
            "references": sum(1 for link in links if link["type"] == "reference"),
        },
        "source_mtime": _max_source_mtime(source_paths),
    }


def write_graph_data() -> dict[str, Any]:
    """Generate and write graph data only when the content changed."""
    data = scan_notebooks()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    current = OUTPUT_PATH.read_text(encoding="utf-8") if OUTPUT_PATH.exists() else ""
    if current != payload:
        OUTPUT_PATH.write_text(payload, encoding="utf-8")

    return data


def on_pre_build(config, **kwargs):
    """
    MkDocs hook: generate knowledge graph data before building.
    """
    data = write_graph_data()

    # 打印一行日志,方便用户确认数据已更新
    print(
        f"[knowledge-graph] 已生成 {data['stats']['nodes']} 个节点, "
        f"{data['stats']['links']} 条关系 -> {OUTPUT_PATH.relative_to(PROJECT_ROOT)}"
    )


# 支持直接运行测试: python hooks/generate_graph_data.py
if __name__ == "__main__":
    data = write_graph_data()
    print(json.dumps(data["stats"], ensure_ascii=False, indent=2))
