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
import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlparse
from urllib.request import Request, urlopen

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
    "blog",  # 博客是独立板块，不属于知识库笔记，不进星图与笔记目录
}
# 排除文件
EXCLUDE_FILES = {"index.md", "knowledge_graph.md"}
# 排除隐藏文件/目录
EXCLUDE_HIDDEN = True
NOTEBOOK_SUFFIXES = {".md", ".ipynb"}
MARKDOWN_LINK_RE = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
WIKI_LINK_RE = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|[^\]]+)?\]\]")
# 引用式链接定义, 例如: [5]: ../path/note.md "标题" 或 [5]: <../path with space/note.md>
REF_LINK_DEF_RE = re.compile(
    r"(?m)^[ \t]{0,3}\[([^\]]+)\]:[ \t]*(?:<([^>]+)>|(\S+))"
)
# 引用式链接使用, 例如: [文字][5] 或折叠形式 [5][]
REF_LINK_USE_RE = re.compile(r"(?<!!)\[([^\]]+)\]\[([^\]]*)\]")
TOP_LEVEL_LABELS = {}
INDEX_PATH = NOTEBOOKS_DIR / "index.md"
NAV_START_MARKER = "<!-- AUTO-GENERATED-NOTE-NAV:START -->"
NAV_END_MARKER = "<!-- AUTO-GENERATED-NOTE-NAV:END -->"

# ---- 浏览量烘焙配置 --------------------------------------------------------
# 构建时通过 Vercount 只读接口抓取各笔记浏览量，烘焙进图谱 JSON，
# 前端侧边栏渲染时直接显示，访客零网络请求、零延迟。
# 缓存 6 小时：mkdocs serve 频繁重建不会重复访问网络，也避免 rebuild 循环。
# 环境变量 KG_PAGEVIEW=off 可完全禁用（离线开发场景）。
PAGEVIEW_API = "https://events.vercount.one/api/v2/log"
SITE_BASE_URL = "https://ranbaigao.github.io/llm_learning/"
PAGEVIEW_CACHE_PATH = NOTEBOOKS_DIR / ".assets" / "data" / "pageview_cache.json"
PAGEVIEW_CACHE_TTL = 6 * 3600
PAGEVIEW_WORKERS = 8
PAGEVIEW_TIMEOUT = 10
# 接口对脚本 UA 有风控，伪装成浏览器
_BROWSER_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)


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


def _strip_index_navigation(content: str) -> str:
    """
    Remove the legacy generated navigation block from notebooks/index.md.

    \u7b14\u8bb0\u76ee\u5f55\u5df2\u8fc1\u79fb\u5230\u661f\u56fe\u5185\u90e8\u7684\u6298\u53e0\u4fa7\u8fb9\u680f\uff08\u7531 knowledge_graph.html \u76f4\u63a5\u6839\u636e
    \u56fe\u8c31 JSON \u6e32\u67d3\uff09\uff0c\u9996\u9875\u4e0d\u518d\u4fdd\u7559\u7ae0\u8282\u76ee\u5f55\uff1b\u6b64\u51fd\u6570\u7528\u4e8e\u6e05\u7406\u5386\u53f2\u9057\u7559\u7684
    AUTO-GENERATED-NOTE-NAV \u533a\u5757\uff0c\u9632\u6b62\u65e7\u5185\u5bb9\u6b8b\u7559\u3002
    """
    if NAV_START_MARKER in content and NAV_END_MARKER in content:
        start = content.index(NAV_START_MARKER)
        end = content.index(NAV_END_MARKER, start) + len(NAV_END_MARKER)
        block_start = content.rfind('<span id="kg-nav"></span>', 0, start)
        if block_start == -1:
            block_start = start
        return content[:block_start].rstrip() + "\n" + content[end:].lstrip("\n")

    match = re.search(r'(?ms)^<span id="kg-nav"></span>\s*## \u7b14\u8bb0\u5bfc\u822a\b.*\Z', content)
    if match:
        return content[: match.start()].rstrip() + "\n"

    return content


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
            links.append(
                {
                    "source": source_id,
                    "target": target_id,
                    "type": "reference",
                }
            )

    return links


def remove_index_navigation() -> bool:
    """Strip the legacy generated navigation block from notebooks/index.md."""
    if not INDEX_PATH.exists():
        return False

    current = INDEX_PATH.read_text(encoding="utf-8")
    next_content = _strip_index_navigation(current)
    if current != next_content:
        INDEX_PATH.write_text(next_content, encoding="utf-8")
        return True
    return False


def _git_last_commit_times() -> dict[str, int]:
    """一次 git log 调用，获取 notebooks/ 下每个文件最近一次提交的时间戳（秒）。

    用于星图「最新更新」面板。注意 CI 部署时 actions/checkout 必须配置
    fetch-depth: 0（完整历史），否则所有文件只会得到最后一次提交的时间。
    """
    try:
        result = subprocess.run(
            ["git", "log", "--format=@%ct", "--name-only", "--", "notebooks"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=PROJECT_ROOT,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    if result.returncode != 0:
        return {}

    times: dict[str, int] = {}
    current = 0
    for line in result.stdout.splitlines():
        if line.startswith("@"):
            try:
                current = int(line[1:])
            except ValueError:
                current = 0
        elif line and current:
            rel = line.strip()
            if rel.startswith("notebooks/"):
                # git log 按时间倒序输出，首次出现即为最近一次提交
                times.setdefault(rel[len("notebooks/"):], current)
    return times


def scan_notebooks() -> dict[str, Any]:
    """
    递归扫描 NOTEBOOKS_DIR,生成 {nodes, links} 格式的图数据。
    """
    nodes: list[dict[str, Any]] = []
    links: list[dict[str, Any]] = []
    note_files: list[Path] = []
    source_paths: list[Path] = []
    # 每篇笔记的最近 git 提交时间（秒），取不到时回退文件 mtime
    commit_times = _git_last_commit_times()

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
                try:
                    file_mtime = int(entry.stat().st_mtime)
                except OSError:
                    file_mtime = 0
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
                        "mtime": commit_times.get(rel_path, file_mtime),
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


def write_graph_data(data: dict[str, Any] | None = None) -> dict[str, Any]:
    """Generate and write graph data only when the content changed."""
    if data is None:
        data = scan_notebooks()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    current = OUTPUT_PATH.read_text(encoding="utf-8") if OUTPUT_PATH.exists() else ""
    if current != payload:
        OUTPUT_PATH.write_text(payload, encoding="utf-8")

    return data


# ---- 浏览量烘焙 ------------------------------------------------------------


def _pageview_get(url: str) -> dict[str, Any] | None:
    """GET Vercount 只读计数接口（不增加计数），失败返回 None。"""
    query = f"{PAGEVIEW_API}?url={quote(url, safe='')}"
    try:
        req = Request(query, headers={"User-Agent": _BROWSER_UA})
        with urlopen(req, timeout=PAGEVIEW_TIMEOUT) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None
    if payload.get("status") == "success" and isinstance(payload.get("data"), dict):
        return payload["data"]
    return None


def _load_pageview_cache() -> dict[str, Any]:
    try:
        return json.loads(PAGEVIEW_CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _write_pageview_cache(cache: dict[str, Any]) -> None:
    """内容变化才写盘，避免触发 mkdocs serve 的多余 rebuild。"""
    payload = json.dumps(cache, ensure_ascii=False, indent=2) + "\n"
    try:
        current = (
            PAGEVIEW_CACHE_PATH.read_text(encoding="utf-8")
            if PAGEVIEW_CACHE_PATH.exists()
            else ""
        )
        if current != payload:
            PAGEVIEW_CACHE_PATH.write_text(payload, encoding="utf-8")
    except OSError:
        pass


def bake_pageviews(data: dict[str, Any]) -> str:
    """
    把浏览量烘焙进图数据：note 节点加 pv 字段，stats 加 site_pv/site_uv/pv_fetched_at。
    返回一行日志说明数据来源。环境变量 KG_PAGEVIEW=off 可禁用网络抓取。
    """
    if os.environ.get("KG_PAGEVIEW", "").lower() == "off":
        return "浏览量: 已禁用 (KG_PAGEVIEW=off)"

    cache = _load_pageview_cache()
    now = int(time.time())
    fetched_at = int(cache.get("fetched_at", 0))
    notes: dict[str, int] = cache.get("notes", {})
    site: dict[str, int] = cache.get("site", {})

    if now - fetched_at >= PAGEVIEW_CACHE_TTL:
        # 缓存超期，重新抓取（8 线程并发，单请求延迟 ~1.5s，58 条共几秒）
        note_urls = {
            node["id"]: SITE_BASE_URL + node["url"]
            for node in data["nodes"]
            if node.get("type") == "note" and node.get("url")
        }
        with ThreadPoolExecutor(max_workers=PAGEVIEW_WORKERS) as pool:
            results = list(
                pool.map(lambda kv: (kv[0], _pageview_get(kv[1])), note_urls.items())
            )
        new_notes = {
            note_id: int(counts.get("page_pv") or 0)
            for note_id, counts in results
            if counts is not None
        }
        site_counts = _pageview_get(SITE_BASE_URL)

        if new_notes:
            notes = {**notes, **new_notes}  # 抓取失败的条目沿用上期缓存
        if site_counts:
            site = {
                "site_pv": int(site_counts.get("site_pv") or 0),
                "site_uv": int(site_counts.get("site_uv") or 0),
            }
        if new_notes or site_counts:
            fetched_at = now
            _write_pageview_cache({"fetched_at": fetched_at, "notes": notes, "site": site})
        source = f"已抓取 {len(new_notes)}/{len(note_urls)} 篇"
    else:
        source = "命中缓存"

    baked = 0
    for node in data["nodes"]:
        pv = notes.get(node["id"])
        if node.get("type") == "note" and pv is not None:
            node["pv"] = pv
            baked += 1
    if site:
        data["stats"]["site_pv"] = site.get("site_pv")
        data["stats"]["site_uv"] = site.get("site_uv")
    if fetched_at:
        data["stats"]["pv_fetched_at"] = fetched_at
    return f"浏览量: {source}, 烘焙 {baked} 篇"


def on_pre_build(config, **kwargs):
    """
    MkDocs hook: generate knowledge graph data before building.
    """
    nav_removed = remove_index_navigation()
    data = scan_notebooks()
    pv_status = bake_pageviews(data)
    write_graph_data(data)

    # 打印一行日志,方便用户确认数据已更新
    nav_status = ", 首页遗留导航已清理" if nav_removed else ""
    print(
        f"[knowledge-graph] 已生成 {data['stats']['nodes']} 个节点, "
        f"{data['stats']['links']} 条关系 -> {OUTPUT_PATH.relative_to(PROJECT_ROOT)}"
        f"{nav_status}; {pv_status}"
    )


# 支持直接运行测试: python hooks/generate_graph_data.py
if __name__ == "__main__":
    nav_removed = remove_index_navigation()
    data = scan_notebooks()
    started = time.time()
    pv_status = bake_pageviews(data)
    elapsed = time.time() - started
    write_graph_data(data)
    print(json.dumps(data["stats"], ensure_ascii=False, indent=2))
    print(f"legacy_index_navigation_removed={nav_removed}")
    print(f"pageview: {pv_status} ({elapsed:.1f}s)")
