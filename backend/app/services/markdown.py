"""Markdown → HTML 渲染服务。

- markdown-it-py + mdit-py-plugins（dollarmath/amsmath/tasklists/anchors/footnote/deflist）
- 代码块用 Pygments 高亮，css 类前缀 highlight（前端需引入对应 pygments css）
- 数学公式保留为 span/div.math 结构，交给前端 MathJax
- 相对路径资源链接改写为 /api/assets/...，站内笔记链接改写为 /notes/{slug}
- 保留原始 HTML（笔记里有 iframe/自定义 HTML 片段）
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlsplit

import frontmatter
from markdown_it import MarkdownIt
from mdit_py_plugins.amsmath import amsmath_plugin
from mdit_py_plugins.anchors import anchors_plugin
from mdit_py_plugins.deflist import deflist_plugin
from mdit_py_plugins.dollarmath import dollarmath_plugin
from mdit_py_plugins.footnote import footnote_plugin
from mdit_py_plugins.tasklists import tasklists_plugin
from pygments import highlight as pygments_highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import TextLexer, get_lexer_by_name
from pygments.util import ClassNotFound

from app.core.config import settings

NOTE_SUFFIXES = {".md", ".ipynb"}


def _highlight(code: str, lang: str, attrs: str) -> str:
    """markdown-it highlight 回调：返回以 <pre 开头的完整高亮块。"""
    try:
        lexer = get_lexer_by_name(lang) if lang else TextLexer()
    except ClassNotFound:
        lexer = TextLexer()
    # nowrap=True 只返回带 span 的代码行，外层 pre/code 自己包，
    # css 类挂在 <pre class="highlight"> 上，兼容标准 pygments css（.highlight .k 等后代选择器）
    inner = pygments_highlight(code, lexer, HtmlFormatter(nowrap=True))
    code_class = f' class="language-{lang}"' if lang else ""
    return f'<pre class="highlight"><code{code_class}>{inner}</code></pre>\n'


def _build_markdown() -> MarkdownIt:
    md = MarkdownIt(
        "default",
        {"highlight": _highlight, "breaks": False, "html": True, "linkify": False},
    )
    return (
        md.use(dollarmath_plugin)
        .use(amsmath_plugin)
        .use(tasklists_plugin)
        .use(anchors_plugin)
        .use(footnote_plugin)
        .use(deflist_plugin)
    )


_md = _build_markdown()

_ATTR_RE = re.compile(r'(\b(?:src|href)\s*=\s*")([^"]*)(")', re.IGNORECASE)

# 旧站 MkDocs pymdownx.snippets 语法：把 .assets 下的交互 HTML 整段内联进文章。
# 新站统一改为「弹窗触发链接」，点击后由前端 HtmlModal 组件以 iframe 打开。
# 只匹配独占一行的指令，避免误伤代码块/正文中的同类文本。
_SNIPPET_RE = re.compile(r'^[ \t]*--8<--\s*["\']([^"\']+)["\'][ \t]*$', re.MULTILINE)


def _snippet_to_modal_link(match: re.Match[str], source_path: Path) -> str:
    """把 --8<-- "path.html" 指令替换为弹窗链接；目标找不到时保留原文方便排查。"""
    ref = match.group(1).strip()
    # 旧站 snippets base_path 是内容根目录，再回退到文章所在目录
    for base in (settings.content_dir, source_path.parent):
        candidate = (base / ref).resolve()
        try:
            rel_assets = candidate.relative_to(settings.assets_dir)
        except ValueError:
            continue
        if candidate.is_file():
            url = "/api/assets/" + quote(rel_assets.as_posix(), safe="/")
            title = candidate.stem.replace("_", " ")
            return (
                f'<a class="html-modal-link" href="{url}" '
                f'data-title="{title}">🔍查看交互图：{title}</a>'
            )
    return match.group(0)


def _rewrite_url(url: str, source_path: Path) -> str:
    """把相对路径链接改写为新站绝对路径。

    - 指向 content/.assets/ 下文件 → /api/assets/{相对路径}
    - 指向其他 .md/.ipynb 笔记 → /notes/{目标 slug}
    - http(s)/#/mailto/绝对路径/不存在或越界的目标 → 原样返回
    """
    url = url.strip()
    if not url or url.startswith("#") or url.startswith("/"):
        return url
    parts = urlsplit(url)
    if parts.scheme or parts.netloc:
        return url
    rel_target = unquote(parts.path)
    if not rel_target:
        return url

    candidate = (source_path.parent / rel_target).resolve()
    content_dir = settings.content_dir

    tail = ""
    if parts.query:
        tail += "?" + parts.query
    if parts.fragment:
        tail += "#" + parts.fragment

    # 1) 静态资产（图片/字体/HTML 片段等）
    try:
        rel_assets = candidate.relative_to(settings.assets_dir)
    except ValueError:
        rel_assets = None
    if rel_assets is not None and candidate.is_file():
        return "/api/assets/" + quote(rel_assets.as_posix(), safe="/") + tail

    # 2) 站内笔记链接
    try:
        rel_content = candidate.relative_to(content_dir)
    except ValueError:
        return url
    if candidate.suffix.lower() in NOTE_SUFFIXES and candidate.is_file():
        slug = rel_content.as_posix()[: -len(candidate.suffix)]
        return "/notes/" + quote(slug, safe="/") + tail

    return url


def rewrite_links(html: str, source_path: Path) -> str:
    """重写渲染后 HTML 中所有 src/href 相对链接（含原始 HTML 片段里的）。"""

    def repl(match: re.Match[str]) -> str:
        prefix, url, suffix = match.groups()
        return prefix + _rewrite_url(url, source_path) + suffix

    return _ATTR_RE.sub(repl, html)


def render_markdown(raw: str, source_path: Path) -> tuple[str, dict[str, Any]]:
    """渲染 Markdown 源码为 HTML，返回 (html, frontmatter_meta)。"""
    try:
        post = frontmatter.loads(raw)
        body = post.content
        meta = dict(post.metadata)
    except Exception:
        body = raw
        meta = {}
    # snippets 指令 → 弹窗链接（在 markdown 渲染前替换，生成的 <a> 作为行内 HTML 原样保留）
    body = _SNIPPET_RE.sub(lambda m: _snippet_to_modal_link(m, source_path), body)
    html = _md.render(body)
    html = rewrite_links(html, source_path)
    return html, meta


def render_article(path: Path) -> tuple[str, dict[str, Any]]:
    """从磁盘读取 .md 并渲染。"""
    try:
        raw = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raw = path.read_text(encoding="utf-8", errors="ignore")
    return render_markdown(raw, path)
