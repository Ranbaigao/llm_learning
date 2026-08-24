"""从自动导航中隐藏 blog 入口（侧边栏只保留知识库笔记）。

material blog 插件会在 on_files（priority -50）强制把博客入口标记为
INCLUDED，因此本钩子必须以更低的优先级（-100，即更晚执行）运行。
插件源码中已明确支持入口不在导航的情形（plugin.py:186-190）。
"""
from mkdocs.plugins import event_priority
from mkdocs.structure.files import InclusionLevel


@event_priority(-100)
def on_files(files, config):
    f = files.get_file_from_path('blog/index.md')
    if f is not None:
        f.inclusion = InclusionLevel.NOT_IN_NAV
