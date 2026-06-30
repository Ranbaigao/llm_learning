这是一个用于自我学习的LLM/VLM学习仓库

他是一个AI与我共创的仓库，需要AI与我一同维护仓库

AI的行动准则参考AGENT.md


项目启动
```shell
mkdocs serve -a 127.0.0.1:6033
```

## 项目技能 (Skills) & 工具

本项目内嵌了一些自定义技能供 AI Agent 使用，以自动化日常工作流：

- **pdf2markdown**: 自动将 `research_files/papers` 目录下的 PDF 论文通过 MinerU 转换为 Markdown 格式，并存放到 `research_files/papers_md` 目录。
- **zotero-pdf-fetcher**: 支持通过关键词快速从本地 Zotero 的库中检索并提取特定的 PDF 到本项目的文献目录中，实现从 Zotero 到项目的自动化同步。

你可以直接对 AI 发送请求，例如：
> "帮我从 Zotero 中把 DeepSeek V4 相关的论文拷贝到项目中，然后再把它转成 Markdown！"