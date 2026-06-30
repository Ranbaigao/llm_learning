---
name: zotero-pdf-fetcher
description: "检索并在Zotero本地存储中查找特定的PDF文献，并将其复制到当前项目的 research_files/papers 目录下。"
---

# Zotero PDF Fetcher

## 简介
这个技能用于在 Zotero 的本地存储目录中，根据关键词（如论文标题、作者名）模糊检索 PDF 文件，并自动将匹配的 PDF 复制到本项目的 `research_files/papers` 目录下。当你需要将 Zotero 中刚刚添加或早已存在的论文导入到当前项目并准备后续转换为 Markdown 时，这个工具会非常有用。

## 默认路径配置
- **Zotero 存储目录**：`C:\Users\10029\Zotero\storage`
- **项目目标目录**：`research_files/papers`

## 使用说明
工具提供了一个 Python 脚本：`scripts/fetch_zotero_pdf.py`，你可以在任何具有 Python 环境的终端中运行它。不需要额外的第三方依赖（仅使用了 Python 标准库）。

### 命令行参数
- `keyword`：位置参数（必填）。用于匹配 PDF 文件名的关键字，不区分大小写。如果是包含空格的关键词，请使用双引号包裹。

### 运行范例

1. **检索标题中包含特定关键词的 PDF 并拷贝**：
```bash
python .\skills\zotero-pdf-fetcher\scripts\fetch_zotero_pdf.py "DeepSeek-V4"
```

2. **检索包含多个词的关键词**：
```bash
python .\skills\zotero-pdf-fetcher\scripts\fetch_zotero_pdf.py "Fast-in-Slow"
```

## AI Agent 使用准则
当你（AI）接到用户请求，比如：“从 Zotero 中把 DeepSeek V4 相关的论文拷贝到项目中”时，请执行以下步骤：
1. 从用户的请求中提取出一个相对唯一的核心关键词（如 `DeepSeek-V4`，而不是简单的 `Deep`，以免匹配过多不相干的论文）。
2. 在项目根目录下，使用 `run_command` 工具执行 `python .\skills\zotero-pdf-fetcher\scripts\fetch_zotero_pdf.py "你的关键词"` 命令。
3. 检查命令行的输出。如果成功拷贝，通知用户已经成功提取了该文献；如果没有匹配到，可以建议用户尝试更简短或更准确的关键词。
