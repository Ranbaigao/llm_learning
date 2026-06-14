```yaml
name: pdf2markdown
description: "使用 MinerU 精准解析 API 将单个 PDF 或目录中的 PDF 批量转换为 Markdown。适用于：整理论文、生成知识库素材、将本地 PDF 转成可读可检索的 md 文件；支持单文件、目录扫描、自定义输出目录、Windows 环境变量读取 Token、异步轮询与结果落盘。"
metadata:
	language: zh-CN
	owner: workspace
	entry: skills/pdf2markdown/scripts/mineru_pdf_to_md.py
	inputs:
		- 单个 PDF 文件路径
		- PDF 目录路径
	outputs:
		- Markdown 文件
		- 可选 zip 结果包
```

# PDF 转 Markdown Skill

使用 `skills/pdf2markdown/scripts/mineru_pdf_to_md.py`，通过 MinerU 精准解析 API 把本地 PDF 转成 Markdown。

适合这些场景：

- 想把论文 PDF 批量转成 Markdown，便于阅读、标注、摘要或入库
- 想临时处理某一个 PDF 文件，而不是整个目录
- 想把结果输出到指定目录，和原始 PDF 分开存放
- 已经有 MinerU Token，希望自动上传、轮询、下载并保存结果

---

## 快速参考

| 场景 | 建议操作 |
| --- | --- |
| 转换单个 PDF | 直接传入 PDF 路径 |
| 转换整个目录 | 传目录路径，或不传使用默认目录 |
| 输出到指定目录 | 加 `--output-dir` |
| 只想先试跑 | 加 `--max-files 1 --force` |
| 已存在 md 但要重跑 | 加 `--force` |
| 需要 OCR | 加 `--ocr` |
| 想保留 MinerU 原始结果包 | 加 `--keep-zips` |
| 提示缺少 Token | 检查 `MINERU_API_TOKEN` 或显式传 `--token` |
| 上传时报签名错误 | 重试；脚本已避免自动附加错误请求头 |

---

## 技能能力

### 输入支持

支持两类输入：

- 单个 PDF 文件路径
- 一个包含多个 PDF 的目录路径（递归扫描）

支持三种传参方式：

1. 位置参数
2. `--input`
3. 兼容旧参数 `--input-dir`

### 输出规则

- 默认输出目录：`files/papers_md`
- 如果输入是目录，则保留相对目录结构
- 如果输入是单个文件，则输出为 `输出目录/文件名.md`
- 如果 Markdown 引用了本地图片资源，脚本会自动提取同级 `images/` 目录
- 如启用 `--keep-zips`，同时保留 `输出目录/文件名.zip`

### Token 读取顺序

脚本按以下顺序获取 Token：

1. `--token`
2. 当前进程环境变量 `MINERU_API_TOKEN`
3. Windows 用户级环境变量
4. Windows 系统级环境变量

这解决了一个常见问题：Windows 下即使 VS Code 终端没有继承最新环境变量，脚本仍可回退读取注册表中的环境变量。

---

## 脚本位置

当前技能脚本位置：

- `skills/pdf2markdown/scripts/mineru_pdf_to_md.py`

推荐解释器示例：

```powershell
D:/env/miniconda/envs/notebook/python.exe
```

推荐调用形式：

```powershell
D:/env/miniconda/envs/notebook/python.exe .\skills\pdf2markdown\scripts\mineru_pdf_to_md.py
```

---

## 最常用命令

### 1. 使用默认目录批量转换

默认扫描 `files/papers`，输出到 `files/papers_md`：

```powershell
D:/env/miniconda/envs/notebook/python.exe .\skills\pdf2markdown\scripts\mineru_pdf_to_md.py
```

### 2. 转换单个 PDF

```powershell
D:/env/miniconda/envs/notebook/python.exe .\skills\pdf2markdown\scripts\mineru_pdf_to_md.py .\files\papers\demo.pdf
```

### 3. 转换单个 PDF 并指定输出目录

```powershell
D:/env/miniconda/envs/notebook/python.exe .\skills\pdf2markdown\scripts\mineru_pdf_to_md.py .\files\papers\demo.pdf --output-dir .\out
```

### 4. 指定输入目录

```powershell
D:/env/miniconda/envs/notebook/python.exe .\skills\pdf2markdown\scripts\mineru_pdf_to_md.py --input .\files\papers
```

### 5. 只试跑一个文件

```powershell
D:/env/miniconda/envs/notebook/python.exe .\skills\pdf2markdown\scripts\mineru_pdf_to_md.py --max-files 1 --force
```

### 6. 强制重跑

```powershell
D:/env/miniconda/envs/notebook/python.exe .\skills\pdf2markdown\scripts\mineru_pdf_to_md.py --force
```

### 7. 显式传入 Token

```powershell
D:/env/miniconda/envs/notebook/python.exe .\skills\pdf2markdown\scripts\mineru_pdf_to_md.py --token "你的_token"
```

### 8. 开启 OCR / 切换模型

```powershell
D:/env/miniconda/envs/notebook/python.exe .\skills\pdf2markdown\scripts\mineru_pdf_to_md.py --ocr
D:/env/miniconda/envs/notebook/python.exe .\skills\pdf2markdown\scripts\mineru_pdf_to_md.py --model-version pipeline
```

---

## 输入与输出示例

### 目录输入

输入：

- `files/papers/a.pdf`
- `files/papers/sub/b.pdf`

输出：

- `files/papers_md/a.md`
- `files/papers_md/sub/b.md`

### 单文件输入

命令：

```powershell
D:/env/miniconda/envs/notebook/python.exe .\skills\pdf2markdown\scripts\mineru_pdf_to_md.py .\files\papers\a.pdf --output-dir .\out
```

输出：

- `out/a.md`

---

## 工作流程

脚本内部执行顺序如下：

1. 解析输入路径（单文件 / 目录 / 默认目录）
2. 发现要处理的 PDF 文件列表
3. 跳过已存在的 `.md`（除非启用 `--force`）
4. 向 MinerU 申请批量上传链接（每批最多 50 个）
5. 上传本地 PDF 到 OSS 签名地址
6. 轮询 MinerU 批量任务状态
7. 下载结果 zip
8. 提取其中的 `full.md`
9. 保存到输出目录

---

## 参数速查

### 输入输出参数

- `input_path`：位置参数，单个 PDF 或目录路径
- `--input`：输入文件或目录
- `--input-dir`：兼容旧参数，等同于 `--input`
- `--output-dir`：输出目录，默认 `files/papers_md`

### 解析参数

- `--model-version {pipeline,vlm}`：模型版本，默认 `vlm`
- `--language`：语言，默认 `ch`
- `--page-ranges`：页码范围，例如 `1-10` 或 `2,4-6`
- `--ocr`：启用 OCR
- `--disable-table`：关闭表格识别
- `--disable-formula`：关闭公式识别

### 运行控制参数

- `--max-files`：最多处理前 N 个 PDF
- `--poll-interval`：轮询间隔秒数，默认 `5`
- `--timeout`：单批次超时时间，默认 `1800`
- `--force`：覆盖已存在结果
- `--keep-zips`：保留 zip 结果包

---

## 常见问题

### 1. 提示“缺少 Token”

排查顺序：

1. 确认变量名是否为 `MINERU_API_TOKEN`
2. 确认 Token 是否已过期
3. 当前终端是否继承了环境变量
4. 直接使用 `--token` 验证是否是环境变量问题

PowerShell 临时注入系统变量：

```powershell
$env:MINERU_API_TOKEN = [Environment]::GetEnvironmentVariable('MINERU_API_TOKEN', 'Machine')
```

### 2. 上传时报 `SignatureDoesNotMatch`

这是 OSS 预签名上传常见问题。当前脚本已避免 Python 标准库自动附加错误的 `Content-Type` 头。

如果仍发生，优先考虑：

- 上传链接已过期
- 网络代理改写了请求
- 服务端临时返回异常签名

建议：直接重试一次；若多次复现，再检查网络代理环境。

### 3. 解析失败或部分文件失败

可能原因：

- PDF 超过 MinerU 限制（200MB / 200 页）
- PDF 文件损坏
- Token 过期
- 服务繁忙

建议：

- 先单文件测试：`--max-files 1 --force`
- 再尝试 `--model-version pipeline`
- 必要时拆分超大 PDF

### 4. 没有生成新的 Markdown

默认会跳过已存在的 `.md`。如果你确认要重新解析，请使用：

```powershell
D:/env/miniconda/envs/notebook/python.exe .\skills\pdf2markdown\scripts\mineru_pdf_to_md.py --force
```

---

## 推荐执行策略

### 最稳妥的方式

1. 先跑一个文件验证链路
2. 再跑全量目录
3. 如需保留原始结果，再加 `--keep-zips`

示例：

```powershell
D:/env/miniconda/envs/notebook/python.exe .\skills\pdf2markdown\scripts\mineru_pdf_to_md.py --max-files 1 --force
D:/env/miniconda/envs/notebook/python.exe .\skills\pdf2markdown\scripts\mineru_pdf_to_md.py --force
```

### 适合 Agent 的调用策略

当任务目标是“把某个 PDF 或某个论文目录转成 Markdown”时，应优先使用这个 skill，而不是临时手写请求脚本。

推荐执行原则：

- 优先复用现成脚本，不重复造轮子
- 单文件优先用位置参数，目录优先用 `--input`
- 首次运行优先加 `--max-files 1`
- 出错时先看 Token、输入路径、输出目录、API 限制
- 只有在脚本能力不够时，才继续扩展脚本本身

---

## 何时使用这个 Skill

当用户提出下面这些需求时，应直接考虑本技能：

- “把这个 PDF 转成 Markdown”
- “把 papers 目录里的论文都转成 md”
- “给我一个批量 PDF 转 md 的脚本”
- “输出到另一个目录，不要覆盖现有文件”
- “我想把论文转成后续能做 RAG 的文本格式”

---

## 相关文件

- 技能说明：`skills/pdf2markdown/SKILL.md`
- 转换脚本：`skills/pdf2markdown/scripts/mineru_pdf_to_md.py`

## 一句话总结

这个 skill 用来稳定地把单个 PDF 或 PDF 目录转换成 Markdown；默认适合论文批处理，也支持单文件和自定义输出目录。
