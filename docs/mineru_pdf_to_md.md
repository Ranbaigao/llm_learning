# MinerU PDF 批量转 Markdown

已新增脚本 `mineru_pdf_to_md.py`，用于把单个 PDF 或 `files/papers` 目录中的 PDF 提交到 MinerU 精准解析 API，并将解析得到的 `full.md` 保存到指定目录或默认目录 `files/papers_md`。

## 功能说明

- 支持传入单个 `.pdf` 文件或目录路径
- 默认扫描 `files/papers` 下所有 `.pdf` 文件（递归扫描）
- 使用 MinerU 精准解析本地文件上传接口：`/api/v4/file-urls/batch`
- 自动分批上传（每批最多 50 个文件）
- 自动轮询批量结果：`/api/v4/extract-results/batch/{batch_id}`
- 从结果 zip 中提取 `full.md`，输出为同名 `.md`
- 默认模型为 `vlm`

## 运行方式

推荐使用环境变量传入 Token：

### PowerShell

```powershell
$env:MINERU_API_TOKEN = "你的 MinerU Token"
python .\mineru_pdf_to_md.py
```

也可以直接通过参数传入：

```powershell
python .\mineru_pdf_to_md.py --token "你的 MinerU Token"
```

## 常用参数

- `--input`：输入文件或目录；兼容旧参数 `--input-dir`；也支持直接传位置参数
- `--output-dir`：输出目录，默认 `files/papers_md`
- `--model-version`：`pipeline` 或 `vlm`，默认 `vlm`
- `--language`：默认 `ch`
- `--page-ranges`：页码范围，例如 `1-10` 或 `2,4-6`
- `--ocr`：开启 OCR
- `--disable-table`：关闭表格识别
- `--disable-formula`：关闭公式识别
- `--force`：即使已有 Markdown 也重新解析
- `--keep-zips`：额外保存每个文件的结果 zip
- `--max-files`：只处理前 N 个文件，便于测试

示例：

```powershell
python .\mineru_pdf_to_md.py --max-files 2 --force
python .\mineru_pdf_to_md.py --model-version pipeline --ocr
python .\mineru_pdf_to_md.py .\files\papers\demo.pdf
python .\mineru_pdf_to_md.py --input .\files\papers\demo.pdf --output-dir .\files\single_md
```

## 输出目录

输入文件：

- `files/papers/a.pdf`
- `files/papers/sub/b.pdf`

或单文件输入：

- `files/papers/a.pdf`

默认输出：

- `files/papers_md/a.md`
- `files/papers_md/sub/b.md`

单文件默认输出：

- `files/papers_md/a.md`

单文件指定输出目录示例：

- 命令：`python .\mineru_pdf_to_md.py --input .\files\papers\a.pdf --output-dir .\out`
- 输出：`out/a.md`

## 注意事项

- 脚本使用的是 **精准解析 API**，适合本地 PDF 上传
- 单文件限制：不超过 200MB、200 页
- 单次申请上传链接上限：50 个文件，脚本会自动分批
- 如果返回失败，请优先检查 Token、文件大小/页数、网络连通性
- MinerU 返回的是 zip 结果包，脚本默认只提取其中的 `full.md`
