import argparse
import shutil
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从 Zotero 存储目录检索并提取匹配关键词的 PDF 文献。"
    )
    parser.add_argument(
        "keyword",
        help="匹配 PDF 文件名的关键字（不区分大小写）。",
    )
    parser.add_argument(
        "--zotero-dir",
        type=Path,
        default=Path(r"C:\Users\10029\Zotero\storage"),
        help="Zotero 的本地存储路径。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("research_files/papers"),
        help="项目内部的目标存储路径。",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="如果目标文件已存在，是否覆盖。",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    keyword = args.keyword.lower()
    zotero_dir: Path = args.zotero_dir
    output_dir: Path = args.output_dir

    if not zotero_dir.exists():
        print(f"错误: 找不到 Zotero 存储目录: {zotero_dir}", file=sys.stderr)
        return 1

    # 确保存储目标目录存在
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"正在扫描 Zotero 存储目录: {zotero_dir}")
    print(f"搜索关键词: '{keyword}'\n")

    # 递归查找所有的 pdf 文件
    # Zotero storage 结构一般是 storage/<random_id>/xxx.pdf
    matches = []
    for pdf_path in zotero_dir.rglob("*.pdf"):
        if keyword in pdf_path.name.lower():
            matches.append(pdf_path)

    if not matches:
        print(f"未找到包含关键词 '{keyword}' 的 PDF 文件。")
        return 0

    print(f"找到 {len(matches)} 个匹配项:")
    success_count = 0
    
    for pdf_path in matches:
        target_path = output_dir / pdf_path.name
        print(f"- 发现: {pdf_path.name}")
        
        if target_path.exists() and not args.force:
            print(f"  -> [跳过] 文件已存在: {target_path}")
            continue

        try:
            shutil.copy2(pdf_path, target_path)
            print(f"  -> [成功] 已复制到: {target_path}")
            success_count += 1
        except Exception as e:
            print(f"  -> [失败] 复制 {pdf_path.name} 时出错: {e}", file=sys.stderr)

    print(f"\n提取完成: 共复制 {success_count} 个文件到 {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
