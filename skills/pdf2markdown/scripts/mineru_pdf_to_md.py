from __future__ import annotations

import argparse
import hashlib
import http.client
import io
import json
import os
import posixpath
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if sys.platform == "win32":
    import winreg

BASE_URL = "https://mineru.net/api/v4"
UPLOAD_BATCH_LIMIT = 50
DEFAULT_INPUT_DIR = Path("research_files/papers")
DEFAULT_OUTPUT_DIR = Path("research_files/papers_md")
TERMINAL_STATES = {"done", "failed"}


@dataclass
class PdfJob:
    source_path: Path
    relative_path: Path
    output_path: Path
    data_id: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将单个 PDF 或目录中的 PDF 通过 MinerU 精准解析 API 转成 Markdown。"
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        type=Path,
        default=None,
        help="可选位置参数：单个 PDF 文件路径或目录路径。",
    )
    parser.add_argument(
        "--input",
        "--input-dir",
        dest="input_path_option",
        type=Path,
        default=None,
        help="待解析的单个 PDF 文件或目录，默认: files/papers",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Markdown 输出目录，默认: files/papers_md",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="MinerU API Token；未提供时尝试读取环境变量 MINERU_API_TOKEN。",
    )
    parser.add_argument(
        "--model-version",
        choices=["pipeline", "vlm"],
        default="vlm",
        help="精准解析模型，默认 vlm。",
    )
    parser.add_argument(
        "--language",
        default="ch",
        help="文档语言，默认 ch。",
    )
    parser.add_argument(
        "--page-ranges",
        default=None,
        help='页码范围，例如 "1-10" 或 "2,4-6"。会应用到本次所有 PDF。',
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=5.0,
        help="轮询间隔秒数，默认 5。",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=1800.0,
        help="单个批次最大等待秒数，默认 1800。",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="最多处理多少个 PDF，默认处理全部。",
    )
    parser.add_argument(
        "--ocr",
        action="store_true",
        help="启用 OCR。",
    )
    parser.add_argument(
        "--disable-table",
        action="store_true",
        help="关闭表格识别。",
    )
    parser.add_argument(
        "--disable-formula",
        action="store_true",
        help="关闭公式识别。",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="即使 Markdown 已存在也重新解析并覆盖。",
    )
    parser.add_argument(
        "--keep-zips",
        action="store_true",
        help="额外保留 MinerU 返回的 zip 结果包。",
    )
    return parser.parse_args()


def get_token(cli_token: str | None) -> str:
    token = cli_token or os.environ.get("MINERU_API_TOKEN") or read_windows_env_var("MINERU_API_TOKEN")
    if not token:
        raise SystemExit("缺少 Token：请通过 --token 或环境变量 MINERU_API_TOKEN 提供。")
    return token


def read_windows_env_var(name: str) -> str | None:
    if sys.platform != "win32":
        return None

    registry_paths = [
        (winreg.HKEY_CURRENT_USER, r"Environment"),
        (winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"),
    ]
    for hive, subkey in registry_paths:
        try:
            with winreg.OpenKey(hive, subkey) as key:
                value, _ = winreg.QueryValueEx(key, name)
        except FileNotFoundError:
            continue
        except OSError:
            continue

        if isinstance(value, str) and value:
            return value
    return None


def ensure_success(payload: dict[str, Any], context: str) -> dict[str, Any]:
    if payload.get("code") != 0:
        raise RuntimeError(f"{context}失败: {payload.get('msg', 'unknown error')} | payload={payload}")
    data = payload.get("data")
    if not isinstance(data, dict):
        raise RuntimeError(f"{context}失败: 响应缺少 data 字段")
    return data


def http_json(method: str, url: str, *, headers: dict[str, str] | None = None, body: dict[str, Any] | None = None) -> dict[str, Any]:
    request_headers = {"Accept": "application/json"}
    if headers:
        request_headers.update(headers)

    data_bytes = None
    if body is not None:
        request_headers["Content-Type"] = "application/json"
        data_bytes = json.dumps(body, ensure_ascii=False).encode("utf-8")

    request = urllib.request.Request(url=url, data=data_bytes, headers=request_headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            response_text = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} for {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"请求失败 {url}: {exc}") from exc

    try:
        return json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"响应不是合法 JSON: {response_text[:500]}") from exc


def http_put_file(url: str, file_path: Path) -> None:
    parsed = urllib.parse.urlsplit(url)
    connection_class = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    request_path = parsed.path
    if parsed.query:
        request_path = f"{request_path}?{parsed.query}"

    file_size = file_path.stat().st_size
    connection = connection_class(parsed.netloc, timeout=300)
    try:
        connection.putrequest("PUT", request_path)
        connection.putheader("Content-Length", str(file_size))
        connection.endheaders()

        with file_path.open("rb") as file_handle:
            while True:
                chunk = file_handle.read(1024 * 1024)
                if not chunk:
                    break
                connection.send(chunk)

        response = connection.getresponse()
        status = response.status
        detail = response.read().decode("utf-8", errors="replace")
    except OSError as exc:
        raise RuntimeError(f"上传失败 {file_path}: {exc}") from exc
    finally:
        connection.close()

    if status not in (200, 201):
        raise RuntimeError(f"上传失败 {file_path}: HTTP {status} {detail}")


def download_bytes(url: str) -> bytes:
    request = urllib.request.Request(url=url, headers={"Accept": "*/*"}, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"下载失败 {url}: HTTP {exc.code} {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"下载失败 {url}: {exc}") from exc


def resolve_input_path(args: argparse.Namespace) -> Path:
    raw_input_path = args.input_path_option or args.input_path or DEFAULT_INPUT_DIR
    return raw_input_path.resolve()


def discover_jobs(input_path: Path, output_dir: Path, force: bool, max_files: int | None) -> list[PdfJob]:
    if not input_path.exists():
        raise SystemExit(f"输入路径不存在: {input_path}")

    if input_path.is_file():
        if input_path.suffix.lower() != ".pdf":
            raise SystemExit(f"输入文件不是 PDF: {input_path}")
        pdf_paths = [input_path]
        input_root = input_path.parent
    else:
        pdf_paths = sorted(path for path in input_path.rglob("*.pdf") if path.is_file())
        input_root = input_path

    if max_files is not None:
        pdf_paths = pdf_paths[:max_files]

    jobs: list[PdfJob] = []
    for pdf_path in pdf_paths:
        relative_path = Path(pdf_path.name) if input_path.is_file() else pdf_path.relative_to(input_root)
        output_path = output_dir / relative_path.with_suffix(".md")
        if output_path.exists() and not force:
            print(f"[skip] 已存在: {output_path}")
            continue
        jobs.append(
            PdfJob(
                source_path=pdf_path,
                relative_path=relative_path,
                output_path=output_path,
                data_id=build_data_id(relative_path),
            )
        )
    return jobs


def build_data_id(relative_path: Path) -> str:
    raw = relative_path.as_posix()
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    normalized = re.sub(r"[^A-Za-z0-9_.-]", "_", raw)
    normalized = normalized.replace("/", "__")
    base = normalized[:100]
    return f"{base}_{digest}"


def chunked(items: list[PdfJob], size: int) -> list[list[PdfJob]]:
    return [items[index:index + size] for index in range(0, len(items), size)]


def build_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def request_upload_batch(token: str, jobs: list[PdfJob], args: argparse.Namespace) -> tuple[str, list[str]]:
    files_payload: list[dict[str, Any]] = []
    for job in jobs:
        item: dict[str, Any] = {
            "name": job.source_path.name,
            "data_id": job.data_id,
            "is_ocr": args.ocr,
        }
        if args.page_ranges:
            item["page_ranges"] = args.page_ranges
        files_payload.append(item)

    payload: dict[str, Any] = {
        "files": files_payload,
        "model_version": args.model_version,
        "language": args.language,
        "enable_table": not args.disable_table,
        "enable_formula": not args.disable_formula,
    }
    data = ensure_success(
        http_json(
            "POST",
            f"{BASE_URL}/file-urls/batch",
            headers=build_headers(token),
            body=payload,
        ),
        "申请上传链接",
    )

    batch_id = data.get("batch_id")
    file_urls = data.get("file_urls")
    if not isinstance(batch_id, str) or not isinstance(file_urls, list):
        raise RuntimeError(f"申请上传链接失败: 返回格式异常 {data}")
    return batch_id, [str(url) for url in file_urls]


def upload_files(jobs: list[PdfJob], file_urls: list[str]) -> None:
    if len(jobs) != len(file_urls):
        raise RuntimeError("上传链接数量与文件数量不一致")
    for job, file_url in zip(jobs, file_urls):
        print(f"[upload] {job.relative_path}")
        http_put_file(file_url, job.source_path)


def poll_batch_results(token: str, batch_id: str, timeout: float, interval: float) -> list[dict[str, Any]]:
    started = time.time()
    while True:
        data = ensure_success(
            http_json(
                "GET",
                f"{BASE_URL}/extract-results/batch/{batch_id}",
                headers=build_headers(token),
            ),
            "查询批量任务",
        )
        results = data.get("extract_result")
        if not isinstance(results, list):
            raise RuntimeError(f"查询批量任务失败: extract_result 缺失 {data}")

        states = [str(item.get("state", "unknown")) for item in results]
        summary = ", ".join(states)
        elapsed = int(time.time() - started)
        print(f"[poll {batch_id}] {elapsed}s -> {summary}")

        if results and all(state in TERMINAL_STATES for state in states):
            return [dict(item) for item in results]

        if time.time() - started > timeout:
            raise TimeoutError(f"批次 {batch_id} 超时，最后状态: {summary}")
        time.sleep(interval)


def find_full_markdown_path(archive: zipfile.ZipFile) -> str:
    candidates = [name for name in archive.namelist() if name.endswith("full.md")]
    if not candidates:
        raise RuntimeError("结果 zip 中未找到 full.md")
    return candidates[0]


def collect_local_asset_paths(markdown_text: str) -> set[str]:
    asset_paths: set[str] = set()
    markdown_links = re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", markdown_text)
    html_images = re.findall(r'<img[^>]+src=["\']([^"\']+)["\']', markdown_text, flags=re.IGNORECASE)

    for raw_path in markdown_links + html_images:
        cleaned = raw_path.strip().strip("<>")
        cleaned = cleaned.split()[0]
        if not cleaned or cleaned.startswith("#"):
            continue
        parsed = urllib.parse.urlparse(cleaned)
        if parsed.scheme or cleaned.startswith("//"):
            continue
        normalized = posixpath.normpath(cleaned)
        if normalized.startswith("../") or normalized == "..":
            continue
        asset_paths.add(normalized)

    return asset_paths


def is_safe_relative_path(relative_path: str) -> bool:
    normalized = posixpath.normpath(relative_path)
    return not (normalized.startswith("../") or normalized == ".." or posixpath.isabs(normalized))


def extract_markdown_and_assets(zip_bytes: bytes, output_path: Path) -> str:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        full_markdown_path = find_full_markdown_path(archive)
        with archive.open(full_markdown_path) as markdown_file:
            markdown_text = markdown_file.read().decode("utf-8", errors="replace")

        referenced_assets = collect_local_asset_paths(markdown_text)
        archive_root = posixpath.dirname(full_markdown_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown_text, encoding="utf-8")

        extracted_assets = 0
        for member in archive.infolist():
            if member.is_dir():
                continue

            member_name = member.filename
            if member_name == full_markdown_path:
                continue

            if archive_root:
                prefix = f"{archive_root}/"
                if not member_name.startswith(prefix):
                    continue
                relative_name = member_name[len(prefix):]
            else:
                relative_name = member_name

            relative_name = posixpath.normpath(relative_name)
            if not is_safe_relative_path(relative_name):
                continue

            if relative_name not in referenced_assets:
                continue

            target_path = output_path.parent / Path(relative_name)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source_file:
                target_path.write_bytes(source_file.read())
            extracted_assets += 1

        if referenced_assets and extracted_assets == 0:
            print(f"[warn] {output_path.name} 引用了本地资源，但未在 zip 中提取到对应文件")

        return markdown_text


def maybe_save_zip(zip_bytes: bytes, output_path: Path) -> None:
    zip_path = output_path.with_suffix(".zip")
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    zip_path.write_bytes(zip_bytes)


def save_markdown(markdown_text: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown_text, encoding="utf-8")
    print(f"[saved] {output_path}")


def process_batch(token: str, jobs: list[PdfJob], args: argparse.Namespace) -> tuple[int, int]:
    batch_id, file_urls = request_upload_batch(token, jobs, args)
    print(f"[batch] 已创建 {batch_id}，准备上传 {len(jobs)} 个文件")
    upload_files(jobs, file_urls)
    print(f"[batch] 上传完成，开始轮询 {batch_id}")

    results = poll_batch_results(token, batch_id, timeout=args.timeout, interval=args.poll_interval)
    jobs_by_data_id = {job.data_id: job for job in jobs}

    success_count = 0
    failed_count = 0
    for result in results:
        data_id = str(result.get("data_id", ""))
        state = str(result.get("state", "unknown"))
        job = jobs_by_data_id.get(data_id)
        if job is None:
            print(f"[warn] 无法匹配返回结果: {result}")
            failed_count += 1
            continue

        if state != "done":
            failed_count += 1
            print(f"[failed] {job.relative_path} -> {result.get('err_msg', '未知错误')}")
            continue

        full_zip_url = result.get("full_zip_url")
        if not isinstance(full_zip_url, str) or not full_zip_url:
            failed_count += 1
            print(f"[failed] {job.relative_path} -> 缺少 full_zip_url")
            continue

        zip_bytes = download_bytes(full_zip_url)
        extract_markdown_and_assets(zip_bytes, job.output_path)
        print(f"[saved] {job.output_path}")
        if args.keep_zips:
            maybe_save_zip(zip_bytes, job.output_path)
        success_count += 1

    return success_count, failed_count


def main() -> int:
    args = parse_args()
    token = get_token(args.token)
    input_path = resolve_input_path(args)
    output_dir = args.output_dir.resolve()

    jobs = discover_jobs(input_path, output_dir, force=args.force, max_files=args.max_files)
    if not jobs:
        print("没有需要处理的 PDF。")
        return 0

    print(f"待处理 PDF 数量: {len(jobs)}")
    total_success = 0
    total_failed = 0

    for batch_jobs in chunked(jobs, UPLOAD_BATCH_LIMIT):
        try:
            success_count, failed_count = process_batch(token, batch_jobs, args)
            total_success += success_count
            total_failed += failed_count
        except Exception as exc:
            total_failed += len(batch_jobs)
            print(f"[batch-error] {exc}", file=sys.stderr)

    print(f"完成：成功 {total_success}，失败 {total_failed}，输出目录: {output_dir}")
    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
