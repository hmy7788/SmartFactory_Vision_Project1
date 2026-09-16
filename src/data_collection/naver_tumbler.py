"""Resumable Naver Shopping image collection and reproducible review/export.

The collector uses Naver's documented Shopping Search API.  It never stores API
credentials, guesses larger image URLs, labels shapes, or treats downloaded
candidates as accepted data before review.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import os
import re
import shutil
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen

from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[2]
API_URL = "https://naverapihub.apigw.ntruss.com/search/v1/image"
DEFAULT_OUTPUT = ROOT / "data" / "naver_tumbler"
DEFAULT_QUERIES = (
    "텀블러", "보온보냉 텀블러", "일자 텀블러", "슬림 텀블러", "머그 텀블러",
    "캠핑 머그", "차량용 텀블러", "대용량 텀블러", "빨대 텀블러", "스테인리스 텀블러",
)
REVIEW_FIELDS = (
    "candidate_id", "filename", "product_id", "group", "query", "title",
    "product_url", "image_url", "width", "height", "near_duplicate_of", "decision", "crop_x",
    "crop_y", "crop_width", "crop_height", "reason", "label", "sha256", "dhash", "collected_at",
)
MANIFEST_FIELDS = REVIEW_FIELDS + ("accepted_filename",)
LOG = logging.getLogger("naver-tumbler")


class CollectionStopped(RuntimeError):
    """Authentication/access restriction or unsafe continuation condition."""


@dataclass(frozen=True)
class Candidate:
    product_id: str
    query: str
    title: str
    product_url: str
    image_url: str
    thumbnail_url: str = ""

    @property
    def candidate_id(self) -> str:
        return hashlib.sha256(f"{self.product_id}|{self.image_url}".encode()).hexdigest()[:24]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean_title(value: str) -> str:
    return re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", "", value))).strip()


def validate_image_url(value: str) -> str:
    parts = urlsplit(value)
    host = (parts.hostname or "").lower()
    if parts.scheme not in ("http", "https") or parts.username or parts.password:
        raise ValueError(f"허용되지 않은 이미지 주소: {value}")
    if not host or host in ("localhost", "127.0.0.1", "::1") or host.startswith(("10.", "192.168.", "172.16.")):
        raise ValueError(f"허용되지 않은 호스트: {value}")
    return value


def parse_api_items(payload: dict, query: str) -> list[Candidate]:
    found: dict[str, Candidate] = {}
    for item in payload.get("items", []):
        link = str(item.get("link", "")).strip()
        thumbnail = str(item.get("thumbnail", "")).strip()
        image = str(item.get("image", "")).strip()
        product_id = str(item.get("productId", "")).strip()
        product_url = link or str(item.get("link", "")).strip()
        title = clean_title(str(item.get("title", "")))

        image_url = link or image or thumbnail
        if not image_url:
            continue

        if not product_id:
            product_id = f"img_{hashlib.sha256(image_url.encode()).hexdigest()[:16]}"

        try:
            validate_image_url(image_url)
        except ValueError:
            if thumbnail:
                try:
                    validate_image_url(thumbnail)
                    image_url = thumbnail
                except ValueError:
                    continue
            else:
                continue

        candidate = Candidate(
            product_id=product_id,
            query=query,
            title=title,
            product_url=product_url or image_url,
            image_url=image_url,
            thumbnail_url=thumbnail,
        )
        found.setdefault(candidate.candidate_id, candidate)
    return list(found.values())


def api_search(query: str, start: int, client_id: str, client_secret: str,
               timeout: int = 30, retries: int = 2) -> dict:
    params = urlencode({"query": query, "display": 100, "start": start, "sort": "sim", "filter": "all"})
    request = Request(f"{API_URL}?{params}", headers={
        "X-NCP-APIGW-API-KEY-ID": client_id,
        "X-NCP-APIGW-API-KEY": client_secret,
        "User-Agent": "TumblerDatasetCollector/1.0",
    })
    for attempt in range(retries + 1):
        try:
            with urlopen(request, timeout=timeout) as response:
                return json.loads(response.read(2_000_000).decode("utf-8"))
        except HTTPError as error:
            if error.code in (401, 403, 429):
                detail = ""
                try:
                    detail = error.read().decode("utf-8", "replace").strip()
                except Exception:
                    pass
                msg = f"네이버 API 오류 HTTP {error.code}"
                if detail:
                    msg += f": {detail}"
                raise CollectionStopped(msg) from error
            if error.code < 500 or attempt >= retries:
                raise
        except (URLError, TimeoutError):
            if attempt >= retries:
                raise
        time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def fetch_image(url: str, timeout: int = 30, retries: int = 2) -> bytes:
    validate_image_url(url)
    request = Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
    })
    for attempt in range(retries + 1):
        try:
            with urlopen(request, timeout=timeout) as response:
                content_type = response.headers.get("Content-Type", "").lower()
                if not content_type.startswith("image/"):
                    raise ValueError("이미지가 아닌 응답입니다.")
                length = int(response.headers.get("Content-Length", "0"))
                if length > 20 * 1024 * 1024:
                    raise ValueError("이미지가 20MB를 초과합니다.")
                body = response.read(20 * 1024 * 1024 + 1)
                if len(body) > 20 * 1024 * 1024:
                    raise ValueError("이미지가 20MB를 초과합니다.")
                return body
        except HTTPError as error:
            if error.code in (401, 403, 429):
                raise CollectionStopped(f"이미지 서버 접근 제한 HTTP {error.code}") from error
            if error.code < 500 or attempt >= retries:
                raise
        except (URLError, TimeoutError):
            if attempt >= retries:
                raise
        time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def normalized_image(body: bytes) -> tuple[Image.Image, str]:
    if len(body) > 20 * 1024 * 1024:
        raise ValueError("이미지가 20MB를 초과합니다.")
    with Image.open(io.BytesIO(body)) as source:
        source.verify()
    with Image.open(io.BytesIO(body)) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    digest = hashlib.sha256(f"{image.width}x{image.height}:".encode() + image.tobytes()).hexdigest()
    return image, digest


def difference_hash(image: Image.Image) -> str:
    gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(gray.get_flattened_data())
    bits = [pixels[row * 9 + col] > pixels[row * 9 + col + 1]
            for row in range(8) for col in range(8)]
    return f"{sum(int(bit) << index for index, bit in enumerate(bits)):016x}"


def hamming(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def write_csv(path: Path, fields, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    with temp.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


class NaverStore:
    def __init__(self, output: Path):
        self.output = output.resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.output / "crawl.sqlite3", check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS candidates (
                candidate_id TEXT PRIMARY KEY, product_id TEXT, query TEXT, title TEXT,
                product_url TEXT, image_url TEXT, status TEXT DEFAULT 'pending',
                filename TEXT, sha256 TEXT, dhash TEXT, width INTEGER, height INTEGER,
                near_duplicate_of TEXT, error TEXT, collected_at TEXT
            );
            CREATE TABLE IF NOT EXISTS searches (
                query TEXT, start INTEGER, item_count INTEGER, complete INTEGER,
                PRIMARY KEY(query,start)
            );
            CREATE INDEX IF NOT EXISTS idx_candidates_sha ON candidates(sha256);
            CREATE INDEX IF NOT EXISTS idx_candidates_product ON candidates(product_id);
        """)
        columns = {row[1] for row in self.db.execute("PRAGMA table_info(candidates)")}
        if "near_duplicate_of" not in columns:
            self.db.execute("ALTER TABLE candidates ADD COLUMN near_duplicate_of TEXT")
        missing = []
        for row in self.db.execute(
                "SELECT candidate_id,query,filename FROM candidates WHERE status='downloaded'"):
            if not row["filename"] or not (self.output / row["filename"]).is_file():
                missing.append((row["candidate_id"], row["query"]))
        for candidate_id, query in missing:
            self.db.execute("UPDATE candidates SET status='failed',error=? WHERE candidate_id=?",
                            ("체크포인트의 원본 파일이 없습니다.", candidate_id))
            self.db.execute("UPDATE searches SET complete=0 WHERE query=?", (query,))
        self.db.commit()

    def close(self) -> None:
        self.db.close()

    def discover(self, candidates: list[Candidate]) -> None:
        with self.db:
            for item in candidates:
                self.db.execute("""INSERT INTO candidates
                    (candidate_id,product_id,query,title,product_url,image_url)
                    VALUES (?,?,?,?,?,?) ON CONFLICT(candidate_id) DO UPDATE SET
                    title=excluded.title,product_url=excluded.product_url""",
                    (item.candidate_id, item.product_id, item.query, item.title,
                     item.product_url, item.image_url))

    def known_product(self, product_id: str) -> bool:
        return bool(self.db.execute(
            "SELECT 1 FROM candidates WHERE product_id=? AND status IN ('downloaded','duplicate')",
            (product_id,)).fetchone())

    def candidate_count(self) -> int:
        return self.db.execute(
            "SELECT COUNT(*) FROM candidates WHERE status='downloaded'").fetchone()[0]

    def search_done(self, query: str, start: int) -> bool:
        row = self.db.execute("SELECT complete FROM searches WHERE query=? AND start=?",
                              (query, start)).fetchone()
        return bool(row and row[0])

    def mark_search(self, query: str, start: int, count: int, complete: bool) -> None:
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO searches VALUES (?,?,?,?)",
                            (query, start, count, int(complete)))

    def candidates_resolved(self, candidates: list[Candidate]) -> bool:
        for item in candidates:
            row = self.db.execute("SELECT status FROM candidates WHERE candidate_id=?",
                                  (item.candidate_id,)).fetchone()
            if not row or row[0] not in ("downloaded", "duplicate"):
                return False
        return True

    def save(self, item: Candidate, body: bytes, min_short: int, min_long: int) -> str:
        image, digest = normalized_image(body)
        short, long = sorted(image.size)
        if short < min_short or long < min_long:
            raise ValueError(f"최소 크기 {min_short}x{min_long}px 미달: {image.width}x{image.height}")
        dhash = difference_hash(image)
        exact = self.db.execute(
            "SELECT filename FROM candidates WHERE sha256=? AND status='downloaded'", (digest,)).fetchone()
        if exact or self.known_product(item.product_id):
            with self.db:
                self.db.execute("""UPDATE candidates SET status='duplicate',sha256=?,dhash=?,
                    width=?,height=?,collected_at=?,error=NULL WHERE candidate_id=?""",
                    (digest, dhash, image.width, image.height, utc_now(), item.candidate_id))
            return "duplicate"
        near_duplicate_of = ""
        for row in self.db.execute(
                "SELECT candidate_id,dhash FROM candidates WHERE status='downloaded' AND dhash IS NOT NULL"):
            if hamming(dhash, row["dhash"]) <= 4:
                near_duplicate_of = row["candidate_id"]
                break
        filename = f"raw/{item.product_id}_{item.candidate_id}.jpg"
        target = self.output / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_suffix(".jpg.part")
        image.save(temp, "JPEG", quality=95, optimize=True)
        temp.replace(target)
        with self.db:
            self.db.execute("""UPDATE candidates SET status='downloaded',filename=?,sha256=?,dhash=?,
                width=?,height=?,near_duplicate_of=?,collected_at=?,error=NULL WHERE candidate_id=?""",
                (filename, digest, dhash, image.width, image.height, near_duplicate_of,
                 utc_now(), item.candidate_id))
        return "downloaded"

    def fail(self, item: Candidate, error: Exception) -> None:
        with self.db:
            self.db.execute("UPDATE candidates SET status='failed',error=? WHERE candidate_id=?",
                            (str(error), item.candidate_id))

    def review_rows(self) -> list[dict]:
        rows = [dict(row) for row in self.db.execute(
            "SELECT * FROM candidates WHERE status='downloaded' ORDER BY collected_at,candidate_id")]
        previous = {}
        review_path = self.output / "review" / "review.csv"
        if review_path.exists():
            with review_path.open(encoding="utf-8-sig", newline="") as stream:
                previous = {row["candidate_id"]: row for row in csv.DictReader(stream)}
        result = []
        for row in rows:
            entry = {field: "" for field in REVIEW_FIELDS}
            entry.update({field: row.get(field, "") for field in REVIEW_FIELDS})
            entry.update(group=f"naver_{row['product_id']}", label="")
            old = previous.get(row["candidate_id"], {})
            for field in ("decision", "crop_x", "crop_y", "crop_width", "crop_height", "reason"):
                entry[field] = old.get(field, entry[field])
            # Shape labels intentionally stay blank for this collection.
            entry["label"] = ""
            result.append(entry)
        return result

    def export_review(self) -> list[dict]:
        rows = self.review_rows()
        write_csv(self.output / "review" / "review.csv", REVIEW_FIELDS, rows)
        return rows


def _download_image(item: Candidate, timeout: int, retries: int) -> bytes:
    try:
        return fetch_image(item.image_url, timeout, retries)
    except Exception as err:
        if item.thumbnail_url and item.thumbnail_url != item.image_url:
            try:
                return fetch_image(item.thumbnail_url, timeout, retries)
            except Exception:
                pass
        raise err


def collect(args) -> dict:
    client_id = (os.environ.get("NAVER_CLIENT_ID") or os.environ.get("NCP_CLIENT_ID") or "").strip()
    client_secret = (os.environ.get("NAVER_CLIENT_SECRET") or os.environ.get("NCP_CLIENT_SECRET") or "").strip()
    if not client_id or not client_secret:
        raise CollectionStopped("NAVER_CLIENT_ID(또는 NCP_CLIENT_ID)와 NAVER_CLIENT_SECRET 환경변수가 필요합니다.")
    store = NaverStore(args.output)
    stop_reason = "search_exhausted"
    try:
        for start in range(1, 1001, 100):
            for query in args.queries:
                if store.candidate_count() >= args.candidate_limit:
                    stop_reason = "candidate_limit"
                    break
                if store.search_done(query, start):
                    continue
                payload = api_search(query, start, client_id, client_secret,
                                     args.timeout, args.retries)
                items = parse_api_items(payload, query)
                store.discover(items)
                pending = [item for item in items if not store.known_product(item.product_id)]
                remaining = args.candidate_limit - store.candidate_count()
                pending = pending[:remaining]
                with ThreadPoolExecutor(max_workers=2) as pool:
                    futures = {pool.submit(_download_image, item, args.timeout, args.retries): item
                               for item in pending}
                    for future in as_completed(futures):
                        item = futures[future]
                        try:
                            status = store.save(item, future.result(), args.min_short, args.min_long)
                            LOG.info("%s %s %s", status, item.product_id, item.title)
                        except CollectionStopped:
                            raise
                        except Exception as error:
                            store.fail(item, error)
                            LOG.warning("실패 %s: %s", item.product_id, error)
                store.mark_search(query, start, len(items), store.candidates_resolved(items))
                store.export_review()
                if len(items) < 100:
                    LOG.info("검색 결과 끝: %s start=%d", query, start)
                time.sleep(args.api_delay)
            if stop_reason == "candidate_limit":
                break
    finally:
        rows = store.export_review()
        summary = summarize(store, stop_reason)
        write_summary(args.output, summary)
        store.close()
    return summary


def summarize(store: NaverStore, stop_reason: str) -> dict:
    counts = {row[0]: row[1] for row in store.db.execute(
        "SELECT status,COUNT(*) FROM candidates GROUP BY status")}
    decisions = {"accept": 0, "reject": 0, "pending": 0}
    for row in store.review_rows():
        decision = row["decision"].strip().lower() or "pending"
        if decision in decisions:
            decisions[decision] += 1
    return {"downloaded_candidates": counts.get("downloaded", 0),
            "duplicates": counts.get("duplicate", 0), "failed": counts.get("failed", 0),
            "review": decisions, "stop_reason": stop_reason, "updated_at": utc_now()}


def write_summary(output: Path, summary: dict) -> None:
    temp = output / "summary.json.part"
    temp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(output / "summary.json")


def parse_int(value: str, field: str) -> int | None:
    value = value.strip()
    if not value:
        return None
    try:
        result = int(value)
    except ValueError as error:
        raise ValueError(f"{field}는 정수여야 합니다: {value}") from error
    if result < 0:
        raise ValueError(f"{field}는 0 이상이어야 합니다.")
    return result


def crop_box(row: dict, width: int, height: int) -> tuple[int, int, int, int]:
    values = [parse_int(row[field], field) for field in
              ("crop_x", "crop_y", "crop_width", "crop_height")]
    if all(value is None for value in values):
        return 0, 0, width, height
    if any(value is None for value in values):
        raise ValueError("크롭 좌표 4개를 모두 입력해야 합니다.")
    x, y, crop_width, crop_height = values
    if crop_width == 0 or crop_height == 0 or x + crop_width > width or y + crop_height > height:
        raise ValueError(f"크롭 좌표가 이미지 범위를 벗어납니다: {width}x{height}")
    return x, y, x + crop_width, y + crop_height


def apply_review(output: Path, min_short: int, min_long: int) -> dict:
    review_path = output / "review" / "review.csv"
    if not review_path.exists():
        raise ValueError("review/review.csv가 없습니다. collect를 먼저 실행하세요.")
    with review_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if tuple(rows[0].keys()) != REVIEW_FIELDS if rows else False:
        raise ValueError("review.csv 열 구성이 변경되었습니다.")
    accepted_dir = output / "accepted"
    accepted_dir.mkdir(parents=True, exist_ok=True)
    errors, accepted = [], []
    seen_pixels: set[str] = set()
    for row in rows:
        decision = row["decision"].strip().lower()
        if decision not in ("", "pending", "accept", "reject"):
            errors.append(f"{row['candidate_id']}: decision={decision}")
            continue
        if decision != "accept":
            continue
        try:
            source = output / row["filename"]
            with Image.open(source) as original:
                image = ImageOps.exif_transpose(original).convert("RGB")
            box = crop_box(row, image.width, image.height)
            cropped = image.crop(box)
            short, long = sorted(cropped.size)
            if short < min_short or long < min_long:
                raise ValueError(f"크롭 후 최소 크기 미달: {cropped.width}x{cropped.height}")
            digest = hashlib.sha256(f"{cropped.width}x{cropped.height}:".encode()
                                    + cropped.tobytes()).hexdigest()
            if digest in seen_pixels:
                raise ValueError("다른 accept 항목과 동일한 픽셀 이미지")
            seen_pixels.add(digest)
            accepted_name = f"unlabeled/{len(accepted)+1:04d}_{row['product_id']}_{digest[:12]}.jpg"
            target = accepted_dir / accepted_name
            target.parent.mkdir(parents=True, exist_ok=True)
            temp = target.with_suffix(".jpg.part")
            cropped.save(temp, "JPEG", quality=95, optimize=True)
            temp.replace(target)
            enriched = dict(row, accepted_filename=f"accepted/{accepted_name}", sha256=digest,
                            dhash=difference_hash(cropped))
            accepted.append(enriched)
        except Exception as error:
            errors.append(f"{row['candidate_id']}: {error}")
    write_csv(output / "accepted_manifest.csv", MANIFEST_FIELDS, accepted)
    result = {"accepted": len(accepted), "review_rows": len(rows), "errors": errors,
              "updated_at": utc_now()}
    if errors:
        (output / "review" / "apply_errors.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def export_final(output: Path, target: int) -> dict:
    manifest = output / "accepted_manifest.csv"
    if not manifest.exists():
        raise ValueError("accepted_manifest.csv가 없습니다. apply-review를 먼저 실행하세요.")
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) < target:
        raise ValueError(f"검수 통과 이미지가 {len(rows)}장입니다. 목표 {target}장에 미달합니다.")
    selected = rows[:target]
    final_dir = output / "final" / "unlabeled"
    final_dir.mkdir(parents=True, exist_ok=True)
    final_rows = []
    for index, row in enumerate(selected, 1):
        source = output / row["accepted_filename"]
        if not source.is_file():
            raise ValueError(f"승인 이미지가 없습니다: {source}")
        target_path = final_dir / f"tumbler_{index:04d}.jpg"
        shutil.copyfile(source, target_path)
        final_rows.append(dict(row, accepted_filename=f"final/unlabeled/{target_path.name}"))
    write_csv(output / "final" / "manifest.csv", MANIFEST_FIELDS, final_rows)
    result = {"exported": len(final_rows), "target": target, "completed": len(final_rows) == target,
              "updated_at": utc_now()}
    (output / "final" / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def make_contact_sheets(output: Path, columns: int = 5, rows_per_sheet: int = 5) -> int:
    store = NaverStore(output)
    try:
        rows = store.export_review()
    finally:
        store.close()
    sheet_dir = output / "review" / "contact_sheets"
    sheet_dir.mkdir(parents=True, exist_ok=True)
    cell_width, cell_height = 240, 290
    page_size = columns * rows_per_sheet
    for page, offset in enumerate(range(0, len(rows), page_size), 1):
        canvas = Image.new("RGB", (columns * cell_width, rows_per_sheet * cell_height), "white")
        draw = ImageDraw.Draw(canvas)
        for slot, row in enumerate(rows[offset:offset + page_size]):
            with Image.open(output / row["filename"]) as source:
                thumb = ImageOps.contain(source.convert("RGB"), (220, 235))
            x = (slot % columns) * cell_width + (cell_width - thumb.width) // 2
            y = (slot // columns) * cell_height + 5
            canvas.paste(thumb, (x, y))
            draw.text(((slot % columns) * cell_width + 8, y + 240),
                      f"{offset + slot + 1:04d} {row['candidate_id'][:8]}", fill="black",
                      font=ImageFont.load_default())
        canvas.save(sheet_dir / f"sheet_{page:03d}.jpg", "JPEG", quality=90)
    return (len(rows) + page_size - 1) // page_size


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="네이버 쇼핑 텀블러 이미지 수집/검수/내보내기")
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    collect_parser = sub.add_parser("collect", parents=[common])
    collect_parser.add_argument("--queries", nargs="+", default=list(DEFAULT_QUERIES))
    collect_parser.add_argument("--candidate-limit", type=int, default=3000)
    collect_parser.add_argument("--min-short", type=int, default=128)
    collect_parser.add_argument("--min-long", type=int, default=256)
    collect_parser.add_argument("--timeout", type=int, default=30)
    collect_parser.add_argument("--retries", type=int, choices=range(4), default=2)
    collect_parser.add_argument("--api-delay", type=float, default=1.0)
    review_parser = sub.add_parser("apply-review", parents=[common])
    review_parser.add_argument("--min-short", type=int, default=128)
    review_parser.add_argument("--min-long", type=int, default=256)
    sheet_parser = sub.add_parser("contact-sheet", parents=[common])
    sheet_parser.add_argument("--columns", type=int, default=5)
    sheet_parser.add_argument("--rows", type=int, default=5)
    export_parser = sub.add_parser("export", parents=[common])
    export_parser.add_argument("--target", type=int, default=500)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        if args.command == "collect":
            if args.candidate_limit < 1 or args.min_short < 1 or args.min_long < args.min_short:
                raise ValueError("수집/크기 한도를 확인하세요.")
            result = collect(args)
        elif args.command == "contact-sheet":
            if args.columns < 1 or args.rows < 1:
                raise ValueError("행과 열은 1 이상이어야 합니다.")
            result = {"contact_sheets": make_contact_sheets(args.output, args.columns, args.rows)}
        elif args.command == "apply-review":
            result = apply_review(args.output, args.min_short, args.min_long)
            if result["errors"]:
                print(json.dumps(result, ensure_ascii=False, indent=2))
                return 1
        else:
            if args.target < 1:
                raise ValueError("target은 1 이상이어야 합니다.")
            result = export_final(args.output, args.target)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (CollectionStopped, ValueError, OSError, sqlite3.Error) as error:
        LOG.error("중단: %s", error)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
