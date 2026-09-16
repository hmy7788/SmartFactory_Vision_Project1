"""Collect public SSG search-card images; see docs/ssg_crawler.md.

No private APIs, guessed image resolutions, stealth, or CAPTCHA handling.
The parser and store work offline; only crawl() requires a browser.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import re
import sqlite3
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urljoin, urlsplit, urlunsplit
from urllib.request import Request, urlopen
from urllib.robotparser import RobotFileParser

from bs4 import BeautifulSoup
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_URL = "https://www.ssg.com/search.ssg?target=all&query=%ED%85%80%EB%B8%94%EB%9F%AC"
BOT_AGENT = "TumblerDatasetBot/1.0"
LOG = logging.getLogger("ssg")
BLOCK_TEXT = ("접속이 잠시 제한", "비정상적인 접근", "자동화된 환경(봇)",
              "access denied", "verify you are human", "captcha verification")
META_FIELDS = ["filename", "session", "group", "label", "count", "value",
               "cond_light", "cond_bg", "cond_angle", "note"]


class CrawlStopped(RuntimeError):
    """Access restriction, unexpected markup, or unsafe continuation."""


@dataclass(frozen=True)
class ProductImage:
    item_id: str
    title: str
    product_url: str
    image_url: str
    image_key: str
    page: int

    @property
    def key(self) -> str:
        return hashlib.sha256(f"{self.item_id}|{self.image_key}".encode()).hexdigest()


def search_url(url: str, page: int) -> str:
    parts = urlsplit(url)
    if parts.scheme != "https" or parts.hostname != "www.ssg.com" or parts.path != "/search.ssg":
        raise ValueError("--url은 https://www.ssg.com/search.ssg 검색 주소여야 합니다.")
    query = parse_qs(parts.query, keep_blank_values=True)
    if not query.get("query", [""])[0]:
        raise ValueError("검색어 query가 필요합니다.")
    query["page"] = [str(page)]
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query, doseq=True), ""))


def asset_key(url: str) -> str:
    """Dedup size variants, without inventing a download URL."""
    parts = urlsplit(url)
    path = re.sub(r"(_i\d+)_\d+(?=\.[a-zA-Z]+$)", r"\1", parts.path)
    return urlunsplit((parts.scheme, parts.netloc, path, parts.query, ""))


def public_image_url(value: str, base_url: str) -> str | None:
    url = urljoin(base_url, value.strip())
    parts = urlsplit(url)
    host = parts.hostname or ""
    if (parts.scheme != "https" or parts.username or parts.password
            or not host.endswith(".ssgcdn.com") or "/item/" not in parts.path):
        return None
    return urlunsplit((parts.scheme, parts.netloc, parts.path, parts.query, ""))


def best_image(img, base_url: str) -> str | None:
    candidates: list[tuple[float, str]] = []
    for attr in ("src", "data-src", "data-original", "data-lazy-src"):
        url = public_image_url(img.get(attr, ""), base_url)
        if url:
            candidates.append((1, url))
    for attr in ("srcset", "data-srcset"):
        for entry in img.get(attr, "").split(","):
            fields = entry.strip().split()
            if not fields:
                continue
            url = public_image_url(fields[0], base_url)
            if not url:
                continue
            try:
                rank = float(fields[1][:-1]) if len(fields) > 1 else 1
            except ValueError:
                rank = 1
            candidates.append((rank, url))
    return max(candidates, key=lambda pair: pair[0])[1] if candidates else None


def extract_products(html: str, page_url: str, page: int) -> list[ProductImage]:
    soup = BeautifulSoup(html, "html.parser")
    found: dict[str, ProductImage] = {}
    for anchor in soup.select('a[href*="itemView.ssg"], a[href*="dealItemView.ssg"]'):
        link = urljoin(page_url, anchor.get("href", ""))
        parts = urlsplit(link)
        if not (parts.hostname == "ssg.com" or (parts.hostname or "").endswith(".ssg.com")):
            continue
        item_id = parse_qs(parts.query).get("itemId", [""])[0]
        if not re.fullmatch(r"\d+", item_id):
            continue
        product_url = f"https://www.ssg.com{parts.path}?itemId={item_id}"
        for img in anchor.select("img"):
            url = best_image(img, page_url)
            if not url:
                continue
            record = ProductImage(item_id, str(img.get("alt", "")).strip(),
                                  product_url, url, asset_key(url), page)
            found.setdefault(record.key, record)
    return list(found.values())


def has_next_page(html: str, page: int) -> bool:
    soup = BeautifulSoup(html, "html.parser")
    numbers = []
    labels = set()
    for node in soup.select("button[aria-label]"):
        if node.has_attr("disabled") or node.get("aria-disabled") == "true":
            continue
        label = node.get("aria-label", "")
        labels.add(label)
        match = re.fullmatch(r"(\d+) 페이지로 이동", label)
        if match:
            numbers.append(int(match[1]))
    if numbers and max(numbers) > page:
        return True
    # SSG leaves an enabled Next button even on page 250 (observed 2026-09-15).
    # In the final number group the Last button disappears. Next alone is unsafe.
    return "다음 페이지로 이동" in labels and "마지막 페이지로 이동" in labels


def check_block(html: str, status: int = 200) -> None:
    # Only visible text, never scripts that may mention CAPTCHA as a feature.
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(" ", strip=True).lower()
    if status in (401, 403, 429) or any(marker in text for marker in BLOCK_TEXT):
        raise CrawlStopped(f"SSG 접근 제한 감지 (HTTP {status}). 재시도/우회 없이 중단합니다.")
    if soup.select('iframe[src*="captcha"], iframe[src*="challenge"], input[name="captcha"]'):
        raise CrawlStopped("CAPTCHA/접근 확인 화면입니다. 수집을 중단합니다.")
    if status >= 400:
        raise CrawlStopped(f"검색 페이지 HTTP {status}")


def policy_report(url: str) -> dict:
    robots_url = "https://www.ssg.com/robots.txt"
    request = Request(robots_url, headers={"User-Agent": BOT_AGENT})
    with urlopen(request, timeout=30) as response:
        content = response.read(512_000).decode("utf-8", errors="replace")
    if "user-agent:" not in content.lower():
        raise CrawlStopped("robots.txt 형식을 확인할 수 없습니다.")
    parser = RobotFileParser(robots_url)
    parser.parse(content.splitlines())
    return {"checked_at": datetime.now(timezone.utc).isoformat(), "url": robots_url,
            "user_agent": BOT_AGENT, "allowed": parser.can_fetch(BOT_AGENT, url),
            "crawl_delay": parser.crawl_delay(BOT_AGENT), "text": content}


class Store:
    """SQLite checkpoint; files first, transaction second for crash recovery."""

    def __init__(self, output: Path, url: str):
        self.output = output.resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.output / "crawl.sqlite3")
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT);
            CREATE TABLE IF NOT EXISTS jobs (
                key TEXT PRIMARY KEY, item_id TEXT, title TEXT, product_url TEXT,
                image_url TEXT, image_key TEXT, page INTEGER, status TEXT DEFAULT 'pending',
                filename TEXT, sha256 TEXT, width INTEGER, height INTEGER, error TEXT);
            CREATE TABLE IF NOT EXISTS assets (
                sha256 TEXT PRIMARY KEY, filename TEXT, width INTEGER, height INTEGER);
            CREATE TABLE IF NOT EXISTS pages (
                page INTEGER PRIMARY KEY, fingerprint TEXT, complete INTEGER);
        """)
        identity = search_url(url, 1)
        previous = self.db.execute("SELECT value FROM settings WHERE key='url'").fetchone()
        if previous and previous[0] != identity:
            self.db.close()
            raise ValueError("다른 검색 조건의 저장 폴더입니다. --output을 새 폴더로 지정하세요.")
        self.db.execute("INSERT OR IGNORE INTO settings VALUES ('url', ?)", (identity,))
        session = datetime.now().strftime("s01_%m%d_ssg")
        self.db.execute("INSERT OR IGNORE INTO settings VALUES ('session', ?)", (session,))
        self.session = self.db.execute("SELECT value FROM settings WHERE key='session'").fetchone()[0]
        self.db.commit()

    def close(self):
        self.db.close()

    def discover(self, products: list[ProductImage]):
        with self.db:
            for p in products:
                self.db.execute("""INSERT INTO jobs
                    (key,item_id,title,product_url,image_url,image_key,page) VALUES (?,?,?,?,?,?,?)
                    ON CONFLICT(key) DO UPDATE SET image_url=excluded.image_url,title=excluded.title
                    """, (p.key, *asdict(p).values()))

    def done(self, product: ProductImage) -> bool:
        row = self.db.execute("SELECT * FROM jobs WHERE key=?", (product.key,)).fetchone()
        return bool(row and row["status"] in ("downloaded", "duplicate")
                    and row["filename"] and (self.output / "raw" / row["filename"]).is_file())

    def save(self, product: ProductImage, body: bytes, min_size: int) -> str:
        if len(body) > 20 * 1024 * 1024:
            raise ValueError("이미지가 20MB 한도를 초과합니다.")
        with Image.open(io.BytesIO(body)) as img:
            img.verify()
        with Image.open(io.BytesIO(body)) as img:
            extension = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "GIF": ".gif"}.get(img.format)
            if not extension:
                raise ValueError(f"지원하지 않는 이미지 형식: {img.format}")
            upright = ImageOps.exif_transpose(img).convert("RGB")
            width, height = upright.size
            if min(width, height) < min_size:
                raise ValueError(f"최소 크기 {min_size}px 미달: {width}x{height}")
            # Pixel hash also catches identical images with different EXIF metadata.
            digest = hashlib.sha256(f"{width}x{height}:".encode() + upright.tobytes()).hexdigest()
        existing = self.db.execute("SELECT filename FROM assets WHERE sha256=?", (digest,)).fetchone()
        filename = existing[0] if existing else f"{self.session}/{product.item_id}_{digest[:16]}{extension}"
        target = self.output / "raw" / filename
        duplicate = target.is_file()
        if not duplicate:
            target.parent.mkdir(parents=True, exist_ok=True)
            temp = target.with_suffix(target.suffix + ".part")
            temp.write_bytes(body)
            temp.replace(target)
        status = "duplicate" if duplicate else "downloaded"
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO assets VALUES (?,?,?,?)", (digest, filename, width, height))
            self.db.execute("""UPDATE jobs SET status=?,filename=?,sha256=?,width=?,height=?,error=NULL
                               WHERE key=?""", (status, filename, digest, width, height, product.key))
        return status

    def fail(self, product: ProductImage, error: Exception):
        with self.db:
            self.db.execute("UPDATE jobs SET status='failed',error=? WHERE key=?", (str(error), product.key))

    def page_done(self, page: int, fingerprint: str, complete: bool):
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO pages VALUES (?,?,?)", (page, fingerprint, int(complete)))

    def resume_page(self) -> int:
        # Missing files or failed downloads force another visit, even on a completed page.
        incomplete = [r[0] for r in self.db.execute("SELECT page FROM pages WHERE complete=0")]
        for row in self.db.execute("SELECT page,status,filename FROM jobs"):
            if row["status"] not in ("downloaded", "duplicate") or not row["filename"] or not (self.output / "raw" / row["filename"]).is_file():
                incomplete.append(row["page"])
        last = self.db.execute("SELECT MAX(page) FROM pages").fetchone()[0] or 1
        return min(incomplete) if incomplete else last

    def all_pages_complete(self) -> bool:
        rows = list(self.db.execute("SELECT page,complete FROM pages ORDER BY page"))
        return bool(rows and [r[0] for r in rows] == list(range(1, rows[-1][0] + 1))
                    and all(r[1] for r in rows))

    def export(self):
        rows = [dict(row) for row in self.db.execute("SELECT * FROM jobs ORDER BY page,key")]
        fields = [column[1] for column in self.db.execute("PRAGMA table_info(jobs)")]
        write_csv(self.output / "manifest.csv", fields, rows)
        meta_path = self.output / "meta.csv"
        old = {}
        if meta_path.exists():
            with meta_path.open(encoding="utf-8-sig", newline="") as stream:
                old = {r["filename"]: r for r in csv.DictReader(stream)}
        meta = []
        seen = set()
        for row in rows:
            filename = row["filename"]
            if (row["status"] not in ("downloaded", "duplicate") or not filename
                    or filename in seen or not (self.output / "raw" / filename).is_file()):
                continue
            seen.add(filename)
            entry = {key: "" for key in META_FIELDS}
            entry.update(filename=filename, session=self.session, group=f"ssg_{row['item_id']}",
                         cond_light="unknown", cond_bg="unknown", cond_angle="unknown",
                         note=f"source=ssg source_domain=web item_id={row['item_id']} {row['product_url']}")
            entry.update({k: v for k, v in old.get(filename, {}).items() if k in META_FIELDS})
            meta.append(entry)
        write_csv(meta_path, META_FIELDS, meta)
        return {"products_images": len(rows), "unique_images": len(meta),
                "failed": sum(row["status"] == "failed" for row in rows),
                "pending": sum(row["status"] == "pending" for row in rows)}


def write_csv(path: Path, fields: list[str], rows: list[dict]):
    temp = path.with_suffix(path.suffix + ".part")
    with temp.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def download(context, url: str, referer: str, delay: float, retries: int) -> bytes:
    """Serial, bounded retries. Rate limiting and access-denial always stop."""
    for attempt in range(retries + 1):
        time.sleep(delay * (2 ** attempt))
        response = None
        try:
            response = context.request.get(url, headers={"Referer": referer}, timeout=30_000)
            if response.status in (401, 403, 429):
                raise CrawlStopped(f"이미지 서버 접근 제한 HTTP {response.status}: {url}")
            if response.status >= 500:
                raise OSError(f"이미지 서버 HTTP {response.status}")
            if not response.ok:
                raise ValueError(f"이미지 HTTP {response.status}")
            if not response.headers.get("content-type", "").lower().startswith("image/"):
                raise ValueError("이미지 대신 HTML 또는 알 수 없는 응답을 받았습니다.")
            if int(response.headers.get("content-length", "0")) > 20 * 1024 * 1024:
                raise ValueError("이미지 응답이 20MB를 초과합니다.")
            return response.body()
        except (CrawlStopped, ValueError):
            raise
        except Exception:
            if attempt >= retries:
                raise
        finally:
            if response is not None:
                response.dispose()
    raise AssertionError("unreachable")


def read_page(page, url: str, timeout: int) -> str:
    response = page.goto(url, wait_until="domcontentloaded", timeout=timeout * 1000)
    status = response.status if response else 200
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        html = page.content()
        check_block(html, status)
        # Lazy images already expose src/srcset. No scrolling or image-load wait needed.
        if extract_products(html, url, 1):
            return html
        if re.search(r"검색\s*결과가\s*없|검색된\s*상품이\s*없", BeautifulSoup(html, "html.parser").get_text(" ")):
            return html
        page.wait_for_timeout(500)
    raise CrawlStopped("상품 목록을 찾지 못했습니다. 페이지 구조 변경/로딩 실패를 확인하세요.")


def crawl(args, store: Store) -> tuple[str, int]:
    from playwright.sync_api import sync_playwright

    page_number = args.start_page or store.resume_page()
    visited = 0
    processed = 0
    fingerprints = set()
    with sync_playwright() as driver:
        options = {"headless": not args.headed}
        if args.browser != "chromium":
            options["channel"] = args.browser
        browser = driver.chromium.launch(**options)
        try:
            context = browser.new_context(locale="ko-KR", viewport={"width": 1440, "height": 1000})
            tab = context.new_page()
            while True:
                url = search_url(args.url, page_number)
                LOG.info("페이지 %d: %s", page_number, url)
                html = read_page(tab, url, args.timeout)
                products = extract_products(html, url, page_number)
                if not products:
                    if page_number == 1:
                        return "empty_search", 0
                    raise CrawlStopped("중간 페이지에 상품이 없습니다. 완료로 처리하지 않습니다.")
                fingerprint = hashlib.sha256("|".join(sorted(p.key for p in products)).encode()).hexdigest()
                if fingerprint in fingerprints:
                    raise CrawlStopped("다른 페이지에서 같은 상품 목록이 반복됩니다. 페이지 이동을 확인하세요.")
                fingerprints.add(fingerprint)
                store.discover(products)
                store.page_done(page_number, fingerprint, False)
                for product in products:
                    if store.done(product) or args.list_only:
                        continue
                    if args.max_images and processed >= args.max_images:
                        return "max_images", 0
                    try:
                        body = download(context, product.image_url, url, args.image_delay, args.retries)
                        status = store.save(product, body, args.min_size)
                        processed += 1
                        LOG.info("%s %s %s", status, product.item_id, product.title)
                    except CrawlStopped:
                        raise
                    except Exception as error:
                        store.fail(product, error)
                        LOG.warning("실패 %s: %s", product.item_id, error)
                complete = not args.list_only and all(store.done(product) for product in products)
                store.page_done(page_number, fingerprint, complete)
                store.export()
                visited += 1
                if not has_next_page(html, page_number):
                    return "last_page", 0
                if args.max_pages and visited >= args.max_pages:
                    return "max_pages", 0
                page_number += 1
                time.sleep(args.page_delay)
        finally:
            browser.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="SSG 텀블러 검색 결과 상품 이미지 수집")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "ssg_tumbler")
    parser.add_argument("--browser", choices=("msedge", "chrome", "chromium"), default="msedge")
    parser.add_argument("--headed", action="store_true", help="브라우저 창 표시")
    parser.add_argument("--start-page", type=int, help="생략 시 체크포인트부터 재개")
    parser.add_argument("--max-pages", type=int, default=0, help="이번 실행 페이지 한도, 0=마지막까지")
    parser.add_argument("--max-images", type=int, default=0, help="이번 실행 신규 처리 이미지 한도, 0=제한 없음")
    parser.add_argument("--page-delay", type=float, default=3.0)
    parser.add_argument("--image-delay", type=float, default=1.0)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--min-size", type=int, default=100)
    parser.add_argument("--list-only", action="store_true", help="상품/이미지 주소만 기록")
    parser.add_argument("--check-policy", action="store_true", help="robots.txt 상태만 확인하고 종료")
    parser.add_argument("--site-permission", action="store_true",
                        help="SSG로부터 자동 수집 허가를 확보한 경우에만 지정 (기술적 차단 우회 없음)")
    parser.add_argument("--html-dir", type=Path, help="저장된 *.html에서 목록만 추출 (네트워크 없음)")
    args = parser.parse_args(argv)
    if args.start_page is not None and args.start_page < 1:
        parser.error("--start-page는 1 이상이어야 합니다.")
    if args.max_pages < 0 or args.max_images < 0 or args.retries not in range(4):
        parser.error("한도는 0 이상, retries는 0~3이어야 합니다.")
    if args.page_delay < 1 or args.image_delay < 0.5 or args.timeout < 1 or args.min_size < 1:
        parser.error("page-delay>=1, image-delay>=0.5, timeout>=1, min-size>=1이 필요합니다.")
    try:
        search_url(args.url, 1)
    except ValueError as error:
        parser.error(str(error))
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    report = None
    if not args.html_dir:
        try:
            report = policy_report(args.url)
            if args.check_policy:
                print(json.dumps(report, ensure_ascii=False, indent=2))
                return 0
            if not report["allowed"] and not args.site_permission:
                LOG.error("robots.txt가 일반 봇의 검색 페이지 수집을 제한합니다. "
                          "SSG 수집 허가가 있는 경우에만 --site-permission을 지정하세요. "
                          "저장된 HTML은 --html-dir로 분석할 수 있습니다.")
                return 2
            if report.get("crawl_delay"):
                args.page_delay = max(args.page_delay, float(report["crawl_delay"]))
        except Exception as error:
            LOG.error("robots.txt 확인 실패: %s", error)
            return 2
    try:
        store = Store(args.output, args.url)
    except (ValueError, OSError, sqlite3.Error) as error:
        LOG.error("저장 폴더 오류: %s", error)
        return 1
    reason, exit_code = "interrupted", 130
    try:
        if args.html_dir:
            files = sorted(args.html_dir.glob("*.html"))
            if not files:
                raise ValueError("--html-dir에 HTML 파일이 없습니다.")
            for index, path in enumerate(files, args.start_page or 1):
                html = path.read_text(encoding="utf-8-sig")
                check_block(html)
                products = extract_products(html, search_url(args.url, index), index)
                if not products:
                    raise ValueError(f"상품 이미지 없음: {path}")
                store.discover(products)
            reason, exit_code = "html_import", 0
        else:
            reason, exit_code = crawl(args, store)
    except KeyboardInterrupt:
        LOG.info("사용자 중단. 체크포인트를 저장합니다.")
    except Exception as error:
        reason, exit_code = str(error), 1
        LOG.error("중단: %s", error)
    finally:
        summary = store.export()
        if summary["failed"]:
            exit_code = exit_code or 1
        summary.update(stop_reason=reason, exit_code=exit_code,
                       finished_at=datetime.now(timezone.utc).isoformat(),
                       search_complete=(reason == "last_page" and store.all_pages_complete()
                                        and not summary["pending"] and not summary["failed"]
                                        and not args.list_only), policy=report)
        (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        store.close()
        print(json.dumps({k: v for k, v in summary.items() if k != "policy"}, ensure_ascii=False, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
