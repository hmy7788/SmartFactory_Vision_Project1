"""End-to-end check using Edge + a local HTTP fixture server, never SSG.

Run: python tests/check_ssg_browser.py
The same crawl loop handles pagination, image HTTP, interruption and resume.
"""
import io
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image
from src.data_collection import ssg_crawler as crawler


def main():
    fixture = (Path(__file__).parent / "fixtures" / "ssg_search.html").read_text(encoding="utf-8")
    second = fixture.replace("1000774926977", "1000000000003").replace("1000043272564", "1000000000004")
    second = second.replace('<button aria-label="마지막 페이지로 이동">last</button>', "")
    pictures = {}
    for color in ("red", "blue"):
        buf = io.BytesIO()
        Image.new("RGB", (464, 464), color).save(buf, "JPEG")
        pictures[color] = buf.getvalue()
    state = {"image_requests": 0, "transient_errors": 0, "pages": []}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/page/"):
                number = int(self.path.rsplit("/", 1)[1])
                state["pages"].append(number)
                content = (fixture if number == 1 else second).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                # Prevent the fixture's visible CDN URLs from making browser requests.
                self.send_header("Content-Security-Policy", "default-src 'none'")
            elif "/item/" in self.path:
                state["image_requests"] += 1
                if state["image_requests"] == 1:
                    state["transient_errors"] += 1
                    self.send_response(503)
                    self.end_headers()
                    return
                blue = "1000043272564" in self.path or "1000000000004" in self.path
                content = pictures["blue" if blue else "red"]
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
            else:
                self.send_response(404)
                self.end_headers()
                return
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    real_download = crawler.download

    def local_download(context, url, referer, delay, retries):
        return real_download(context, base + urlsplit(url).path, referer, delay, retries)

    try:
        with tempfile.TemporaryDirectory(prefix="ssg-browser-test-") as folder:
            args = crawler.parse_args(["--output", folder, "--max-images", "1"])
            store = crawler.Store(Path(folder), crawler.DEFAULT_URL)
            try:
                with patch.object(crawler, "search_url", side_effect=lambda _url, n: f"{base}/page/{n}"), \
                     patch.object(crawler, "download", side_effect=local_download), \
                     patch.object(crawler.time, "sleep"):
                    reason, code = crawler.crawl(args, store)
                    assert (reason, code) == ("max_images", 0), (reason, code)
                    assert store.export()["unique_images"] == 1
                    assert not store.all_pages_complete()
                    args.max_images = 0
                    reason, code = crawler.crawl(args, store)
                    assert (reason, code) == ("last_page", 0), (reason, code)
                    summary = store.export()
                    assert summary == {"products_images": 4, "unique_images": 2, "failed": 0, "pending": 0}, summary
                    assert store.all_pages_complete()
                    before = state["image_requests"]
                    crawler.crawl(args, store)
                    assert state["image_requests"] == before, "Resuming downloaded images must not redownload"
                    assert state["transient_errors"] == 1
                print("PASS: Edge + local HTTP, 2 pages, 503 retry, partial stop, resume, pixel dedup, last-page detection")
                print(summary)
                print(state)
            finally:
                store.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


if __name__ == "__main__":
    main()
