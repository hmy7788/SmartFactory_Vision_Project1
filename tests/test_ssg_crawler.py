import csv
import io
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from bs4 import BeautifulSoup
from PIL import Image

from src.data_collection.ssg_crawler import (
    DEFAULT_URL, CrawlStopped, Store, asset_key, best_image, check_block,
    download, extract_products, has_next_page, main, search_url,
)

FIXTURE = Path(__file__).parent / "fixtures" / "ssg_search.html"


def image_bytes(color="red", size=(160, 200)):
    stream = io.BytesIO()
    Image.new("RGB", size, color).save(stream, "PNG")
    return stream.getvalue()


class ParserTests(unittest.TestCase):
    def setUp(self):
        self.html = FIXTURE.read_text(encoding="utf-8")

    def test_real_markup_lazy_srcset_and_badges(self):
        items = extract_products(self.html, DEFAULT_URL, 1)
        self.assertEqual(len(items), 2)
        self.assertTrue(all(p.image_url.endswith("_464.jpg") for p in items))
        self.assertEqual(items[0].item_id, "1000774926977")
        self.assertEqual(items[1].title, "글라스락 샤이닝 텀블러 500ml")

    def test_lazy_data_srcset(self):
        tag = BeautifulSoup('<img src="data:image/gif,x" data-srcset="//sitem.ssgcdn.com/a/item/a.jpg 320w, //sitem.ssgcdn.com/a/item/b.jpg 640w">', "html.parser").img
        self.assertEqual(best_image(tag, DEFAULT_URL), "https://sitem.ssgcdn.com/a/item/b.jpg")

    def test_non_cdn_and_non_product_links_are_excluded(self):
        html = '<a href="https://evil.test/item/itemView.ssg?itemId=1"><img src="https://sitem.ssgcdn.com/item/a.jpg"></a>'
        html += '<a href="https://www.ssg.com/item/itemView.ssg?itemId=2"><img src="https://evil.test/item/a.jpg"></a>'
        self.assertEqual(extract_products(html, DEFAULT_URL, 1), [])

    def test_url_and_size_identity(self):
        self.assertIn("page=2", search_url(DEFAULT_URL + "&page=20", 2))
        self.assertNotIn("page=20", search_url(DEFAULT_URL + "&page=20", 2))
        self.assertEqual(asset_key("https://sitem.ssgcdn.com/item/123_i1_232.jpg"),
                         asset_key("https://sitem.ssgcdn.com/item/123_i1_464.jpg"))
        with self.assertRaises(ValueError):
            search_url("http://localhost/search.ssg?query=x", 1)

    def test_pagination_group_boundary_and_last_page(self):
        self.assertTrue(has_next_page(self.html, 1))
        self.assertTrue(has_next_page(self.html, 2))  # Last button means another group exists.
        end = '<button aria-label="249 페이지로 이동">249</button><button aria-label="250 페이지로 이동">250</button><button aria-label="다음 페이지로 이동">next</button>'
        self.assertTrue(has_next_page(end, 249))
        self.assertFalse(has_next_page(end, 250))  # Enabled Next remains on the actual last page.

    def test_access_block_is_not_empty_search(self):
        for html, status in [("접속이 잠시 제한되었습니다", 200), ("", 429), ("", 403),
                             ('<iframe src="/captcha"></iframe>', 200)]:
            with self.subTest(status=status), self.assertRaises(CrawlStopped):
                check_block(html, status)
        check_block('<script>const text="접속이 잠시 제한";</script><div>상품</div>')


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.store = Store(self.root, DEFAULT_URL)
        self.items = extract_products(FIXTURE.read_text(encoding="utf-8"), DEFAULT_URL, 1)
        self.store.discover(self.items)

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_identical_pixels_dedup_but_keep_both_sources(self):
        self.assertEqual(self.store.save(self.items[0], image_bytes(), 100), "downloaded")
        self.assertEqual(self.store.save(self.items[1], image_bytes(), 100), "duplicate")
        summary = self.store.export()
        self.assertEqual(summary["unique_images"], 1)
        self.assertEqual(summary["products_images"], 2)
        self.assertEqual(len(list((self.root / "raw").rglob("*.png"))), 1)

    def test_resume_retry_and_missing_file(self):
        self.store.save(self.items[0], image_bytes(), 100)
        self.store.fail(self.items[1], ValueError("temporary"))
        self.store.page_done(1, "fp", False)
        self.assertTrue(self.store.done(self.items[0]))
        self.assertFalse(self.store.done(self.items[1]))
        self.assertEqual(self.store.resume_page(), 1)
        self.store.save(self.items[1], image_bytes("blue"), 100)
        self.store.page_done(1, "fp", True)
        self.assertTrue(self.store.all_pages_complete())
        filename = self.store.db.execute("SELECT filename FROM jobs WHERE key=?", (self.items[0].key,)).fetchone()[0]
        (self.root / "raw" / filename).unlink()  # Only test-owned temporary data.
        self.assertFalse(self.store.done(self.items[0]))
        self.assertEqual(self.store.resume_page(), 1)

    def test_different_resolution_and_false_success(self):
        self.store.page_done(250, "last", True)
        self.assertFalse(self.store.all_pages_complete())
        for data in [b"<html>blocked</html>", image_bytes(size=(10, 10))]:
            with self.assertRaises(Exception):
                self.store.save(self.items[0], data, 100)
        self.assertFalse(self.store.done(self.items[0]))

    def test_preserve_reviewed_labels(self):
        self.store.save(self.items[0], image_bytes(), 100)
        self.store.export()
        path = self.root / "meta.csv"
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        rows[0]["label"] = "mug"
        rows[0]["group"] = "same_product_001"
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        self.store.export()
        with path.open(encoding="utf-8-sig", newline="") as stream:
            row = next(csv.DictReader(stream))
        self.assertEqual(row["label"], "mug")
        self.assertEqual(row["group"], "same_product_001")
        self.assertEqual(row["filename"].split("/")[0], row["session"])

    def test_reopen_and_query_mismatch(self):
        self.store.save(self.items[0], image_bytes(), 100)
        reopened = Store(self.root, DEFAULT_URL)
        try:
            self.assertTrue(reopened.done(self.items[0]))
        finally:
            reopened.close()
        with self.assertRaises(ValueError):
            Store(self.root, "https://www.ssg.com/search.ssg?query=other")


class DownloadTests(unittest.TestCase):
    def test_server_retry_but_no_access_control_retry(self):
        from unittest.mock import MagicMock
        context = MagicMock()
        server_error = MagicMock(status=503)
        good = MagicMock(status=200, ok=True, headers={"content-type": "image/png"})
        good.body.return_value = image_bytes()
        context.request.get.side_effect = [server_error, good]
        with patch("src.data_collection.ssg_crawler.time.sleep"):
            self.assertEqual(download(context, "url", "referer", 1, 2), image_bytes())
        self.assertEqual(context.request.get.call_count, 2)
        context.request.get.reset_mock(side_effect=True)
        context.request.get.return_value = MagicMock(status=429)
        with patch("src.data_collection.ssg_crawler.time.sleep"), self.assertRaises(CrawlStopped):
            download(context, "url", "referer", 1, 2)
        self.assertEqual(context.request.get.call_count, 1)

    def test_policy_denial_does_not_launch_crawler(self):
        with patch("src.data_collection.ssg_crawler.policy_report", return_value={"allowed": False}), \
             patch("src.data_collection.ssg_crawler.crawl") as crawler:
            self.assertEqual(main([]), 2)
            crawler.assert_not_called()

    def test_offline_cli_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            code = main(["--html-dir", str(FIXTURE.parent), "--output", folder])
            self.assertEqual(code, 0)
            with (Path(folder) / "manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 2)


if __name__ == "__main__":
    unittest.main()
