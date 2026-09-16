import csv
import io
import os
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from src.data_collection.naver_tumbler import (
    Candidate, CollectionStopped, NaverStore, REVIEW_FIELDS, apply_review,
    collect, crop_box, difference_hash, export_final, parse_api_items,
    validate_image_url,
)


def image_bytes(color="white", size=(300, 400)):
    stream = io.BytesIO()
    Image.new("RGB", size, color).save(stream, "JPEG")
    return stream.getvalue()


def candidate(product_id="100", query="텀블러", suffix="100.jpg"):
    return Candidate(product_id, query, "<b>테스트</b> 텀블러",
                     f"https://shopping.naver.com/product/{product_id}",
                     f"https://shopping-phinf.pstatic.net/main/{suffix}")


class ApiParserTests(unittest.TestCase):
    def test_parse_documented_payload_and_strip_html(self):
        payload = {"items": [{
            "title": "<b>보온</b> &amp; 보냉 텀블러",
            "link": "https://example.com/images/tumbler123.jpg",
            "thumbnail": "https://search.pstatic.net/common/?src=http%3A%2F%2Fexample.com%2Fimages%2Ftumbler123.jpg",
            "sizewidth": "800", "sizeheight": "1000",
        }, {
            "title": "허용되지 않은 내부 호스트",
            "link": "http://localhost/evil.jpg",
            "thumbnail": "http://127.0.0.1/evil.jpg",
        }]}
        rows = parse_api_items(payload, "텀블러")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].title, "보온 & 보냉 텀블러")
        self.assertEqual(rows[0].query, "텀블러")

    def test_image_url_validation(self):
        self.assertTrue(validate_image_url("https://search.pstatic.net/a.jpg"))
        self.assertTrue(validate_image_url("https://cdn.example.com/tumbler.jpg"))
        for url in ("ftp://example.com/a.jpg", "http://user:pass@evil.com/a.jpg",
                    "http://localhost/a.jpg", "http://127.0.0.1/a.jpg"):
            with self.assertRaises(ValueError):
                validate_image_url(url)

    def test_difference_hash_is_stable(self):
        image = Image.new("RGB", (300, 400), "white")
        self.assertEqual(difference_hash(image), difference_hash(image.copy()))


class StoreAndReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.output = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_exact_and_product_duplicates_do_not_inflate_count(self):
        store = NaverStore(self.output)
        try:
            first = candidate("100", suffix="first.jpg")
            exact = candidate("200", suffix="second.jpg")
            product_repeat = candidate("100", suffix="third.jpg")
            store.discover([first, exact, product_repeat])
            self.assertEqual(store.save(first, image_bytes("red"), 128, 256), "downloaded")
            self.assertEqual(store.save(exact, image_bytes("red"), 128, 256), "duplicate")
            self.assertEqual(store.save(product_repeat, image_bytes("blue"), 128, 256), "duplicate")
            self.assertEqual(store.candidate_count(), 1)
        finally:
            store.close()

    def test_review_is_preserved_but_label_forced_blank(self):
        store = NaverStore(self.output)
        try:
            item = candidate()
            store.discover([item])
            store.save(item, image_bytes(), 128, 256)
            rows = store.export_review()
            rows[0].update(decision="accept", crop_x="10", crop_y="20",
                           crop_width="250", crop_height="350", label="mug")
            review = self.output / "review" / "review.csv"
            with review.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=REVIEW_FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            updated = store.export_review()[0]
            self.assertEqual(updated["decision"], "accept")
            self.assertEqual(updated["crop_x"], "10")
            self.assertEqual(updated["label"], "")
        finally:
            store.close()

    def test_near_duplicate_is_flagged_for_visual_review(self):
        store = NaverStore(self.output)
        try:
            first = candidate("100", suffix="first.jpg")
            similar = candidate("200", suffix="second.jpg")
            store.discover([first, similar])
            store.save(first, image_bytes("red"), 128, 256)
            store.save(similar, image_bytes("blue"), 128, 256)
            rows = store.export_review()
            self.assertEqual(rows[1]["near_duplicate_of"], first.candidate_id)
        finally:
            store.close()

    def test_missing_download_is_retried_on_reopen(self):
        store = NaverStore(self.output)
        item = candidate()
        store.discover([item])
        store.save(item, image_bytes(), 128, 256)
        store.mark_search(item.query, 1, 1, True)
        filename = store.db.execute(
            "SELECT filename FROM candidates WHERE candidate_id=?", (item.candidate_id,)).fetchone()[0]
        store.close()
        (self.output / filename).unlink()
        reopened = NaverStore(self.output)
        try:
            self.assertFalse(reopened.search_done(item.query, 1))
            status = reopened.db.execute(
                "SELECT status FROM candidates WHERE candidate_id=?", (item.candidate_id,)).fetchone()[0]
            self.assertEqual(status, "failed")
        finally:
            reopened.close()

    def test_apply_review_crop_and_export_target(self):
        store = NaverStore(self.output)
        item = candidate()
        try:
            store.discover([item])
            store.save(item, image_bytes(size=(400, 500)), 128, 256)
            rows = store.export_review()
            rows[0].update(decision="accept", crop_x="40", crop_y="50",
                           crop_width="300", crop_height="400")
            write_review(self.output, rows)
        finally:
            store.close()
        applied = apply_review(self.output, 128, 256)
        self.assertEqual(applied["accepted"], 1)
        self.assertFalse(applied["errors"])
        exported = export_final(self.output, 1)
        self.assertTrue(exported["completed"])
        final = self.output / "final" / "unlabeled" / "tumbler_0001.jpg"
        self.assertTrue(final.is_file())
        with Image.open(final) as image:
            self.assertEqual(image.size, (300, 400))

    def test_partial_or_out_of_bounds_crop_is_rejected(self):
        row = {"crop_x": "1", "crop_y": "", "crop_width": "10", "crop_height": "10"}
        with self.assertRaises(ValueError):
            crop_box(row, 100, 100)
        row = {"crop_x": "90", "crop_y": "0", "crop_width": "20", "crop_height": "100"}
        with self.assertRaises(ValueError):
            crop_box(row, 100, 100)

    def test_export_refuses_shortfall(self):
        (self.output / "accepted_manifest.csv").write_text(
            ",".join(REVIEW_FIELDS) + ",sha256,dhash,collected_at,accepted_filename\n",
            encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "목표 500장에 미달"):
            export_final(self.output, 500)


class CredentialTests(unittest.TestCase):
    def test_collect_requires_environment_credentials_before_network(self):
        args = Namespace(output=self.output if hasattr(self, "output") else Path("unused"),
                         queries=["텀블러"], candidate_limit=1, min_short=128,
                         min_long=256, timeout=1, retries=0, api_delay=1)
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(CollectionStopped):
            collect(args)


def write_review(output, rows):
    path = output / "review" / "review.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
