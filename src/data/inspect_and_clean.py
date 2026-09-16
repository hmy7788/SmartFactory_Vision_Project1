"""데이터 검수, 중복 제거, Perceptual Hash 기반 제품 그룹화, 시각적 검수 연락판 생성 모듈."""
from __future__ import annotations

import os
import hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import cv2

from src.cv_utils.segmentation import extract_tumbler_mask, create_nukki_preview, imread_unicode, imwrite_unicode

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent.parent
RAW_DIR = WORKSPACE_ROOT / "raw"
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
REPORTS_DIR = Path(__file__).resolve().parent.parent.parent / "reports" / "figures"
CLASSES = ["straight", "taper_smooth", "taper_step", "mug"]


def calculate_sha256(filepath: Path) -> str:
    """파일의 SHA256 해시를 계산합니다."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def calculate_dhash(image: Image.Image, hash_size: int = 8) -> str:
    """이미지의 Difference Hash(dHash)를 계산합니다."""
    gray = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = np.array(gray)
    diff = pixels[:, 1:] > pixels[:, :-1]
    decimal_val = 0
    hex_chars = []
    flat = diff.flatten()
    for i, val in enumerate(flat):
        if val:
            decimal_val += 2 ** (i % 8)
        if (i % 8) == 7:
            hex_chars.append(f"{decimal_val:02x}")
            decimal_val = 0
    return "".join(hex_chars)


def hamming_distance(h1: str, h2: str) -> int:
    """두 16진수 해시 문자열 간의 해밍 거리(비트 차이)를 계산합니다."""
    return bin(int(h1, 16) ^ int(h2, 16)).count("1")


def cluster_into_product_groups(records: list[dict], max_dist: int = 4) -> dict[str, str]:
    """동일 클래스 내에서 dHash 해밍 거리가 임계치(<=4) 이하인 이미지들을 동일 product_group으로 묶습니다."""
    by_class = defaultdict(list)
    for r in records:
        by_class[r["label"]].append(r)

    group_map = {}
    for cls, items in by_class.items():
        n = len(items)
        # Disjoint-set union (DSU)
        parent = list(range(n))

        def find(i: int) -> int:
            if parent[i] == i:
                return i
            parent[i] = find(parent[i])
            return parent[i]

        def union(i: int, j: int):
            root_i, root_j = find(i), find(j)
            if root_i != root_j:
                parent[root_j] = root_i

        for i in range(n):
            for j in range(i + 1, n):
                if hamming_distance(items[i]["dhash"], items[j]["dhash"]) <= max_dist:
                    union(i, j)

        groups = defaultdict(list)
        for i in range(n):
            groups[find(i)].append(items[i]["image_id"])

        for g_idx, (_, img_ids) in enumerate(groups.items(), start=1):
            group_name = f"pg_{cls}_{g_idx:03d}"
            for img_id in img_ids:
                group_map[img_id] = group_name

    return group_map


def generate_visual_contact_sheets(records: list[dict], dup_pairs: list[tuple[dict, dict]]):
    """시각 검수를 위한 리포트 이미지들을 렌더링하여 reports/figures/에 저장합니다."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 중복 이미지 비교 시각화 (duplicate_pairs.png)
    if dup_pairs:
        dup_canvas_rows = []
        for d1, d2 in dup_pairs:
            im1 = imread_unicode(RAW_DIR / d1["rel_path"])
            im2 = imread_unicode(RAW_DIR / d2["rel_path"])
            if im1 is None or im2 is None:
                continue
            im1_res = cv2.resize(im1, (200, 200))
            im2_res = cv2.resize(im2, (200, 200))
            cv2.putText(im1_res, f"{d1['rel_path']}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(im2_res, f"{d2['rel_path']} (EXCLUDED)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            pair_img = np.hstack([im1_res, im2_res])
            dup_canvas_rows.append(pair_img)
        if dup_canvas_rows:
            dup_grid = np.vstack(dup_canvas_rows)
            imwrite_unicode(REPORTS_DIR / "duplicate_pairs.png", dup_grid)
            print(f"Saved duplicate comparison: {REPORTS_DIR / 'duplicate_pairs.png'}")

    # 2. 클래스별 대표 이미지 샘플 그리드 (dataset_samples_by_class.png)
    by_class = defaultdict(list)
    for r in records:
        if not r["is_duplicate"]:
            by_class[r["label"]].append(r)

    rows = []
    for cls in CLASSES:
        samples = by_class[cls][:4]
        cls_imgs = []
        for s in samples:
            img = imread_unicode(RAW_DIR / s["rel_path"])
            if img is not None:
                img = cv2.resize(img, (180, 180))
            else:
                img = np.zeros((180, 180, 3), dtype=np.uint8)
            cv2.putText(img, f"{cls}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cls_imgs.append(img)
        # 부족하면 빈 타일
        while len(cls_imgs) < 4:
            cls_imgs.append(np.zeros((180, 180, 3), dtype=np.uint8))
        rows.append(np.hstack(cls_imgs))

    samples_grid = np.vstack(rows)
    imwrite_unicode(REPORTS_DIR / "dataset_samples_by_class.png", samples_grid)
    print(f"Saved dataset samples: {REPORTS_DIR / 'dataset_samples_by_class.png'}")

    # 3. 누끼 및 실루엣 세그멘테이션 샘플 미리보기 (nukki_segmentation_preview.png)
    nukki_previews = []
    for cls in CLASSES:
        if by_class[cls]:
            sample = by_class[cls][0]
            img_path = RAW_DIR / sample["rel_path"]
            pil_img = Image.open(img_path)
            mask, rgba = extract_tumbler_mask(pil_img)
            preview = create_nukki_preview(pil_img, mask, rgba, target_size=(200, 200))
            nukki_previews.append(preview)

    if nukki_previews:
        # 2x2 그리드로 배치
        row1 = np.hstack(nukki_previews[:2])
        row2 = np.hstack(nukki_previews[2:4])
        nukki_grid = np.vstack([row1, row2])
        # BGR 변환 후 저장 (create_nukki_preview는 RGB 배열)
        imwrite_unicode(REPORTS_DIR / "nukki_segmentation_preview.png", cv2.cvtColor(nukki_grid, cv2.COLOR_RGB2BGR))
        print(f"Saved nukki preview: {REPORTS_DIR / 'nukki_segmentation_preview.png'}")


def inspect_and_clean_dataset() -> pd.DataFrame:
    """445개 이미지를 전수 조사하여 검수 메타데이터와 시각 자료를 생성합니다."""
    print("=== Starting Dataset Inspection & Cleaning ===")
    records = []
    sha_map = defaultdict(list)

    for cls in CLASSES:
        folder = RAW_DIR / cls
        if not folder.exists():
            continue
        for f in sorted(folder.glob("*.jpg")):
            rel_path = f"{cls}/{f.name}"
            image_id = f"{cls}_{f.stem}"
            sha = calculate_sha256(f)

            with Image.open(f) as img:
                w, h = img.size
                dh = calculate_dhash(img)

            is_small = (w < 300 or h < 300)
            rec = {
                "image_id": image_id,
                "label": cls,
                "filename": f.name,
                "rel_path": rel_path,
                "full_path": str(f.resolve()),
                "width": w,
                "height": h,
                "aspect_ratio": round(w / h, 3),
                "is_small": is_small,
                "sha256": sha,
                "dhash": dh,
                "is_duplicate": False,
                "duplicate_of": None,
                "use_for_training": True,
            }
            records.append(rec)
            sha_map[sha].append(rec)

    # 중복 이미지 식별 (첫 번째 이미지는 유지, 이후는 중복으로 플래그)
    dup_pairs = []
    dup_count = 0
    for sha, group in sha_map.items():
        if len(group) > 1:
            primary = group[0]
            for dup in group[1:]:
                dup["is_duplicate"] = True
                dup["duplicate_of"] = primary["image_id"]
                dup["use_for_training"] = False
                dup_count += 1
                dup_pairs.append((primary, dup))

    print(f"Total images scanned: {len(records)}")
    print(f"Duplicate images excluded: {dup_count} (Unique remaining: {len(records) - dup_count})")
    small_count = sum(1 for r in records if r["is_small"])
    print(f"Images < 300x300: {small_count}")

    # Perceptual hash 기반 product_group 부여
    group_map = cluster_into_product_groups(records, max_dist=4)
    for r in records:
        r["product_group"] = group_map.get(r["image_id"], f"pg_{r['label']}_000")

    num_groups = len(set(r["product_group"] for r in records if not r["is_duplicate"]))
    print(f"Formed {num_groups} independent product groups across classes")

    # 시각화 검수 연락판 생성
    generate_visual_contact_sheets(records, dup_pairs)

    # CSV 저장
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    csv_path = DATA_DIR / "metadata.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"Saved metadata: {csv_path}")
    return df


if __name__ == "__main__":
    inspect_and_clean_dataset()
