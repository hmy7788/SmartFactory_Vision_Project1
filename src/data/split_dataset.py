"""제품 그룹 단위 누수 없는 Stratified Group Split 모듈."""
from __future__ import annotations

import json
import random
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
REPORTS_DIR = Path(__file__).resolve().parent.parent.parent / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
CLASSES = ["straight", "taper_smooth", "taper_step", "mug"]


def split_by_product_group(
    df: pd.DataFrame,
    val_ratio: float = 0.20,
    seed: int = 42,
) -> pd.DataFrame:
    """동일 product_group의 모든 이미지가 Train 또는 Val 중 한 곳에만 배정되도록 분할합니다."""
    rng = random.Random(seed)
    valid_mask = ~df["is_duplicate"]
    valid_df = df[valid_mask].copy()

    split_assignments = {}
    summary = {}

    for cls in CLASSES:
        cls_df = valid_df[valid_df["label"] == cls]
        # product_group 별 이미지 리스트 수집
        group_to_imgs = defaultdict(list)
        for _, row in cls_df.iterrows():
            group_to_imgs[row["product_group"]].append(row["image_id"])

        groups = list(group_to_imgs.keys())
        rng.shuffle(groups)

        total_cls_imgs = len(cls_df)
        target_val_imgs = int(round(total_cls_imgs * val_ratio))

        val_groups = set()
        val_imgs_count = 0

        # 그리디 배정: 20%에 가장 근접하도록 그룹 할당
        for g in groups:
            g_len = len(group_to_imgs[g])
            # 최소 1개 그룹은 할당하되, 너무 초과하지 않도록
            if val_imgs_count == 0 or (val_imgs_count + g_len <= target_val_imgs + max(2, g_len // 2)):
                val_groups.add(g)
                val_imgs_count += g_len
            if val_imgs_count >= target_val_imgs:
                break

        # 안전장치: 모든 그룹이 val에 가버리면 안 되므로
        if len(val_groups) == len(groups):
            val_groups.pop()

        train_groups = set(groups) - val_groups

        for g in train_groups:
            for img_id in group_to_imgs[g]:
                split_assignments[img_id] = "train"

        for g in val_groups:
            for img_id in group_to_imgs[g]:
                split_assignments[img_id] = "val"

        summary[cls] = {
            "total_images": total_cls_imgs,
            "train_images": total_cls_imgs - val_imgs_count,
            "val_images": val_imgs_count,
            "val_actual_ratio": round(val_imgs_count / total_cls_imgs, 3),
            "total_groups": len(groups),
            "train_groups": len(train_groups),
            "val_groups": len(val_groups),
        }

    # DataFrame에 split 컬럼 반영
    df["split"] = "excluded_duplicate"
    for img_id, sp in split_assignments.items():
        df.loc[df["image_id"] == img_id, "split"] = sp

    # 검증: Train과 Val 간 그룹 중복 0건 확인
    train_groups_all = set(df[df["split"] == "train"]["product_group"])
    val_groups_all = set(df[df["split"] == "val"]["product_group"])
    leak = train_groups_all & val_groups_all
    assert len(leak) == 0, f"Critical Data Leakage: Product groups in both train and val: {leak}"

    # 검증: 4개 클래스 모두 train, val에 존재 확인
    train_classes = set(df[df["split"] == "train"]["label"])
    val_classes = set(df[df["split"] == "val"]["label"])
    assert train_classes == set(CLASSES), f"Train missing classes: {set(CLASSES) - train_classes}"
    assert val_classes == set(CLASSES), f"Val missing classes: {set(CLASSES) - val_classes}"

    print("=== Group-based Split Completed Successfully ===")
    total_train = sum(1 for s in split_assignments.values() if s == "train")
    total_val = sum(1 for s in split_assignments.values() if s == "val")
    print(f"Train: {total_train} ({total_train / (total_train + total_val) * 100:.1f}%), Val: {total_val} ({total_val / (total_train + total_val) * 100:.1f}%)")
    for cls, s in summary.items():
        print(f"  [{cls}] Train: {s['train_images']} (Groups: {s['train_groups']}) | Val: {s['val_images']} (Groups: {s['val_groups']}) | Val Ratio: {s['val_actual_ratio']:.1%}")

    # 리포트 및 시각화 저장
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "data_split_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 분할 시각화 차트 생성
    plot_split_distribution(summary)

    # metadata.csv 갱신
    csv_path = DATA_DIR / "metadata.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"Updated metadata with split: {csv_path}")
    return df


def plot_split_distribution(summary: dict):
    """Train과 Val의 클래스별 장수 분포를 막대 그래프로 시각화합니다."""
    classes = list(summary.keys())
    train_counts = [summary[c]["train_images"] for c in classes]
    val_counts = [summary[c]["val_images"] for c in classes]

    x = np.arange(len(classes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    rects1 = ax.bar(x - width / 2, train_counts, width, label="Train", color="#2a78d6")
    rects2 = ax.bar(x + width / 2, val_counts, width, label="Val", color="#eb6834")

    ax.set_ylabel("Number of Images")
    ax.set_title("Dataset Split by Class (Product-Group Stratified, Seed 42)")
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=10)

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    plot_path = FIGURES_DIR / "split_distribution.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved split distribution chart: {plot_path}")


if __name__ == "__main__":
    csv_path = DATA_DIR / "metadata.csv"
    if not csv_path.exists():
        from src.data.inspect_and_clean import inspect_and_clean_dataset
        df = inspect_and_clean_dataset()
    else:
        df = pd.read_csv(csv_path)
    split_by_product_group(df, val_ratio=0.20, seed=42)
