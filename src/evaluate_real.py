"""팀원 실촬영 텀블러 사진에 대한 6조건별 최종 테스트 및 고정 4종 Macro F1 평가 모듈."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import TumblerPredictor
from src.models.vit_classifier import CLASSES

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
REPORTS_DIR = PROJECT_ROOT / "reports"
REAL_DATA_DIR = PROJECT_ROOT / "data" / "real_test"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

CONDITIONS = [
    "plain_normal_front",          # 단순 배경 + 기본 조명 정면
    "plain_normal_oblique_left",   # 단순 배경 + 기본 조명 좌측 사선
    "plain_normal_oblique_right",  # 단순 배경 + 기본 조명 우측 사선
    "daily_front",                 # 생활 배경 정면
    "light_front",                 # 조명 변경 정면
    "handheld",                    # 손에 든 상태 (몸통 윤곽 충분히 노출)
]


def evaluate_real_dataset(
    data_dir: Path | str,
    checkpoint_path: Path | str,
    output_prefix: str = "real_test",
) -> dict:
    """실촬영 데이터 디렉토리를 평가하여 Macro F1, 혼동행렬, 조건별 성능, 오분류 사례를 분석합니다."""
    data_dir = Path(data_dir)
    predictor = TumblerPredictor(checkpoint_path)

    print(f"=== Starting Real Photo Evaluation: {data_dir} ===")
    meta_path = data_dir / "meta.csv"

    records = []
    if meta_path.is_file():
        df_meta = pd.read_csv(meta_path)
        for _, row in df_meta.iterrows():
            img_p = data_dir / row["filename"]
            if img_p.is_file():
                records.append({
                    "path": img_p,
                    "filename": row["filename"],
                    "product_id": str(row.get("product_id", row.get("group", "unknown"))),
                    "condition": str(row.get("condition", "plain_normal_front")),
                    "label": str(row["label"]),
                })
    else:
        # 폴더 기반 자동 스캔 (data_dir/label/ 또는 data_dir/condition/)
        for img_p in sorted(data_dir.glob("**/*.*")):
            if img_p.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                parent_name = img_p.parent.name
                lbl = parent_name if parent_name in CLASSES else "unknown"
                records.append({
                    "path": img_p,
                    "filename": str(img_p.relative_to(data_dir)),
                    "product_id": img_p.stem.split("_")[0],
                    "condition": "plain_normal_front",
                    "label": lbl,
                })

    if not records:
        print(f"Warning: No images found in {data_dir}.")
        return {}

    y_true = []
    y_pred = []
    condition_results = defaultdict(lambda: {"true": [], "pred": []})
    misclassified = []
    product_ids = set()

    for r in records:
        product_ids.add(r["product_id"])
        with Image.open(r["path"]) as img:
            res = predictor.predict(img)

        pred_lbl = res["label"]
        true_lbl = r["label"]

        if true_lbl in CLASSES:
            y_true.append(CLASSES.index(true_lbl))
            y_pred.append(CLASSES.index(pred_lbl))
            condition_results[r["condition"]]["true"].append(CLASSES.index(true_lbl))
            condition_results[r["condition"]]["pred"].append(CLASSES.index(pred_lbl))

            if pred_lbl != true_lbl:
                misclassified.append({
                    "filename": r["filename"],
                    "product_id": r["product_id"],
                    "condition": r["condition"],
                    "true_label": true_lbl,
                    "pred_label": pred_lbl,
                    "pred_score": res["score"],
                    "probs": res["probs"],
                })

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 1. 전체 핵심 지표 (Macro F1 고정)
    overall_macro_f1 = float(f1_score(y_true, y_pred, labels=[0, 1, 2, 3], average="macro", zero_division=0))
    overall_acc = float(accuracy_score(y_true, y_pred))
    prec, rec, f1_cls, sup = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2, 3], zero_division=0)
    conf_mat = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])

    per_class_summary = {}
    for i, c in enumerate(CLASSES):
        per_class_summary[c] = {
            "precision": round(float(prec[i]), 4),
            "recall": round(float(rec[i]), 4),
            "f1": round(float(f1_cls[i]), 4),
            "support": int(sup[i]),
        }

    # 2. 6조건별 성적 집계
    cond_summary = {}
    for cond_name in CONDITIONS:
        cond_data = condition_results.get(cond_name, {"true": [], "pred": []})
        c_true = np.array(cond_data["true"])
        c_pred = np.array(cond_data["pred"])
        n_imgs = len(c_true)
        if n_imgs > 0:
            c_f1 = float(f1_score(c_true, c_pred, labels=[0, 1, 2, 3], average="macro", zero_division=0))
            c_acc = float(accuracy_score(c_true, c_pred))
            classes_present = [CLASSES[i] for i in set(c_true)]
        else:
            c_f1 = 0.0
            c_acc = 0.0
            classes_present = []

        cond_summary[cond_name] = {
            "n_images": n_imgs,
            "classes_present": classes_present,
            "macro_f1": round(c_f1, 4),
            "accuracy": round(c_acc, 4),
        }

    summary = {
        "n_total_images": len(records),
        "n_distinct_products": len(product_ids),
        "overall_macro_f1": round(overall_macro_f1, 4),
        "overall_accuracy": round(overall_acc, 4),
        "per_class": per_class_summary,
        "conditions": cond_summary,
        "misclassified_count": len(misclassified),
        "misclassified_samples": misclassified,
    }

    # 혼동행렬 시각화 플롯 저장
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 6))
    sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title(f"Real Photo Confusion Matrix (Macro F1: {overall_macro_f1:.4f})")
    plt.xlabel("Predicted Shape")
    plt.ylabel("True Shape")
    plt.tight_layout()
    conf_path = FIGURES_DIR / f"{output_prefix}_confusion_matrix.png"
    plt.savefig(conf_path, dpi=150)
    plt.close()

    # 리포트 마크다운 작성
    report_md_path = REPORTS_DIR / f"{output_prefix}_report.md"
    write_real_test_markdown_report(report_md_path, summary)

    print(f"\n=======================================================")
    print(f" Real Test Completed: {len(records)} photos ({len(product_ids)} distinct tumblers)")
    print(f" Overall Macro F1:  {overall_macro_f1:.4f}")
    print(f" Overall Accuracy:  {overall_acc:.4f}")
    print(f" Misclassifications: {len(misclassified)} photos")
    print(f" Report saved: {report_md_path}")
    print(f" Confusion matrix: {conf_path}")
    print(f"=======================================================\n")
    return summary


def write_real_test_markdown_report(path: Path, summary: dict):
    """실촬영 평가 종합 리포트 마크다운 파일 작성."""
    lines = [
        "# 실촬영 텀블러 4종 형태 분류 최종 평가 보고서",
        "",
        f"- **총 평가 사진 수**: {summary['n_total_images']} 장",
        f"- **고유 실제 텀블러 제품 수**: {summary['n_distinct_products']} 개",
        f"- **주요 평가 지표 (Macro F1)**: **`{summary['overall_macro_f1']:.4f}`**",
        f"- **보조 평가 지표 (Accuracy)**: `{summary['overall_accuracy']:.4f}`",
        "",
        "## 1. 클래스별 상세 성능",
        "",
        "| 클래스 | Precision | Recall | F1-Score | 사진 수(Support) |",
        "|---|---|---|---|---|",
    ]
    for c, d in summary["per_class"].items():
        lines.append(f"| `{c}` | {d['precision']:.4f} | {d['recall']:.4f} | {d['f1']:.4f} | {d['support']} |")

    lines.extend([
        "",
        "## 2. 6조건별 성능 분포",
        "",
        "| 촬영 조건 | 사진 수 | 포함된 클래스 | Macro F1 | Accuracy |",
        "|---|---|---|---|---|",
    ])
    for cond, d in summary["conditions"].items():
        classes_str = ", ".join(d["classes_present"]) if d["classes_present"] else "(없음)"
        lines.append(f"| `{cond}` | {d['n_images']} | {classes_str} | {d['macro_f1']:.4f} | {d['accuracy']:.4f} |")

    lines.extend([
        "",
        "## 3. 오분류 사례 분석",
        "",
        f"총 {summary['misclassified_count']}건 오분류 발생:",
        "",
        "| 파일명 | 제품 ID | 촬영 조건 | 정답 라벨 | 예측 라벨 | 예측 확신도 |",
        "|---|---|---|---|---|---|",
    ])
    for m in summary["misclassified_samples"][:20]:
        lines.append(f"| `{m['filename']}` | `{m['product_id']}` | `{m['condition']}` | `{m['true_label']}` | **`{m['pred_label']}`** | {m['pred_score']:.1%} |")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def generate_mock_pilot_testset():
    """실제 팀원 촬영 데이터가 아직 준비 중일 때 파이프라인 검증을 위한 파일럿 테스트셋을 생성합니다."""
    from src.models.transforms import AspectRatioPadResize
    REAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(PROJECT_ROOT / "data" / "metadata.csv")
    val_df = df[df["split"] == "val"]

    mock_rows = []
    idx = 0
    for cls in CLASSES:
        cls_rows = val_df[val_df["label"] == cls].head(3)
        for _, r in cls_rows.iterrows():
            orig_p = Path(r["full_path"])
            cond = CONDITIONS[idx % len(CONDITIONS)]
            dest_name = f"{cls}_{cond}_{orig_p.name}"
            dest_p = REAL_DATA_DIR / dest_name
            with Image.open(orig_p) as im:
                im.save(dest_p)
            mock_rows.append({
                "filename": dest_name,
                "product_id": f"pilot_{cls}_{idx // 2}",
                "condition": cond,
                "label": cls,
            })
            idx += 1

    pd.DataFrame(mock_rows).to_csv(REAL_DATA_DIR / "meta.csv", index=False, encoding="utf-8-sig")
    print(f"Created pilot real test set with {len(mock_rows)} photos in {REAL_DATA_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=str(REAL_DATA_DIR))
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--generate-pilot", action="store_true")
    args = parser.parse_args()

    if args.generate_pilot:
        generate_mock_pilot_testset()
        sys.exit(0)

    ckpt_path = args.checkpoint
    if ckpt_path is None:
        candidates = list(CHECKPOINTS_DIR.glob("*.pt"))
        if candidates:
            ckpt_path = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]
        else:
            print("Error: No checkpoint found.")
            sys.exit(1)

    evaluate_real_dataset(args.data_dir, ckpt_path)
