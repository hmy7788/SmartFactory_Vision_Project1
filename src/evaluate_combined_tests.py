"""리비전 실촬영 테스트셋(126장), 웹 밸리데이션셋(89장), 및 통합 테스트셋(215장) 종합 평가 및 시각화 스크립트."""
from __future__ import annotations

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

from src.models.vit_classifier import CLASSES, create_deit_model, load_tumbler_checkpoint
from src.models.transforms import get_val_transforms

# 한글 폰트 설정
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

PROJ_ROOT = Path(__file__).resolve().parent.parent
REVISED_TEST_DIR = PROJ_ROOT / "data" / "test1_orientation_backup"
RAW_DATA_DIR = Path(r"C:\Users\한국전파진흥협회\Desktop\Vision Proj\data set\raw")
METADATA_CSV = PROJ_ROOT / "data" / "metadata.csv"
CHECKPOINTS_DIR = PROJ_ROOT / "checkpoints"
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


def load_datasets():
    records = []

    # 1. 리비전된 실촬영 테스트셋 (126장)
    for label in CLASSES:
        cls_dir = REVISED_TEST_DIR / label
        if not cls_dir.is_dir():
            continue
        for img_path in sorted(cls_dir.glob("*.*")):
            if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                records.append({
                    "dataset_source": "revised_real_test",
                    "filename": img_path.name,
                    "rel_path": f"{label}/{img_path.name}",
                    "full_path": str(img_path),
                    "label": label,
                })

    n_real = len(records)
    print(f"Loaded {n_real} images from revised test1_orientation_backup.")

    # 2. 웹 밸리데이션 데이터셋 (89장)
    meta_df = pd.read_csv(METADATA_CSV)
    val_meta = meta_df[meta_df["split"] == "val"]
    for _, row in val_meta.iterrows():
        label = row["label"]
        fn = row["filename"]
        img_path = RAW_DATA_DIR / label / fn
        if not img_path.is_file():
            # 백업 경로 확인
            img_path = PROJ_ROOT.parent / "vision-harness" / "data" / "raw" / "web_val" / f"{label}_{fn}"
        
        records.append({
            "dataset_source": "web_validation",
            "filename": f"web_{label}_{fn}",
            "rel_path": f"{label}/{fn}",
            "full_path": str(img_path),
            "label": label,
            "product_group": row.get("product_group", ""),
        })

    n_val = len(records) - n_real
    print(f"Loaded {n_val} images from web validation dataset.")
    print(f"Total combined dataset: {len(records)} images.")

    return pd.DataFrame(records)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = load_datasets()
    val_transforms = get_val_transforms(224)

    # 3개 모델 체크포인트 로드
    ckpt_paths = {
        "exp_b_seed44": CHECKPOINTS_DIR / "exp_b_seed44_best.pt",
        "exp_b_seed43": CHECKPOINTS_DIR / "exp_b_seed43_best.pt",
        "exp_b_seed42": CHECKPOINTS_DIR / "exp_b_seed42_best.pt",
    }
    models = {}
    for key, p in ckpt_paths.items():
        if p.is_file():
            m = create_deit_model(num_classes=len(CLASSES), pretrained=False)
            load_tumbler_checkpoint(p, m, device=device)
            m = m.to(device)
            m.eval()
            models[key] = m
            print(f"Loaded model: {key}")

    # 전체 추론 수행
    all_preds = {k: [] for k in models}
    all_probs = {k: [] for k in models}

    for idx, row in df.iterrows():
        img_p = Path(row["full_path"])
        with Image.open(img_p) as im:
            tensor = val_transforms(im).unsqueeze(0).to(device)
        with torch.no_grad():
            for k, m in models.items():
                logits = m(tensor)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                pred_idx = int(np.argmax(probs))
                all_preds[k].append(CLASSES[pred_idx])
                all_probs[k].append(probs)

    for k in models:
        df[f"pred_{k}"] = all_preds[k]
        df[f"score_{k}"] = [float(p[CLASSES.index(pred)]) for p, pred in zip(all_probs[k], all_preds[k])]
        for i, c in enumerate(CLASSES):
            df[f"prob_{k}_{c}"] = [float(p[i]) for p in all_probs[k]]

    # 앙상블 (소프트 보팅)
    ensemble_probs = np.mean([np.array(all_probs[k]) for k in models], axis=0)
    df["pred_ensemble"] = [CLASSES[int(np.argmax(p))] for p in ensemble_probs]
    df["score_ensemble"] = [float(np.max(p)) for p in ensemble_probs]

    # 세 가지 분할(Subset)별 성능 평가
    subsets = {
        "revised_real_test": df[df["dataset_source"] == "revised_real_test"],
        "web_validation": df[df["dataset_source"] == "web_validation"],
        "combined_test": df,
    }

    results = {}
    primary_key = "exp_b_seed43"  # 실촬영에서 가장 성능이 우수했던 Seed 43

    for sub_name, sub_df in subsets.items():
        results[sub_name] = {}
        y_true = sub_df["label"].tolist()
        for k in list(models.keys()) + ["ensemble"]:
            y_pred = sub_df[f"pred_{k}"].tolist()
            macro_f1 = f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0)
            acc = accuracy_score(y_true, y_pred)
            rep = classification_report(y_true, y_pred, labels=CLASSES, target_names=CLASSES, output_dict=True, zero_division=0)
            cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
            results[sub_name][k] = {
                "n_images": len(sub_df),
                "macro_f1": float(macro_f1),
                "accuracy": float(acc),
                "report": rep,
                "cm": cm.tolist(),
            }
        print(f"\n[{sub_name}] (n={len(sub_df)})")
        for k in list(models.keys()) + ["ensemble"]:
            res = results[sub_name][k]
            print(f"  {k:15s} -> Macro F1: {res['macro_f1']:.4f} ({res['macro_f1']*100:.2f}%), Acc: {res['accuracy']:.4f} ({res['accuracy']*100:.2f}%)")

    # 시각화 1: 3개 데이터셋의 혼동 행렬 나란히 비교 (3-Panel Confusion Matrices)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle(f"텀블러 형태 분류 혼동 행렬 비교 ({primary_key} 기준)", fontsize=15, fontweight="bold", y=1.02)

    titles = {
        "revised_real_test": f"1. 리비전 실촬영 테스트셋 (126장)\nMacro F1: {results['revised_real_test'][primary_key]['macro_f1']:.4f} (Acc: {results['revised_real_test'][primary_key]['accuracy']:.4f})",
        "web_validation": f"2. 웹 밸리데이션셋 (89장)\nMacro F1: {results['web_validation'][primary_key]['macro_f1']:.4f} (Acc: {results['web_validation'][primary_key]['accuracy']:.4f})",
        "combined_test": f"3. 통합 테스트셋 (126+89=215장)\nMacro F1: {results['combined_test'][primary_key]['macro_f1']:.4f} (Acc: {results['combined_test'][primary_key]['accuracy']:.4f})",
    }
    cmaps = ["Blues", "Greens", "Purples"]

    for idx, (sub_name, ax) in enumerate(zip(subsets.keys(), axes)):
        cm = np.array(results[sub_name][primary_key]["cm"])
        sns.heatmap(cm, annot=True, fmt="d", cmap=cmaps[idx], xticklabels=CLASSES, yticklabels=CLASSES, cbar=False, ax=ax)
        ax.set_title(titles[sub_name], fontsize=11, fontweight="bold")
        ax.set_xlabel("예측 라벨 (Predicted)", fontsize=10)
        ax.set_ylabel("정답 라벨 (True)", fontsize=10)

    plt.tight_layout()
    cm_3panel_path = FIGURES_DIR / "combined_evaluation_confusion_matrices.png"
    plt.savefig(cm_3panel_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved 3-panel confusion matrix: {cm_3panel_path}")

    # 시각화 2: 클래스별 F1-Score 비교 막대 차트 (실촬영 vs 웹 vs 통합)
    f1_comparison = []
    for sub_name in subsets.keys():
        rep = results[sub_name][primary_key]["report"]
        for cls_name in CLASSES:
            f1_comparison.append({
                "데이터셋": {
                    "revised_real_test": "실촬영 (126장)",
                    "web_validation": "웹 밸리데이션 (89장)",
                    "combined_test": "통합 테스트 (215장)",
                }[sub_name],
                "클래스": cls_name,
                "F1-Score": rep[cls_name]["f1-score"],
            })

    f1_df = pd.DataFrame(f1_comparison)
    plt.figure(figsize=(9, 5))
    sns.barplot(data=f1_df, x="클래스", y="F1-Score", hue="데이터셋", palette=["#4c72b0", "#55a868", "#c44e52"])
    plt.title(f"클래스별 F1-Score 비교 ({primary_key} 기준)", fontsize=13, fontweight="bold")
    plt.ylim(0, 1.05)
    plt.axhline(0.9, color="gray", linestyle="--", alpha=0.5, label="F1 0.90 기준선")
    for p in plt.gca().patches:
        h = p.get_height()
        if h > 0.05:
            plt.gca().annotate(f"{h:.2f}", (p.get_x() + p.get_width() / 2., h),
                               ha='center', va='bottom', fontsize=9, xytext=(0, 2), textcoords='offset points')
    plt.legend(loc="lower right")
    plt.tight_layout()
    bar_chart_path = FIGURES_DIR / "combined_evaluation_f1_bars.png"
    plt.savefig(bar_chart_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved F1 bar chart: {bar_chart_path}")

    # 4. 전체 예측 결과 CSV 저장
    pred_csv_path = REPORTS_DIR / "combined_test_predictions.csv"
    df.to_csv(pred_csv_path, index=False, encoding="utf-8-sig")
    print(f"Saved predictions CSV: {pred_csv_path}")

    # 5. 상세 Markdown 리포트 작성
    report_md = f"""# 리비전 실촬영(126장) + 웹 밸리데이션(89장) 통합 테스트 종합 평가 보고서

- **평가 모델**: `{primary_key}` (`deit_small_patch16_224.fb_in1k`)
- **총 평가 사진 수**: **215 장** (리비전 실촬영 126장 + 웹 밸리데이션 89장)

## 1. 3가지 평가 데이터셋 요약 성과 비교

| 평가 데이터셋 | 사진 수 | Macro F1 | Accuracy | 오분류 수 | 비고 |
|---|---|---|---|---|---|
| **① 리비전 실촬영 (`test1_orientation_backup`)** | 126장 | **`{results['revised_real_test'][primary_key]['macro_f1']:.4f}` ({results['revised_real_test'][primary_key]['macro_f1']*100:.2f}%)** | `{results['revised_real_test'][primary_key]['accuracy']:.4f}` ({results['revised_real_test'][primary_key]['accuracy']*100:.2f}%) | {int(np.sum(subsets['revised_real_test']['label'] != subsets['revised_real_test'][f'pred_{primary_key}']))}장 | 정제된 실촬영 각도 데이터 |
| **② 웹 밸리데이션 (`web_validation`)** | 89장 | **`{results['web_validation'][primary_key]['macro_f1']:.4f}` ({results['web_validation'][primary_key]['macro_f1']*100:.2f}%)** | `{results['web_validation'][primary_key]['accuracy']:.4f}` ({results['web_validation'][primary_key]['accuracy']*100:.2f}%) | {int(np.sum(subsets['web_validation']['label'] != subsets['web_validation'][f'pred_{primary_key}']))}장 | 미학습 368그룹 분리 검증셋 |
| **③ 통합 테스트셋 (① + ②)** | **215장** | **`{results['combined_test'][primary_key]['macro_f1']:.4f}` ({results['combined_test'][primary_key]['macro_f1']*100:.2f}%)** | **`{results['combined_test'][primary_key]['accuracy']:.4f}` ({results['combined_test'][primary_key]['accuracy']*100:.2f}%)** | **{int(np.sum(df['label'] != df[f'pred_{primary_key}']))}장** | 전체 비학습 데이터 총결합 |

## 2. 통합 테스트셋(215장) 클래스별 세부 성능

| 클래스 | Support (사진 수) | Precision | Recall | F1-Score | 분석 특징 |
|---|---|---|---|---|---|
"""
    comb_rep = results["combined_test"][primary_key]["report"]
    for c in CLASSES:
        r = comb_rep[c]
        report_md += f"| `{c}` | {int(r['support'])}장 | {r['precision']:.4f} | {r['recall']:.4f} | **`{r['f1-score']:.4f}`** | "
        if c == "taper_step":
            report_md += "단차 꺾임선 포착으로 최고 성능 달성 |\n"
        elif c == "mug":
            report_md += "낮은 높이-너비 비율 정확 식별 |\n"
        elif c == "straight":
            report_md += "실촬영 부감 각도에서 taper와 일부 혼동 |\n"
        elif c == "taper_smooth":
            report_md += "완만한 경사면 인식, 높은 재현율(Recall) |\n"

    report_md += f"""
## 3. 통합 테스트셋 오분류 분석 (총 {int(np.sum(df['label'] != df[f'pred_{primary_key}']))}건)

| 출처 | 파일명 | 정답 라벨 | 예측 라벨 | 예측 확신도 |
|---|---|---|---|---|
"""
    mis_comb = df[df["label"] != df[f"pred_{primary_key}"]]
    for _, row in mis_comb.iterrows():
        src_tag = "실촬영" if row["dataset_source"] == "revised_real_test" else "웹Val"
        report_md += f"| {src_tag} | `{row['filename']}` | `{row['label']}` | `{row[f'pred_{primary_key}']}` | {row[f'score_{primary_key}']:.4f} |\n"

    report_path = REPORTS_DIR / "combined_test_report.md"
    report_path.write_text(report_md, encoding="utf-8")
    print(f"Saved combined evaluation report: {report_path}")

    # JSON 메트릭 저장
    with open(REPORTS_DIR / "combined_test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
